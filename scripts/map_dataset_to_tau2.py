#!/usr/bin/env python
"""Map each Act-PRM dataset task (unique_data_sample_id) -> its tau2-bench task id, by
matching UNIQUE identifiers (reservation ids from the trajectory's tool-call args + the
served user_id) against each tau2 task's evaluation_criteria (ground-truth actions) +
known_info. Then report the COMPLEMENT: tau2 tasks NOT covered by the dataset — candidate
"never-seen" rl_eval tasks.

CPU-only (reads the cached parquet + tau2 registry) — safe to run alongside GPU training.

Usage (offline):
  UV_PROJECT_ENVIRONMENT=.venv-tau2 HF_HOME=/data/users/$USER/models/hf_cache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TAU2_DATA_DIR=$PWD/tau2-bench/data CUDA_VISIBLE_DEVICES= \
  uv run --no-sync python scripts/map_dataset_to_tau2.py --domain airline \
      --dataset mzio/aprm-tau2-airline-gpt5m_med-gs4-s0-train --out data/splits/tau2_airline_taskmap.json
"""
import argparse, glob, json, os, re, collections
from pathlib import Path

RES = re.compile(r"\b[A-Z0-9]{6}\b")                 # reservation id, e.g. XEHM4B
UID = re.compile(r"\b[a-z]+_[a-z]+_\d{3,4}\b")       # user id, e.g. aarav_garcia_1177


def res_ids(s: str) -> set:
    # keep alphanumeric mixes (drop pure-digit / pure-alpha 6-grams to cut noise)
    return {r for r in RES.findall(s) if any(c.isdigit() for c in r) and any(c.isalpha() for c in r)}


def arg_reservation_ids(tool_calls: list) -> set:
    """The reservation_id ARG values across a list of {name, arguments} tool calls —
    the task-target reservation(s) the agent/GT explicitly acted on (much cleaner than
    regex over observations/DB listings)."""
    out = set()
    for a in tool_calls or []:
        args = a.get("arguments") or {}
        rid = args.get("reservation_id")
        if isinstance(rid, str) and res_ids(rid):
            out.add(rid)
    return out


def flight_keys(tool_calls: list) -> set:
    """(flight_number, date) pairs from book/update tool-call args — unique per itinerary,
    so they disambiguate booking tasks that share a user_id and have no reservation_id."""
    out = set()
    for a in tool_calls or []:
        for f in (a.get("arguments") or {}).get("flights") or []:
            fn, dt = f.get("flight_number"), f.get("date")
            if fn:
                out.add((fn, dt))
    return out


def dataset_tool_calls(messages: list) -> list:
    """Parse assistant <tool_call>{...}</tool_call> JSON blocks into {name, arguments}."""
    calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for block in re.findall(r"<tool_call>(.*?)</tool_call>", m.get("content") or "", re.DOTALL):
            try:
                calls.append(json.loads(block.strip()))
            except Exception:
                pass
    return calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import act_prm.environments.tau2bench.env as _e  # noqa: F401 (installs tau2 voice stubs)
    from tau2 import registry
    from datasets import load_dataset

    tasks = registry.get_tasks_loader(args.domain)()
    splits = registry.get_task_splits_loader(args.domain)()
    split_of = {}
    for name, ids in splits.items():
        if name == "base":
            continue
        for i in ids:
            split_of[str(i)] = name

    # tau2 task identity: reservations + flights in the ground-truth eval actions + known_info user_id
    t_res, t_uid, t_flt = {}, {}, {}
    for t in tasks:
        d = t.model_dump()
        acts = (d.get("evaluation_criteria") or {}).get("actions") or []
        ki = (((d.get("user_scenario") or {}).get("instructions") or {}).get("known_info") or "")
        t_res[t.id] = arg_reservation_ids(acts)          # task-target reservation(s)
        t_flt[t.id] = flight_keys(acts)                  # task-target itinerary (booking tasks)
        m = UID.findall(ki)
        t_uid[t.id] = m[0] if m else None

    # dataset task identity: reservations + user_ids across the whole logged trajectory
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = args.dataset.split("/", 1)
    pq = sorted(glob.glob(str(Path(hf_home) / "hub" / f"datasets--{org}--{name}" / "snapshots" / "*" / "**" / "*.parquet"), recursive=True))
    ds = load_dataset("parquet", data_files=pq, split="train")
    d_res, d_uid = collections.defaultdict(set), collections.defaultdict(collections.Counter)
    d_flt = collections.defaultdict(set)
    for r in ds:
        u = r["unique_data_sample_id"]
        calls = dataset_tool_calls(list(r.get("state") or []) + [r.get("action")])
        d_res[u] |= arg_reservation_ids(calls)           # reservation_ids the agent acted on
        d_flt[u] |= flight_keys(calls)                   # itineraries the agent booked/changed
        for x in UID.findall(json.dumps(r.get("state"), default=str) + json.dumps(r.get("action"), default=str)):
            d_uid[u][x] += 1

    # match each dataset uid -> best tau2 task (reservation overlap weighted high; user_id as tiebreak)
    mapping, ambiguous = {}, {}
    for u in sorted(d_res):
        prim_uid = d_uid[u].most_common(1)[0][0] if d_uid[u] else None
        scored = []
        for t in tasks:
            s = 5 * len(d_res[u] & t_res[t.id]) + 4 * len(d_flt[u] & t_flt[t.id])
            if t_uid[t.id] and (t_uid[t.id] in d_uid[u]):
                s += 1
            if s > 0:
                scored.append((s, t.id))
        scored.sort(reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            mapping[u] = scored[0][1]
        elif scored:
            ambiguous[u] = [(t, s) for s, t in scored[:3]]

    all_ids = [t.id for t in tasks]
    # ROBUST covered set (does NOT need the per-uid bijection): a tau2 task is "seen" if
    # its unique target reservation-id OR itinerary appears in ANY log trajectory. rl_eval
    # candidates = the complement. Conservative: a task is unseen only if NEITHER matches.
    all_d_res = set().union(*d_res.values()) if d_res else set()
    all_d_flt = set().union(*d_flt.values()) if d_flt else set()
    covered_any = {t.id for t in tasks if (t_res[t.id] & all_d_res) or (t_flt[t.id] & all_d_flt)}
    unseen = [i for i in all_ids if i not in covered_any]  # RL-eval on these (never in the logs)
    out = {
        "domain": args.domain, "dataset": args.dataset,
        "n_dataset_tasks": len(d_res), "n_mapped": len(mapping),
        "n_ambiguous": len(ambiguous), "n_tau2_tasks": len(all_ids),
        "n_covered_any": len(covered_any),
        "uid_to_tau2_id": {str(k): v for k, v in mapping.items()},   # confident 1:1 (best-effort)
        "ambiguous": {str(k): v for k, v in ambiguous.items()},
        "covered_tau2_ids": sorted(covered_any),          # tasks present in the logs -> RL train
        "unseen_tau2_ids": unseen,                        # NOT in the logs -> RL eval (never-seen)
        "unseen_by_split": {s: [i for i in unseen if split_of.get(i) == s] for s in set(split_of.values())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"dataset tasks={len(d_res)} mapped={len(mapping)} ambiguous={len(ambiguous)} "
          f"(covered_any={len(covered_any)})")
    print(f"unseen tau2 tasks (candidate rl_eval): {len(unseen)} -> {unseen}")
    print(f"unseen by canonical split: { {s: len(v) for s,v in out['unseen_by_split'].items()} }")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
