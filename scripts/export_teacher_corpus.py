#!/usr/bin/env python3
"""Turn Claude-teacher rollouts into an SFT corpus, keeping only SUCCESSFUL trajectories.

    uv run python scripts/export_teacher_corpus.py \
        --runs "logs/act_prm_snorkel_insurance_gym/hf_qwen3_4b_instruct/insurance_teacher_*/" \
        --out data/sft_corpus/snorkel_insurance/claude_s46

Reads each run's ``trajectories.jsonl`` (written when ACT_PRM_DUMP_TRAJECTORIES=1) and
emits ``train.json`` / ``eval.json`` in the same shape as every other corpus in
``data/sft_corpus/**`` -- ``{uid, generation_id, system_prompt, messages}``, messages
starting at the first user turn -- so the existing Stage-2 path consumes it unchanged.

NOT filtered by split. ``rl.py`` produces both train- and eval-split rollouts, and we want
demonstrations for EVERY sample, so ``split`` only decides which output file a trajectory
lands in. (The split filter belongs in the rollout *scoring* path, where mixing the two
inflates a success denominator -- a different job; see export_rollout_results.py.)

Success is ``final_reward > 0``, not ``!= 0``: the snorkel gyms run negative_rewards=True,
so a failure is -1.0 and any sum- or truthiness-based test miscounts.

Selection: one trajectory per (split, task_id) by default, the successful one with the
FEWEST steps -- the cleanest demonstration, and it matches the existing corpora's one
trajectory per question. ``--per_task 0`` keeps every success instead.
"""
import argparse
import collections
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_runs(patterns):
    rows = []
    for pat in patterns:
        for d in sorted(glob.glob(pat if os.path.isabs(pat) else os.path.join(REPO, pat))):
            fp = os.path.join(d, "trajectories.jsonl")
            if not os.path.exists(fp):
                continue
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["_run"] = os.path.basename(d.rstrip("/"))
                rows.append(r)
    return rows


def to_corpus_entry(r):
    """Trajectory record -> corpus trajectory. Drops the leading system message: the
    corpus keeps the system prompt in its own field and starts messages at the user turn.
    """
    msgs = [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in r.get("messages", [])
        if m.get("role") != "system"
    ]
    sysp = r.get("system_prompt") or next(
        (m.get("content") for m in r.get("messages", []) if m.get("role") == "system"), ""
    )
    return {
        "uid": str(r.get("task_id")),
        "generation_id": r.get("gen_id", 0),
        "system_prompt": sysp,
        "messages": msgs,
    }


def n_tool_calls(r):
    return sum(
        1 for m in r.get("messages", [])
        if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or "")
    )


def thought_frac(r):
    tc = [m for m in r.get("messages", [])
          if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or "")]
    if not tc:
        return 0.0
    w = sum(1 for m in tc
            if len(m["content"][: m["content"].find("<tool_call>")].strip()) >= 10)
    return w / len(tc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="glob(s) matching run dirs that contain trajectories.jsonl")
    ap.add_argument("--out", required=True, help="output corpus dir")
    ap.add_argument("--per_task", type=int, default=1,
                    help="successes to keep per (split, task); 0 = keep all")
    ap.add_argument("--min_thought_frac", type=float, default=0.0,
                    help="drop successes whose tool calls carry a thought less often than this")
    args = ap.parse_args()

    rows = load_runs(args.runs)
    if not rows:
        raise SystemExit(f"no trajectories.jsonl found under {args.runs}")

    total = len(rows)
    wins = [r for r in rows if (r.get("final_reward") or 0) > 0]
    kept_frac = [r for r in wins if thought_frac(r) >= args.min_thought_frac]
    dropped_thin = len(wins) - len(kept_frac)

    by_task = collections.defaultdict(list)
    for r in kept_frac:
        by_task[(r.get("split") or "train", str(r.get("task_id")))].append(r)

    out = collections.defaultdict(list)
    for (split, _task), cand in sorted(by_task.items()):
        cand.sort(key=n_tool_calls)  # fewest steps = cleanest demonstration
        chosen = cand if args.per_task <= 0 else cand[: args.per_task]
        out[split].extend(to_corpus_entry(r) for r in chosen)

    attempts = collections.defaultdict(int)
    solved = collections.defaultdict(int)
    for r in rows:
        k = (r.get("split") or "train", str(r.get("task_id")))
        attempts[k] += 1
        solved[k] += (r.get("final_reward") or 0) > 0
    covered = sum(1 for k in attempts if solved[k] > 0)

    os.makedirs(args.out, exist_ok=True)
    for split, entries in out.items():
        fp = os.path.join(args.out, f"{split}.json")
        with open(fp, "w") as f:
            json.dump(entries, f)
        print(f"  wrote {fp}  ({len(entries)} trajectories)")

    meta = {
        "source_runs": sorted({r["_run"] for r in rows}),
        "n_trajectories_seen": total,
        "n_successful": len(wins),
        "n_dropped_low_thought_frac": dropped_thin,
        "min_thought_frac": args.min_thought_frac,
        "per_task": args.per_task,
        "tasks_seen": len(attempts),
        "tasks_with_at_least_one_success": covered,
        "task_coverage": round(covered / max(1, len(attempts)), 4),
        "counts_by_split": {k: len(v) for k, v in out.items()},
        "selection": "fewest tool calls among successes",
        "success_rule": "final_reward > 0 (snorkel gyms use negative_rewards, failure = -1.0)",
        "note": "Claude-teacher demonstrations: assistant content = thought + <tool_call>.",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  success {len(wins)}/{total} trajectories; "
          f"task coverage {covered}/{len(attempts)} ({100*covered/max(1,len(attempts)):.1f}%)")
    if dropped_thin:
        print(f"  dropped {dropped_thin} successes below --min_thought_frac {args.min_thought_frac}")
    missing = [k for k in attempts if solved[k] == 0]
    if missing:
        print(f"  {len(missing)} task(s) with NO success -- re-run those with more attempts:")
        print("    " + " ".join(sorted(t for _s, t in missing))[:300])


if __name__ == "__main__":
    main()
