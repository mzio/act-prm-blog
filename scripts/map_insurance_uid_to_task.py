#!/usr/bin/env python
"""Map Act-PRM insurance uids -> gym `company_task_id`, so the rollout can target the
held-out 41 questions.

The pools key trajectories by `unique_data_sample_id` (0..261, from the GPT-5-mini rollout
dataset); the gym keys tasks by `company_task_id` (300 entries in
data/snorkel_insurance/task_data/downsampled_task_set.json). Without a mapping the rollout
falls back to the env's own fractional split and silently evaluates the wrong tasks -- which
is exactly what happened on finance, where an undeclared `eval_query_ids` key meant all ten
runs scored the env's 11-question frac split instead of the intended fair-10/hard-29.

The HF prompt is rendered FROM the gym's user_information, so the join is on content:
company name (exact) plus the LOB and the user task text to disambiguate the several tasks
that share a company. Anything that does not resolve to exactly one candidate is reported,
not guessed.

Writes data/splits/snorkel_insurance_uid_to_task.json.

Usage: uv run --no-project python scripts/map_insurance_uid_to_task.py
"""
import json
import re
from collections import defaultdict


def field(text: str, name: str) -> str | None:
    m = re.search(rf"^{re.escape(name)}:\s*(.+)$", text or "", re.M)
    return m.group(1).strip() if m else None


def main() -> None:
    tasks = json.load(open("data/snorkel_insurance/task_data/downsampled_task_set.json"))
    by_company = defaultdict(list)
    for t in tasks:
        ui = t.get("user_information") or {}
        if isinstance(ui, str):
            try:
                ui = json.loads(ui.replace("'", '"'))
            except Exception:
                ui = {}
        by_company[str(ui.get("company_name", "")).strip()].append((t, ui))

    pools = {}
    for split in ("train", "eval"):
        for r in json.load(open(f"data/snorkel_insurance_split/{split}.json")):
            pools.setdefault(r["uid"], (split, r))

    # The ROLLOUT questions are deliberately absent from the pools (never seen by Stage 1
    # or Stage 2), so their prompts have to come from the source dataset -- otherwise the
    # very questions the rollout targets are the only ones left unmapped.
    sp0 = json.load(open("data/splits/snorkel_insurance.json"))
    need = [q for q in sp0["rollout_questions"] if q not in pools]
    if need:
        from datasets import load_dataset
        d = load_dataset(sp0["source"], split="train")
        want = set(need)
        for i in range(len(d)):
            uid = d[i]["unique_data_sample_id"]
            if uid in want and uid not in pools:
                pools[uid] = ("rollout", {"messages": list(d[i]["state"])})
                want.discard(uid)
                if not want:
                    break
        print(f"pulled {len(need) - len(want)}/{len(need)} rollout prompts from {sp0['source']}")

    mapping, ambiguous, missing = {}, [], []
    for uid, (split, r) in sorted(pools.items()):
        user = next((m["content"] for m in r["messages"] if m.get("role") == "user"), "")
        company = field(user, "Company")
        lob = (field(user, "Line of Business (LOB)") or "").lower()
        cands = by_company.get(company or "", [])
        if not cands:
            missing.append((uid, company))
            continue
        if len(cands) > 1 and lob:
            narrowed = [c for c in cands if str(c[1].get("lob", "")).lower() == lob]
            if narrowed:
                cands = narrowed
        if len(cands) > 1:
            # last resort: the task text is echoed verbatim at the end of the HF prompt
            narrowed = [c for c in cands if (c[0].get("user_task") or "")[:40] in user]
            if narrowed:
                cands = narrowed
        if len(cands) == 1:
            mapping[str(uid)] = int(cands[0][0]["company_task_id"])
        else:
            ambiguous.append((uid, company, [int(c[0]["company_task_id"]) for c in cands]))

    sp = json.load(open("data/splits/snorkel_insurance.json"))
    out = {
        "note": ("uid -> gym company_task_id. Joined on company name + LOB + user_task text; "
                 "the HF prompt is rendered from the gym's user_information."),
        "uid_to_task_id": mapping,
        "n_mapped": len(mapping),
        "n_ambiguous": len(ambiguous),
        "n_missing": len(missing),
        "ambiguous": ambiguous[:20],
        "missing": missing[:20],
    }
    for name in ("train", "eval", "rollout"):
        qs = sp[f"{name}_questions"]
        ids = [mapping[str(q)] for q in qs if str(q) in mapping]
        out[f"{name}_task_ids"] = ids
        out[f"{name}_unmapped"] = [q for q in qs if str(q) not in mapping]

    json.dump(out, open("data/splits/snorkel_insurance_uid_to_task.json", "w"), indent=2)
    print(f"mapped {len(mapping)}/{len(pools)} uids   ambiguous {len(ambiguous)}   missing {len(missing)}")
    for name in ("train", "eval", "rollout"):
        print(f"  {name:8}: {len(out[f'{name}_task_ids'])}/{len(sp[f'{name}_questions'])} mapped"
              + (f"   UNMAPPED {out[f'{name}_unmapped'][:8]}" if out[f"{name}_unmapped"] else ""))
    if ambiguous:
        print("\n  ambiguous (first 5):")
        for uid, c, ids in ambiguous[:5]:
            print(f"    uid {uid}  company={c!r}  candidates={ids}")
    if missing:
        print("\n  missing (first 5):", missing[:5])
    # the rollout ids must be disjoint from everything the model trains on
    tr, ev, ro = set(out["train_task_ids"]), set(out["eval_task_ids"]), set(out["rollout_task_ids"])
    print(f"\n  task-id overlap  train&rollout {len(tr&ro)}   eval&rollout {len(ev&ro)}   train&eval {len(tr&ev)}")
    print("  wrote data/splits/snorkel_insurance_uid_to_task.json")


if __name__ == "__main__":
    main()
