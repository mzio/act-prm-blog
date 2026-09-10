#!/usr/bin/env python3
"""Export every rollout eval to CSV: one row per run, plus a per-arm summary.

Writes
  results/rollouts_all.csv      one row per rollout RUN (domain, arm, snapshot, seed, ...)
  results/rollouts_by_arm.csv   per (domain, arm, snapshot): n runs, mean, sd, pooled rate

Scoring follows scripts/score_rollouts.py and encodes the three corrections that each
silently corrupted a reading at some point:
  * rollouts_per_task.jsonl mixes eval AND train rows (rl.py generates train rollouts even
    under --no_train) -> filter split == "eval".
  * task_id is null on the snorkel envs (they store tasks as dicts) -> fall back to
    sample_id, else every task collapses into one bucket.
  * the snorkel gyms run negative_rewards=True, so a failure is -1.0, not 0.0. Summing
    final_reward gives a negative count; success is final_reward > 0.

Seeds matter differently per harness (see CLAUDE.md): the snorkel gyms are deterministic
given a seed, so same-seed reruns are duplicates and are marked is_dup=1 here. tau2 calls
an unseeded external user simulator, so its repeats are independent regardless of seed.
"""
import csv
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results")
os.makedirs(OUT, exist_ok=True)

# (glob, domain, harness). One entry per gym we ever rolled out in.
SOURCES = [
    ("logs/tau2bench_retail_rlvr/hf_qwen3_4b_instruct/retail_rollout_*/",       "retail",    "tau2"),
    ("logs/tau2bench_airline_rlvr/hf_qwen3_4b_instruct/airline_rollout_*/",     "airline",   "tau2"),
    ("logs/act_prm_snorkel_insurance_gym/hf_qwen3_4b_instruct/insurance_rollout_*/", "insurance", "snorkel"),
    ("logs/act_prm_snorkel_finance_gym/hf_qwen3_4b_instruct/finance_rollout_*/",     "finance",   "snorkel"),
]

ARMS = ["thoughts_policy_adamw30", "thoughts_base_adamw30", "expert_thoughts_all",
        "expert_thoughts", "thoughts_policy_g8top1", "thoughts_policy_g8top2",
        "thoughts_policy_g8top4", "thoughts_policy_g8top8", "thoughts_policy_1gen",
        "thoughts_policy", "thoughts_base", "actions_only", "BASE"]


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def parse_tag(tag, domain):
    """(arm, snapshot, seed) from a run tag. Longest arm name wins so that e.g.
    thoughts_policy_adamw30 is not shadowed by thoughts_policy."""
    body = re.sub(rf"^{domain}_rollout_", "", tag)
    arm = next((a for a in sorted(ARMS, key=len, reverse=True) if body.startswith(a)), "")
    snap = "step_best"
    m = re.search(r"(step_\d+|step_best|step_last)", body)
    if m:
        snap = m.group(1)
    # seed appears either as an explicit _sNNNN / _seedNNNN tag or in the run dir (s=NN)
    seed = None
    m = re.search(r"_s(?:eed)?(\d{3,})", body)
    if m:
        seed = int(m.group(1))
    # RECIPE = the checkpoint generation this rollout used (everything between the arm
    # name and the step_* suffix). Without it, rows from different Stage-2 runs collapse
    # together -- e.g. retail thoughts_policy_adamw30 at seed 42 spans the SGD 1e-3 arm
    # (~31%), the AdamW 1e-3 arm (~12%) and a 4-task smoke, which is meaningless pooled.
    rest = body[len(arm):].lstrip("_") if arm and body.startswith(arm) else body
    rest = re.sub(r"_?(step_\d+|step_best|step_last).*$", "", rest)
    # strip the seed token whether or not a leading underscore survived the lstrip above
    rest = re.sub(r"_?s(?:eed)?\d{3,}", "", rest).strip("_")
    # The seed chains (chain_extra_seeds.sh / chain_tau2_seeds.sh) tagged insurance runs
    # as <arm>_seed<N>_step_0020 with no recipe token, but every one of them passed
    # CKPT_PAT=lr1e_3_nb200_flat32sgd -- i.e. the same checkpoints as the sgdlr1e_3 runs.
    # Without this they land in a separate group and the per-arm n stays at 1.
    if not rest and re.search(r"_s(?:eed)?\d{3,}_step_", body):
        rest = "sgdlr1e_3"
    return arm or body, snap, seed, (rest or "default")


def score(d):
    p = os.path.join(d, "rollouts_per_task.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    ev = [r for r in rows if r.get("split") == "eval"]
    seen = {}
    for r in ev:
        k = r.get("task_id")
        k = str(k) if k is not None else f's{r.get("sample_id")}'
        seen.setdefault(k, r["final_reward"])
    v = list(seen.values())
    return sum(1 for x in v if x > 0), len(v)


def main():
    runs = []
    for pat, domain, harness in SOURCES:
        for d in sorted(glob.glob(os.path.join(REPO, pat))):
            base = os.path.basename(d.rstrip("/"))
            tag = base.split("-act-prm")[0]
            sc = score(d)
            if not sc or sc[1] == 0:
                continue
            k, n = sc
            arm, snap, seed, recipe = parse_tag(tag, domain)
            m = re.search(r"-s=(\d+)", base)
            if seed is None and m:
                seed = int(m.group(1))
            lo, hi = wilson(k, n)
            runs.append(dict(domain=domain, harness=harness, arm=arm, recipe=recipe,
                             is_smoke=1 if ("smoke" in tag or n <= 6) else 0, snapshot=snap,
                             seed=seed if seed is not None else 42, solved=k, n_tasks=n,
                             pct=round(100 * k / n, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                             run_tag=tag))

    # snorkel is deterministic given a seed -> flag same-(domain,arm,snapshot,seed) repeats
    cnt = defaultdict(int)
    for r in sorted(runs, key=lambda x: x["run_tag"]):
        key = (r["domain"], r["arm"], r["recipe"], r["snapshot"], r["seed"])
        cnt[key] += 1
        r["is_dup"] = 1 if (r["harness"] == "snorkel" and cnt[key] > 1) else 0

    runs.sort(key=lambda r: (r["domain"], r["arm"], r["recipe"], r["snapshot"], r["seed"]))
    cols = ["domain", "harness", "arm", "recipe", "snapshot", "seed", "solved", "n_tasks",
            "pct", "ci_lo", "ci_hi", "is_dup", "is_smoke", "run_tag"]
    f1 = os.path.join(OUT, "rollouts_all.csv")
    with open(f1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(runs)

    # per-arm summary over INDEPENDENT samples only (drop snorkel same-seed dups)
    g = defaultdict(list)
    for r in runs:
        if r["is_dup"] or r["is_smoke"]:
            continue
        g[(r["domain"], r["arm"], r["recipe"], r["snapshot"])].append(r)
    f2 = os.path.join(OUT, "rollouts_by_arm.csv")
    with open(f2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "arm", "recipe", "snapshot", "n_runs", "mean_pct", "sd_pct",
                    "min_pct", "max_pct", "pooled_solved", "pooled_n", "pooled_pct", "seeds"])
        for (dom, arm, recipe, snap), rs in sorted(g.items()):
            p = [r["pct"] for r in rs]
            ks, ns = sum(r["solved"] for r in rs), sum(r["n_tasks"] for r in rs)
            w.writerow([dom, arm, recipe, snap, len(rs), round(st.mean(p), 2),
                        round(st.pstdev(p), 2) if len(p) > 1 else "",
                        min(p), max(p), ks, ns, round(100 * ks / ns, 2),
                        " ".join(str(r["seed"]) for r in sorted(rs, key=lambda x: x["seed"]))])

    print(f"  wrote {f1}  ({len(runs)} runs)")
    print(f"  wrote {f2}  ({len(g)} arm/snapshot groups)")


if __name__ == "__main__":
    main()
