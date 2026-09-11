#!/usr/bin/env python3
"""Export every training run's metrics to CSV.

Writes
  results/training_runs.csv    one row per run: config + best/final eval metrics
  results/training_curves.csv  one row per (run, eval batch): the learning curves

Reads logs/<env>/<model>/<run_tag>-*/{config.json,metrics.jsonl}, which are gitignored
runtime artifacts -- these CSVs are the committable summary of them.

The metric of record is eval_actiononly_ppl: perplexity over the ACTION sub-span only.
That is the only span comparable across arms, because actions_only has no reasoning
prefix while the thought arms' whole-target span includes tokens the baseline never has
to model. eval_action_ppl (no "only") is the whole thought+action span and is NOT
comparable -- do not mix them.

optimizer/learning_rate come from each run's own config.json, so SGD and AdamW
generations are always distinguishable (they were conflated once when inferred from
run-tag strings: several tags carry no "sgd"/"adamw" marker at all).
"""
import csv
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results")
os.makedirs(OUT, exist_ok=True)

CFG_KEYS = ["run_tag", "env_config", "trainer_name", "optimizer", "learning_rate",
            "num_batches", "steps_per_batch", "batch_size", "group_size", "lora_config",
            "dataset_path", "hide_observations", "require_thought", "advantage_mode",
            "length_penalty", "score_with_base", "best_metric", "seed",
            "keep_expert_thoughts", "infer_thoughts", "no_train"]

# eval metric -> short column name
EVAL = {
    "eval/eval_actiononly_ppl": "eval_actiononly_ppl",
    "eval/eval_actiononly_accuracy": "eval_actiononly_acc",
    "eval/eval_action_accuracy": "eval_action_acc",
    "eval/eval_ppl": "eval_ppl",
    "eval/eval_n_action_tokens": "eval_n_action_tokens",
    "eval/eval_n_label_tokens": "eval_n_label_tokens",
}


def flat(o, p=""):
    out = {}
    for k, v in (o or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def stage_of(tag, cfg):
    """s1em / s1relabel / s2 / rollout, from the run tag (falls back to config)."""
    for s in ("s1em", "s1relabel", "s2", "rollout"):
        if f"_{s}_" in tag or tag.endswith(f"_{s}"):
            return s
    return "rollout" if cfg.get("no_train") else "other"


def main():
    runs, curves = [], []
    for cfgf in sorted(glob.glob(os.path.join(REPO, "logs/*/*/*/config.json"))):
        d = os.path.dirname(cfgf)
        mf = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(mf):
            continue
        try:
            cfg = flat(json.load(open(cfgf)))
        except Exception:
            continue
        rows = []
        for l in open(mf):
            l = l.strip()
            if l:
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
        if not rows:
            continue
        base = os.path.basename(d)
        tag = cfg.get("run_tag") or base.split("-act-prm")[0]
        env = d.split(os.sep)[-3]
        rec = {"env_dir": env, "stage": stage_of(str(tag), cfg), "log_dir": base}
        for k in CFG_KEYS:
            rec[k] = cfg.get(k, "")
        ev = [(r.get("progress/batch"), r) for r in rows if "eval/eval_actiononly_ppl" in r]
        rec["n_evals"] = len(ev)
        rec["last_batch"] = rows[-1].get("progress/batch", "")
        if ev:
            b_best, r_best = min(ev, key=lambda x: x[1]["eval/eval_actiononly_ppl"])
            rec["best_actiononly_ppl"] = round(r_best["eval/eval_actiononly_ppl"], 4)
            rec["best_batch"] = b_best
            rec["first_actiononly_ppl"] = round(ev[0][1]["eval/eval_actiononly_ppl"], 4)
            rec["final_actiononly_ppl"] = round(ev[-1][1]["eval/eval_actiononly_ppl"], 4)
            rec["eval_n_action_tokens"] = r_best.get("eval/eval_n_action_tokens", "")
            for b, r in ev:
                row = {"log_dir": base, "run_tag": tag, "env_dir": env,
                       "stage": rec["stage"], "optimizer": rec["optimizer"],
                       "learning_rate": rec["learning_rate"], "batch": b}
                for src, dst in EVAL.items():
                    v = r.get(src)
                    row[dst] = round(v, 6) if isinstance(v, float) else (v if v is not None else "")
                curves.append(row)
        runs.append(rec)

    if runs:
        cols = (["env_dir", "stage"] + CFG_KEYS +
                ["n_evals", "last_batch", "best_batch", "best_actiononly_ppl",
                 "first_actiononly_ppl", "final_actiononly_ppl", "eval_n_action_tokens",
                 "log_dir"])
        f1 = os.path.join(OUT, "training_runs.csv")
        with open(f1, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(runs, key=lambda r: (r["env_dir"], r["stage"], str(r["run_tag"]))))
        print(f"  wrote {f1}  ({len(runs)} runs)")
    if curves:
        f2 = os.path.join(OUT, "training_curves.csv")
        cols2 = ["env_dir", "stage", "run_tag", "optimizer", "learning_rate", "batch"] + list(EVAL.values()) + ["log_dir"]
        with open(f2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols2, extrasaction="ignore")
            w.writeheader()
            w.writerows(curves)
        print(f"  wrote {f2}  ({len(curves)} eval points)")


if __name__ == "__main__":
    main()
