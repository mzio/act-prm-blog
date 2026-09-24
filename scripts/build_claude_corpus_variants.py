#!/usr/bin/env python3
"""Split a Claude-teacher collection into train/eval tasks and emit the arm variants.

    uv run python scripts/build_claude_corpus_variants.py \
        --corpus /tmp/ins_claude_all --domain snorkel_insurance --eval_frac 0.15

Produces, under ``data/<domain>_claude/``:
    split.json              task-level train / eval partition (seeded, reproducible)
    expert_thoughts/        assistant content = thought + <tool_call>   (as collected)
    actions_only/           assistant content = <tool_call> only        (thoughts stripped)

``actions_only`` doubles as the **Stage-1 EM input**: the Act-PRM E-step infers a thought
for every logged action, so it must not see the teacher's. (In the ``act_prm_traces`` env
this is the ``keep_expert_thoughts=False`` path -- ``environments/act_prm_traces/data.py:47``
-- but Stage-2 SFT reads a fixed ``--dataset_path`` corpus VERBATIM, so the stripped form
has to exist on disk too.)

Both variants share one task partition so the arms stay comparable: the same task is in
train for every variant, or in eval for every variant. Splitting per-variant would let an
arm train on a task another arm is evaluated on.

The eval tasks are held out for ROLLOUTS. A further step-wise split (state-action tuples
within the train tasks, for PPL / action-accuracy) is a separate concern and is NOT done
here -- see --help on the follow-up script when it exists.
"""
import argparse
import collections
import json
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOS = "<tool_call>"


def strip_thought(content):
    """Action-only content: everything from the first <tool_call> on."""
    i = content.find(BOS)
    return content[i:].strip() if i >= 0 else content.strip()


def variant(trajs, keep_thoughts):
    out = []
    for t in trajs:
        msgs = []
        for m in t["messages"]:
            c = m.get("content") or ""
            if m.get("role") == "assistant" and BOS in c and not keep_thoughts:
                c = strip_thought(c)
            msgs.append({"role": m["role"], "content": c})
        out.append({**t, "messages": msgs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir from export_teacher_corpus.py")
    ap.add_argument("--domain", required=True, help="e.g. snorkel_insurance / tau2_retail")
    ap.add_argument("--eval_frac", type=float, default=0.15,
                    help="fraction of TASKS held out for rollout eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default="data")
    args = ap.parse_args()

    trajs = []
    for s in ("train", "eval"):
        fp = os.path.join(args.corpus, f"{s}.json")
        if os.path.exists(fp):
            trajs += json.load(open(fp))
    if not trajs:
        raise SystemExit(f"no trajectories in {args.corpus}")

    # One trajectory per uid is the expected shape (export_teacher_corpus --per_task 1).
    # If several share a uid they must not straddle the split, so partition by uid.
    by_uid = collections.defaultdict(list)
    for t in trajs:
        by_uid[str(t["uid"])].append(t)
    uids = sorted(by_uid, key=lambda u: (len(u), u))
    rng = random.Random(args.seed)
    rng.shuffle(uids)
    n_eval = max(1, round(len(uids) * args.eval_frac))
    eval_uids, train_uids = set(uids[:n_eval]), set(uids[n_eval:])

    root = os.path.join(args.out_root, f"{args.domain}_claude")
    os.makedirs(root, exist_ok=True)
    split = {
        "note": ("Task-level split of the Claude-teacher corpus. eval tasks are held out "
                 "for ROLLOUT eval and are never trained on by any arm."),
        "source_corpus": args.corpus,
        "seed": args.seed,
        "eval_frac": args.eval_frac,
        "n_tasks": len(uids),
        "train_tasks": sorted(train_uids, key=lambda u: (len(u), u)),
        "eval_tasks": sorted(eval_uids, key=lambda u: (len(u), u)),
    }
    with open(os.path.join(root, "split.json"), "w") as f:
        json.dump(split, f, indent=2)
    print(f"  split: {len(train_uids)} train / {len(eval_uids)} eval tasks -> {root}/split.json")

    for name, keep in (("expert_thoughts", True), ("actions_only", False)):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        for sname, uset in (("train", train_uids), ("eval", eval_uids)):
            sel = [t for u in sorted(uset, key=lambda x: (len(x), x)) for t in by_uid[u]]
            data = variant(sel, keep)
            with open(os.path.join(d, f"{sname}.json"), "w") as f:
                json.dump(data, f)
            steps = sum(
                1 for t in data for m in t["messages"]
                if m.get("role") == "assistant" and BOS in (m.get("content") or "")
            )
            withz = sum(
                1 for t in data for m in t["messages"]
                if m.get("role") == "assistant" and BOS in (m.get("content") or "")
                and len((m["content"][: m["content"].find(BOS)]).strip()) >= 10
            )
            print(f"    {name:16s} {sname:5s}: {len(data):4d} traj  {steps:5d} steps  "
                  f"thought={100*withz/max(1,steps):5.1f}%")
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({
                "built_from": args.corpus,
                "variant": name,
                "keep_thoughts": keep,
                "split_file": os.path.join(root, "split.json"),
                "note": ("actions_only is ALSO the Stage-1 EM input: the E-step must infer "
                         "thoughts, so it must not see the teacher's."
                         if not keep else
                         "assistant content = teacher thought + <tool_call>."),
            }, f, indent=2)
    print(f"  done -> {root}/{{expert_thoughts,actions_only}}")


if __name__ == "__main__":
    main()
