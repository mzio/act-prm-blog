#!/usr/bin/env python
"""Offline re-grade of every recorded finance rollout answer. No GPU, no re-rollout.

The env captures the judge's text into `grader_text` and then throws it away, so the six
completed runs kept only a pass/fail bit -- and that bit was produced by a parser that
defaulted to "no" whenever the judge did not emit a literal `correct:` line (see
act_prm/graders/snorkel_finance.parse_verdict). Every parse failure biases the score DOWN.

This walks the rollout replay buffers, recovers (question, gold answer, model response) for
each episode that actually answered, re-runs the SAME judge, and records BOTH parses of the
same judge text:
  verdict_old -- the strict `correct:\\s*(yes|no)` regex, defaulting to "no"
  verdict_new -- the fall-through parser, plus whether anything matched at all
so the cost of the bug is measurable rather than assumed.

Also emits, per answer, whether the numeric figures in the gold answer appear in the model's
response. That is the concrete version of "the policy never computes the derived quantity" --
it can be checked per row in the viewer instead of taken on faith from one example.

Writes data/finance_regrade.jsonl (one row per answered episode).

Usage:
  export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080
  CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync \\
      python scripts/regrade_finance.py [--limit N]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers(text: str) -> list[str]:
    """Numeric tokens, normalised so 1,234.50 == 1234.5 and trailing zeros don't matter."""
    out = []
    for tok in NUM.findall(text or ""):
        t = tok.replace(",", "").rstrip(".")
        if not t or t in {"-"}:
            continue
        try:
            f = float(t)
        except ValueError:
            continue
        out.append(f"{f:g}")
    return out


def extract(limit: int | None):
    from datasets import load_from_disk

    rows = list(csv.DictReader(open("data/snorkel_finance/benchmark/finqa_reasoning.csv")))
    by_q = {r["question"].strip(): r["answer"] for r in rows}

    recs = []
    bufs = sorted(
        glob.glob(
            "checkpoints_lora/act_prm_snorkel_finance_gym/hf_qwen3_4b_instruct/"
            "finance_rollout_*_v3_*/replay_buffer"
        ),
        key=os.path.getmtime,
    )
    for buf in bufs:
        tag = buf.split("/")[-2].split("-act-prm")[0]
        m = re.match(r"finance_rollout_(.+)_v3_(fair|hard)$", tag)
        if not m:
            continue
        arm, setname = m.group(1), m.group(2)
        try:
            ds = load_from_disk(buf)
        except Exception:
            continue
        seen = set()
        for row in ds:
            if row.get("is_train"):
                continue
            key = (row.get("sample_id"), row.get("generation_id"))
            if key in seen:
                continue
            seen.add(key)

            q = None
            for msg in row.get("state") or []:
                mm = re.search(r"here is the question\s*:\s*(.+)", msg.get("content") or "", re.S)
                if mm:
                    q = mm.group(1).strip()
                    break
            if not q:
                continue
            if q not in by_q:
                q = next((qq for qq in by_q if qq[:80] == q[:80]), None)
                if not q:
                    continue

            resp, n_turns = None, 0
            for msg in row.get("final_outcome") or []:
                if msg.get("role") != "assistant":
                    continue
                n_turns += 1
                if '"respond_user"' in (msg.get("content") or ""):
                    mm = re.search(r'"text"\s*:\s*"(.*?)"\s*}\s*}', msg["content"], re.S)
                    if mm:
                        resp = mm.group(1).encode().decode("unicode_escape", "ignore")
            recs.append({
                "arm": arm, "set": setname, "sample_id": row.get("sample_id"),
                "generation_id": row.get("generation_id"), "question": q,
                "gold": by_q[q], "response": resp, "answered": resp is not None,
                "n_assistant_turns": n_turns,
            })
            if limit and len(recs) >= limit:
                return recs
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap episodes scanned (debug)")
    ap.add_argument("--out", default="data/finance_regrade.jsonl")
    args = ap.parse_args()

    recs = extract(args.limit)
    answered = [r for r in recs if r["answered"]]
    print(f"episodes: {len(recs)}   answered: {len(answered)}", flush=True)
    if not answered:
        print("nothing to re-grade")
        return 1

    from act_prm.graders.snorkel_finance import SnorkelFinanceGrader, parse_verdict

    grader = SnorkelFinanceGrader(
        grader_model_config={"name": "claude_agent_sdk",
                             "model_config": {"model": "claude-sonnet-4-5", "max_turns": 1}},
        num_samples=1, verbose=False,
    )
    OLD = re.compile(r"correct:\s*(yes|no)\b", re.IGNORECASE)

    for i, r in enumerate(answered, 1):
        try:
            _, raw = grader.grade_sample(
                question=r["question"], correct_answer=r["gold"], response=r["response"])
        except Exception as e:  # a judge outage must not look like a wrong answer
            r["judge_error"] = str(e)[:200]
            raw = ""
        om = OLD.search(raw or "")
        vn, parsed = parse_verdict(raw)
        r["raw_judge"] = raw
        r["verdict_old"] = om.group(1).lower() if om else "no"
        r["old_parsed"] = bool(om)
        r["verdict_new"] = vn
        r["new_parsed"] = parsed
        g, m = numbers(r["gold"]), set(numbers(r["response"]))
        r["gold_numbers"] = g
        r["gold_numbers_found"] = [x for x in g if x in m]
        print(f"  [{i}/{len(answered)}] {r['arm']}/{r['set']} old={r['verdict_old']}"
              f"({'ok' if om else 'DEFAULT'}) new={vn}({'ok' if parsed else 'DEFAULT'})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    flips = [r for r in answered if r["verdict_old"] != r["verdict_new"]]
    unparsed_old = [r for r in answered if not r["old_parsed"]]
    print(f"\nwrote {args.out}  ({len(recs)} episodes, {len(answered)} answered)")
    print(f"  old parser failed to find a verdict : {len(unparsed_old)}/{len(answered)}")
    print(f"  verdict CHANGED by the new parser   : {len(flips)}/{len(answered)}")
    for v in ("yes", "no"):
        print(f"  new verdict '{v}': {sum(1 for r in answered if r['verdict_new']==v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
