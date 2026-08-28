#!/usr/bin/env python
"""Replay the finance LLM judge offline on recorded (question, gold, response) triples.

Why: every finance arm scores 0, and the env throws `grader_text` away -- it is captured
from the grader and never stored or logged -- so the judge's actual verdicts are not
recoverable from a finished run. This re-runs the SAME grader on answers pulled from the
replay buffers and prints the RAW judge output next to the parsed verdict.

The specific worry: SnorkelFinanceGrader parses `correct:\\s*(yes|no)` and defaults to "no"
on any parse failure, but the system prompt is self-contradictory -- it asks for
    correct: yes/no
    rationale: ...
while its own worked example shows
    Your response : {True, 'Yes they both match'}
A judge that follows the example produces text the regex cannot match, and every answer is
silently graded incorrect. That would make 0/10 an artifact rather than a finding. This
script distinguishes the two: it prints whether the regex matched, so a systematic
parse failure is visible.

Also included: a CONTROL where the gold answer is submitted as the model response. If the
judge marks the gold answer itself incorrect, grading is broken; if it marks it correct,
the judge works and the arms genuinely fail.

Usage: UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python scripts/probe_finance_grader.py [--n 5]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys


def load_triples(n: int):
    """(question, gold, model_response) from the newest finance rollout replay buffers."""
    from datasets import load_from_disk

    rows = list(csv.DictReader(open("data/snorkel_finance/benchmark/finqa_reasoning.csv")))
    by_q = {r["question"].strip(): r["answer"] for r in rows}

    out = []
    bufs = sorted(
        glob.glob(
            "checkpoints_lora/act_prm_snorkel_finance_gym/hf_qwen3_4b_instruct/"
            "finance_rollout_*_v3_fair-*/replay_buffer"
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    seen = set()
    for buf in bufs:
        arm = buf.split("/")[-2].split("-act-prm")[0].replace("finance_rollout_", "")
        try:
            ds = load_from_disk(buf)
        except Exception:
            continue
        for row in ds:
            if row.get("is_train"):
                continue
            msgs = row.get("final_outcome") or []
            resp = None
            for m in reversed(msgs):
                if m.get("role") == "assistant" and '"respond_user"' in (m.get("content") or ""):
                    mm = re.search(r'"text"\s*:\s*"(.*?)"\s*}\s*}', m["content"], re.S)
                    if mm:
                        resp = mm.group(1).encode().decode("unicode_escape", "ignore")
                    break
            if not resp:
                continue
            # The question is a user message in `state`, prefixed by the company:
            #   "For company `at_t`, here is the question : <question>"
            q = None
            for m in row.get("state") or []:
                c = m.get("content") or ""
                mm = re.search(r"here is the question\s*:\s*(.+)", c, re.S)
                if mm:
                    q = mm.group(1).strip()
                    break
            if q and q not in by_q:
                q = next((qq for qq in by_q if qq[:80] == q[:80]), None)
            if not q or (arm, q) in seen:
                continue
            seen.add((arm, q))
            out.append((arm, q, by_q[q], resp))
            if len(out) >= n:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    triples = load_triples(args.n)
    if not triples:
        print("no (question, gold, response) triples recoverable from the replay buffers")
        return 1

    from act_prm.graders.snorkel_finance import SnorkelFinanceGrader

    grader = SnorkelFinanceGrader(
        grader_model_config={"name": "claude_agent_sdk",
                             "model_config": {"model": "claude-sonnet-4-5", "max_turns": 1}},
        num_samples=1,
        verbose=False,
    )

    for arm, q, gold, resp in triples:
        for label, answer in (("MODEL", resp), ("CONTROL(gold as response)", gold)):
            verdict, raw = grader.grade_sample(question=q, correct_answer=gold, response=answer)
            parsed = re.search(r"correct:\s*(yes|no)\b", raw or "", flags=re.IGNORECASE)
            print("=" * 78)
            print(f"arm      : {arm}")
            print(f"question : {q[:150]}")
            print(f"gold     : {str(gold)[:150]}")
            print(f"{label:24} : {str(answer)[:200]}")
            print(f"verdict  : {verdict}   regex matched: {bool(parsed)}")
            print(f"raw judge: {(raw or '')[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
