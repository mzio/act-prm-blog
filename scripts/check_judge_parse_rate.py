#!/usr/bin/env python
"""Report how often the LLM judge's verdict failed to parse in a rollout run.

Why this exists: SnorkelFinanceGrader parsed only `correct:\\s*(yes|no)` and DEFAULTED TO
"no" on anything else, while the judge's system prompt demonstrates a different shape
(`{True, 'Yes they both match'}`). On the finance rollouts 12 of 18 judgements (67%) hit
that default. Every parse failure biases the score DOWN, which is the direction that
manufactures a null result -- so a silent 67% failure rate is indistinguishable from "the
model is bad" unless it is measured.

`parse_verdict` now falls through several shapes and logs
    "Grader verdict UNPARSEABLE -- defaulting to 'no'"
whenever none matches. This counts those warnings against the number of graded answers, so
the rate is visible per run instead of being rediscovered later.

Exit code 1 if the failure rate exceeds --max-frac, so a driver can refuse to bank a run
whose scores are mostly unread defaults.

Usage: uv run --no-project python scripts/check_judge_parse_rate.py <run.log> [--max-frac 0.2]
"""
import argparse
import re
import sys

UNPARSEABLE = re.compile(r"Grader verdict UNPARSEABLE", re.I)
# the env appends this to the transcript once per graded answer
GRADED = re.compile(r"# RESULT: (CORRECT|INCORRECT)!")
NO_RESPONSE = re.compile(r"Grader returned no actions", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="the run's stdout log (or its logs.log)")
    ap.add_argument("--max-frac", type=float, default=0.20)
    args = ap.parse_args()

    try:
        text = open(args.log, errors="replace").read()
    except OSError as e:
        print(f"  judge-parse: cannot read {args.log}: {e}")
        return 0  # missing log is not itself a grading failure

    n_unparsed = len(UNPARSEABLE.findall(text))
    n_graded = len(GRADED.findall(text))
    n_noresp = len(NO_RESPONSE.findall(text))

    if n_graded == 0 and n_unparsed == 0:
        print("  judge-parse: no graded answers in this log (nothing reached the judge)")
        return 0

    denom = max(n_graded, n_unparsed)
    frac = n_unparsed / denom if denom else 0.0
    print(f"  judge-parse: {n_unparsed}/{denom} verdicts unparseable ({100*frac:.0f}%)"
          + (f", {n_noresp} with no judge response" if n_noresp else ""))
    if n_noresp:
        print("               'no judge response' means the judge call itself failed -- "
              "those are graded 'no' regardless of the answer.")
    if frac > args.max_frac:
        print(f"               FAIL: above the {100*args.max_frac:.0f}% threshold. Scores are "
              "mostly unread defaults, biased toward incorrect; do not bank this run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
