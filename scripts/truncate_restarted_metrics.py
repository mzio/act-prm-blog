#!/usr/bin/env python
"""Drop the abandoned prefix from a metrics.jsonl that two runs wrote into.

A run's log directory name is derived from its run tag plus a hash of the config. Re-running
the SAME tag with an UNCHANGED config therefore resolves to the SAME directory, and the
second run appends to the first one's metrics.jsonl. The curve then reads as
b0..b10, b0..b149 -- and every "last row" / "best over the file" reader silently mixes the
two. This has bitten the sweep three times: two duplicate-run races (fixed in the drivers)
and the finance expert_thoughts_all retry after its empty-minibatch crash.

A restart is detectable without any bookkeeping: progress/batch is monotonically
non-decreasing within a run, so any row whose batch is LOWER than the previous row's begins
a new run. This keeps the rows from the last restart onward and moves the abandoned prefix
to metrics.jsonl.abandoned-<n> so nothing is destroyed.

Usage:
  uv run --no-project python scripts/truncate_restarted_metrics.py <run_dir> [...]   # fix
  uv run --no-project python scripts/truncate_restarted_metrics.py --check <glob>    # report
"""
import argparse
import glob
import json
import os
import sys


def resets(rows: list[dict]) -> list[int]:
    """Row indices where progress/batch goes backwards, i.e. a new run began."""
    out, prev = [], None
    for i, r in enumerate(rows):
        b = r.get("progress/batch")
        if b is None:
            continue
        if prev is not None and b < prev:
            out.append(i)
        prev = b
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="run dirs or globs")
    ap.add_argument("--check", action="store_true", help="report only, do not modify")
    args = ap.parse_args()

    dirs: list[str] = []
    for p in args.paths:
        dirs.extend(sorted(glob.glob(p)) or [p])

    n_bad = 0
    for d in dirs:
        f = os.path.join(d, "metrics.jsonl") if os.path.isdir(d) else d
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        rs = resets(rows)
        if not rs:
            continue
        n_bad += 1
        cut = rs[-1]
        tag = os.path.basename(os.path.dirname(f)).split("-act-prm")[0]
        print(f"{tag}: {len(rows)} rows, restart(s) at {rs} -> keeping {len(rows) - cut}")
        if args.check:
            continue
        n = 0
        while os.path.exists(f"{f}.abandoned-{n}"):
            n += 1
        with open(f"{f}.abandoned-{n}", "w") as fh:
            for r in rows[:cut]:
                fh.write(json.dumps(r) + "\n")
        with open(f, "w") as fh:
            for r in rows[cut:]:
                fh.write(json.dumps(r) + "\n")
        print(f"  wrote abandoned prefix -> {os.path.basename(f)}.abandoned-{n}")

    if not n_bad:
        print("no restarted metrics files found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
