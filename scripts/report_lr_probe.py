#!/usr/bin/env python
"""Report how far each LR probe's LoRA adapter moved, and whether eval followed.

Pairs every `<dom>_s2probe_*` checkpoint with its run log and prints, per LR:
  max|dW|   the largest weight delta the adapter applies, = max|(alpha/r) * B @ A|
            over all LoRA modules. Compare against base weights of order 1e-2:
            the lr=4e-5 runs peaked at ~4e-5 (a ~0.4% perturbation on one layer,
            ~0.01% typical), i.e. numerically a no-op.
  ||B||     largest LoRA-B norm. B is zero-initialised, so this is a direct read
            on how far training escaped the cold start (dL/dA is proportional to
            B, so A cannot move until B does).
  eval PPL  first -> last held-out action PPL, and the % swing over the probe.

Usage: uv run --no-project python scripts/report_lr_probe.py [ckpt_root] [log_root]
"""
import glob
import json
import re
import sys
from pathlib import Path

from safetensors import safe_open

CK = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_lora"
LG = sys.argv[2] if len(sys.argv) > 2 else "logs"
SCALE = 16 / 8  # lora_alpha / lora_rank, r8_a16_*


def adapter_stats(path):
    mx = nb = 0.0
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if "lora_B" not in k:
                continue
            B = f.get_tensor(k).float()
            A = f.get_tensor(k.replace("lora_B", "lora_A")).float()
            d = (B @ A) * SCALE
            mx = max(mx, d.abs().max().item())
            nb = max(nb, B.norm().item())
    return mx, nb


def eval_curve(run_dir):
    m = Path(run_dir) / "metrics.jsonl"
    if not m.exists():
        return None
    pts = {}
    for line in m.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        v = r.get("eval/eval_action_ppl")
        if v is not None and r.get("progress/batch") is not None:
            pts[r["progress/batch"]] = v
    if not pts:
        return None
    ys = [pts[x] for x in sorted(pts)]
    swing = 100 * (max(ys) - min(ys)) / min(ys) if min(ys) else 0.0
    return ys[0], ys[-1], swing, len(ys)


def main():
    cks = sorted(glob.glob(f"{CK}/*/*/*_s2probe_*/step_last/adapter_model.safetensors"))
    if not cks:
        print(f"no probe checkpoints under {CK}/*/*/*_s2probe_*/ — did the probe run?")
        return
    print(f"{'run':44} {'max|dW|':>11} {'||B||':>11}   eval PPL first->last (swing)")
    print("-" * 104)
    for p in cks:
        run = Path(p).parents[1].name
        tag = run.split("-act-prm")[0]
        mx, nb = adapter_stats(p)
        logdir = sorted(glob.glob(f"{LG}/*/*/{tag}-*/"))
        ev = eval_curve(logdir[0]) if logdir else None
        s = f"{ev[0]:.4f} -> {ev[1]:.4f}  ({ev[2]:.2f}% over {ev[3]} pts)" if ev else "(no eval points)"
        lr = (re.search(r"_lr([0-9.e\-]+)", tag) or [None, "?"])[1]
        print(f"{tag[:44]:44} {mx:11.3e} {nb:11.3e}   {s}")
    print("\nReference — the shipped lr=4e-5 Stage-2 runs: max|dW| 4.5e-05, held-out swing 0.08-0.26%.")
    print("A working LR should move max|dW| by orders of magnitude and make the eval curve bend.")


if __name__ == "__main__":
    main()
