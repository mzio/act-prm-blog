#!/usr/bin/env python
"""How far did a LoRA adapter actually travel from its initialisation?

`lora_B` is zero-initialised, so `B @ A` IS the adapter's entire contribution to the model.
If max|B@A| is orders of magnitude below the base weight scale (~1e-2 for Qwen3-4B), the
adapter is a numerical no-op no matter what the loss curve looked like. This is the
measurement that caught the 08-22 Stage-2 LR bug and, on 08-25, showed the Stage-1 EM
adapter was a no-op in every domain (insurance: max|B@A| = 2.8e-06).

Reports per checkpoint, so different (rank, lr) settings can be compared directly. Note
max|B@A| is scale-comparable ACROSS ranks: it is the induced weight delta, already summed
over the rank dimension, and PEFT's alpha/r scaling is applied at forward time (all configs
here hold alpha/r = 2).

Usage: uv run --no-project python scripts/report_lora_movement.py [--glob PATTERN]
"""
import argparse
import glob
import json
import os
import statistics

import safetensors.torch as st

BASE_SCALE = 1e-2  # order of magnitude of Qwen3-4B base weights


def measure(ckpt_dir):
    f = os.path.join(ckpt_dir, "adapter_model.safetensors")
    if not os.path.exists(f):
        return None
    W = st.load_file(f)
    B = [v.float() for k, v in W.items() if "lora_B" in k]
    if not B:
        return None
    mx_ba, deltas = 0.0, []
    for k in W:
        if "lora_A" not in k:
            continue
        kb = k.replace("lora_A", "lora_B")
        if kb in W:
            d = (W[kb].float() @ W[k].float()).abs().max().item()
            deltas.append(d)
            mx_ba = max(mx_ba, d)
    return {"n_modules": len(deltas),
            "max_B": max(b.abs().max().item() for b in B),
            "max_BA": mx_ba,
            "median_BA": statistics.median(deltas) if deltas else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="checkpoints_lora/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/s1probe_*/step_*")
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.glob))
    if not dirs:
        print(f"no checkpoints matching {args.glob}")
        return
    print(f"  {'checkpoint':34} {'modules':>8} {'max|B|':>11} {'max|B@A|':>11} {'median':>11}  verdict")
    for d in dirs:
        m = measure(d)
        if not m:
            continue
        tag = os.path.basename(os.path.dirname(d.rstrip("/"))).split("-act-prm")[0][:26]
        step = os.path.basename(d.rstrip("/"))
        rel = m["max_BA"] / BASE_SCALE
        verdict = ("NO-OP (<0.1% of base)" if rel < 1e-3 else
                   "marginal (<1% of base)" if rel < 1e-2 else "MOVED")
        print(f"  {tag+'/'+step:34} {m['n_modules']:>8} {m['max_B']:>11.3e} "
              f"{m['max_BA']:>11.3e} {m['median_BA']:>11.3e}  {verdict}")
    print(f"\n  base weights are order {BASE_SCALE:g}; max|B@A| is the induced weight delta and is")
    print("  comparable across ranks (already summed over the rank dimension).")


if __name__ == "__main__":
    main()
