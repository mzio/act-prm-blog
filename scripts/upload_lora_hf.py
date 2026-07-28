#!/usr/bin/env python
"""Upload finished Act-PRM SFT LoRA checkpoints to the HF Hub (public), mirroring the
airline repo layout: <domain>/<variant_regime>/{adapter_config.json,
adapter_model.safetensors, README.md} + a top-level model card with the action-only
results table. Idempotent: re-run to add newly-finished checkpoints.

Env: needs a WRITE HF token + fwdproxy + HF_HUB_DISABLE_XET=1 (Xet is blocked here).
Usage:  MODEL_CFG=hf_qwen3_4b_instruct uv run --no-sync python scripts/upload_lora_hf.py \
          --env act_prm/tau2_retail --repo mzio/aprm-sft-tau2-retail
Safe to run alongside training (read-only on checkpoints + network upload; no GPU).
"""
import argparse, glob, json, os, re
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="act_prm/tau2_retail")
ap.add_argument("--repo", default="mzio/aprm-sft-tau2-retail")
ap.add_argument("--model_cfg", default=os.environ.get("MODEL_CFG", "hf_qwen3_4b_instruct"))
ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
args = ap.parse_args()

ENVDIR = args.env.replace("/", "_")
DOM = args.env.split("/")[-1].replace("tau2_", "")          # retail
ROOT = f"checkpoints_lora/{ENVDIR}/{args.model_cfg}"
LOGROOT = f"logs/{ENVDIR}/{args.model_cfg}"
api = HfApi()

def best_metrics(tag):
    runs = glob.glob(f"{LOGROOT}/{tag}-*/metrics.jsonl")
    if not runs: return {}
    R = [json.loads(l) for l in open(runs[0]) if l.strip()]
    ao = [r for r in R if "eval/eval_actiononly_ppl" in r]
    if not ao: return {}
    b = min(ao, key=lambda r: r["eval/eval_actiononly_ppl"])
    return {"action_ppl": b["eval/eval_actiononly_ppl"],
            "action_acc": b.get("eval/eval_actiononly_accuracy"),
            "whole_ppl": min((r["eval/eval_action_ppl"] for r in R if "eval/eval_action_ppl" in r), default=None),
            "step": b.get("progress/batch")}

VARIANT_DESC = {
    "actions_only": "SFT on expert **action-only** targets (no thoughts) — baseline.",
    "expert_thoughts": "SFT on the **original expert reasoning + action** (oracle upper-bound).",
    "thoughts_policy": "SFT on **Act-PRM inferred thought + action**, best thought scored by the **policy**.",
    "thoughts_base": "SFT on **Act-PRM inferred thought + action**, best thought scored by the **base** model.",
    "thoughts_policy_last": "As thoughts_policy, thoughts relabeled from the EM **step_last** checkpoint.",
    "thoughts_base_last": "As thoughts_base, thoughts relabeled from the EM **step_last** checkpoint.",
}

api.create_repo(args.repo, private=False, exist_ok=True)
uploaded = []
for ck in sorted(glob.glob(f"{ROOT}/retail_s2_*/step_best" if DOM == "retail" else f"{ROOT}/{DOM}_s2_*/step_best")):
    if not os.path.isfile(os.path.join(ck, "adapter_model.safetensors")):
        continue
    run = os.path.basename(os.path.dirname(ck))
    tag = re.sub(r"-act-prm.*", "", run).replace(f"{DOM}_s2_", "")   # e.g. thoughts_policy_heldout_fullctx
    variant = tag.replace("_heldout_fullctx", "").replace("_heldout", "")
    regime = "full-context" if tag.endswith("_fullctx") else "hide-observations"
    m = best_metrics(f"{DOM}_s2_{tag}")
    metric_md = ""
    if m:
        metric_md = (f"\n**Held-out eval (action-only, best step {m.get('step')}):** "
                     f"PPL {m['action_ppl']:.3f}, action-token accuracy {m['action_acc']:.3f} "
                     f"(whole-target PPL {m['whole_ppl']:.3f}).\n")
    readme = (f"# {DOM} · {tag}\n\n"
              f"Act-PRM SFT LoRA adapter ({args.model_cfg} = `{args.base_model}`, r8_a16 on all linear).\n\n"
              f"- **Variant:** {VARIANT_DESC.get(variant, variant)}\n"
              f"- **Context regime:** {regime}\n{metric_md}\n"
              f"Load:\n```python\nfrom peft import PeftModel\nfrom transformers import AutoModelForCausalLM\n"
              f"m = AutoModelForCausalLM.from_pretrained('{args.base_model}')\n"
              f"m = PeftModel.from_pretrained(m, '{args.repo}', subfolder='{DOM}/{tag}')\n```\n")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo=f"{DOM}/{tag}/README.md",
                    repo_id=args.repo, commit_message=f"readme {tag}")
    for fn in ("adapter_model.safetensors", "adapter_config.json"):
        p = os.path.join(ck, fn)
        if os.path.isfile(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{DOM}/{tag}/{fn}",
                            repo_id=args.repo, commit_message=f"upload {tag}/{fn}")
    uploaded.append((tag, m))
    print(f"  uploaded {DOM}/{tag}" + (f"  (action-PPL {m['action_ppl']:.3f})" if m else ""))

# top-level card with the results table
_rl = []
for t, mm in sorted(uploaded, key=lambda x: x[0]):
    ppl = f"{mm['action_ppl']:.3f}" if mm else "—"
    acc = f"{mm['action_acc']:.3f}" if mm else "—"
    _rl.append(f"| {t} | {ppl} | {acc} |")
rows = "\n".join(_rl)
card = (f"# Act-PRM SFT LoRA checkpoints — tau2 {DOM}\n\n"
        f"LoRA adapters (r8_a16, base `{args.base_model}`) from **Act-PRM** supervised fine-tuning on "
        f"tau2-bench {DOM}. Variants: `actions_only` (baseline), `expert_thoughts` (oracle), "
        f"`thoughts_{{policy,base}}[_last]` (Act-PRM inferred thoughts), each in hide-observations and "
        f"full-context regimes. Adapters live under `{DOM}/<variant_regime>/`.\n\n"
        f"## Held-out action-only eval (lower PPL / higher acc = better next-action fit)\n"
        f"| variant_regime | action-only PPL | action-acc |\n|---|---|---|\n{rows}\n\n"
        f"See the project for methodology (Act-PRM: infer latent thoughts behind action-only demos via offline EM).\n")
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                repo_id=args.repo, commit_message="update model card")
print(f"\nDONE: {len(uploaded)} checkpoints -> https://huggingface.co/{args.repo}")
