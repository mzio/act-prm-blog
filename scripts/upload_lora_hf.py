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
import argparse, glob, json, os, re, subprocess
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
# Model-tagged destination so multiple models coexist in one repo without collision:
# retail/ for the default 4B-instruct, retail-<short>/ (e.g. retail-8b) otherwise.
_short = args.model_cfg.replace("hf_qwen3_", "").replace("_instruct", "")   # 4b / 8b / 4b_base
DEST = DOM if args.model_cfg == "hf_qwen3_4b_instruct" else f"{DOM}-{_short}"
# base model: read from the model config yaml if present (so 8B gets Qwen3-8B)
try:
    from omegaconf import OmegaConf
    _mc = OmegaConf.load(f"configs/model/{args.model_cfg}.yaml")
    args.base_model = _mc["model_config"]["pretrained_model_name_or_path"]
except Exception:
    pass
FORCE = os.environ.get("UPLOAD_FORCE", "0") == "1"
api = HfApi()

# Local upload marker: {run_tag: adapter_mtime}. Re-upload only when the checkpoint
# CHANGES (so a run's FINAL step_best replaces any mid-run copy), and never upload a
# run that's still training (avoids publishing a non-final early-stop checkpoint).
MARKER = f"/tmp/aprm/uploaded_lora_{DEST}.json"
marker = {}
if os.path.exists(MARKER):
    try: marker = json.load(open(MARKER))
    except Exception: marker = {}

def _is_training(run_tag: str) -> bool:
    try:
        out = subprocess.run(["pgrep", "-af", "main_pytorch.py"], capture_output=True, text=True).stdout
        return any(f"run_tag {run_tag}" in ln for ln in out.splitlines())
    except Exception:
        return False

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
for ck in sorted(glob.glob(f"{ROOT}/{DOM}_s2_*/step_best")):
    adapter = os.path.join(ck, "adapter_model.safetensors")
    if not os.path.isfile(adapter):
        continue
    run = os.path.basename(os.path.dirname(ck))
    tag = re.sub(r"-act-prm.*", "", run).replace(f"{DOM}_s2_", "")   # e.g. thoughts_policy_heldout_fullctx
    run_tag = f"{DOM}_s2_{tag}"
    if _is_training(run_tag):
        continue  # still training — wait for the FINAL step_best before publishing
    mt = f"{os.path.getmtime(adapter):.0f}"
    if not FORCE and marker.get(tag) == mt:
        continue  # already uploaded this exact (final) checkpoint
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
              f"m = PeftModel.from_pretrained(m, '{args.repo}', subfolder='{DEST}/{tag}')\n```\n")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo=f"{DEST}/{tag}/README.md",
                    repo_id=args.repo, commit_message=f"readme {tag}")
    for fn in ("adapter_model.safetensors", "adapter_config.json"):
        p = os.path.join(ck, fn)
        if os.path.isfile(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{DEST}/{tag}/{fn}",
                            repo_id=args.repo, commit_message=f"upload {tag}/{fn}")
    uploaded.append((tag, m))
    marker[tag] = mt
    print(f"  uploaded {DEST}/{tag}" + (f"  (action-PPL {m['action_ppl']:.3f})" if m else ""))

os.makedirs("/tmp/aprm", exist_ok=True)
json.dump(marker, open(MARKER, "w"))

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
        f"full-context regimes. Adapters live under `{DEST}/<variant_regime>/`.\n\n"
        f"## Held-out action-only eval (lower PPL / higher acc = better next-action fit)\n"
        f"| variant_regime | action-only PPL | action-acc |\n|---|---|---|\n{rows}\n\n"
        f"See the project for methodology (Act-PRM: infer latent thoughts behind action-only demos via offline EM).\n")
# default 4B -> top-level model card; other models -> a per-model sub-card (avoids clobber)
card_path = "README.md" if DEST == DOM else f"{DEST}/README.md"
if uploaded or FORCE:
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo=card_path,
                    repo_id=args.repo, commit_message=f"update {DEST} model card")
print(f"\nDONE: {len(uploaded)} newly-uploaded checkpoints ({DEST}) -> https://huggingface.co/{args.repo}")
