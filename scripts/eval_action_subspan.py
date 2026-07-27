#!/usr/bin/env python
"""Post-hoc, action-SUBSPAN eval for Act-PRM Stage-2 SFT checkpoints.

Why this exists
---------------
``scripts/analyze_sft.py`` reports ``eval_action_ppl`` over the *whole* SFT target
span. For the thought variants that span is ``thought + action`` (or, for
``expert_thoughts``, ``reasoning + action``), while for ``actions_only`` it is just
the action. Those spans are NOT comparable — a model that spends tokens on
reasoning is scored on different tokens than the actions-only baseline.

This script makes the variants comparable on a *same-span* basis: it teacher-forces
each eval example and measures next-token perplexity + accuracy over ONLY the
explicit **action** tokens — the ``<tool_call> ... </tool_call>`` block (or a
``Final Answer: ...`` suffix) — excluding the thought / reasoning tokens. It also
reports the whole-target PPL for reference (this reproduces ``analyze_sft``'s
``eval_action_ppl`` at the same checkpoint).

No retraining: it loads each saved LoRA checkpoint and forward-passes the eval
corpus. It reuses the exact same tokenization/scoring convention as the
``ActPrmGenerator`` E-step (``base.py._score_action_only`` / ``build_scoring_messages``):
render ``[system] + state + [assistant: target]`` via ``apply_chat_template``, take
the target span as ``input_ids[state_len:]`` where ``state_len`` is the
generation-prompt boundary, and (here) further isolate the action sub-span.

Usage
-----
    # full eval of all retail SFT checkpoints for the 4B model (GPU):
    CUDA_VISIBLE_DEVICES=0 MODEL_CFG=hf_qwen3_4b_instruct \
        uv run --no-sync python scripts/eval_action_subspan.py act_prm/tau2_retail

    # CPU correctness self-test (tiny model, synthetic corpus, no checkpoints):
    MODEL_CFG=hf_qwen3_0_6b HF_HUB_OFFLINE=1 SELFTEST=1 \
        uv run --no-sync python scripts/eval_action_subspan.py act_prm/tau2_retail

Env vars: ``MODEL_CFG`` (default ``hf_qwen3_4b_instruct``) selects the model config
+ the ``<MODEL>`` checkpoint/log dir; ``LORA_CFG`` (default ``r8_a16_linear``);
``SELFTEST=1`` runs the CPU sanity check instead of the full eval.
"""
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

# Reuse the repo's action extractor + model/LoRA loaders + context compaction so the
# eval context matches training exactly (no reinvented tokenization).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from act_prm.environments.act_prm_traces.data import compact_observations, extract_action  # noqa: E402
from act_prm.llm_handlers import load_llm  # noqa: E402
from act_prm.lora import get_lora_model, load_lora  # noqa: E402

ENVCFG = sys.argv[1] if len(sys.argv) > 1 else "act_prm/tau2_retail"
ENVDIR = ENVCFG.replace("/", "_")               # act_prm_tau2_retail
ENVNAME = ENVCFG.split("/")[-1]                  # tau2_retail
DOM = ENVNAME.replace("tau2_", "")              # retail / airline
DEFAULT_MODEL = "hf_qwen3_4b_instruct"
MODEL = os.environ.get("MODEL_CFG", DEFAULT_MODEL)
LORA = os.environ.get("LORA_CFG", "r8_a16_linear")
ENABLE_THINKING = False  # matches the training default (cfg.enable_thinking=False)

CKPT_ROOT = Path("checkpoints_lora") / ENVDIR / MODEL
LOGROOT = Path("logs") / ENVDIR / MODEL


# ---------------------------------------------------------------------------
# tokenization + span isolation
# ---------------------------------------------------------------------------
def _ids(out):
    """Normalize apply_chat_template / tokenizer output to a flat list[int]."""
    if not isinstance(out, (list, tuple)):  # BatchEncoding
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def render_example(tokenizer, system_prompt, state_messages, target_content):
    """Render ``[system] + state + [assistant: target]`` and return
    ``(input_ids, offsets, state_len, full_text)``.

    ``state_len`` is the generation-prompt boundary (where the assistant target
    begins) — identical to the ActPrmGenerator convention. Both the state prefix
    and the full sequence go through the *same* tokenizer-on-rendered-text path so
    ``input_ids[:state_len]`` is guaranteed to be the state token prefix.
    """
    scoring_state = [{"role": "system", "content": system_prompt}] + [
        m for m in state_messages if m["role"] != "system"
    ]
    state_text = tokenizer.apply_chat_template(
        scoring_state, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING
    )
    full_msgs = scoring_state + [{"role": "assistant", "content": target_content}]
    full_text = tokenizer.apply_chat_template(
        full_msgs,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
        enable_thinking=ENABLE_THINKING,
    )
    state_len = len(_ids(tokenizer(state_text)))
    enc = tokenizer(full_text, return_offsets_mapping=True)
    input_ids = _ids(enc)
    offsets = enc["offset_mapping"]
    if isinstance(offsets[0], (list, tuple)) and offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    return input_ids, offsets, state_len, full_text


def action_start_token(full_text, offsets, target_content, state_len):
    """Index of the first token of the explicit action within the target span.

    The action (``<tool_call>...</tool_call>`` block or ``Final Answer:`` suffix) is
    always a *suffix* of the assistant content, so the action sub-span is taken from
    this token to the END of the target (naturally including the trailing
    ``</tool_call>`` + ``<|im_end|>`` tokens). Returns ``state_len`` (i.e. the whole
    target = action) when there is no separable reasoning prefix, so an
    actions-only-style target yields action-span == whole-target span.
    """
    action_str = extract_action(target_content)
    if not action_str:
        return state_len  # no separable action -> whole target span
    # Locate the action substring in the rendered text (last occurrence = the
    # final assistant turn). Fall back to whole-target if it can't be found.
    pos = full_text.rfind(action_str)
    if pos < 0:
        # try the raw tool_call/Final-Answer marker as a looser anchor
        for marker in ("<tool_call>", "Final Answer:"):
            p = full_text.rfind(marker)
            if p >= 0:
                pos = p
                break
    if pos < 0:
        return state_len
    # First real (non-special, offsets[i][1] > offsets[i][0]) token at/after `pos`.
    for i in range(state_len, len(offsets)):
        s, e = offsets[i]
        if e > s and s >= pos:
            return i
    return state_len


@torch.no_grad()
def eval_example(model, tokenizer, system_prompt, state_messages, target_content):
    """Teacher-force one example; return summed CE / #correct / #tokens for the
    whole-target span and the action sub-span, or ``None`` if unscorable."""
    input_ids, offsets, state_len, full_text = render_example(
        tokenizer, system_prompt, state_messages, target_content
    )
    if state_len >= len(input_ids):
        return None
    a_start = action_start_token(full_text, offsets, target_content, state_len)

    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    logits = model(input_ids=ids, use_cache=False).logits[0, :-1, :]  # predicts token t from t-1
    labels = ids[0, 1:]

    def span(start_tok):
        # target token index t is predicted by logits[t-1]; here labels[k]=input_ids[k+1]
        lo = max(0, start_tok - 1)
        tl, lb = logits[lo:], labels[lo:]
        if lb.numel() == 0:
            return 0.0, 0, 0
        ce = F.cross_entropy(tl.float(), lb, reduction="sum").item()
        correct = int((tl.argmax(dim=-1) == lb).sum().item())
        return ce, correct, int(lb.numel())

    whole = span(state_len)
    action = span(a_start)
    return {"whole": whole, "action": action, "a_start": a_start, "state_len": state_len,
            "n_input": len(input_ids), "offsets": offsets, "input_ids": input_ids,
            "full_text": full_text}


# ---------------------------------------------------------------------------
# run discovery + eval-corpus resolution
# ---------------------------------------------------------------------------
def parse_run(run_dir):
    """(variant, base_variant, corpus, regime) from a checkpoint run dir name,
    following analyze_sft.py's tag parsing. ``corpus`` = best|last (which Stage-1
    relabel corpus the SFT trained on), ``regime`` = hide|full."""
    tag = run_dir.name.split("-act-prm")[0]     # retail_s2_thoughts_policy_heldout[_fullctx]
    regime = "full" if tag.endswith("_fullctx") else "hide"
    variant = re.sub(r"^%s_s2_" % DOM, "", tag).replace("_heldout_fullctx", "").replace("_heldout", "")
    if variant.endswith("_last"):
        corpus, base = "last", variant[:-len("_last")]
    else:
        corpus, base = "best", variant
    return variant, base, corpus, regime


def eval_pool_path(base_variant, corpus):
    """Map a variant to the eval.json it trained/evaluated against.

    - actions_only    -> data/<env>/eval.json                 (action-only pool)
    - expert_thoughts -> data/<env>_expert_thoughts/eval.json (reasoning+action)
    - thoughts_policy -> data/sft_corpus/<env>/policy[_last]/eval.json (thought+action)
    - thoughts_base   -> data/sft_corpus/<env>/base[_last]/eval.json
    """
    if base_variant == "actions_only":
        return Path("data") / ENVNAME / "eval.json"
    if base_variant == "expert_thoughts":
        return Path("data") / f"{ENVNAME}_expert_thoughts" / "eval.json"
    if base_variant.startswith("thoughts_"):
        sub = base_variant[len("thoughts_"):]  # policy | base
        if corpus == "last":
            sub = f"{sub}_last"
        return Path("data") / "sft_corpus" / ENVNAME / sub / "eval.json"
    return None


def load_eval_examples(pool_path, regime, obs_max_chars, first_obs, last_obs):
    """Flatten an eval pool (list of trajectories) into per-assistant-turn examples,
    applying the run's hide_observations regime to each turn's context — mirroring
    the ActPrmGenerator E-step iteration over assistant turns."""
    trajs = json.loads(Path(pool_path).read_text())
    examples = []
    hide_middle = regime == "hide"
    for traj in trajs:
        messages = traj["messages"]
        system_prompt = traj.get("system_prompt") or "You are a helpful assistant."
        for idx, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            state = compact_observations(
                messages[:idx], obs_max_chars, first_to_show=first_obs,
                last_to_show=last_obs, hide_middle=hide_middle,
            )
            examples.append((system_prompt, state, m["content"]))
    return examples


def aggregate(examples, model, tokenizer):
    """Micro-average CE/accuracy over a list of (system, state, target) examples."""
    w_ce = w_cor = w_tok = 0.0
    a_ce = a_cor = a_tok = 0.0
    n = 0
    for system_prompt, state, target in examples:
        try:
            r = eval_example(model, tokenizer, system_prompt, state, target)
        except Exception as e:  # keep going on a single bad example
            print(f"  ! skipped an example: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        w_ce += r["whole"][0]; w_cor += r["whole"][1]; w_tok += r["whole"][2]
        a_ce += r["action"][0]; a_cor += r["action"][1]; a_tok += r["action"][2]
        n += 1
    if a_tok == 0 or w_tok == 0:
        return None
    return {
        "n_examples": n,
        "action_ppl": math.exp(a_ce / a_tok),
        "action_acc": a_cor / a_tok,
        "action_tokens": int(a_tok),
        "whole_target_ppl": math.exp(w_ce / w_tok),
        "whole_target_acc": w_cor / w_tok,
        "whole_tokens": int(w_tok),
    }


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_base_peft():
    """Load the base model (per MODEL_CFG) wrapped as a PEFT model with one adapter
    slot; per-checkpoint weights are then loaded IN PLACE via load_lora."""
    model_cfg = OmegaConf.load(f"./configs/model/{MODEL}.yaml")
    lora_cfg = OmegaConf.load(f"./configs/lora/{LORA}.yaml")
    llm = load_llm(**model_cfg)
    llm.model = get_lora_model(llm.model, **lora_cfg)
    llm.model.eval()
    return llm


# ---------------------------------------------------------------------------
# CPU self-test (no checkpoints needed)
# ---------------------------------------------------------------------------
def selftest():
    print(f"[selftest] loading base model {MODEL} (LoRA {LORA}) ...")
    llm = load_base_peft()
    model, tok = llm.model, llm.tokenizer
    system = "You are a helpful assistant."
    state = [{"role": "user", "content": "What is acme's 2024 revenue growth?"}]

    ok = True

    # (A) thought + <tool_call> action: the action span must EXCLUDE the reasoning.
    reasoning = "Let me think about this. I should look up the income statement first."
    action = '<tool_call>\n{"name": "query_table", "arguments": {"table": "acme_IncomeStatement"}}\n</tool_call>'
    target = f"{reasoning}\n\n{action}"
    r = eval_example(model, tok, system, state, target)
    offs, ids = r["offsets"], r["input_ids"]
    a_tokens = ids[r["a_start"]:]
    a_text = tok.decode(a_tokens)
    reason_free = "Let me think" not in a_text and "look up" not in a_text
    has_toolcall = "<tool_call>" in a_text
    smaller = r["action"][2] < r["whole"][2]
    print(f"[A] tool_call target: action-span tokens={r['action'][2]} whole={r['whole'][2]} "
          f"a_start={r['a_start']} state_len={r['state_len']}")
    print(f"    decoded action-span (head): {a_text[:80]!r}")
    print(f"    excludes reasoning={reason_free}  has <tool_call>={has_toolcall}  "
          f"action<whole={smaller}")
    ok &= reason_free and has_toolcall and smaller

    # (B) Final Answer suffix: action span must start at 'Final Answer:'.
    target_fa = "First I reason through the numbers carefully.\n\nFinal Answer: revenue grew 18%."
    r_fa = eval_example(model, tok, system, state, target_fa)
    fa_text = tok.decode(r_fa["input_ids"][r_fa["a_start"]:])
    fa_ok = fa_text.lstrip().startswith("Final Answer:") and "First I reason" not in fa_text
    print(f"[B] Final Answer target: decoded action-span (head): {fa_text[:60]!r}  ok={fa_ok}")
    ok &= fa_ok

    # (C) actions-only-style target (action only): action-span PPL == whole-target PPL.
    r_ao = eval_example(model, tok, system, state, action)
    a_ppl = math.exp(r_ao["action"][0] / r_ao["action"][2])
    w_ppl = math.exp(r_ao["whole"][0] / r_ao["whole"][2])
    same_span = r_ao["a_start"] == r_ao["state_len"]
    equal_ppl = abs(a_ppl - w_ppl) < 1e-6
    print(f"[C] actions-only target: a_start==state_len={same_span}  "
          f"action_ppl={a_ppl:.6f} whole_ppl={w_ppl:.6f} equal={equal_ppl}")
    ok &= same_span and equal_ppl

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    if os.environ.get("SELFTEST") == "1" or "--selftest" in sys.argv:
        sys.exit(selftest())

    # env-config context knobs (obs cap + which observations to keep)
    env_cfg = OmegaConf.load(f"./configs/environments/{ENVCFG}.yaml")
    obs_max_chars = env_cfg.get("obs_max_chars", None)
    first_obs = env_cfg.get("first_obs_to_show", 1)
    last_obs = env_cfg.get("last_obs_to_show", 1)

    if not CKPT_ROOT.exists():
        print(f"no checkpoints at {CKPT_ROOT}")
        return
    runs = sorted(d for d in CKPT_ROOT.glob(f"{DOM}_s2_*") if d.is_dir())
    if not runs:
        print(f"no Stage-2 SFT runs under {CKPT_ROOT}")
        return

    print(f"loading base model {MODEL} (LoRA {LORA}) ...")
    llm = load_base_peft()
    model, tok = llm.model, llm.tokenizer

    # cache eval-example lists per (pool_path, regime) so we don't re-tokenize corpora
    ex_cache = {}
    results = []
    for run in runs:
        variant, base, corpus, regime = parse_run(run)
        pool = eval_pool_path(base, corpus)
        if pool is None or not Path(pool).is_file():
            print(f"[skip] {run.name}: no eval corpus at {pool}")
            continue
        key = (str(pool), regime)
        if key not in ex_cache:
            ex_cache[key] = load_eval_examples(pool, regime, obs_max_chars, first_obs, last_obs)
        examples = ex_cache[key]

        for ckpt_name in ("step_best", "step_last"):
            ckpt = run / ckpt_name
            if not (ckpt / "adapter_model.safetensors").is_file():
                print(f"[skip] {run.name}/{ckpt_name}: no adapter")
                continue
            print(f"[eval] {variant:24s} regime={regime:4s} corpus={corpus:4s} "
                  f"{ckpt_name}  ({len(examples)} eval trajs-flattened examples)")
            try:
                load_lora(model, str(ckpt), is_trainable=False)
                model.eval()
                agg = aggregate(examples, model, tok)
            except Exception as e:
                print(f"  ! failed: {type(e).__name__}: {e}")
                continue
            if agg is None:
                print("  ! no scorable tokens")
                continue
            row = {
                "variant": base, "corpus": corpus, "regime": regime, "checkpoint": ckpt_name,
                "action_ppl": agg["action_ppl"], "action_acc": agg["action_acc"],
                "whole_target_ppl": agg["whole_target_ppl"], "whole_target_acc": agg["whole_target_acc"],
                "action_tokens": agg["action_tokens"], "whole_tokens": agg["whole_tokens"],
                "n_examples": agg["n_examples"], "run_dir": str(run),
            }
            results.append(row)
            print(f"    action_ppl={agg['action_ppl']:.4f} action_acc={agg['action_acc']:.4f} "
                  f"whole_ppl={agg['whole_target_ppl']:.4f}")

    if not results:
        print("no checkpoints produced results")
        return

    order = {"actions_only": 0, "expert_thoughts": 1, "thoughts_policy": 2, "thoughts_base": 3}
    results.sort(key=lambda r: (order.get(r["variant"], 9), r["corpus"] != "best",
                                r["regime"] != "hide", r["checkpoint"]))

    LOGROOT.mkdir(parents=True, exist_ok=True)
    csv_path = LOGROOT / f"{DOM}_action_subspan.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    lines = [
        f"# cc-3.1 — {ENVCFG} action-subspan eval ({MODEL})",
        "",
        "Auto-generated by `scripts/eval_action_subspan.py`. **This is the FAIR variant",
        "comparison**: next-token PPL + accuracy over ONLY the explicit action tokens",
        "(`<tool_call>...</tool_call>` block, or a `Final Answer:` suffix) — NOT the",
        "thought/reasoning tokens — so `expert_thoughts` / `thoughts_{policy,base}` and the",
        "`actions_only` baseline are all scored on the SAME action span. `whole_target_ppl`",
        "is the full-target PPL for reference (reproduces `analyze_sft`'s `eval_action_ppl`",
        "at that checkpoint; for `actions_only` it equals `action_ppl`). Lower PPL / higher",
        "accuracy = better next-action fit.",
        "",
        "| variant | corpus | regime | ckpt | action PPL | action acc | whole-target PPL | #act tok | #ex |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['corpus']} | {r['regime']} | {r['checkpoint']} | "
            f"{fmt(r['action_ppl'])} | {fmt(r['action_acc'])} | {fmt(r['whole_target_ppl'])} | "
            f"{r['action_tokens']} | {r['n_examples']} |"
        )
    lines += [
        "",
        f"CSV: `{csv_path}`.",
        "",
        "Reading: `actions_only` is the no-thoughts baseline; `expert_thoughts` the oracle",
        "upper-bound; `thoughts_{policy,base}` the Act-PRM inferred-thought arms (`corpus` =",
        "which Stage-1 relabel pass the SFT corpus came from: best|last). The question: on the",
        "SAME action tokens, do inferred thoughts improve next-action prediction over",
        "actions-only, and does it hold under both context regimes (hide vs full)?",
    ]
    note = Path("notes") / f"cc-3.1-{DOM}_action_subspan.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {note} and {csv_path}")


if __name__ == "__main__":
    main()
