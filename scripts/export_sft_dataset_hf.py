#!/usr/bin/env python
"""
Publish the FULL multi-thought Act-PRM SFT thought datasets to the HF Hub.

Act-PRM infers the latent *thoughts* behind logged, action-only agent
demonstrations via an offline EM: for each logged action ``x`` in state ``s`` we
sample ``G`` candidate thoughts ``z``, score each by the length-penalized action
likelihood ``reward = p(x|s,z) - 0.15 * len_frac``, and mark the ``best`` (argmax
reward) thought. This script takes the Stage-1 *relabel* passes'
``generations.jsonl`` and publishes **every** sampled thought (all ``G``) with its
per-thought ``rewards`` / ``likelihoods`` / ``thought_tokens`` and the
``best_index`` — i.e. the full E-step, not just the committed top-1.

Unlike ``scripts/export_sft_corpus.py`` (which commits only ``thoughts[best]`` into
an ``act_prm_traces`` pool for SFT), this keeps the whole candidate set so the
published dataset can be re-derived downstream (top-1, top-half, EM-weighted, ...).

Self-containment
----------------
``generations.jsonl`` rows carry only the sampled thoughts + the logged
``target_action`` (``x``), NOT the observations (state ``s``). Like
``export_sft_corpus.py`` we recover the state from the **source pools** (the env's
``--dataset_path`` dir, ``train.json`` / ``eval.json``) — reusing
``act_prm_traces.data.load_pools``. The env's dataloader *shuffles* the pool
between epochs, so ``pool[sample_id % len]`` is NOT reliable once ``sample_id``
wraps past the pool size; instead we resolve the source trajectory for each
``(split, sample_id)`` group by matching its logged ``target_action`` **sequence**
against the pool (unique in practice). We then attach, per row: the
``system_prompt``, the ``messages`` (context up to — not including — the action)
and the trajectory ``uid``.

Variants
--------
Four relabel passes (2 scorers x 2 EM checkpoints), one file each:
  scorer     = policy | base   (E-step reweighting used the trained POLICY LoRA
                                or the frozen BASE model to score p(x|s,z))
  em_checkpoint = best | last  (the EM checkpoint the scorer was resumed from:
                                the best-metric step, or the last step)
On disk these are the ``<domain>_s1relabel_<scorer>_<ckpt>_heldout-*`` run dirs.
(NB: the ``<domain>_s1relabel_<scorer>_heldout`` dirs WITHOUT ``_best``/``_last``
are a separate eval-only single-thought pass and are intentionally NOT used here.)

Example
-------
    export HF_HOME=/data/users/mzio/models/hf_cache \\
           https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 \\
           HF_HUB_DISABLE_XET=1 HF_TOKEN=$(cat ~/models/token)
    uv run python scripts/export_sft_dataset_hf.py \\
        --env act_prm/tau2_retail --domain retail \\
        --repo mzio/aprm-sft-thoughts-tau2-retail

Airline reuse:
    uv run python scripts/export_sft_dataset_hf.py \\
        --env act_prm/tau2_airline --domain airline --source-pools data/tau2_airline \\
        --repo mzio/aprm-sft-thoughts-tau2-airline
"""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LENGTH_PENALTY = 0.15  # reward = p(x|s,z) - LENGTH_PENALTY * len_frac

# (scorer, em_checkpoint) -> run_tag suffix component
VARIANTS = [
    ("policy", "best"),
    ("base", "best"),
    ("policy", "last"),
    ("base", "last"),
]


def _load_data_module():
    """Import ``act_prm_traces.data`` directly by file path.

    Avoids importing the package ``__init__`` (which pulls in numpy/torch): this
    script only needs ``load_pools`` + ``extract_action`` and stays CPU/dep-light.
    """
    path = REPO_ROOT / "src/act_prm/environments/act_prm_traces/data.py"
    spec = importlib.util.spec_from_file_location("aprm_traces_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_run(logs_root: Path, env_dir: str, model_cfg: str, run_tag: str) -> Path:
    """Resolve ``<logs_root>/<env_dir>/<model_cfg>/<run_tag>-*/generations.jsonl``.

    Picks the newest generations.jsonl if several run dirs share the tag prefix.
    """
    base = logs_root / env_dir / model_cfg
    cands = sorted(base.glob(f"{run_tag}-*/generations.jsonl"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no generations.jsonl for run_tag {run_tag!r} under {base}")
    return cands[-1]


def _load_rows(gen_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(gen_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _action_seq(traj: dict[str, Any]) -> list[str]:
    return [m["content"] for m in traj["messages"] if m.get("role") == "assistant"]


def _resolve_source(pool: list[dict[str, Any]], steps: dict[int, str]) -> int | None:
    """Return the unique pool index whose action sequence matches every
    (timestep -> target_action) in ``steps``; None if 0 or >1 candidates."""
    cands = []
    for i, traj in enumerate(pool):
        acts = _action_seq(traj)
        if all(t < len(acts) and acts[t] == a for t, a in steps.items()):
            cands.append(i)
    return cands[0] if len(cands) == 1 else None


def build_variant(
    data_mod,
    gen_file: Path,
    source_pools: str,
    domain: str,
    scorer: str,
    em_checkpoint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the self-contained rows for one relabel variant + a stats dict."""
    train_pool, eval_pool = data_mod.load_pools(source_pools)
    src = {"train": train_pool, "eval": eval_pool}

    rows = _load_rows(gen_file)
    # Group rows by (split, sample_id); keep the LAST row per (split, sample_id,
    # timestep) so a multi-batch / re-run relabel keeps the freshest generation.
    groups: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        split = r.get("split", "train")
        groups[(split, int(r["sample_id"]))][int(r["timestep"])] = r

    out: list[dict[str, Any]] = []
    n_unresolved = 0
    for (split, sample_id), steps in sorted(groups.items()):
        pool = src.get(split)
        if not pool:
            raise SystemExit(f"source pools have no {split!r} split at {source_pools}")
        target_by_t = {t: r["target_action"] for t, r in steps.items()}
        idx = _resolve_source(pool, target_by_t)
        if idx is None:
            n_unresolved += len(steps)
            continue
        traj = pool[idx]
        messages = traj["messages"]
        action_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        uid = traj.get("uid")
        system_prompt = traj.get("system_prompt") or "You are a helpful assistant."
        for t, r in sorted(steps.items()):
            if t >= len(action_indices):
                n_unresolved += 1
                continue
            state = messages[: action_indices[t]]  # context up to (not incl.) the action
            out.append(
                {
                    "domain": domain,
                    "scorer": scorer,
                    "em_checkpoint": em_checkpoint,
                    "split": split,
                    "uid": uid,
                    "sample_id": sample_id,
                    "timestep": t,
                    "system_prompt": system_prompt,
                    "messages": [
                        {"role": m.get("role"), "content": m.get("content") or ""} for m in state
                    ],
                    "target_action": r["target_action"],
                    "thoughts": r.get("thoughts") or [],
                    "likelihoods": r.get("likelihoods") or [],
                    "rewards": r.get("rewards") or [],
                    "thought_tokens": r.get("thought_tokens") or [],
                    "best_index": int(r.get("best", 0)),
                }
            )

    stats = _variant_stats(out)
    stats["n_unresolved"] = n_unresolved
    stats["gen_file"] = str(gen_file)
    return out, stats


def _variant_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Row count + per-thought-count breakdown + integrity flags."""
    n_by_thoughts: dict[int, int] = defaultdict(int)
    n_by_split: dict[str, int] = defaultdict(int)
    bad_lists = 0
    bad_best = 0
    best_is_argmax = 0
    for r in rows:
        g = len(r["thoughts"])
        n_by_thoughts[g] += 1
        n_by_split[r["split"]] += 1
        if not (g == len(r["rewards"]) == len(r["likelihoods"]) == len(r["thought_tokens"]) >= 1):
            bad_lists += 1
            continue
        bi = r["best_index"]
        if not (0 <= bi < g):
            bad_best += 1
            continue
        if abs(r["rewards"][bi] - max(r["rewards"])) < 1e-9:
            best_is_argmax += 1
    return {
        "n_rows": len(rows),
        "n_by_thoughts": dict(sorted(n_by_thoughts.items())),
        "n_by_split": dict(sorted(n_by_split.items())),
        "bad_lists": bad_lists,
        "bad_best_index": bad_best,
        "best_is_argmax": best_is_argmax,
    }


DATASET_CARD = """\
---
license: mit
task_categories:
- text-generation
tags:
- act-prm
- process-reward-model
- agent
- reasoning
- tau2-bench
- %%DOMAIN%%
pretty_name: Act-PRM SFT thoughts (tau2 %%DOMAIN%%)
configs:
- config_name: default
  data_files:
  - split: policy_best
    path: policy_best.jsonl
  - split: base_best
    path: base_best.jsonl
  - split: policy_last
    path: policy_last.jsonl
  - split: base_last
    path: base_last.jsonl
---

# Act-PRM SFT thoughts — tau2-bench %%DOMAIN%%

**Act-PRM** (Action Process Reward Models) infers the latent *thoughts* behind
logged, **action-only** agent demonstrations via an offline **EM**. For each
logged action `x` in state `s` we sample `G=%%GROUP_SIZE%%` candidate thoughts `z`,
score each by the **length-penalized action likelihood**

```
reward(z) = p(x | s, z) - %%LENGTH_PENALTY%% * len_frac
```

(`len_frac` grows with the thought's token length), and mark the `best` thought
(argmax `reward`). The `(thought + action)` span is then what downstream SFT / RL
trains on.

This dataset publishes the **full E-step**: every one of the `G` sampled thoughts
per logged action, each with its `reward`, `likelihood` (`p(x|s,z)`) and
`thought_tokens`, plus the `best_index`. That is, it is **not** reduced to the
top-1 — you can re-derive top-1, top-half, EM-weighted, or any other selection
downstream (see below).

## Variants (files / splits)

Four relabel passes = **2 scorers x 2 EM checkpoints**:

| file / split | scorer | em_checkpoint |
|---|---|---|
| `policy_best.jsonl`  | policy | best |
| `base_best.jsonl`    | base   | best |
| `policy_last.jsonl`  | policy | last |
| `base_last.jsonl`    | base   | last |

- **scorer** — which model computed `p(x|s,z)` in the E-step: the trained
  **policy** LoRA, or the frozen **base** model (`--score_with_base`).
- **em_checkpoint** — which EM training checkpoint the scorer was resumed from:
  the **best**-metric step, or the **last** step.

Every row also carries `scorer` / `em_checkpoint` columns, so you can also
concatenate all four files and filter.

> Note: `train`-split rows have the full `G=%%GROUP_SIZE%%` sampled thoughts; the
> held-out `eval`-split rows were relabeled with a single sample (`G=1`), so their
> `thoughts`/`rewards` lists have length 1 and `best_index == 0`.

## Schema (one row per relabel-variant x logged action-step)

| field | type | meaning |
|---|---|---|
| `domain` | str | `{domain}` |
| `scorer` | str | `policy` or `base` |
| `em_checkpoint` | str | `best` or `last` |
| `split` | str | `train` or `eval` (eval = held-out; `G=1`) |
| `uid` | str/int | source trajectory id |
| `sample_id` | int | dataloader sample id within the relabel pass |
| `timestep` | int | index of this action among the trajectory's actions |
| `system_prompt` | str | the agent system prompt |
| `messages` | list[{role, content}] | state `s`: context **up to** (not incl.) the action |
| `target_action` | str | the logged ground-truth action `x` (a `<tool_call>...</tool_call>`) |
| `thoughts` | list[str] | the `G` sampled candidate thoughts `z` |
| `likelihoods` | list[float] | `p(x | s, z)` for each thought |
| `rewards` | list[float] | length-penalized reward for each thought |
| `thought_tokens` | list[int] | token length of each thought |
| `best_index` | int | argmax-reward thought index |

## Selecting thoughts

```python
from datasets import load_dataset
ds = load_dataset("%%REPO%%", split="policy_best")   # or base_best / policy_last / base_last
row = ds[0]

# top-1 (what the paper's SFT commits): the best length-penalized thought
top1 = row["thoughts"][row["best_index"]]
assert row["rewards"][row["best_index"]] == max(row["rewards"])

# the SFT target span is thought + action:
sft_target = top1 + "\\n\\n" + row["target_action"]

# top-half: keep the thoughts whose reward is in the top 50%
import numpy as np
order = np.argsort(row["rewards"])[::-1]
top_half = [row["thoughts"][i] for i in order[: max(1, len(order) // 2)]]

# EM weights: group-normalized (softmax-like) weights over all G thoughts,
# e.g. a temperature-1 softmax over rewards (or normalize exp(likelihood)):
import math
r = row["rewards"]
Z = sum(math.exp(x) for x in r)
em_weights = [math.exp(x) / Z for x in r]   # weight every (thought+action) span
```

## Provenance

Generated by Act-PRM Stage-1 relabel passes over tau2-bench %%DOMAIN%% expert
demonstrations (`no_train` generate-only passes), model `%%MODEL_CFG%%`,
`group_size=%%GROUP_SIZE%%`, `length_penalty=%%LENGTH_PENALTY%%`. The held-out
RL-eval tasks are excluded. Built with `scripts/export_sft_dataset_hf.py`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--env",
        default="act_prm/tau2_retail",
        help="env_config (e.g. act_prm/tau2_retail); used to locate the log dir "
        "logs/<env_with_/_as_>/<model_cfg>.",
    )
    ap.add_argument("--domain", default="retail", help="Domain tag + run-tag prefix (retail/airline).")
    ap.add_argument(
        "--source-pools",
        default=None,
        help="Source pools dir (train.json/eval.json). Default: data/<domain-derived>.",
    )
    ap.add_argument("--model_cfg", default="hf_qwen3_4b_instruct", help="Model config (log subdir).")
    ap.add_argument("--logs-root", default=str(REPO_ROOT / "logs"), help="Root logs dir.")
    ap.add_argument(
        "--run-tag-prefix",
        default=None,
        help="Relabel run_tag prefix; default '<domain>_s1relabel'. Variants are "
        "'<prefix>_<scorer>_<ckpt>_heldout'.",
    )
    ap.add_argument("--repo", default="mzio/aprm-sft-thoughts-tau2-retail", help="Target HF dataset repo.")
    ap.add_argument("--group_size", type=int, default=4, help="G (train), for the card only.")
    ap.add_argument(
        "--out",
        default=None,
        help="Local staging dir for the jsonl + README before upload. "
        "Default: data/hf_export/<repo-name>.",
    )
    ap.add_argument("--no-push", action="store_true", help="Build + verify locally; skip HF upload.")
    args = ap.parse_args()

    env_dir = args.env.replace("/", "_")
    if env_dir.startswith("act_prm_"):
        pass  # already the log convention (ec=act_prm_tau2_retail)
    domain = args.domain
    source_pools = args.source_pools or str(REPO_ROOT / "data" / f"tau2_{domain}")
    run_tag_prefix = args.run_tag_prefix or f"{domain}_s1relabel"
    logs_root = Path(args.logs_root)
    out_dir = Path(args.out or (REPO_ROOT / "data/hf_export" / args.repo.split("/")[-1]))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_mod = _load_data_module()

    print(f"== Act-PRM SFT thoughts export : {args.repo} ==")
    print(f"   domain={domain}  env_dir={env_dir}  model_cfg={args.model_cfg}")
    print(f"   source_pools={source_pools}")
    print(f"   run_tag_prefix={run_tag_prefix}")
    print(f"   out_dir={out_dir}\n")

    all_ok = True
    manifest: list[dict[str, Any]] = []
    for scorer, ckpt in VARIANTS:
        name = f"{scorer}_{ckpt}"
        run_tag = f"{run_tag_prefix}_{scorer}_{ckpt}_heldout"
        gen_file = _find_run(logs_root, env_dir, args.model_cfg, run_tag)
        rows, stats = build_variant(data_mod, gen_file, source_pools, domain, scorer, ckpt)
        fp = out_dir / f"{name}.jsonl"
        with open(fp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_multi = sum(v for k, v in stats["n_by_thoughts"].items() if k > 1)
        integrity_ok = (
            stats["bad_lists"] == 0
            and stats["bad_best_index"] == 0
            and stats["best_is_argmax"] == stats["n_rows"]
            and stats["n_unresolved"] == 0
        )
        all_ok = all_ok and integrity_ok
        print(f"[{name}]  {stats['n_rows']} rows -> {fp.name}")
        print(f"    run: {Path(stats['gen_file']).parent.name}")
        print(f"    n_by_split={stats['n_by_split']}  n_by_thoughts={stats['n_by_thoughts']}  (multi-thought rows: {n_multi})")
        print(
            f"    integrity: bad_lists={stats['bad_lists']} bad_best_index={stats['bad_best_index']} "
            f"best==argmax(reward): {stats['best_is_argmax']}/{stats['n_rows']} unresolved={stats['n_unresolved']}  -> {'OK' if integrity_ok else 'FAIL'}"
        )
        manifest.append({"variant": name, **stats})

    # dataset card
    readme = out_dir / "README.md"
    card = DATASET_CARD
    for token, val in {
        "%%DOMAIN%%": domain,
        "%%GROUP_SIZE%%": str(args.group_size),
        "%%LENGTH_PENALTY%%": str(LENGTH_PENALTY),
        "%%REPO%%": args.repo,
        "%%MODEL_CFG%%": args.model_cfg,
    }.items():
        card = card.replace(token, val)
    readme.write_text(card)
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote README.md + export_manifest.json to {out_dir}")

    if not all_ok:
        print("\n!! integrity checks FAILED — not pushing.", file=sys.stderr)
        sys.exit(1)

    if args.no_push:
        print("\n--no-push set; skipping HF upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(out_dir),
        commit_message="Add Act-PRM full multi-thought SFT datasets (all G thoughts + rewards/likelihoods/best-index)",
    )
    print(f"\nPushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
