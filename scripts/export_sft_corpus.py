#!/usr/bin/env python
"""
Stage 1.5 — export an SFT corpus from a Stage-1 Act-PRM relabel pass.

Turns a generate-only relabel pass's ``generations.jsonl`` (one row per logged
action-step, with the sampled candidate thoughts, their rewards/weights and the
selected ``best`` index — see ``act_prm.generator.act_prm.base``) into
``act_prm_traces`` trajectory pools that Stage-2 SFT reuses via the existing
loader (``--dataset_path``).

For each logged step we commit the **best-scored thought** (``thoughts[best]``)
and set that step's assistant/target content to ``thought + "\\n\\n" + action`` —
exactly how the generator builds the (thought+action) span it scores (see
``base.py``: ``action={"role":"assistant","content": f"{thoughts[g]}\\n\\n{x_t}"}``).

``generations.jsonl`` rows carry only ``target_action`` + the thoughts, NOT the
observations (user / tool messages), so the original state is recovered from the
**source pools** the relabel pass ran over (the env's ``dataset_path`` dir, with
``train.json`` / ``eval.json``). Rows are mapped back by their ``split`` +
``sample_id`` (the env indexes ``pool[sample_id % len(pool)]``) and ``timestep``
(the t-th assistant/action message). The rest of each trajectory (observations,
system prompt, uid) is preserved so the SFT loader + ``prepare_minibatch``
supervise only the target span.

Actions-only rows (``thoughts == []``, from ``--no-infer_thoughts``) fall back to
the bare logged action as the target.

Example
-------
    uv run python scripts/export_sft_corpus.py \
        --generations logs/<run>/generations.jsonl \
        --source-pools data/tau2_retail \
        --out data/sft_corpus/tau2_retail/policy

Then SFT on it:
    ./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy \
        --dataset_path data/sft_corpus/tau2_retail/policy
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from act_prm.environments.act_prm_traces.data import (
    extract_action,
    load_pools,
    save_pools,
)


def _find_generations(path: str) -> Path:
    """Resolve ``--generations`` to a generations.jsonl file (accepts a run dir)."""
    p = Path(path)
    if p.is_dir():
        cand = p / "generations.jsonl"
        if not cand.is_file():
            raise SystemExit(f"--generations: no generations.jsonl under {p}")
        return cand
    if not p.is_file():
        raise SystemExit(f"--generations: not a file or dir: {p}")
    return p


def _load_rows(gen_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(gen_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _committed_target(row: dict[str, Any]) -> str:
    """Best-scored (thought + action) target for one logged step.

    ``best`` is the selected candidate index the generator already marks; for
    actions-only rows (no thoughts) the target is the bare logged action.
    """
    thoughts = row.get("thoughts") or []
    action = row["target_action"]
    if not thoughts:
        return action
    best = int(row.get("best", 0))
    best = best if 0 <= best < len(thoughts) else 0
    return f"{thoughts[best]}\n\n{action}"


def export(
    gen_file: Path,
    source_pools: str,
    out_dir: str,
    splits: list[str],
    strict: bool = False,
) -> dict[str, int]:
    """Build {split}.json pools; return per-split trajectory counts."""
    rows = _load_rows(gen_file)
    train_pool, eval_pool = load_pools(source_pools)
    source = {"train": train_pool, "eval": eval_pool}

    # Group rows by (split, sample_id) -> {timestep: row}. Keep the LAST row per
    # (split, sample_id, timestep) so a re-run / multi-batch relabel keeps the
    # freshest commit for that step.
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for r in rows:
        split = r.get("split", "train")
        if split not in splits:
            continue
        key = (split, int(r["sample_id"]))
        grouped.setdefault(key, {})[int(r["timestep"])] = r

    out: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    n_mismatch = 0
    for (split, sample_id), steps in sorted(grouped.items()):
        pool = source.get(split)
        if not pool:
            raise SystemExit(f"source pools have no non-empty {split!r} split for {source_pools}")
        traj = copy.deepcopy(pool[sample_id % len(pool)])
        messages = traj["messages"]
        action_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        for t, row in sorted(steps.items()):
            if t >= len(action_indices):
                continue  # relabel covered more steps than this source traj has
            idx = action_indices[t]
            # Sanity: the row's target_action should match the source's stripped
            # action at this step (they came from the same traces).
            src_action = extract_action(messages[idx].get("content") or "") or messages[idx].get("content")
            if src_action != row["target_action"]:
                n_mismatch += 1
                if strict:
                    raise SystemExit(
                        f"target_action mismatch at {split} sample {sample_id} step {t}:\n"
                        f"  source: {src_action!r}\n  row:    {row['target_action']!r}"
                    )
            messages[idx]["content"] = _committed_target(row)
        out[split].append(traj)

    if n_mismatch:
        print(f"WARNING: {n_mismatch} step(s) had a source/row target_action mismatch "
              f"(source pools may differ from the relabel pass; used the row's action).",
              file=sys.stderr)

    save_pools(
        out_dir,
        out.get("train", []),
        out.get("eval", []),
        meta={
            "exported_from": str(gen_file),
            "source_pools": source_pools,
            "splits": splits,
            "n_train": len(out.get("train", [])),
            "n_eval": len(out.get("eval", [])),
            "note": "SFT corpus: assistant content = committed (best) thought + action.",
        },
    )
    return {s: len(v) for s, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--generations",
        required=True,
        help="Path to generations.jsonl (or a run log dir containing it).",
    )
    ap.add_argument(
        "--source-pools",
        required=True,
        help="Dir with the relabel pass's source pools (train.json / eval.json) — "
        "the env's --dataset_path. Provides the observations the generations rows omit.",
    )
    ap.add_argument("--out", required=True, help="Output dir for {train,eval}.json pools.")
    ap.add_argument(
        "--splits",
        default="train,eval",
        help="Comma-separated splits to export (default: train,eval).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Error (instead of warn) if a row's target_action doesn't match the source action.",
    )
    args = ap.parse_args()

    gen_file = _find_generations(args.generations)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    counts = export(gen_file, args.source_pools, args.out, splits, strict=args.strict)
    total = sum(counts.values())
    print(f"Exported {total} trajectories to {args.out}")
    for s in splits:
        print(f"  {s}: {counts.get(s, 0)}")


if __name__ == "__main__":
    main()
