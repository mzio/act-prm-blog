# Handoff — Act-PRM **snorkel_finance** pipeline, for a Claude Code session on devvm40015 (2×H100)

Run the SAME Act-PRM experiments as the tau2 airline box, but for the
**snorkel_finance** environment. Repo = `act-prm-blog`, branch `act-prm-pytorch`.
This handoff assumes the airline box's fixes are already in the code (they are — see
"What's already fixed", below). You are `mzio` on `devvm40015.ftw0.facebook.com`, which
has `~/projects/` and 2×H100 (same 80 GiB).

## 0. Get the code
The airline box backs up a git bundle into dotsynced `~/.claude/` — it should appear here:
```bash
ls ~/.claude/act-prm-backups/act-prm-blog-devvm27322.bundle    # (dotsync; wait if not yet present)
git clone ~/.claude/act-prm-backups/act-prm-blog-devvm27322.bundle ~/projects/act-prm-blog
cd ~/projects/act-prm-blog && git checkout act-prm-pytorch
```
If dotsync hasn't propagated, scp it from your own shell:
`scp mzio@devvm27322.rva0.facebook.com:~/.claude/act-prm-backups/act-prm-blog-devvm27322.bundle .`
then `git clone <bundle> ~/projects/act-prm-blog`.

## 1. Box setup (offline is the norm here)
Your **agent egress is filtered** (fwdproxy 403s HF's CDN/Xet for `agent_id:claude_code`),
so large downloads must be done by YOU in a plain shell; the agent runs everything OFFLINE.
```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
bash scripts/setup_new_box.sh    # uv (installs from PyPI wheel if astral.sh/pip blocked) + uv sync --frozen
```
If `uv sync` trips on the `./tau2-bench` path dep (not needed for finance): it's fine —
`UV_FROZEN=1 uv sync --frozen` uses the committed lock. tau2-bench is only for tau2 Stage-3.

**Prefetch (YOU, plain shell — reaches the CDN):**
```bash
export HF_HOME=/data/users/$USER/models/hf_cache https_proxy=http://fwdproxy:8080 HF_HUB_DISABLE_XET=1
uv run hf download Qwen/Qwen3-4B-Instruct-2507        # paper model
uv run hf download Qwen/Qwen3-8B                       # for the 8B arm
uv run hf download --repo-type dataset mzio/aprm-snorkelai_agent_finance_reasoning
```
(Model configs' `cache_dir` = `/data/users/mzio/models/hf_cache/hub` — keep HF_HOME there.)

## 2. Splits + offline pools (clean train / eval / rl_eval)
The finance split is committed: `data/splits/snorkel_finance.json` = **116 act_prm_train /
25 act_prm_eval / 26 rl_eval** (167 usable, seed 0). rl_eval is held out for later RL eval
and NEVER seen by act-prm. To rebuild/verify: `uv run python scripts/make_split.py
--dataset mzio/aprm-snorkelai_agent_finance_reasoning --name snorkel_finance ...` (needs
the proxy). Env config: `act_prm/snorkel_finance_split` (uses the split_file + a persisted
`dataset_path=data/snorkel_finance_split`).

**Build the offline pools from the cached parquet** (the env's streaming load fails offline):
```bash
export HF_HOME=/data/users/$USER/models/hf_cache UV_FROZEN=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
uv run python scripts/prebuild_pools_offline.py --split_file data/splits/snorkel_finance.json --dataset_path data/snorkel_finance_split
uv run python scripts/prebuild_pools_offline.py --split_file data/splits/snorkel_finance.json --dataset_path data/snorkel_finance_split_expert_thoughts --keep_expert_thoughts
```
Verify each prints `21/… train / … eval` matching the split (train 116 / eval 25).

## 3. Run the full pipeline (one command per model)
Everything (Stage 1 EM → Stage 1.5 relabel best+last + export → Stage 2 SFT ×12) is in a
single env+model-parameterized script. It's offline, uncapped + gradient-checkpointing,
resumable, 2-GPU, and sizes the relabel from the split automatically:
```bash
mkdir -p /tmp/aprm
MODEL=hf_qwen3_4b_instruct nohup ./scripts/run_actprm_pipeline.sh act_prm/snorkel_finance_split > /tmp/aprm/fin_4b.log 2>&1 &
# after 4B finishes (or on the other GPUs), the 8B arm:
MODEL=hf_qwen3_8b        nohup ./scripts/run_actprm_pipeline.sh act_prm/snorkel_finance_split > /tmp/aprm/fin_8b.log 2>&1 &
```
Watch: `tail -f /tmp/aprm/snorkel_finance_split_4b/orchestrator.log`. It produces:
- corpora `data/sft_corpus/snorkel_finance_split[_8b]/{policy,base,policy_last,base_last}`
- 12 SFT checkpoints per model: {actions_only, expert_thoughts, thoughts_policy(best/last),
  thoughts_base(best/last)} × {hide-obs, full-context}, early-stopped on eval_action_ppl.

## 4. The key metric (action-subspan) — read the SFT curves right
`eval_action_ppl` scores the WHOLE (thought+action) span, so it's inflated by verbose
thoughts (expert_thoughts looks "worst"). The trustworthy metric is the **action subspan**
(`eval_actiononly_ppl` / `eval_actiononly_accuracy`, logged every `eval_every` batches in
each run's `metrics.jsonl`) — it scores only the `<tool_call>` tokens. On airline this
FLIPPED the story: thoughts (expert + Act-PRM) beat actions_only on action prediction.
Build a notebook `notebooks/cc-finance-1.x-*` plotting whole-span vs action-subspan
ppl/accuracy curves per variant×regime, 4B vs 8B. Keep per-domain artifacts prefixed
`cc-finance-*` to avoid clobbering airline's.

## What's already fixed in the code (hard-won on the airline box)
- **Offline everything**: run with `HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  HF_HUB_DISABLE_XET=1 UV_FROZEN=1` (the script exports these).
- **Relabel corpus corruption FIXED**: `--no_train` now (a) generates+saves TRAIN rollouts
  (was eval-only → corpus train:0), and (b) does NOT shuffle the pool (a mid-run shuffle
  misaligned generation `sample_id` vs export `pool[sample_id%n]`, grafting thoughts onto
  the wrong task's observations). So any `num_batches` yields a correct corpus.
- **Qwen3-8B tokenizer hack REMOVED** (2 sites): it swapped Qwen3-8B's tokenizer for
  Qwen2.5-3B-Instruct — wrong vocab + not cached offline. Now uses the model's own tokenizer.
- **SFT eval** logs train {loss, ppl, action_accuracy} + eval {action_ppl, action_accuracy,
  action_loss} + the action-subspan {actiononly_ppl, actiononly_accuracy}; early-stop on
  eval_action_ppl; run_name >255-char crash guarded.
- **VRAM**: uncapped + `--gradient_checkpointing` (peaks ~34 GiB/run; ~6-8× activation
  savings). One run per GPU; the pipeline runs 2 in parallel.

## Gotchas / ops
- github is BLOCKED by fwdproxy; push from your own shell. Back up: `./scripts/snapshot.sh
  "msg"` (per-host bundle in dotsynced ~/.claude).
- Kill runs: `pkill -9 -f '[m]ain_pytorch.py'` (bracket avoids self-match; without it, a
  literal `main_pytorch.py` in the pkill arg matches the shell itself and kills your session).
- The pipeline is resumable — re-run the same command to continue after a crash.
- **Stage 3 (env RL)**: finance has NO tau2-style gym env, so Stage 3 (interactive RL) is
  N/A / TBD for finance — focus is Stage 1 (thought-gen) + Stage 2 (SFT) + the offline
  action-subspan comparison. (rl_eval tasks are still held out in case a finance env appears.)
- Finance is bigger than airline (116 vs 21 train tasks) — the script auto-scales the EM /
  relabel `num_batches`; SFT stays 60 batches (early-stop handles convergence).
