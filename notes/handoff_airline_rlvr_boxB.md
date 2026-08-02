# Handoff — airline Stage-3 RLVR, box B (2 GPUs)

Supersedes `handoff_airline_rlvr_actions_only.md`. Box A runs `thoughts_policy`;
box B runs `actions_only` + `base`, one per GPU.

**If you already started an arm at `batch_size 1`: kill it and restart** — see the
correction below. Also move its checkpoint dir out of
`checkpoints_lora/tau2bench_airline_rlvr/<model>/` (e.g. to `_bs1_superseded/`), or the
runner sees `step_best` and silently skips the arm.

## Run it

```bash
cd ~/projects/act-prm-blog && git pull
mkdir -p /tmp/aprm/airline_rlvr_pair
ARMS="actions_only base" setsid nohup ./scripts/run_airline_rlvr_pair.sh \
  > /tmp/aprm/airline_rlvr_pair/run.log 2>&1 &

# durability: logs/ and checkpoints_lora/ are gitignored (we lost a fleet to a reimage)
setsid nohup ./scripts/backup_results_daemon.sh > /tmp/backup_results.daemon.log 2>&1 &
```

The daemon commits every 30 min but **cannot push** (fwdproxy blocks agent traffic, 403) —
ask the user to `git push`.

## Config — do not change any of it

Box A runs `thoughts_policy` with exactly this; the comparison is only valid if they match.

| knob | value | why |
|---|---|---|
| `--group_size 8 --batch_size 4` | 4 tasks × 8 rollouts / update | bs1 gave 72% zero-gradient updates |
| `--max_turns 30`, `--max_tokens 2048` | | |
| `--learning_rate 1e-4`, `--num_batches 50`, `--eval_every 10`, patience 3 | | |
| env `tau2bench/airline_rlvr` | `hide_observations:true`, `last_obs_to_show:1`, `negative_rewards:false` | hide-obs must match Stage-2 SFT |
| generator `hf_rlvr` | `mean_center:false`, `discount_factor:1.0` | advantage == +1 success / 0 fail |
| `max_seq_len` (pg.yaml) | 32768 | was 8192, silently dropping ~6% of episodes |
| warm-start | `airline_s2_<arm>_heldout/step_best` | never `_fullctx`; `base` uses none |

## Why the two fixes (don't revert them)

1. **`batch_size 1` → 72% of updates had zero gradient.** Under RLVR (advantage +1/0, no
   baseline), a batch where every rollout fails has all-zero advantages and teaches nothing.
   bs1 = one *task* per update, and airline tasks are near-bimodal, so 29/40 updates were
   no-ops (measured from the replay buffer). Box A's bs1 `thoughts_policy` early-stopped at
   b40 with eval declining `0.611 → 0.556 → 0.500 → 0.444` — that's a broken signal, **not**
   evidence Act-PRM is worse. Do not cite it.
2. **`max_seq_len` 8192 → 32768.** Longer episodes were silently dropped from the gradient
   (logged as `n_skipped_seq_len`). Observed max is ~12.7k.

**`batch_size` is not a VRAM increase** — rollouts are generated *sequentially* per task
(`_dispatch_rollouts → run_rollouts`, the per-sample loop), and the backward runs **one
sequence at a time** (`dataloader_batch_size` hard-pinned to 1 in `rl.py` + grad
accumulation). So peak memory scales with `group_size`, not `batch_size × group_size`. It
costs wall-clock (~4×/update). Same reason `max_seq_len 32768` is nearly free.

## User simulator (if LLM errors appear)

tau2 calls `litellm.completion()` internally; we register a litellm **custom provider**
`claude_agent_sdk` (`src/act_prm/environments/tau2bench/litellm_claude_agent_sdk.py`) that
routes to the Claude Agent SDK using the **ambient Claude Code login**. No `ANTHROPIC_API_KEY`,
no `LLAMA_API_KEY`; set `CLAUDE_CODE_OAUTH_TOKEN` in `.env` only if there's no ambient login.
Deliberate oddities: a 0-priced model entry (litellm has no pricing for custom providers and
tau2 computes cost), a worker-thread dispatch (else `asyncio.run` raises inside a running
loop), and a `tau2.config` monkey-patch (tau2 hardcodes gpt-4.1 as the judge default).
Don't switch to the `metagen/` backend — it changes the user sim and breaks comparability.

## Fresh-box preflight (all hit on box A)

- `nvidia-smi -L` — expect 2 GPUs; one run per card.
- venv check: `.venv-tau2/bin/python -c "from tau2.evaluator.evaluator import evaluate_simulation; from tau2.gym.gym_agent import AgentGymEnv; print('ok')"`
  - fails on `pyaudio` → no linux wheel, no portaudio headers, no sudo. tau2 only uses it in
    voice playback: write a stub `site-packages/pyaudio.py` with `paInt8/16/32` + a `PyAudio`
    class that raises. Then `python -m ensurepip` and install what else is missing
    (`elevenlabs`, `rank_bm25`).
- `uv` may be absent — fine, the script falls back to `.venv-tau2/bin/python`.
- model cache `/data/users/$USER/models/hf_cache` needs `models--Qwen--Qwen3-4B-Instruct-2507`.
- `tau2-bench/data` must exist (github unreachable from the devserver).

## Verify after launch

- orchestrator log shows one line per arm **with both running**
- run log shows `batch_size: 4` and `max_seq_len: 32768`
- rollout turns advance (proves the user sim authenticated)
- after a few batches: load `<ckpt>/replay_buffer` with `datasets.load_from_disk`, confirm
  `advantage` ∈ {0.0, 1.0} and that per-batch success isn't almost always 0
- `n_skipped_seq_len` should now be ~0

## Report

Per arm: eval trend by batch (`eval/try_0/final_reward` = held-out success fraction over 18
tasks), best + its step, train reward trend, `n_skipped_seq_len`.

**Timing:** ~2 h/update, so ~3.5 days to early-stop near b40; both arms run concurrently.

## Open risk

Every arm so far peaked at or near its warm-start. If box A's bs4 `thoughts_policy` also
declines through its first two evals (~40 h in), the problem is the setup — most likely
`lr 1e-4` being too aggressive for a strong SFT init — and box B should pause rather than
spend three more GPU-days.
