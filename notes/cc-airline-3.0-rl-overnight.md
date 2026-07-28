# cc-airline-3.0 — Stage-3 RL fleet (overnight autonomous run)

**Started:** 2026-07-28 ~06:11. Branch `act-prm-pytorch`, commit `a47d856`.

## What's running
5 RL runs (GRPO/RLVR on tau2-airline gym), Qwen3-4B, warm-started from each
variant's **hide-regime SFT `step_best`**. 1 run/GPU serial:
- GPU0: `base` → `thoughts_policy` → `thoughts_base`
- GPU1: `actions_only` → `expert_thoughts`

Config: `group_size 4`, `batch_size 2`, `max_turns 20`, `max_tokens 2048`,
`num_batches 100`, `eval_every 10`, **`early_stop_patience 3`** (stop if eval
`final_reward` flat for 3 evals = 30 steps), train=32 / eval=18 (taskmap).
User-sim = `claude_agent_sdk/claude-haiku-4-5`.

## The comparison
Does RL from *inferred*-thought SFT (`thoughts_policy`/`thoughts_base`) land
between `actions_only` (baseline) and `expert_thoughts` (ceiling) on task success
(eval `final_reward` over the 18 never-seen tasks). `base` = RL floor (no SFT).

## Fixes landed this session (all in commit a47d856)
1. **Concurrent `env.step`** (generator base.py): the group's `env.step`/user-sim
   calls now run via `asyncio.gather` over `AsyncTau2BenchEnv.step_async` instead of
   a serial list comp. **~4.2× speedup** (11 obs/min vs 2.6). GPU generation was
   already batched (`group_size × ctx`, num_return_sequences=1); the serial part was
   the N user-sim round-trips (~30–74s each).
2. **Removed 6 headless `breakpoint()` landmines** (action_utils, rl.py×2, gen base,
   args) that dropped background runs into pdb → EOF → death.
3. **Fixed `get_messages_from_text` crash** on a non-dict tool-call body (int/list)
   → routes to `invalid_tool_call`.
4. **Eval-based early-stop** added to RLTrainer (`early_stop_patience`, wired via
   pg.yaml + args). Eval reward (not train) = generalization-correct signal.

## VRAM note
1 run/GPU is required: a single rollout balloons to ~44–68 GB at max_turns 20
(2/GPU OOMs — that killed 3 runs earlier). Watchdog warns >76 GB.

## When you wake — how to check
- Status trail: `/tmp/aprm/monitor_overnight.log` (health + backups every 20 min).
- Per-run curves: `logs/act_prm_tau2bench_airline/.../airline_rl_*/metrics.jsonl`
  → plot `eval/try_0/final_reward` and `eval/try_0/final_reward` best vs step.
- Live logs: `/tmp/aprm/airline_rl/airline_rl_<variant>.log`.
- Expected timing: SFT'd runs ~12 min/step → early-stop ~6 h each; `base` slower.

## Durability
- Code committed (`a47d856`) + dotsync bundle `~/.claude/act-prm-backups/act-prm-blog-devvm27322.bundle` (refreshed every 20 min by the monitor).
- Artifacts (metrics/notes/corpora/splits) tarred to `~/.claude/act-prm-backups/artifacts/` every 20 min.
- SFT ckpts: public HF `mzio/aprm-sft-tau2-airline` + `~/aprm-sft-airline-ckpts-devvm27322.tar.gz`.

## NOT running (deliberate)
- 8B EM pipeline (killed — it sniped GPUs and OOM'd the RL fleet earlier). Re-queue
  only after RL, pinned so its wait_gpu_free can't contend.
