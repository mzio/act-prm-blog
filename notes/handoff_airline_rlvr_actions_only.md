# Handoff — airline Stage-3 RLVR, `actions_only` arm (box B)

Created 2026-08-01. Box A (`devvm54227`) is running `thoughts_policy`; this hands the
matched baseline arm `actions_only` to a second GPU box.

## Coordination

- **Box B runs `actions_only` only.** Do not run `thoughts_policy` — box A has it (started
  08-01 02:24, ~31 min/batch).
- Once box B is confirmed running, box A must **drop `actions_only` from its queue**,
  otherwise it launches a duplicate when `thoughts_policy` finishes.
- Next most valuable arm if a third GPU appears: `expert_thoughts` (oracle upper bound) —
  same prompt with `ARMS=expert_thoughts`.
- `git push` from box A before box B pulls — box B needs the `ARMS` selector and the
  `hf_rlvr` generator config.

## The prompt to paste into Claude Code on box B

```
You're picking up the Act-PRM airline Stage-3 RL work on a fresh devserver. Another box is
already running the `thoughts_policy` arm — your job is the matched baseline arm,
`actions_only`. Do NOT run thoughts_policy; it would duplicate 20h of compute.

Repo: ~/projects/act-prm-blog, branch act-prm-pytorch. Read CLAUDE.md. FIRST: `git pull` —
the arm-selection support and the corrected RLVR config landed today and you need them.

THE EXPERIMENT
Act-PRM infers latent thoughts behind action-only expert demos (Stage 1 EM), SFTs on
thought+action (Stage 2), then RLs online (Stage 3). Stage 3 tests whether the offline
Act-PRM advantage transfers. The comparison is thoughts_policy vs actions_only, both in the
hide-observations regime, both warm-started from their MATCHING hide-obs SFT checkpoint.

WHAT TO RUN (one command, after preflight below):
  cd ~/projects/act-prm-blog
  mkdir -p /tmp/aprm/airline_rlvr_pair
  ARMS=actions_only setsid nohup ./scripts/run_airline_rlvr_pair.sh \
    > /tmp/aprm/airline_rlvr_pair/run.log 2>&1 &

Config is baked into the script and must NOT be changed (it has to match the other arm
exactly): env tau2bench/airline_rlvr (hide_observations:true, last_obs_to_show:1,
negative_rewards:false), generator hf_rlvr (mean_center:false, discount_factor:1.0 -> the
advantage is exactly +1 on success / 0 on failure, no baseline, no discounting),
--group_size 8 --batch_size 1 --max_turns 30 --learning_rate 1e-4 --num_batches 100
--gradient_checkpointing, eval every 10 on the 18 never-seen tau2 tasks, early-stop
patience 3. No observation truncation.

PREFLIGHT — verify these before launching. I hit every one of them on a fresh box today:
1. GPU: `nvidia-smi -L`. One run needs a whole 80GB card (~60 GiB peak at the M-step).
   Never co-locate two gs8 runs on one GPU. The runner reads GPU indices from nvidia-smi;
   set GPU_LIST="0" to pin.
2. venv: the per-box .venv-tau2 may be an incomplete dotsync copy. Check:
     .venv-tau2/bin/python -c "from tau2.evaluator.evaluator import evaluate_simulation; \
       from tau2.gym.gym_agent import AgentGymEnv; print('ok')"
   If it fails on `pyaudio`: there are no linux wheels and portaudio headers aren't
   installed (no sudo). tau2 only uses pyaudio inside voice-playback functions, so write a
   stub module at .venv-tau2/lib/python3.12/site-packages/pyaudio.py defining paInt8/16/32
   and a PyAudio class that raises. Then `python -m ensurepip` and pip install whatever else
   is missing — for me that was `elevenlabs` and `rank_bm25`.
3. uv may not be installed. Not required — the script falls back to .venv-tau2/bin/python.
   (`curl -LsSf https://astral.sh/uv/install.sh | sh` works from YOUR shell if you want it.)
4. Model cache is per-box: /data/users/$USER/models/hf_cache must have
   models--Qwen--Qwen3-4B-Instruct-2507, else re-download via the proxy.
5. tau2-bench/data must exist (github is unreachable from the devserver, so clone it from a
   github-capable shell if missing).
6. Warm-start ckpt must exist:
     checkpoints_lora/act_prm_tau2_airline/hf_qwen3_4b_instruct/airline_s2_actions_only_heldout-*/step_best
   It MUST be the `_heldout` (hide-obs) one, never `_heldout_fullctx` — the script's glob
   already enforces this. Do not relax it.
7. STALE-CHECKPOINT TRAP: the runner skips an arm if checkpoints_lora/tau2bench_airline_rlvr/
   <model>/airline_rlvr_actions_only-*/step_best already exists. Old `gc=hf_grpo` (mean-
   centered, superseded) checkpoints can arrive via dotsync and cause a silent skip. If you
   see "skip (already has step_best)", move the gc=hf_grpo dirs into a _stale_hf_grpo/
   subfolder and relaunch. Verify the arm actually started.

DURABILITY — do this before the long run, we already lost a full fleet to a reimage:
  logs/ and checkpoints_lora/ are GITIGNORED, so results exist only on local disk.
  setsid nohup ./scripts/backup_results_daemon.sh > /tmp/backup_results.daemon.log 2>&1 &
  It copies logs/*/metrics.jsonl into tracked results/ and commits every 30 min. It CANNOT
  push: the fwdproxy rejects agent traffic ({"agent_id":"agent:claude_code"} -> 403), so ask
  me to run `git push` periodically. Also confirm the daemon is still alive on each check —
  mine died once silently.

REPORTING
eval/try_0/final_reward is now the held-out SUCCESS FRACTION directly (0-1 over 18 tasks;
multiply by 18 for the count). Report the eval trend per batch, the best and its step, and
the train reward trend. Reference points from the other box's thoughts_policy run:
eval 11/18 @b10, 10/18 @b20, train reward 0.113 -> 0.375 by b26. Superseded mean-centered
GRPO runs gave actions_only best 13/18 @b30 and base 13/18 @b10 — do not compare those as
final numbers; they used a different advantage.

Expect ~31 min/batch, so ~20h if early-stop fires near b40. Launch, verify the first batch
starts cleanly, then report status and wait — don't start additional arms without asking.
```

## Why the config is what it is (don't "fix" these)

| knob | value | reason |
|---|---|---|
| `negative_rewards` | `false` | RLVR: +1 success / 0 fail (not ±1) |
| `mean_center` | `false` | no GRPO baseline — raw verifiable reward |
| `discount_factor` | `1.0` | generator default 0.9 would make earlier steps `0.9^k · r` |
| `hide_observations` | `true` | must match the Stage-2 SFT regime |
| `last_obs_to_show` | `1` | keeps the latest tool result actionable; older obs → `"..."` |
| warm-start | `*_heldout` | hide-obs RL must start from hide-obs SFT, never `_fullctx` |

Consequence of RLVR with no baseline: **only successful rollouts produce gradient**;
failures contribute exactly zero (rather than being pushed down as GRPO would).

## Status at handoff (box A, 08-01 15:51)

| arm | generator | batches | eval (n/18) | best |
|---|---|---|---|---|
| thoughts_policy | `hf_rlvr` | 26/100 | 10→11/18, 20→10/18 | 0.611 @10 |
| actions_only | `hf_rlvr` | — | — | **this handoff** |

Superseded (mean-centered GRPO, do not cite): base 13/18 @b40 stop, actions_only 13/18
@b30 (crashed b42), thoughts_policy 8/18 @b13, expert_thoughts 7/18 @b13.
