> **SUPERSEDED** by `handoff_airline_rlvr_boxB.md` (batch_size 4 + max_seq_len 32768).
> Kept for history; do not run from this file.

# Handoff — airline Stage-3 RLVR on a 2-GPU box (`actions_only` + `base`)

Created 2026-08-01. Box A (`devvm54227`, 1 GPU) is running `thoughts_policy`. Box B has
**2 GPUs**, so it takes the two reference arms in parallel — one per card:

| GPU | arm | warm-start |
|---|---|---|
| 0 | `actions_only` | hide-obs SFT `airline_s2_actions_only_heldout/step_best` |
| 1 | `base` | **none** — fresh LoRA on the base policy (tests whether SFT helps at all) |

One gs8 run peaks ~60 GiB at the M-step, so **one run per 80 GiB card — never co-locate two**.

## Coordination

- **Box B runs `actions_only` and `base` only.** Do not run `thoughts_policy` — box A has it
  (started 08-01 02:24, ~31 min/batch).
- Once box B is confirmed running, box A must **drop `actions_only` from its queue**,
  otherwise it launches a duplicate when `thoughts_policy` finishes.
- Remaining unrun arms after this: `expert_thoughts` (oracle upper bound — highest value)
  and `thoughts_base`. Same prompt, `ARMS=expert_thoughts`.
- `git push` from box A before box B pulls — box B needs the `ARMS` selector, the `base`-arm
  support, and the `hf_rlvr` generator config.

## The prompt to paste into Claude Code on box B

```
You're picking up the Act-PRM airline Stage-3 RL work on a fresh 2-GPU devserver. Another box
is already running the `thoughts_policy` arm — your job is the two reference arms,
`actions_only` (GPU 0) and `base` (GPU 1), in parallel. Do NOT run thoughts_policy; it would
duplicate ~20h of compute.

Repo: ~/projects/act-prm-blog, branch act-prm-pytorch. Read CLAUDE.md. FIRST: `git pull` —
arm selection, the `base` arm, and the corrected RLVR config all landed today.

THE EXPERIMENT
Act-PRM infers latent thoughts behind action-only expert demos (Stage 1 EM), SFTs on
thought+action (Stage 2), then RLs online (Stage 3). Stage 3 tests whether the offline
Act-PRM advantage transfers. The comparison is thoughts_policy vs actions_only, all in the
hide-observations regime, each warm-started from its MATCHING hide-obs SFT checkpoint.
`base` is the no-SFT control: it shows whether any SFT init helps online at all.

WHAT TO RUN (one command, after preflight below):
  cd ~/projects/act-prm-blog
  mkdir -p /tmp/aprm/airline_rlvr_pair
  ARMS="actions_only base" setsid nohup ./scripts/run_airline_rlvr_pair.sh \
    > /tmp/aprm/airline_rlvr_pair/run.log 2>&1 &

Arms are round-robined across the GPUs nvidia-smi reports, so with 2 visible GPUs
actions_only lands on GPU 0 and base on GPU 1, running concurrently. Confirm BOTH started
(the orchestrator log prints one line per arm with its GPU). If only one GPU is visible they
will run serially instead — that's still correct, just slower.

Config is baked into the script and must NOT be changed (it has to match box A's arm
exactly): env tau2bench/airline_rlvr (hide_observations:true, last_obs_to_show:1,
negative_rewards:false), generator hf_rlvr (mean_center:false, discount_factor:1.0 -> the
advantage is exactly +1 on success / 0 on failure, no baseline, no discounting),
--group_size 8 --batch_size 1 --max_turns 30 --learning_rate 1e-4 --num_batches 100
--gradient_checkpointing, eval every 10 on the 18 never-seen tau2 tasks, early-stop
patience 3. No observation truncation. The `base` arm is identical except it passes no
--resume_from.

PREFLIGHT — verify these before launching. I hit every one of them on a fresh box today:
1. GPUs: `nvidia-smi -L` — expect 2. One run needs a whole 80GB card (~60 GiB peak at the
   M-step); never put two gs8 runs on one GPU. Pin explicitly with GPU_LIST="0 1" if needed.
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
6. Warm-start ckpt for actions_only must exist:
     checkpoints_lora/act_prm_tau2_airline/hf_qwen3_4b_instruct/airline_s2_actions_only_heldout-*/step_best
   It MUST be the `_heldout` (hide-obs) one, never `_heldout_fullctx` — the script's glob
   already enforces this. Do not relax it. (`base` needs no checkpoint.)
7. STALE-CHECKPOINT TRAP: the runner skips an arm if checkpoints_lora/tau2bench_airline_rlvr/
   <model>/airline_rlvr_<arm>-*/step_best already exists. Old `gc=hf_grpo` (mean-centered,
   superseded) checkpoints can arrive via dotsync and cause a silent skip — this bit us for
   both thoughts_policy and base. If you see "skip (already has step_best)", move the
   gc=hf_grpo dirs into a _stale_hf_grpo/ subfolder and relaunch. Verify each arm started.

USER SIMULATOR / JUDGE — it is litellm AND claude_agent_sdk, don't "fix" it:
tau2 calls litellm.completion() internally for its user simulator, NL-assertions judge, and
env-interface LLM. We register a litellm CUSTOM PROVIDER named `claude_agent_sdk` (see
src/act_prm/environments/tau2bench/litellm_claude_agent_sdk.py), so the model string
`claude_agent_sdk/claude-haiku-4-5` routes to the Claude Agent SDK and authenticates with
the AMBIENT CLAUDE CODE LOGIN on the devserver (it shells out to the bundled Claude Code
CLI). There is NO ANTHROPIC_API_KEY and NO LLAMA_API_KEY. If the box has no ambient login,
set CLAUDE_CODE_OAUTH_TOKEN in .env. Registration is lazy/idempotent and fires from env.py
only because user_llm / nl_assertions_llm / env_interface_llm all start with
`claude_agent_sdk/` in airline_rlvr.yaml — do not change those values.
Two details that look odd but are deliberate: the provider registers a 0-priced model entry
(litellm has no pricing map for custom providers and tau2 computes per-call cost), and
_complete() runs in a worker thread when already inside an event loop (else asyncio.run
raises). env.py also monkey-patches tau2.config because tau2 hardcodes gpt-4.1 as the
default judge/env-interface model, which would otherwise be used silently and fail.
VERIFY AUTH before the long run: after launching, confirm the log shows rollout turns
advancing (e.g. "Generating rollout 0: 4/4"). Turns only advance if the user simulator
replied, so that alone proves auth works; "Using bundled Claude Code CLI" confirms the path.
Do NOT switch to the metagen/<model> (Llama-API) backend unless OAuth is truly unavailable —
it changes the user simulator and results stop being comparable to box A.

DURABILITY — do this before the long run, we already lost a full fleet to a reimage:
  logs/ and checkpoints_lora/ are GITIGNORED, so results exist only on local disk.
  setsid nohup ./scripts/backup_results_daemon.sh > /tmp/backup_results.daemon.log 2>&1 &
  It copies logs/*/metrics.jsonl into tracked results/ and commits every 30 min. It CANNOT
  push: the fwdproxy rejects agent traffic ({"agent_id":"agent:claude_code"} -> 403), so ask
  me to run `git push` periodically. Also confirm the daemon is still alive on each check —
  mine died once silently.

REPORTING
eval/try_0/final_reward is now the held-out SUCCESS FRACTION directly (0-1 over 18 tasks;
multiply by 18 for the count). Report, per arm: the eval trend by batch, the best and its
step, and the train reward trend. Reference points from box A's thoughts_policy run:
eval 11/18 @b10, 10/18 @b20; train reward 0.113 -> 0.375 by b26. Superseded mean-centered
GRPO runs gave actions_only best 13/18 @b30 and base 13/18 @b10 — do NOT cite those as
final numbers; they used a different advantage.

Expect ~31 min/batch, so ~20h per arm if early-stop fires near b40; both arms run
concurrently so wall-clock is ~20h total. Launch, verify the first batch of BOTH arms starts
cleanly, then report status and wait — don't start additional arms without asking.
```

## How the tau2 user simulator / judge is wired (read before debugging LLM errors)

It is **litellm AND claude_agent_sdk** — not one or the other:

- tau2 internally calls `litellm.completion(model=...)` for its **user simulator**,
  **NL-assertions judge**, and **env-interface** LLM (`tau2/utils/llm_utils.py`). We never
  patch those call sites.
- `src/act_prm/environments/tau2bench/litellm_claude_agent_sdk.py` registers a litellm
  **custom provider** (`CustomLLM`) named `claude_agent_sdk`. So the model string
  `claude_agent_sdk/claude-haiku-4-5` routes into `ClaudeAgentSDKLLM._complete()` →
  `ClaudeQueryLLM.sample()` → Claude Agent SDK → **Claude Code OAuth**.
- Auth is the **ambient Claude Code login on the devserver** (it shells out to the bundled
  Claude Code CLI). **No `ANTHROPIC_API_KEY` and no `LLAMA_API_KEY` are needed.** For a
  headless box with no ambient login, set `CLAUDE_CODE_OAUTH_TOKEN` in `.env`.
- Registration is lazy + idempotent, triggered in `env.py` only when one of
  `user_llm` / `nl_assertions_llm` / `env_interface_llm` starts with `claude_agent_sdk/`.
  All three are set to `claude_agent_sdk/claude-haiku-4-5` in `airline_rlvr.yaml`.
- Two implementation details that exist for a reason — don't "simplify" them:
  - it registers a **0-priced model entry**, because litellm has no pricing map for a
    custom provider and tau2 computes cost per call;
  - `_complete()` dispatches to a **worker thread** when already inside a running event
    loop, else `asyncio.run()` raises "cannot be called from a running event loop".
- `env.py` also **monkey-patches `tau2.config`**, because tau2 hardcodes `gpt-4.1` as the
  default NL-assertions / env-interface model — without the patch the judge silently uses
  gpt-4.1 (and fails with no OpenAI key).
- Alternative backend if OAuth is unavailable: `metagen/<model>` →
  `litellm_metagen.py` (Llama-API passthrough, needs `LLAMA_API_KEY=LLM|<id>|<secret>`).
  Only switch if you must — it changes the user simulator, so results stop being
  comparable to box A.

Verify it works on box B **before** the long run: launch, then confirm the log shows
rollouts advancing turns (e.g. `Generating rollout 0: 4/4`). Turns only advance if the user
simulator actually replied, so that is sufficient proof of auth. A line like
`Using bundled Claude Code CLI: ...` confirms the SDK path.

## Why the config is what it is (don't "fix" these)

| knob | value | reason |
|---|---|---|
| `negative_rewards` | `false` | RLVR: +1 success / 0 fail (not ±1) |
| `mean_center` | `false` | no GRPO baseline — raw verifiable reward |
| `discount_factor` | `1.0` | generator default 0.9 would make earlier steps `0.9^k · r` |
| `hide_observations` | `true` | must match the Stage-2 SFT regime |
| `last_obs_to_show` | `1` | keeps the latest tool result actionable; older obs → `"..."` |
| warm-start | `*_heldout` | hide-obs RL must start from hide-obs SFT, never `_fullctx` |
| `base` arm | no `--resume_from` | control: no SFT init at all |

Consequence of RLVR with no baseline: **only successful rollouts produce gradient**;
failures contribute exactly zero (rather than being pushed down as GRPO would).

## Status at handoff (box A, 08-01 15:51)

| arm | generator | batches | eval (n/18) | best | where |
|---|---|---|---|---|---|
| thoughts_policy | `hf_rlvr` | 26/100 | 10→11/18, 20→10/18 | 0.611 @10 | box A |
| actions_only | `hf_rlvr` | — | — | — | **box B GPU 0** |
| base | `hf_rlvr` | — | — | — | **box B GPU 1** |
| expert_thoughts | `hf_rlvr` | — | — | — | unrun (next priority) |
| thoughts_base | `hf_rlvr` | — | — | — | unrun |

Superseded (mean-centered GRPO, do not cite): base 13/18 @b40 stop, actions_only 13/18
@b30 (crashed b42), thoughts_policy 8/18 @b13, expert_thoughts 7/18 @b13.
