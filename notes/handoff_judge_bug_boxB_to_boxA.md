# Box B → Box A: the NL-assertion judge silently scores ~10% of episodes as reward=0

Created 2026-08-01 by box B (`devvm26907`, 2 GPUs). **Box B has NOT launched the bs4 arms
yet** — it is holding until you agree on this fix, because fixing the judge changes the
scoring function and would break the head-to-head if only one box has it.

## TL;DR

A judge crash is scored identically to a failed episode. It hit **26 episodes (~10%)** of
box B's bs1 run and it has been happening since **07-27** (172 records, 28–38/day), so it
affects `thoughts_policy`, every superseded GRPO run, and any bs4 run you start now.

Decision needed: **do we fix the judge fleet-wide and restart, or keep it as shared noise?**

## The bug (verified end to end)

1. `env.py` calls `evaluate_simulation(... ALL_WITH_NL_ASSERTIONS)`.
2. The judge goes through our litellm custom provider, which built the handler as
   `ClaudeQueryLLM(model, max_turns=1)` — hardcoded.
3. The SDK raises `Reached maximum number of turns (1)` whenever the model's first turn
   isn't a final text answer.
4. `ClaudeQueryLLM._query` catches **every** exception, `print`s
   `ClaudeQueryLLM query error: ...`, and returns `None`.
5. `_reply_text(handler, None)` → `""` → `message.content = ""`.
6. tau2 does `json.loads("")` → `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
   — that exact message is the fingerprint of parsing an empty string, which is how this
   was tracked down.
7. `env.py` caught it and returned **`reward = 0.0`**.

So the reward says "the agent failed" when the truth is "the judge never ran".

## Evidence

- `logs/tau2_judge_failures.jsonl`: 172 records, 07-27 → 08-01. 26 since box B launched.
- Failures come in contiguous per-task blocks — one batch's 8 rollouts partially failing:
  `task 14 ×6 (17:19–17:46)`, `task 15 ×4 (18:16–18:30)`, `task 23 ×10 (21:37–22:38)`,
  plus singles on 17/20/21/24 and eval tasks 2/39.
- It is **task-linked but stochastic** — roughly 4–6 of a task's 8 rollouts fail, not 8/8.
- Tasks 14 and 23 are the two gift-card tasks, both asserting *"total … is $327"*.
  Plausible (unproven) mechanism: asking the judge to verify a numeric total nudges it to
  attempt a computation step, which consumes turn 1 and needs turn 2.

Ruled out: markdown-fenced JSON (the `_tolerant_loads` shim already handles it), trajectory
length (2/10/30/60-turn prompts all pass), 8-way concurrency on the shared cached handler
(8/8 pass), assertion count (task 15 has only 2).

## Impact on the numbers

Mostly training, not eval — **24 of 26 hit training tasks, 2 hit held-out tasks**:

- box B `actions_only` **11/18 is clean** (0 judge failures inside its eval window).
- box B `base` 6/18 is under-counted — both eval failures landed in its window, so the true
  value is **6–8 of 18**.
- Under RLVR (advantage +1/0, no baseline) a false zero removes that episode's gradient
  *and* counts against the success rate. At bs1 one bad task killed a whole update; at bs4
  it degrades to ~10% label noise, so **bs4 already mitigates most of the damage**.

Worth noting for your bs1 post-mortem: box B measured **92–93% zero-gradient updates**
(vs your 72%), and `n_skipped_seq_len` of **100 and 90** out of ~104–120 episodes — i.e. the
8192 `max_seq_len` was dropping nearly every episode. Your two fixes address the dominant
cause; this judge bug is a smaller, independent one.

## The fix (implemented on box B, not yet launched)

1. `litellm_claude_agent_sdk.py` — `max_turns=1` → `_JUDGE_MAX_TURNS = 4`.
   Verified free on the happy path: identical output shape/length at 1 vs 4.
2. `litellm_claude_agent_sdk.py` — raise `ClaudeAgentSDKError` when the handler returns
   `None`, instead of returning `""`. Verified: surfaces as
   `APIConnectionError: Claude Agent SDK returned no response for model='claude-haiku-4-5'`
   rather than a silent empty parse.
3. `env.py` — retry the evaluator `JUDGE_MAX_ATTEMPTS = 3` times (failures are stochastic,
   so most clear on retry); if all attempts fail, log it and return `judge_failed: True` in
   the info payload **instead of fabricating `reward=0.0`**.
4. `env.py` — the tolerant-JSON shim now stashes the raw judge text so
   `tau2_judge_failures.jsonl` records `judge_output` / `judge_output_empty`. The old
   records only had the exception type, which is why this took reconstruction.

Caveat: I could not reproduce the SDK turn-limit error in ~20 controlled calls, so fix (1)
is a well-motivated but **unproven** root-cause fix. Fixes (2)–(4) are unconditional
improvements — they make the failure loud and non-destructive regardless of the trigger.

Still open: `judge_failed` is reported but the generator does not yet **drop** those episodes
from the batch. Worth doing if we restart, so unscored episodes don't dilute the update.

## What box B needs from you

- **Agree or reject the fix.** If you agree, take these changes and restart `thoughts_policy`
  so all arms share one scoring function; box B will launch `actions_only` + `base` at the
  same time. We're hours into the bs4 fleet, not days — this is the cheapest moment to do it.
- If you reject it, box B will launch the bs4 arms **unfixed** to match you exactly, and we
  treat the ~10% as shared noise. Say so and box B launches immediately.

Box B is otherwise ready: merge resolved, bs1 arms killed, their checkpoints moved to
`checkpoints_lora/tau2bench_airline_rlvr/<model>/_bs1_superseded/`, both GPUs idle.
