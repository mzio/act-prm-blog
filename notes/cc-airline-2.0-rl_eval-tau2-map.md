# cc-airline-2.0 — rl_eval → tau2 task-index map (Stage-3 prerequisite)

**Goal:** to eval Stage-3 RL strictly on the held-out `rl_eval` tasks, map each
`rl_eval` `unique_data_sample_id` (from `mzio/aprm-tau2-airline-gpt5m_med-gs4-s0-train`)
→ the corresponding **tau2-bench airline gym task id** (`registry.get_tasks_loader("airline")()`).
airline split: `rl_eval = [16, 1, 20, 30, 2, 5]`.

## What it is NOT (all tested + disproved)
- `uid == registry load-order index`: **NO**. Counterexample: uid 15 = *book NY→Seattle
  May 20*; `tau2[15]` = *change ATL→PHL (Aarav Garcia)*. Different task.
- `uid == np.random.seed(0) shuffle index`: **NO** (0/13 keyword match).
- `uid == train-split position`: **NO**. tau2 airline `task_splits` = {train:30 (id
  strings like `train[15]='23'`), test, base}; positions don't match.
- Provenance unknown (dataset author unsure how uid was assigned).

## Why heuristic matching is hard
- `user_id` is NOT unique per task: 34 unique user_ids across 50 tasks (multiple tasks
  per customer).
- Trajectories reference MANY users/reservations (the agent queries the DB), incl. a
  recurring default-ish `sara_doe_496`, so "primary user_id" is noisy.
- Generic keyword overlap (cancel/refund/reservation) collides many uids onto one task.

## Partial result (needs verification before use)
High-confidence rl_eval matches by combined user_id+reservation overlap:
`uid 16→task 43`, `uid 1→task 37`, `uid 20→task 5`, `uid 2→task 26`.
Unresolved: `uid 30`, `uid 5` (both surfaced the shared default `sara_doe_496`).

## Robust path (do at Stage-3 time — only 6 tasks)
1. From each rl_eval trajectory, extract the **actually-modified reservation id** (the
   arg to the write tool call: book/cancel/update_reservation) + the served user_id.
2. Match to the tau2 task whose **evaluation/action-checks** reference that same
   reservation id / user_id (the task's ground-truth target — unique).
3. OR a quick LLM match: rl_eval opening request → tau2 task `reason_for_call`.
4. Then make the Stage-3 gym env eval on those specific task ids (it currently
   shuffle-splits by count via `data_seed`, so this needs a small env change to accept
   an explicit task-id list).

Status: deferred (Stage 3 pending; Stage 1–2 for 4B + 8B running). Not a blocker yet.
