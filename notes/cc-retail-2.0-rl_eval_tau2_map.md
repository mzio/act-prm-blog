# cc-retail-2.0 — aprm retail uid → tau2 task map (Stage-3 RL eval unblock, task #24)

Read-only forensic pass. Goal: reliable map from the aprm retail dataset's
`unique_data_sample_id` (uid, 0..71) → tau2-bench retail `task id` (0..113), and a
coverage answer for the Stage-3 RL eval set.

Artifacts produced:
- `data/splits/tau2_retail_uid_to_tau2id.json` — `{uid: tau2_task_id}` + `coverage`,
  `rl_eval`, `contamination_analysis`, and per-uid `evidence` blocks.
- This note.

## Method

Naive action-sequence matching fails (expert GPT-5-mini actions ≠ a task's
`evaluation_criteria.actions`; users are reused across up to 5 tasks). Instead I
matched on **stable user identity + the specific write action**:

1. **Canonical `user_id` string as the primary key.** Extracted the tau2 `user_id`
   (e.g. `yusuf_rossi_9620`) on both sides:
   - Dataset: the `find_user_id_*` tool RESULT in `next_obs` plus downstream action
     args (`get_user_details(user_id=…)`, etc.). Regex `[a-z]+_[a-z]+_\d{3,5}`,
     majority vote across the 8 generations. **All 72 uids resolved, dominant count
     (no ambiguity).**
   - tau2 tasks: `find_user_id_by_name_zip` / `find_user_id_by_email` in
     `evaluation_criteria.actions`, else the `user_id`/email/name+zip in
     `user_scenario.known_info` canonicalized through `domains/retail/db.json`
     (name+zip → user_id, email → user_id). Emails in ~11 tasks are fictional/"unknown"
     and don't match the db, so those were resolved by name + order-overlap vs
     `db.users[*].orders`. **All 114 tasks resolved.**
2. **Disambiguate a user's multiple tasks** by exact **write-action** overlap:
   the set of `(action_name, order_id)` for the mutating tools
   (`cancel/modify/exchange/return_*`, `modify_user_address`). Dataset write actions
   taken in ≥2 of 8 gens vs the task's `evaluation_criteria` write actions.
   Tie-break: `reason_for_call` token overlap with the logged user turns.
3. **Per-user greedy bijective assignment** (highest score first, no task reused).

## Result: 72 / 72 matched, bijective, no duplicates

- Every one of the 12 `rl_eval` uids matched with an **exact write-action
  (name + order_id)** hit — high confidence (spot-checked all 12 against their
  runner-up candidates; each assignment is the uniquely-best write-set match).
- 4 uids whose users appear only in email-only tasks (lucas_brown, liam_moore,
  amelia_gonzalez, james_kim) matched once those tasks were name-resolved.

### uid → tau2_id relationship: **scattered, NOT identity/positional**
Examples: uid 0→84, 3→102, 5→3, 24→85, 37→104, 43→35, 58→112. The
dataset uid ordering has no relation to tau2 task ids. (Full map in the json.)

## `rl_eval` uids → tau2 task ids (deliverable 2)

| uid | user_id | tau2 id | tau2 split | write-exact |
|----:|---------|--------:|:----------:|:-----------:|
| 43 | aarav_santos_2259 | 35 | train | 2 |
| 3  | noah_ito_3850 | 102 | **test** | 3 |
| 37 | lucas_brown_6720 | 104 | train | 5 |
| 65 | ivan_johnson_6036 | 77 | **test** | 1 |
| 44 | sofia_hernandez_5364 | 23 | train | 3 |
| 56 | aarav_anderson_8794 | 49 | **test** | 1 |
| 66 | isabella_johansson_2152 | 26 | **test** | 1 |
| 69 | liam_thomas_7882 | 100 | **test** | 2 |
| 4  | mei_ahmed_4909 | 91 | train | 2 |
| 13 | sofia_li_9219 | 99 | train | 2 |
| 58 | yara_silva_7567 | 112 | train | 3 |
| 54 | mei_davis_8935 | 19 | train | 1 |

rl_eval tau2 ids = **{19, 23, 26, 35, 49, 77, 91, 99, 100, 102, 104, 112}**
— 7 in tau2's train split, 5 in tau2's test split.

## Coverage — does the dataset = the tau2 train split? (deliverable 3)

**No.** The 72-uid dataset is a 72-task subset that **spans BOTH tau2 splits**:

| | count |
|---|---|
| distinct tau2 tasks covered | 72 |
| …in tau2 **train** split (74 total) | **46** covered / **28 missing** |
| …in tau2 **test** split (40 total) | **26** covered / 14 not in dataset |

- **The 40 test tasks are NOT entirely uncovered** — 26 of them are in the dataset.
- **Missing tau2 train ids (28):** 2, 8, 10, 13, 16, 20, 22, 24, 25, 44, 46, 48, 50,
  58, 59, 66, 67, 73, 82, 87, 88, 89, 93, 103, 105, 106, 110, 113.
- **Test ids never in the dataset (14, truly unseen by anything):** 5, 12, 33, 36,
  39, 51, 64, 65, 68, 70, 71, 90, 94, 101.

### Contamination of the tau2 test split by our Stage-1/2 training
"Seen" = tau2 ids reached by `act_prm_train ∪ act_prm_eval` dataset uids (used in the
Act-PRM EM + the SFT). rl_eval uids are held out.

- test tasks **seen/contaminated** (20): 9, 17, 18, 32, 38, 40, 42, 45, 53, 55, 56,
  60, 61, 62, 74, 79, 86, 97, 108, 111.
- test tasks in our **rl_eval hold-out** (5, clean + intended): 26, 49, 77, 100, 102.
- test tasks **never in the dataset** (14, cleanest): listed above.

So of the 40 tau2 test tasks, **20 are contaminated**, and **19 are clean** for us
(5 rl_eval-test + 14 never-seen; note 100 appears once).

## RECOMMENDATION for the Stage-3 RL eval set

Two candidate eval sets, plus a recommended hybrid:

**(a) Map `rl_eval` uids → their 12 tau2 ids, eval on those 12.**
- Pros: exactly the hold-out we already carved out; never touched by our Stage-1 EM or
  Stage-2 SFT; the mapping is clean/high-confidence; directly comparable to the SFT
  hold-out story. We also have the expert (GPT-5-mini) demos for these (held out) as a
  ceiling.
- Cons: only 12 tasks (noisy reward estimate); 7 of the 12 sit in tau2's *train*
  split, so numbers aren't the "standard tau2 test benchmark" figure.

**(b) Use the tau2 `test` split (40 truly-standard tasks).**
- Pros: standard, larger, comparable to published τ²-bench numbers.
- Cons: **20 of the 40 are contaminated** — their expert thoughts+actions were in
  `act_prm_train`/`act_prm_eval` and thus trained on in Stage-1/2. Reporting the full
  40 overstates generalization.

**Recommended: hybrid — report the CLEAN, standard-comparable set and the hold-out
separately.**
1. Primary generalization number: the **19 clean tau2 test tasks**
   `{5,12,26,33,36,39,49,51,64,65,68,70,71,77,90,94,100,101,102}`
   (= 14 never-in-dataset test + 5 rl_eval-test). These are (i) in the standard tau2
   test split and (ii) unseen by our Stage-1/2 training → an honest, benchmark-aligned
   eval.
2. Secondary: the **12 rl_eval tasks** as the pipeline-internal hold-out (consistent
   with the SFT eval, and gives a same-distribution comparison).
3. Do **not** report the full 40-task tau2 test split as a generalization metric
   without excluding the 20 contaminated ids (or clearly flagging them).

If a single set is required, prefer the **19 clean test tasks** — it maximizes both
standard comparability and freedom from Stage-1/2 leakage. If simplicity/consistency
with the SFT hold-out matters more, use the **12 rl_eval tasks**.

### Open item
tau2's `AgentGymEnv` is indexed by task position/id; confirm the env loads
`domains/retail/tasks.json` in id order (id == list index here, ids are "0".."113"
contiguous) so the recommended id set can be passed straight through as the eval task
indices.

## DECISION (user-confirmed): hybrid eval, label which is which
Stage-3 RL evaluates on BOTH, reported separately:
- **Primary (benchmark generalization):** the **19 clean tau2 test tasks** — never-in-dataset ∪ rl_eval-in-test — `{5,12,26,33,36,39,49,51,64,65,68,70,71,77,90,94,100,101,102}`. Uncontaminated by Stage-1/2 training.
- **Internal hold-out:** the **12 `rl_eval` tasks** → tau2 ids `{19,23,26,35,49,77,91,99,100,102,104,112}`.
- **Never** report the full 40-task tau2 test split (20/40 are contaminated: their trajectories were in `act_prm_train ∪ act_prm_eval`).
Remaining wiring (#24): make the tau2 `AgentGymEnv` eval on an explicit task-id list (ids are contiguous 0..113 = list index, verified) — add an `eval_task_ids` path + thread the two labeled sets into `run_retail_stage3.sh`. Shared with airline → coordinate the env edit to avoid divergence.

## ADDENDUM: never-in-logs complement (purer RL hold-out)
Reconciled: 72 uids → 72 distinct tau2 tasks (clean bijection). **42 tau2 retail tasks are NEVER in
the logs** (114 − 72), split 28 tau2-train + 14 tau2-test. IDs in `never_in_logs` of the map json.
- **Recommended RL design (matches airline):** train RL on the 72 logged tasks, eval on the 42
  never-in-logs tasks (truly unseen by logs AND SFT). Purer + more training data than carving
  `rl_eval` from the logs. No regeneration needed — the map gives the complement directly.
- Reporting options: all 42 (purest generalization) or the 14 test-split-complement (benchmark-aligned).
- Supersedes the earlier "19 clean test + 12 rl_eval" hybrid if we adopt this; the 14 test-complement ⊂ the 19.
