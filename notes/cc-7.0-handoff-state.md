# cc-7.0 — Complete handoff (written for a cold pickup after context compaction)

> **RETRACTION (08-24) — rollout/completion numbers only.** All task-completion figures
> below are superseded; see [cc-8.0](cc-8.0-rollout-confound.md). The airline
> `thoughts_policy` 72.2% (+22.2pp) did not replicate — 55.6% re-running the SAME seed,
> 44.4–50.0% at seeds 0/1/7 — and at 3 rollouts/task the pooled difference is +0.6pp
> (p=0.765). The harness also confounds arm with measurement time (`actions_only` scored
> 53.7% at 09:39 and 33–39% at 17:50 the same day). Teacher-forced Stage-2 PPL/accuracy in
> this note are computed offline from fixed checkpoints and are NOT affected.



**Read this first.** Companions: **[cc-5.0](cc-5.0-sft-lr-investigation.md)** = running
numbers (auto-refreshed table), **[cc-6.0](cc-6.0-session-report.md)** = narrative report.
This file is self-contained: what happened, what is true, what is running, and how to
operate it.

---

## 0. Sixty-second orientation

Everything runs **from cron**, never from an agent shell:

```
*/5  scripts/sweep_guard.sh        # the queue. sources auth env, back-fills .done, runs stages in order
*/20 scripts/snapshot_results.sh   # notes/CSVs/figures, dotsync backup, auto git commit
*/15 scripts/upload_all_envs.sh    # publish adapters to 3 public HF repos
```

Status check:
```bash
tail -5 /tmp/aprm/guard.log
ps -eo pid,etime,args | grep -E "main_pytorch|scripts/run_" | grep -v grep
uv run --no-project python scripts/report_sft_sweep.py --match _lr3e_3
uv run --no-project python scripts/report_rollout_eval.py
for g in control expert_all finance_v3 finance_rollout multirollout bestgen rollout; do
  [ -f /tmp/aprm/$g/ALLDONE ] && echo "DONE $g" || echo "pending $g"; done
```

**Never launch a driver from an agent shell.** It dies with the process group, and a launch
the guard cannot see causes duplicate runs that corrupt finished metrics (§6.3).

---

## 1. What this session found (the short version)

1. **Stage-2 SFT never trained.** Everything shipped at `lr 4e-5` barely moved the model:
   max |ΔW| 3.7e-5 against base weights ~1e-2, `lora_A` still at its seeded init. All prior
   Stage-2 tables were invalid, and it explains the Stage-3 RL null result (all four RLVR
   arms warm-started from ~the same base model).
2. **lr 3e-3 works.** Chosen by probing a *thought* arm; `actions_only` alone would have
   picked 1e-3. All 12 hide arms then converged at 150 batches.
3. **Rollout inverts the perplexity ranking.** `expert_thoughts` has the best action-token
   PPL in every domain yet gives +0.0pp completion on retail and −16.7pp on airline.
   `thoughts_policy` is 1.8× baseline on retail and 1.44× on airline.
4. **Why the oracle fails**: ~50% of GPT-5-mini's logged actions carry no reasoning
   (retail 318/635, airline 46%, finance 71%), so `expert_thoughts` was trained to *not*
   think and reasons before only 0–8% of its tool calls at rollout.
5. **Not significant yet.** Paired McNemar: retail p=0.29, airline p=0.29, pooled **0.077**.
6. **Volume confound closed** on both domains and both metrics (§3).
7. **Finance v1 results withdrawn** — 76% question-level eval contamination (§4).

---

## 2. Current numbers

**Teacher-forced, hide, lr 3e-3, 150 batches** (action-subspan PPL / accuracy):

| domain | actions_only | thoughts_base | thoughts_policy | expert_thoughts |
|---|---|---|---|---|
| retail | 2.3422 / .7726 | 2.2003 / .7890 | 2.2059 / .7925 | **2.1071 / .7973** |
| airline | 2.4506 / .7629 | 2.3981 / .7644 | 2.3868 / .7691 | **2.3383 / .7732** |
| finance | *withdrawn — contaminated eval* | | | |

Act-PRM recovers **58% (retail) / 57% (airline)** of the oracle PPL gap.

**Rollout, task completion on never-in-logs tasks:**

| arm | retail (42) | airline (18) |
|---|---|---|
| actions_only | 11.9% (5/42) | 50.0% (9/18) |
| expert_thoughts | 11.9% (+0.0) | 33.3% (**−16.7pp**) |
| thoughts_base | 16.7% (+4.8) | 44.4% (−5.6) |
| **thoughts_policy** | **21.4% (+9.5pp)** | **72.2% (+22.2pp)** |

**Mechanism check** (reasoning that actually precedes a tool call — NOT talk-only turns,
which an earlier buggy measurement miscounted):

| | reasoning before call | tool calls used | completion |
|---|---|---|---|
| retail actions_only | 7% | 3905 | 11.9% |
| retail thoughts_policy | 21% | 2909 | 21.4% |
| airline actions_only | 2% | 966 | 50.0% |
| airline expert_thoughts | 0% | 1939 | 33.3% |
| airline thoughts_policy | 7% | 690 | 72.2% |

Supported claim: thought-training yields a **more efficient** policy (25–29% fewer tool
calls at higher completion). NOT supported: "Act-PRM policies reason before acting" — at
most 21% of calls carry any reasoning.

---

## 3. The volume-matched control (closed)

Retail thought corpora had 1043 supervised steps vs the baseline's 635 (the relabel emitted
up to 2 generations/task; the export kept both). Retrained `thoughts_policy` on one
generation per task:

| retail | steps | PPL / acc | completion | McNemar p |
|---|---|---|---|---|
| actions_only | 635 | 2.3422 / .7726 | 5/42 = 11.9% | — |
| **matched** | **627** | **2.2119 / .7918** | **8/42 = 19.0%** | 0.453 |
| as run | 1043 | 2.2059 / .7925 | 9/42 = 21.4% | 0.289 |

Survives on both metrics (+7.1pp matched vs +9.5pp unmatched), and the two Act-PRM runs
solve overlapping task sets (25, 65, 73). Airline was already matched at 221 steps.

---

## 4. The finance data leak (why v1 is withdrawn, what v3 is)

The aprm finance split partitioned `unique_data_sample_id`s, but ~2.5 uids share each
`finqa_reasoning` question, so questions leaked across splits: **76% of act_prm_eval and
91% of rl_eval were also in act_prm_train**. Retail/airline are clean (zero overlap).

Second discovery: the 29 questions with "no usable trajectory" have 123 trajectories, **all
reward=0** — the split's usable filter is `done + reward>0`, so those are exactly the
questions GPT-5-mini *failed*. They are a difficulty-selected tail, NOT a clean hold-out.

**v3 fix** (pure subselection — no relabel, no regeneration): question text matches
`finqa_reasoning.csv` at 100%, resolving the `uid → CSV row` TODO. The 50 demo-bearing
questions are split **40 train / 10 eval by question**; each arm's pool filtered by question.
Verified zero overlap on all four arms.

```
train 40 Q: 116 traj / 1347 steps   (thought arms 122 / 1409)
eval  10 Q:  25 traj /  363 steps   (thought arms  27 /  398)
hard  29 Q: expert-failure, rollout only
```

Finance rollout uses **two sets, reported separately, never pooled**: the fair 10 (expert
solved them; same questions the SFT curves score) and the hard 29 (labelled as such).

---

## 5. Queue: order, scope, ETAs

Gates in `sweep_guard.sh`, each writing `ALLDONE` under `/tmp/aprm/<name>/`.

| # | stage | scope | ETA |
|---|---|---|---|
| running | **expert_thoughts_all** | retail + airline: 2 SFT + 2 rollouts | ~8–11h |
| 2f | **finance v3** | 5 arms SFT, clean eval | ~13.5h |
| 2g | **finance rollout** | 5 arms × {fair 10, hard 29} | ~6h |
| 2g2 | **multi-rollout (3/task)** | baseline + thoughts_policy, both domains | ~11h |
| 2h | best-generation Act-PRM | 6 SFT + 6 rollouts | ~25h |
| 2z | full-context rollout | 8 checkpoints — curiosity, last | ~8h |
| 3 | full-context SFT matrix | 12 arms | ~40h |

**Measured unit costs**: retail SFT 3.5h, airline SFT 2.8h, finance SFT 2.7h, retail rollout
(42 tasks) 74 min, airline rollout (18 tasks) 38 min.

**expert_thoughts_all scope**: ALL training tasks in both domains (retail 49, airline 21).
Filtering is *within* tasks — only turns with reasoning become targets: retail 635 → **317**
(50%), airline 221 → **119** (54%). Finance excluded here because `run_finance_v3.sh`
already trains it against the clean eval.

**Multi-rollout is the one that matters for significance.** `eval_group_size: 1` in
`configs/trainer/pg.yaml` is why every eval did one attempt per task; `--eval_group_size 3`
gives 3 per task in one pass (n: 42→126 retail, 18→54 airline). No other pending stage can
move p=0.077 — they add interpretation, not evidence.

---

## 6. Traps already hit — do not re-learn these

1. **cron's stripped env breaks Claude auth.** `clicat` fails → the `claude` CLI errors →
   every episode dies on turn 1 → recorded as a plausible `0/N`. This silently produced
   8 invalid rollouts before it was caught. Fixed by sourcing `/tmp/aprm/agent_env.sh` in
   the guard. **If rollouts show 1.00 generate-calls/task again the snapshot is stale —
   refresh from an interactive shell:**
   ```bash
   env | grep -vE "^(_|PWD|OLDPWD|SHLVL)=" > /tmp/aprm/agent_env.sh
   ```
2. **A dead judge is invisible to the calls-per-task test.** The policy acts normally; only
   scoring collapses (`grade_sample` returns `"no"` on failure). `check_rollout_valid.py`
   also flags >10% query errors with zero score. Most dangerous on the hard-29 set where
   near-zero is expected. Judge verified working (gold→yes, nonsense→no).
3. **Duplicate runs corrupt finished metrics.** Two causes, both fixed: `run_one` checked
   `.done` *before* `wait_gpu_free` (a driver could wait hours then relaunch a finished
   arm); and the guard's driver-detection regex omitted a driver. Symptom: the batch counter
   resets to 0 mid-file. Detect via `[r['progress/batch'] for r in rows]`, truncate at reset.
4. **`step_best` is written at the FIRST eval** — it never means "finished". Completion is
   an explicit `.done` marker, with a back-fill pass in the guard.
5. **`NUM_BATCHES` / `LR` must ride in the run tag**, else a budget change silently skips.
6. **Run dirs sanitise `-` and `.` to `_`** (`lr1e-4` → `lr1e_4`). Globs must match.
7. **github is blocked** from the agent session (fwdproxy 403); HF and the Meta proxy work.
   The user pushes; adapters auto-publish to HF.
8. **Measure before claiming.** Three claims in this session were wrong on first pass and
   corrected only because they were checked: tokens/call as evidence of "thinking"
   (counted talk-only turns), the 29 questions as a "clean hold-out" (they are
   expert-failures), and "transient outage" for what was a reproducible cron-env bug.

---

## 7. Scripts

| script | purpose |
|---|---|
| `sweep_guard.sh` | the queue |
| `snapshot_results.sh` | notes/figures/backup/commit |
| `upload_all_envs.sh` | HF publish, 3 domains |
| `run_sft_lr_matrix.sh` / `run_sft_sweep.sh` | Stage-2 matrix (`LR=`,`VARIANTS=`,`NUM_BATCHES=`,`REGIMES=`,`ACTION_ONLY=`,`PATIENCE=`) |
| `run_matched_control.sh` | volume-matched control (done) |
| `run_expert_all.sh` | expert_thoughts_all (retail, airline) |
| `run_finance_v3.sh` | finance Stage-2, clean eval |
| `run_finance_rollout.sh` | finance rollout, fair-10 + hard-29 |
| `run_multirollout.sh` | 3 rollouts/task (`NTRIES=`, `ARMS=`) |
| `run_bestgen_sft.sh` | best-generation Act-PRM, all domains |
| `run_sft_rollout_eval.sh` | tau2 rollout (`REGIME=hide\|full`, `SMOKE=1`) |
| `probe_sft_lr.sh` / `report_lr_probe.py` | LR probe |
| `check_rollout_valid.py` | **validity gate** |
| `report_sft_sweep.py` / `report_rollout_eval.py` | results tables |
| `plot_sft_curves.py` / `plot_lr_comparison.py` | figures → `notebooks/figs_sft/`, mirrored to `~/.claude/act-prm-figs/` |
| `make_bestgen_corpus.py` / `make_matched_corpus.py` / `make_finance_pools_v3.py` | corpus subselection |
| `dump_rollout_generations.py` | read generations back from replay buffers |
| `make_finance_split_v2.py` / `make_finance_clean_eval_corpora.py` | superseded by v3; kept for provenance |

---

## 8. Code / config changes made this session

- **lr 3e-3** is the working LR (yamls still ship 4e-5; drivers override).
- `--require_thought` — train only on turns with reasoning (targets only, context intact).
- `--train_action_only` — ablation, **off by default**; loss is on the full thought+action span.
- `action_mask` in `prepare_minibatch` → `train/actiononly_{ppl,accuracy}` (metrics only,
  never touches the gradient).
- `action_start_token` moved to `trainer/utils.py`, shared by loss mask and eval so the
  trained span and scored span cannot drift.
- `--eval_query_ids` + explicit question-id selection in the snorkel_finance env.
- `claude_agent_sdk` lazily registered in `llm_handlers/__init__.py`; finance judge resolves
  to `ClaudeQueryLLM` / claude-sonnet-4-5.
- `train_rl_finance.sh` runs in **`.venv-tau2`** (SDK absent from the base venv).
- Merge resolution of `origin/act-prm-pytorch` (7 conflicts) + port of the snorkel_finance
  gym from the recovered `bk40015/mz-airline` branch.

## 9. Data artifacts created

```
data/splits/snorkel_finance_uid_to_qid.json      # 357 uids -> question ids (100% match)
data/splits/snorkel_finance_v3.json              # 40/10 questions + 29 hard
data/splits/snorkel_finance_clean_eval.json      # the 29 expert-failure questions
data/splits/snorkel_finance_unseen_by_sft.json   # unseen by the v1 checkpoints
data/sft_corpus/*/{policy,base}_bestgen/         # best generation per task (EM likelihood)
data/sft_corpus/tau2_retail/policy_1gen/         # volume-matched control corpus
data/{snorkel_finance_split,..._expert_thoughts}_v3/  # finance v3 per-arm pools
data/snorkel_finance/{raw,benchmark}/            # FinQABenchmark (user cloned from github)
```

## 10. Open items

- **Multi-rollout (3/task)** — queued at 2g2. The only route to significance.
- **`thoughts_base` is unreliable** (+4.8pp retail, −5.6pp airline). Claims should name
  `thoughts_policy` specifically.
- **Public HF repos still host the broken 4e-5 adapters** alongside the good ones, with a
  model card reporting the invalid numbers. User's call: delete, deprecate, or rewrite once
  the sweep finishes.
- **Stage-3 RL has not been re-run** at the working LR. Everything so far is Stage-2 plus
  rollout eval of Stage-2 checkpoints.
