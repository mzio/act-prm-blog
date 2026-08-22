# cc-7.0 — Live handoff state (for pickup after context compaction)

Companion notes: **[cc-5.0](cc-5.0-sft-lr-investigation.md)** = running numbers (auto-refreshed
results table), **[cc-6.0](cc-6.0-session-report.md)** = narrative report of the whole session.
This file is the operational state: what is running, what is queued, what every script does,
and the traps that have already bitten.

---

## 0. Thirty-second orientation

Everything runs **from cron**, not from an agent shell. Two entries plus an uploader:

```
*/5  scripts/sweep_guard.sh        # advances the queue; sources the auth env; flock-guarded
*/20 scripts/snapshot_results.sh   # notes/CSVs/figures, dotsync backup, git commit
*/15 scripts/upload_all_envs.sh    # publishes adapters to HF (3 domains)
```

Check state with:
```bash
tail -5 /tmp/aprm/guard.log
ps -eo pid,etime,args | grep -E "main_pytorch|scripts/run_" | grep -v grep
uv run --no-project python scripts/report_sft_sweep.py --match _lr3e_3
uv run --no-project python scripts/report_rollout_eval.py
for g in control expert_all finance_v3 finance_rollout bestgen rollout; do
  [ -f /tmp/aprm/$g/ALLDONE ] && echo "DONE $g" || echo "pending $g"; done
```

**Do not launch drivers from an agent shell.** They get killed with the process group, and
worse, a manual launch the guard cannot see causes duplicate runs (see §5).

---

## 1. Queue order and ETAs

Gates live in `sweep_guard.sh` and fire in this order. Each writes `ALLDONE` under
`/tmp/aprm/<name>/` when finished.

| # | stage | scope | ETA |
|---|---|---|---|
| running | **control** rollout re-run | retail `thoughts_policy_1gen`, 42 tasks | ~50 min left |
| 2e | **expert_thoughts_all** | retail + airline: SFT + rollout | **~8–11h** |
| 2f | **finance v3** | 5 arms SFT on the corrected split | **~13.5h** |
| 2g | **finance rollout** | 5 arms × {fair 10, hard 29} | **~6h** |
| 2h | **best-generation Act-PRM** | 6 SFT + 6 rollouts, 3 domains | ~25h |
| 2z | full-context rollout | 8 checkpoints — curiosity, deliberately last | ~8h |
| 3 | full-context SFT matrix | 12 arms | ~40h |

**≈28h** to the end of the finance rollout (stage 2g), which is the last stage carrying
new scientific weight.

Measured per-unit costs: retail SFT 3.5h, airline SFT 2.8h, finance SFT 2.7h, retail
rollout (42 tasks) 74 min, airline rollout (18 tasks) 38 min. Finance rollout estimated
~20 min (10 q) + ~55 min (29 q) per arm.

### expert_thoughts_all scope
**All training tasks in both domains** — retail 49, airline 21. What is filtered is which
*turns within* each task become targets: only those carrying reasoning before the action.
retail 635 → **317** targets (50%), airline 221 → **119** (54%). Finance is deliberately
excluded here because `run_finance_v3.sh` already trains it against the clean eval;
including it would spend 2.7h on a number scored against the 76%-contaminated v1 eval.

---

## 2. Scripts, and what each is for

| script | purpose |
|---|---|
| `sweep_guard.sh` | the queue. Sources `/tmp/aprm/agent_env.sh`, back-fills `.done`, runs each stage in order |
| `snapshot_results.sh` | refresh notes/CSVs/figures, dotsync backup, auto-commit |
| `upload_all_envs.sh` | publish adapters to the 3 public HF repos |
| `run_sft_lr_matrix.sh` / `run_sft_sweep.sh` | the Stage-2 matrix (regime-major; `LR=`, `VARIANTS=`, `NUM_BATCHES=`, `REGIMES=`, `ACTION_ONLY=`) |
| `run_matched_control.sh` | retail `thoughts_policy` at matched volume + rollout |
| `run_expert_all.sh` | `expert_thoughts_all` SFT + rollout (retail, airline) |
| `run_finance_v3.sh` | finance Stage-2, 5 arms, clean v3 eval |
| `run_finance_rollout.sh` | finance rollout on fair-10 and hard-29, reported separately |
| `run_bestgen_sft.sh` | Act-PRM with the best generation per task, all domains + rollouts |
| `run_sft_rollout_eval.sh` | tau2 rollout eval (`REGIME=hide|full`, `SMOKE=1`) |
| `check_rollout_valid.py` | **gate**: rejects auth-failed rollouts (see §5) |
| `report_sft_sweep.py` / `report_rollout_eval.py` / `report_lr_probe.py` | results tables |
| `plot_sft_curves.py` / `plot_lr_comparison.py` | figures → `notebooks/figs_sft/`, mirrored to `~/.claude/act-prm-figs/` |
| `make_bestgen_corpus.py` / `make_matched_corpus.py` / `make_finance_pools_v3.py` | corpus subselection |
| `make_finance_split_v2.py` / `make_finance_clean_eval_corpora.py` | superseded by v3; kept for provenance |

---

## 3. Data artifacts created this session

```
data/splits/snorkel_finance_uid_to_qid.json      # 357 uids -> finqa question ids (100% match)
data/splits/snorkel_finance_v3.json              # 40 train / 10 eval questions + 29 hard
data/splits/snorkel_finance_clean_eval.json      # the 29 expert-FAILURE questions
data/splits/snorkel_finance_unseen_by_sft.json   # unseen by the v1 checkpoints
data/sft_corpus/*/{policy,base}_bestgen/         # best generation per task (EM likelihood)
data/sft_corpus/tau2_retail/policy_1gen/         # volume-matched control corpus
data/{snorkel_finance_split,...}_v3/             # finance v3 per-arm pools
data/snorkel_finance/{raw,benchmark}/            # FinQABenchmark (user cloned)
```

---

## 4. Config / code changes worth knowing

- **lr 3e-3** is the working LR (`configs/trainer/{sft,pg}.yaml` still ship 4e-5; drivers override).
- `--require_thought` — train only on turns with reasoning (targets only; context intact).
- `--train_action_only` — ablation, **off**; loss is on the full thought+action span.
- `action_mask` in `prepare_minibatch` → `train/actiononly_{ppl,accuracy}` (metrics only).
- `action_start_token` lives in `trainer/utils.py`, shared by loss mask and eval so they cannot drift.
- `--eval_query_ids` + explicit id selection in the snorkel_finance env.
- `claude_agent_sdk` lazily registered in `llm_handlers/__init__.py` (finance judge → `ClaudeQueryLLM`).
- Finance gym runs in **`.venv-tau2`** (the SDK is not in the base venv).

---

## 5. Traps that have already bitten — do not re-learn these

1. **cron's stripped env breaks Claude auth.** `clicat` fails → the `claude` CLI errors →
   every episode dies on turn 1 → recorded as a plausible `0/N`. Fixed by sourcing
   `/tmp/aprm/agent_env.sh` in the guard. **If rollouts start showing 1.00 generate
   calls/task again, the snapshot has gone stale — refresh it from an interactive shell:**
   ```bash
   env | grep -vE "^(_|PWD|OLDPWD|SHLVL)=" > /tmp/aprm/agent_env.sh
   ```
2. **A dead judge is invisible to the calls-per-task test.** The policy acts normally and
   only scoring collapses; `grade_sample` returns `"no"` on failure. `check_rollout_valid.py`
   now also flags >10% query errors with zero score. Especially dangerous on the hard-29 set
   where near-zero is expected. Judge verified working (gold→yes, nonsense→no).
3. **Duplicate runs corrupt finished metrics.** Two separate causes, both fixed: `run_one`
   checked `.done` *before* `wait_gpu_free`; and the guard's driver-detection regex omitted
   a driver. Symptom: batch counter resets to 0 mid-file. Detect with
   `[r['progress/batch'] for r in rows]` and truncate at the reset.
4. **`step_best` is written at the FIRST eval**, so it never means "finished". Completion is
   an explicit `.done` marker.
5. **`NUM_BATCHES`/`LR` must ride in the run tag**, or a budget change silently skips.
6. **github is blocked** from the agent session (fwdproxy 403); HF and the Meta proxy work.
   The user pushes. Adapters go to HF automatically.
7. **Run dirs sanitise `-` and `.` to `_`** (`lr1e-4` → `lr1e_4`) — globs must match the
   written form.

---

## 6. Results so far (details in cc-5.0 / cc-6.0)

- Stage-2 at 4e-5 never trained (max |ΔW| 3.7e-5 vs base ~1e-2); all prior tables invalid;
  explains the Stage-3 RL null.
- At 3e-3 all 12 hide arms converged. Act-PRM recovers 58% (retail) / 57% (airline) of the
  oracle PPL gap.
- **Rollout inverts the PPL ranking.** `thoughts_policy` 21.4% retail (vs 11.9% baseline)
  and 72.2% airline (vs 50.0%); `expert_thoughts` has the best PPL everywhere yet gives
  +0.0pp retail and **−16.7pp** airline. Cause: its targets are ~50% bare `<tool_call>`.
- Paired McNemar pooled **p = 0.077** — suggestive, not significant. ≥3 rollouts/task is
  the cheapest fix and is **not yet queued**.
- Volume confound closed: matched-volume retail control reproduces the effect
  (2.2119 vs 2.2059; baseline 2.3422).
- **Finance v1 numbers withdrawn** — 76% question-level eval contamination.

## 7. Not yet done

- ≥3 rollouts per task (would move p=0.077 to a real result) — **highest value remaining**.
- Finance rollout depends on stages 2f→2g completing.
- `thoughts_base` is unreliable (+4.8pp retail, −5.6pp airline); claims should name
  `thoughts_policy` specifically.
