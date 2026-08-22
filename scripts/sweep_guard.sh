#!/usr/bin/env bash
# Cron-driven guard for the Stage-2 SFT sweep. Idempotent and safe to run every 5 min.
#
# Why cron: backgrounded shells launched from the agent session kept getting killed with
# their process group, which silently stalled the queue. Cron re-checks independently of
# any shell, so the sweep survives.
#
# Order of business each tick:
#   1. If a trainer is already running, do nothing.
#   2. If the thoughts_base LR probe hasn't run yet, run it (it informs the LR for every
#      remaining arm, so it goes before the rest of the matrix).
#   3. Otherwise keep the LR matrix moving. Both are resumable and skip finished work.
# Cron entry: */5 * * * * /home/mzio/projects/act-prm-blog/scripts/sweep_guard.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 0
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
G=/tmp/aprm; mkdir -p "$G"
LOCK="$G/guard.lock"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$G/guard.log"; }

# single-instance
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

# 1. a trainer is live -> nothing to do
if pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; then exit 0; fi
# a driver shell is mid-launch -> let it be
# Any of OUR driver shells mid-launch -> let it be. This list must include every driver
# the guard can start; omitting one (run_matched_control) let cron launch a SECOND copy
# of a control run that was already going, and both appended to the same metrics.jsonl.
if ps -eo args | grep -qE 'scripts/(run_sft_lr_matrix|run_sft_sweep|probe_sft_lr|run_matched_control|run_expert_all|run_sft_rollout_eval|run_finance_v3|run_finance_rollout)\.sh'; then exit 0; fi

# 2. thoughts_base LR probe (once)
if [ ! -f "$G/lrprobe/thoughts_base.done" ]; then
  log "starting thoughts_base LR probe (1e-4 / 1e-3 / 3e-3, 30 batches)"
  mkdir -p "$G/lrprobe"
  VARIANT=thoughts_base BATCHES=30 EVAL_EVERY=10 LRS="1e-4 1e-3 3e-3" \
    ./scripts/probe_sft_lr.sh >> "$G/lrprobe/driver.log" 2>&1
  touch "$G/lrprobe/thoughts_base.done"
  log "thoughts_base probe finished"
  exit 0
fi

# 2b. Back-fill .done for arms whose metrics reached the final batch but were never
# marked (driver restarted mid-arm). Without this a reorder or restart silently re-runs
# finished work -- ~2.5h each.
uv run --no-project python - <<'BACKFILL' >> "$G/guard.log" 2>&1 || true
import glob, json, os, re
for env, dom in [("act_prm_tau2_retail","retail"),("act_prm_tau2_airline","airline"),
                 ("act_prm_snorkel_finance_split","snorkel_finance_split")]:
    mdir = f"/tmp/aprm/sft_sweep_{'tau2_'+dom if dom in ('retail','airline') else dom}"
    if not os.path.isdir(mdir):
        continue
    for d in glob.glob(f"logs/{env}/hf_qwen3_4b_instruct/{dom}_s2_*_lr*_heldout*/"):
        tag = os.path.basename(d.rstrip('/')).split('-act-prm')[0]
        marker = f"{mdir}/{tag}.done"
        if os.path.exists(marker):
            continue
        m = re.search(r'-nb=(\d+)', d)
        if not m:
            continue
        nb = int(m.group(1))
        try:
            rows = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
        except Exception:
            continue
        last = max((r.get("progress/batch", -1) for r in rows), default=-1)
        if last >= nb - 1:
            open(marker, "w").close()
            print(f"back-filled .done: {tag} (reached b{last}/{nb})")
BACKFILL

# 2c. Rollout eval of the finished hide-regime SFT checkpoints, BEFORE the full-context
# arms. Scores TASK COMPLETION in the live tau2 gym rather than teacher-forced PPL, on
# never-in-logs tasks (retail 42 / airline 18). Prioritised because it is the metric the
# Act-PRM story is actually about, and the hide matrix it evaluates is already complete.
if [ ! -f "$G/rollout/ALLDONE" ]; then
  log "advancing the SFT rollout eval (task completion)"
  ./scripts/run_sft_rollout_eval.sh >> "$G/rollout/driver.log" 2>&1
  n=$(ls "$G"/rollout/*_lr3e_3.done 2>/dev/null | wc -l)
  if [ "$n" -ge 8 ]; then
    # hide pass done -> run the same 8 checkpoints with FULL context at rollout time.
    # Deliberate train/test mismatch: do policies trained on a compacted context
    # generalise when handed the whole thing?
    log "hide rollout complete ($n/8) -> starting the full-context rollout pass"
    REGIME=full ./scripts/run_sft_rollout_eval.sh >> "$G/rollout/driver_full.log" 2>&1
    m=$(ls "$G"/rollout/*_lr3e_3_fullctx.done 2>/dev/null | wc -l)
    [ "$m" -ge 8 ] && { touch "$G/rollout/ALLDONE"; log "rollout eval complete: hide $n/8, full $m/8"; }
  fi
  exit 0
fi

# 2d. Volume-matched control for the retail Act-PRM result (see run_matched_control.sh).
if [ ! -f "$G/control/ALLDONE" ]; then
  log "advancing the volume-matched control"
  ./scripts/run_matched_control.sh >> "$G/control/driver.log" 2>&1
  exit 0
fi

# 2e. expert_thoughts_all: SFT on expert reasoning+action but only on turns that HAVE
# reasoning, then rollout-eval. Tests whether expert thoughts help once the arm is actually
# trained to produce them (the plain arm was ~50% bare-action targets).
if [ ! -f "$G/expert_all/ALLDONE" ]; then
  log "advancing expert_thoughts_all"
  ./scripts/run_expert_all.sh >> "$G/expert_all/driver.log" 2>&1
  exit 0
fi

# 2f. Finance v3: QUESTION-level 40/10 split, subselected from the existing pools (v1 eval
# was 76% contaminated at the question level). Honest train/eval action-span curves +
# final checkpoints for rollout on the 10 eval questions and the 29 expert-failure questions.
if [ ! -f "$G/finance_v3/ALLDONE" ]; then
  log "advancing finance v3 Stage-2"
  ./scripts/run_finance_v3.sh >> "$G/finance_v3/driver.log" 2>&1
  exit 0
fi

# 2g. Finance rollout eval on the two v3 sets: the 10 fair eval questions (expert solved
# them; same questions the SFT curves score) and the 29 expert-FAILURE questions as a
# labelled hard test. Reported separately, never pooled.
if [ -f "$G/finance_v3/ALLDONE" ] && [ ! -f "$G/finance_rollout/ALLDONE" ]; then
  log "advancing the finance rollout eval"
  ./scripts/run_finance_rollout.sh >> "$G/finance_rollout/driver.log" 2>&1
  exit 0
fi

# 3. keep the matrix moving
if [ -f "$G/lrmatrix/DONE" ]; then exit 0; fi
log "matrix idle -> advancing it"
./scripts/run_sft_lr_matrix.sh >> "$G/lrmatrix_driver.log" 2>&1
