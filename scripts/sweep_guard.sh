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
# CRITICAL: the tau2 user simulator and the finance judge shell out to the `claude` CLI,
# which mints Meta auth tokens via `clicat`. Under cron's stripped environment clicat
# fails ("No CATs were created / clicat create-all returned empty output"), the CLI
# returns an error, and EVERY episode dies on its first turn -- recorded as a plausible
# 0/N. That is exactly what happened to the full-context rollout pass and the matched
# control's rollout. Sourcing a snapshot of the interactive environment fixes it
# (verified: FAILED(None) under `env -i`, OK with the snapshot).
# Refresh with:  env | grep -vE "^(_|PWD|OLDPWD|SHLVL)=" > /tmp/aprm/agent_env.sh
[ -f "$G/agent_env.sh" ] && { set -a; . "$G/agent_env.sh" 2>/dev/null; set +a; }
# /tmp REAPING. Completion markers live under /tmp/aprm and the system reaps old files
# there: on 08-25 the whole /tmp/aprm/lrprobe dir vanished, so the thoughts_base LR probe
# marker from 08-22 was gone, gate 2 re-fired, and it started re-running a probe finished
# three days earlier -- burning GPU AND blocking every gate below it (including a completed
# insurance EM waiting to hand off). Refresh every marker's mtime each tick so they never
# age out, and keep a durable copy under the repo's state dir as a backstop.
find "$G" -name '*.done' -o -name 'ALLDONE' 2>/dev/null | xargs -r touch 2>/dev/null || true
mkdir -p .guard_markers 2>/dev/null
( cd "$G" 2>/dev/null && find . \( -name '*.done' -o -name 'ALLDONE' \) -print0 2>/dev/null ) \
  | ( cd "$G" 2>/dev/null && xargs -0 -r -I{} sh -c 'mkdir -p "$OLDPWD/.guard_markers/$(dirname {})"; touch "$OLDPWD/.guard_markers/{}"' ) 2>/dev/null || true
# Restore any marker that exists in the durable copy but was reaped from /tmp.
if [ -d .guard_markers ]; then
  ( cd .guard_markers && find . \( -name '*.done' -o -name 'ALLDONE' \) -print ) 2>/dev/null | while read -r m; do
    [ -f "$G/$m" ] || { mkdir -p "$G/$(dirname "$m")"; touch "$G/$m"; echo "[$(date '+%m-%d %H:%M:%S')] restored reaped marker: $m" >> "$G/guard.log"; }
  done
fi

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
if ps -eo args | grep -qE 'scripts/(run_sft_lr_matrix|run_sft_sweep|probe_sft_lr|run_matched_control|run_expert_all|run_sft_rollout_eval|run_finance_v3|run_finance_rollout|run_bestgen_sft|run_multirollout|run_base_rollout|run_insurance_pipeline|run_x1_replicate|probe_stage1_lora_rank|run_stage1_r32)\.sh'; then exit 0; fi

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
#
# The driver touches ALLDONE unconditionally, even when an arm FAILED -- expert_thoughts_all
# crashed on 08-23 (empty minibatch; 56% of finance expert trajectories carry no reasoning).
# Retract ALLDONE whenever an arm marker is missing so the driver is re-invoked and picks up
# just the failed arm; its per-arm .done files make that a cheap no-op for the rest.
if [ -f "$G/finance_v3/ALLDONE" ]; then
  for _a in actions_only expert_thoughts expert_thoughts_all thoughts_policy thoughts_base; do
    if [ ! -f "$G/finance_v3/snorkel_finance_split_s2_${_a}_v3_lr3e_3_nb150_heldout.done" ]; then
      log "finance v3: arm '$_a' never completed -- retracting ALLDONE for a retry"
      rm -f "$G/finance_v3/ALLDONE"; break
    fi
  done
fi
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

# 2g1. RE-RUN retail+airline expert_thoughts_all. The first pass ran with --require_thought
# filtering the EVAL targets as well as the train targets, so its teacher-forced PPL was
# scored on a thought-bearing (longer, harder) subset -- mean eval target 497.7 tokens vs
# 293.6 for the same pool without the flag -- and is not comparable to the other arms.
# base.py now gates the filter on split=="train". Rollout numbers from the first pass are
# unaffected (gym task completion never touches that path), but step_best was selected
# against the filtered metric, so the rollouts re-run off the corrected checkpoints.
# Placed after the finance work to preserve the requested ordering.
if [ -f "$G/finance_rollout/ALLDONE" ] && [ ! -f "$G/expert_all_fix/ALLDONE" ]; then
  log "advancing the expert_thoughts_all re-run (comparable eval)"
  mkdir -p "$G/expert_all_fix"
  MDIR="$G/expert_all_fix" TAGSFX="_fixeval" \
    ./scripts/run_expert_all.sh >> "$G/expert_all_fix/driver.log" 2>&1
  exit 0
fi

# 2g1b. BASE-MODEL rollout: the missing control. Every other rollout loads a Stage-2
# adapter, so nothing measures the raw instruct model in the gym and we cannot say what
# Stage-2 SFT bought -- only how the arms compare to each other. ~1.9h for both domains.
# Ordered before the multi-rollout so the baseline lands early and cheaply.
if [ -f "$G/finance_rollout/ALLDONE" ] && [ ! -f "$G/base_rollout/ALLDONE" ]; then
  log "advancing the base-model rollout (no Stage-2 adapter)"
  mkdir -p "$G/base_rollout"
  ./scripts/run_base_rollout.sh >> "$G/base_rollout/driver.log" 2>&1
  exit 0
fi

# 2g1c. X1 SEED SWEEP. Airline thoughts_policy scored 72.2% at 1 rollout/task but 55.6/38.9/
# 50.0 at 3 -- x1 sits ~2.9 SD above its own x3 samples (~0.4% if one distribution), and all
# three x3 streams used more tool calls than x1 did. Batch truncation, differing configs and
# a positional padding artifact are all ruled out, so either x1 was a lucky draw or the
# batched path depresses scores. That distinction decides whether the 3-rollout null is a
# real result or an artifact, so it runs BEFORE the new domain. ~2.5h for 2 arms x 2 reps.
# Seed 42 is included deliberately, as a REPRODUCIBILITY probe rather than a new draw: the
# original 72.2% was measured at seed 42, so re-running it tests whether an outcome is
# pinned by the seed at all. If it reproduces, outcomes are seed-determined and the spread
# ACROSS seeds is the real uncertainty. If it does NOT reproduce, CUDA nondeterminism alone
# moves the score ~20pp, which is a worse problem than the seed story and would mean single
# -rollout evals cannot be replicated even in principle.
X1_SEEDS="0 1 7 42"
X1_ARMS="thoughts_policy actions_only"
if [ -f "$G/x1replicate/ALLDONE" ]; then
  for _a in $X1_ARMS; do for _s in $X1_SEEDS; do
    [ -f "$G/x1replicate/airline_rollout_${_a}_x1seed${_s}.done" ] || {
      log "x1replicate: ${_a}/seed${_s} missing -- retracting ALLDONE"; rm -f "$G/x1replicate/ALLDONE"; break 2; }
  done; done
fi
if [ -f "$G/finance_rollout/ALLDONE" ] && [ ! -f "$G/x1replicate/ALLDONE" ]; then
  log "advancing the x1 seed sweep (seeds: $X1_SEEDS)"
  mkdir -p "$G/x1replicate"
  SEEDS="$X1_SEEDS" ARMS="$X1_ARMS" ./scripts/run_x1_replicate.sh >> "$G/x1replicate/driver.log" 2>&1
  exit 0
fi

# 2f2b. Back-fill insurance Stage-1a .done markers. The insurance driver writes a marker
# only after its train.sh call RETURNS, so stopping the driver to reorder work leaves a
# completed EM run unmarked and the gates below never fire. Derive the marker from the run's
# own metrics instead: if it reached its final batch, it is done.
uv run --no-project python - <<'INSBF' >> "$G/guard.log" 2>&1 || true
import glob, json, os, re
for scorer in ("policy", "base"):
    marker = f"/tmp/aprm/insurance/insurance_s1em_{scorer}.done"
    if os.path.exists(marker):
        continue
    for d in glob.glob(f"logs/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/insurance_s1em_{scorer}-*/"):
        m = re.search(r"-nb=(\d+)", d)
        f = d + "metrics.jsonl"
        if not m or not os.path.exists(f):
            continue
        nb = int(m.group(1))
        try:
            rows = [json.loads(l) for l in open(f) if l.strip()]
        except Exception:
            continue
        last = max((r.get("progress/batch", -1) for r in rows), default=-1)
        if last >= nb - 1:
            open(marker, "w").close()
            print(f"back-filled insurance EM marker: {scorer} (reached b{last}/{nb})")
            break
INSBF

# 2f3. STAGE-1 @ RANK 32, all domains, regenerating the Act-PRM corpora. Every Stage-1 EM
# run to date used lr=4e-5 / r8_a16_linear and the adapter is a MEASURED no-op in every
# domain with a surviving checkpoint (retail 6.7e-07, finance 4.1e-07, insurance 2.8e-06 vs
# base weights ~1e-2), so the EM-trained generator has always been the base model. MZ's
# hypothesis: keep lr=4e-5, raise the rank to 32.
#
# The separate 3-arm probe was folded into this: the driver writes a checkpoint at batch 5
# (--save_every) and measures max|B@A| there, aborting the whole sweep if it is still a
# no-op. That buys the same de-risking for ~50 min of a run we want anyway, instead of 2.5h
# of probes. Full sweep is ~35h for the policy scorer, ~70h for both.
# Gated on insurance's Stage-1a BASE EM only, not the whole insurance pipeline: that EM
# completes the 4e-5 parity set, but the 8.5h relabel that follows it is worth deferring
# until we know whether rank 32 works. Insurance's Stage-1a .done is already written, so
# re-invoking its driver later resumes cleanly at Stage-1b.
if [ -f "$G/insurance/insurance_s1em_base.done" ] && [ ! -f "$G/stage1_r32/ALLDONE" ] \
   && [ ! -f "$G/stage1_r32/ABORTED" ]; then
  log "advancing Stage-1 @ rank32 (early-abort check at batch 5)"
  mkdir -p "$G/stage1_r32"
  ./scripts/run_stage1_r32.sh >> "$G/stage1_r32/driver.log" 2>&1
  exit 0
fi

# 2f2. INSURANCE: the full pipeline on a fourth domain (Stage-1 EM -> Stage-2 SFT ->
# rollout on 41 held-out questions). Ordered after the multi-rollout because that stage is
# what makes the EXISTING result reportable, whereas this adds a new domain; a fourth
# domain is worth much less if the three we have cannot clear the noise floor.
if [ ! -f "$G/insurance/ALLDONE" ]; then
  log "advancing the insurance pipeline (stage 1 -> 2 -> 3)"
  mkdir -p "$G/insurance"
  ./scripts/run_insurance_pipeline.sh >> "$G/insurance/driver.log" 2>&1
  exit 0
fi

# 2g2. 3 rollouts per task for the key arms. Ordered AFTER the finance work and BEFORE
# best-gen because it is the only stage that can make the headline result significant
# (pooled McNemar is currently p=0.077); every other pending stage adds interpretation,
# not evidence.
# expert_thoughts was ADDED to the set on 08-24. The base-model control showed it is the
# WORST arm -- -8.3pp vs base pooled, -22.2pp on airline -- which makes it the largest
# effect in the project, and it rested on ONE rollout per task against a measured 22.2pp
# replicate floor. The originally-queued pair (actions_only vs thoughts_policy) came back
# null on retail at 3 rollouts (+3.2pp, p=0.28), so the unverified arm now carries more
# weight than the verified one.
MR_ARMS="actions_only thoughts_policy expert_thoughts"
# Retract ALLDONE if any (arm, domain) marker is missing, so adding an arm re-invokes the
# driver; its per-run .done files make the finished runs no-ops.
if [ -f "$G/multirollout/ALLDONE" ]; then
  for _a in $MR_ARMS; do
    for _d in retail airline; do
      [ -f "$G/multirollout/${_d}_rollout_${_a}_x3.done" ] || {
        log "multirollout: ${_d}/${_a} missing -- retracting ALLDONE to cover the added arm"
        rm -f "$G/multirollout/ALLDONE"; break 2; }
    done
  done
fi
if [ -f "$G/x1replicate/ALLDONE" ] && [ ! -f "$G/multirollout/ALLDONE" ]; then
  log "advancing the multi-rollout (3 per task, arms: $MR_ARMS)"
  ARMS="$MR_ARMS" ./scripts/run_multirollout.sh >> "$G/multirollout/driver.log" 2>&1
  exit 0
fi

# 2h. Act-PRM Stage-2 with ONE generation per task (the best, by mean EM likelihood),
# volume-matched to actions_only in every domain, then rollout eval per environment.
if [ -f "$G/finance_rollout/ALLDONE" ] && [ ! -f "$G/bestgen/ALLDONE" ]; then
  log "advancing the best-generation Act-PRM SFT"
  ./scripts/run_bestgen_sft.sh >> "$G/bestgen/driver.log" 2>&1
  exit 0
fi

# 2z. FULL-CONTEXT rollout pass. Deliberately LAST: these checkpoints were trained with
# --hide_observations, so running them on full context is a generalisation curiosity, not
# part of the main comparison. It must not preempt the control / expert_thoughts_all /
# finance work.
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


# 3. keep the matrix moving
if [ -f "$G/lrmatrix/DONE" ]; then exit 0; fi
log "matrix idle -> advancing it"
./scripts/run_sft_lr_matrix.sh >> "$G/lrmatrix_driver.log" 2>&1
