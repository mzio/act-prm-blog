#!/usr/bin/env bash
# Regenerate the insurance Stage-1 thought corpus under the PER-DOMAIN few-shot, then
# Stage-2 SFT it and roll out step_0020. Tagged `_fs2` throughout so nothing existing is
# overwritten -- the poisoned `policy_adamw30` corpus and its checkpoints stay on disk as
# the comparison arm.
#
# WHY: until 2026-09-23 the Stage-1 reversal prompt was seeded with ONE worked example --
# the finance/`acme` SQL task -- for every domain. Insurance shares that task modality
# (list_tables / get_table_descriptions / read_query) but not its subject, so the E-step
# reused the seed's content instead of grounding: 77.7% of insurance `policy` thoughts
# carry finance vocabulary, and `base` has 308 verbatim "I need revenue figures for acme"
# openers with 58.9% duplicate prefixes. Retail/airline are CRUD/API domains with a
# disjoint tool vocabulary and were untouched (0.0% n-gram overlap).
#
# WHY IT MATTERS: insurance is the ONLY domain whose noise floor (seed sd ~3.0pt, n=40)
# resolves the ~5pt effects at issue -- retail is 9pt, airline 22pt, finance n=10 has no
# information. Every "corpus choice is irrelevant" conclusion was measured on insurance,
# i.e. on a corpus that is largely boilerplate contradicting its own actions. This run is
# the first clean test of whether Act-PRM thoughts help.
#
# Stage-1 recipe is copied verbatim from insurance_s1em/s1relabel_policy_adamw30 (r32_a32
# LoRA, adamw 4e-5, length_penalty 0, advantage_mode action_probs then best). The ONLY
# change is the few-shot, which now resolves from --env_config. Stage-2 and the rollouts
# match the shipped sgdlr1e_3 generation exactly, so the new arm drops straight into
# results/rollouts_by_arm.csv beside the old one.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false ACT_PRM_DUMP_TRAJECTORIES=1

G=/tmp/aprm/ins_fs2; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

ENV=act_prm/snorkel_insurance
ENVDIR=act_prm_snorkel_insurance
MODEL=hf_qwen3_4b_instruct
CORPUS=data/sft_corpus/snorkel_insurance/policy_fs2
CKPAT="lr1e_3_nb200_flat32sgd"

# ---- 0. wait for the GPU. WAIT_PID lets this be queued behind a running chain; the
# pgrep fallback catches a bare main_pytorch started by something else.
WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  log "waiting for PID $WAIT_PID to exit before starting"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi
while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 120; done
sleep 30
log "GPU free -- starting insurance few-shot-v2 regeneration"

# ---- 1. Stage-1 EM (train the thought policy)
if [ ! -f "$G/s1em.done" ]; then
  log "S1 EM  insurance_s1em_policy_fs2"
  uv run --no-sync python main_pytorch.py \
      --env_config "$ENV" --generator_config act_prm --trainer_config pg \
      --model_config "$MODEL" --lora_config r32_a32_linear --replay_buffer_config default \
      --no-score_with_base --run_tag insurance_s1em_policy_fs2 \
      --group_size 4 --batch_size 4 --num_batches 30 --learning_rate 4e-5 \
      --length_penalty 0 --advantage_mode action_probs --optimizer adamw \
      --save_generations --no_initial_eval --eval_every 30 --gradient_checkpointing \
      --save_every 10 --verbose >>"$G/s1em.log" 2>&1
  log "  rc=$?"; touch "$G/s1em.done"; reap
fi

# ---- 2. Stage-1 relabel (generate-only pass -> generations.jsonl)
if [ ! -f "$G/s1relabel.done" ]; then
  CK=$(newest "checkpoints_lora/$ENVDIR/$MODEL/insurance_s1em_policy_fs2-*/step_best")
  if [ -z "$CK" ]; then
    log "S1 relabel: no step_best from the EM pass -- ABORT"; exit 1
  fi
  log "S1 relabel  insurance_s1relabel_policy_fs2  (from $CK)"
  uv run --no-sync python main_pytorch.py \
      --env_config "$ENV" --generator_config act_prm --trainer_config pg \
      --model_config "$MODEL" --lora_config r32_a32_linear --replay_buffer_config default \
      --no-score_with_base --no_train --resume_from "$CK" \
      --advantage_mode best --group_size 4 --batch_size 4 --num_batches 45 \
      --no_initial_eval --length_penalty 0 --save_generations \
      --run_tag insurance_s1relabel_policy_fs2 --verbose >>"$G/s1relabel.log" 2>&1
  log "  rc=$?"; touch "$G/s1relabel.done"; reap
fi

# ---- 3. export the SFT corpus
if [ ! -f "$G/export.done" ]; then
  GEN=$(newest "logs/$ENVDIR/$MODEL/insurance_s1relabel_policy_fs2-*/generations.jsonl")
  [ -z "$GEN" ] && { log "export: no generations.jsonl -- ABORT"; exit 1; }
  log "export  $GEN -> $CORPUS"
  uv run --no-sync python scripts/export_sft_corpus.py \
      --generations "$GEN" --source-pools data/snorkel_insurance_split \
      --out "$CORPUS" >>"$G/export.log" 2>&1
  log "  rc=$?"; touch "$G/export.done"
fi

# ---- 4. GATE: did the fix actually take? Abort before spending ~7h downstream if not.
log "auditing $CORPUS for few-shot contamination"
uv run --no-sync python scripts/audit_thought_corpus.py "$CORPUS" --gate 2>&1 | tee -a "$G/chain.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  log "GATE FAILED -- the new corpus is still contaminated. Stopping before Stage-2."
  log "  inspect: uv run python scripts/audit_thought_corpus.py $CORPUS"
  exit 1
fi
log "gate passed"

# STOP_AFTER_GATE=1 ends here, leaving the corpus built and audited. Used to interleave:
# Stage-1 is the long pole (10.7h) and the thing worth starting early, while Stage-2 +
# rollouts (6.9h) can wait behind other queued work. Re-running without the flag resumes
# at Stage-2 -- every step below is .done-guarded.
if [ "${STOP_AFTER_GATE:-0}" = 1 ]; then
  log "=== STOP_AFTER_GATE: Stage-1 complete, corpus at $CORPUS ==="
  exit 0
fi

# ---- 5. Stage-2 SFT (matches the shipped sgdlr1e_3 generation exactly)
TAG="snorkel_insurance_s2_thoughts_policy_fs2_${CKPAT}_heldout"
if [ ! -f "$G/sft.done" ]; then
  log "S2 SFT  $TAG"
  TRAINER_CFG=sft_flat ./scripts/train_sft.sh "$ENV" thoughts_policy \
      --dataset_path "$CORPUS" --run_tag "$TAG" \
      --best_metric eval_actiononly_ppl --learning_rate 1e-3 --optimizer sgd \
      --num_batches 200 --eval_every 5 --early_stop_patience 6 --steps_per_batch 32 \
      >>"$G/sft.log" 2>&1
  log "  rc=$?"; touch "$G/sft.done"; reap
fi

# ---- 6. rollouts at step_0020, 5 seeds (snorkel gyms are deterministic given a seed;
# same-seed repeats are byte-identical, so samples MUST vary --seed)
INS_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids']))")
for SEED in 1234 777 555 999 42; do
  m="$G/roll.$SEED.done"; [ -f "$m" ] && continue
  CK=$(newest "checkpoints_lora/$ENVDIR/$MODEL/${TAG}-*/step_0020")
  [ -z "$CK" ] && { log "rollout: no step_0020 -- skip seed $SEED"; touch "$m"; continue; }
  log "ROLLOUT thoughts_policy_fs2 seed=$SEED step_0020"
  CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
      --env_config act_prm/snorkel_insurance_gym --model_config "$MODEL" \
      --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
      --replay_buffer_config default --resume_from "$CK" \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --max_tokens 2048 --hide_observations \
      --run_tag "insurance_rollout_thoughts_policy_fs2_seed${SEED}_step_0020" \
      --eval_task_ids $INS_IDS --seed "$SEED" --verbose >>"$G/roll.$SEED.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

# ---- 7. re-export
log "re-exporting CSVs"
uv run --no-sync python scripts/export_training_results.py  >>"$G/chain.log" 2>&1
uv run --no-sync python scripts/export_rollout_results.py   >>"$G/chain.log" 2>&1
uv run --no-sync python scripts/export_trajectories.py      >>"$G/chain.log" 2>&1
uv run --no-sync python scripts/measure_thought_emission.py >>"$G/chain.log" 2>&1
log "=== insurance few-shot-v2 complete ==="
