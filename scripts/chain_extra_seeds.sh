#!/usr/bin/env bash
# Extra rollout samples at step_0020 for the three main arms, all three usable domains.
#
# WHY, AND WHY THE TWO DOMAINS ARE HANDLED DIFFERENTLY:
#   insurance (snorkel gym) is DETERMINISTIC given a seed -- no simulated user, just tools
#     plus a judge. Re-running at the default seed=42 reproduces the same number (measured:
#     40/40 identical tasks). So extra samples MUST vary --seed. run_insurance_rollout.sh
#     has no EXTRA_ARGS passthrough, so these call main_pytorch.py directly.
#   retail / airline (tau2) call an EXTERNAL Claude user simulator we do not seed, so plain
#     repeats are already independent -- they just need a distinct CKPT_TAG so the .done
#     markers and log dirs do not collide. Measured same-checkpoint spread: retail 16.6pt
#     (sd 9.0), airline 22.2pt. See CLAUDE.md.
#
# COVERAGE BEFORE THIS RUN (insurance step_0020): actions_only 2 runs but BOTH at seed 42
# (= 1 effective sample), expert_thoughts_all 1, thoughts_policy 4 distinct seeds,
# thoughts_base 4 distinct seeds. So the two under-sampled arms get 4 seeds each here.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false ACT_PRM_DUMP_TRAJECTORIES=1
export https_proxy="${https_proxy:-http://fwdproxy:8080}" http_proxy="${http_proxy:-http://fwdproxy:8080}"
G=/tmp/aprm/extra_seeds; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
CKPAT="lr1e_3_nb200_flat32sgd"

# ---------------- insurance: seed-varied (the only domain that resolves ~3pt effects)
INS_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids']))")
for arm in actions_only expert_thoughts_all; do
  for SEED in 1234 777 555 999; do
    m="$G/ins.$arm.$SEED.done"; [ -f "$m" ] && continue
    CK=$(newest "checkpoints_lora/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/snorkel_insurance_s2_${arm}_${CKPAT}_heldout-*/step_0020")
    [ -z "$CK" ] && { log "ins $arm: no step_0020, skip"; touch "$m"; continue; }
    log "INSURANCE $arm seed=$SEED"
    CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
        --env_config act_prm/snorkel_insurance_gym --model_config hf_qwen3_4b_instruct \
        --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
        --replay_buffer_config default --resume_from "$CK" \
        --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
        --max_tokens 2048 --hide_observations \
        --run_tag "insurance_rollout_${arm}_seed${SEED}_step_0020" \
        --eval_task_ids $INS_IDS --seed "$SEED" --verbose >>"$G/ins.$arm.$SEED.log" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
done

# ---------------- retail + airline: plain repeats (one driver call does BOTH domains)
for rep in rep2 rep3; do
  for arm in actions_only thoughts_policy_adamw30 expert_thoughts_all; do
    m="$G/tau2.$arm.$rep.done"; [ -f "$m" ] && continue
    log "TAU2 (retail+airline) $arm $rep"
    VARIANTS="$arm" CKPT_PAT="$CKPAT" CKPT_STEP="step_0020" \
    CKPT_TAG="sgdlr1e_3_$rep" ./scripts/run_sft_rollout_eval.sh >>"$G/chain.log" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
done
log "=== extra seeds complete ==="
