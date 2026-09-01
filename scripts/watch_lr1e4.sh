#!/usr/bin/env bash
# As each lr 1e-4 arm finishes, append its full selection grid (PPL / accuracy /
# max|B@A| / median|B@A| per snapshot) to one summary file.
#
# The grid is what this sweep is FOR: the collapse tracks how completely the SFT target
# distribution is fit, and step_best (chosen on PPL) was the worst rollout checkpoint at
# lr 1e-3. Computing it per arm as they land avoids a slow batch job at the end -- each
# grid loads ~15 checkpoints' safetensors and takes a couple of minutes.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
OUT=/tmp/aprm/stage2_lr1e4/grids.txt
mkdir -p /tmp/aprm/stage2_lr1e4
done_seen=""
while true; do
  for m in /tmp/aprm/sft_sweep_tau2_retail/*_lr1e_4_adamw_nb200_flat32_heldout.done; do
    [ -f "$m" ] || continue
    tag=$(basename "$m" .done)
    case " $done_seen " in *" $tag "*) continue;; esac
    done_seen="$done_seen $tag"
    {
      echo "================================================================"
      echo "$(date '+%m-%d %H:%M')  $tag"
      uv run --no-project python scripts/report_checkpoint_grid.py --run "${tag}-*" 2>&1
    } >> "$OUT"
  done
  # stop once all four arms are marked done
  n=$(ls /tmp/aprm/sft_sweep_tau2_retail/*_lr1e_4_adamw_nb200_flat32_heldout.done 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && { echo "[$(date '+%m-%d %H:%M')] all 4 arms done" >> "$OUT"; exit 0; }
  sleep 120
done
