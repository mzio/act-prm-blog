#!/usr/bin/env bash
# Milestone backup of the SMALL, valuable Act-PRM finance artifacts — the curve
# data (metrics.jsonl), corpora, pools, run configs, pipeline logs, notes, and
# notebooks — i.e. everything EXCEPT the large checkpoint tensors. Produces:
#   1. A dotsynced copy under ~/.claude/act-prm-backups/finance_results/ (survives
#      a devserver RECYCLE, since ~/.claude is dotsync-backed and ~/projects is not).
#      This copy ALSO includes the step_best LoRA adapters (~66MB each) for full
#      recovery — small enough for dotsync, large enough to skip in the scp tarball.
#   2. A single tarball ~/aprm_finance_artifacts.tgz (non-checkpoint only) for a
#      one-line `scp`/`rsync` PULL from a laptop (devserver->laptop is firewalled,
#      so you pull; see the command this script prints).
#
# Idempotent + rerunnable — call at each milestone (EM done, corpora exported,
# SFT done). Safe to run while the pipeline is live.
#
# Usage:  ./scripts/backup_results.sh
set -uo pipefail
cd "$(dirname "$0")/.."
shopt -s nullglob

DEST="$HOME/.claude/act-prm-backups/finance_results"
TARBALL="$HOME/aprm_finance_artifacts.tgz"
HOST="$(hostname -f 2>/dev/null || hostname -s)"
mkdir -p "$DEST"

# --- collect existing light (non-checkpoint) artifact paths (repo-relative) ---
LIGHT=()
for p in \
  data/snorkel_finance_split data/snorkel_finance_split_expert_thoughts \
  data/sft_corpus/snorkel_finance_split data/sft_corpus/snorkel_finance_split_8b \
  data/splits/snorkel_finance.json; do
  [ -e "$p" ] && LIGHT+=("$p")
done
for p in logs/act_prm_snorkel_finance_split* notes/cc-finance-* notebooks/cc-finance-*; do
  LIGHT+=("$p")
done

# --- tarball: light artifacts (repo tree) + the /tmp pipeline logs, minus big files ---
TMP_DIRS=()
for d in /tmp/aprm/snorkel_finance_split_4b /tmp/aprm/snorkel_finance_split_8b; do
  [ -d "$d" ] && TMP_DIRS+=("aprm/$(basename "$d")")
done
tar czf "$TARBALL" \
  --exclude='*.safetensors' --exclude='*.bin' --exclude='replay_buffer' --exclude='*.arrow' \
  "${LIGHT[@]}" \
  ${TMP_DIRS:+-C /tmp "${TMP_DIRS[@]}"} 2>/dev/null || true
echo "tarball -> $TARBALL ($(du -h "$TARBALL" 2>/dev/null | cut -f1))"

# --- dotsync copy: same light artifacts, expanded, for durable recovery ---
rm -rf "$DEST/artifacts"; mkdir -p "$DEST/artifacts"
tar xzf "$TARBALL" -C "$DEST/artifacts" 2>/dev/null || true

# --- dotsync also keeps the step_best adapters (small) for full re-use offline ---
n_ckpt=0
while IFS= read -r f; do
  [ -f "$f" ] || continue
  rel="${f#checkpoints_lora/}"; mkdir -p "$DEST/checkpoints/$(dirname "$rel")"
  cp -f "$f" "$DEST/checkpoints/$rel"; n_ckpt=$((n_ckpt+1))
done < <(find checkpoints_lora/act_prm_snorkel_finance_split* \
           \( -path '*step_best/adapter_model.safetensors' -o -path '*step_best/adapter_config.json' \) 2>/dev/null)
echo "dotsync -> $DEST (artifacts + $n_ckpt step_best adapter files)"

# --- print the laptop PULL command ---
cat <<EOF

  # ---- run THIS from your laptop to pull the non-checkpoint artifacts ----
  scp ${USER}@${HOST}:~/aprm_finance_artifacts.tgz .
  # or, for incremental syncs of the expanded results dir:
  rsync -avz ${USER}@${HOST}:~/.claude/act-prm-backups/finance_results/ ./finance_results/
EOF
