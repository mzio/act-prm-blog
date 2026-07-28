#!/usr/bin/env bash
# Back up the SMALL, non-checkpoint experiment artifacts (results) into the dotsynced
# ~/.claude dir so they survive a devserver loss and can be pulled to a laptop.
# Includes: per-run metrics.jsonl (training/eval curves), the exported SFT corpora, the
# splits + task maps, and the notes/notebooks/figs. EXCLUDES: LoRA checkpoints, model
# cache, venvs, tau2-bench, wandb (large / reproducible).
#
# Uniquely named per host + timestamp so backups from different boxes/runs never collide.
# Run periodically (e.g. folded into monitoring) or at milestones:
#   ./scripts/backup_artifacts.sh
set -uo pipefail
cd "$(dirname "$0")/.."

HOST=$(hostname -s)
TS=$(date +%Y%m%d-%H%M%S)
BK="$HOME/.claude/act-prm-backups/artifacts"
mkdir -p "$BK"
OUT="$BK/aprm-artifacts-${HOST}-${TS}.tar.gz"

# Collect metrics + generations across all run dirs (small-ish JSONL), + corpora/splits/notes.
LIST=$(mktemp)
find logs -name 'metrics.jsonl' -type f 2>/dev/null                 >> "$LIST" || true
find logs -name 'generations.jsonl' -type f 2>/dev/null             >> "$LIST" || true
# Include the human-facing + data artifacts if present.
for d in data/sft_corpus data/splits notes notebooks; do
  [ -e "$d" ] && find "$d" -type f 2>/dev/null >> "$LIST"
done

if [ ! -s "$LIST" ]; then echo "no artifacts found to back up"; rm -f "$LIST"; exit 0; fi

tar czf "$OUT" -T "$LIST" 2>/dev/null
rm -f "$LIST"
SZ=$(du -h "$OUT" | cut -f1)
echo "backed up $(tar tzf "$OUT" | wc -l) files -> $OUT ($SZ)"

# Keep only the most recent 5 backups for THIS host (bound dotsync size).
ls -1t "$BK"/aprm-artifacts-"${HOST}"-*.tar.gz 2>/dev/null | tail -n +6 | xargs -r rm -f

# Manifest so a laptop pull knows what's here.
{ echo "# aprm artifact backups (host=$HOST)"; ls -1t "$BK"/aprm-artifacts-*.tar.gz 2>/dev/null | sed 's|.*/||'; } \
  > "$BK/MANIFEST-${HOST}.txt"
