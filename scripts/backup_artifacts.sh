#!/usr/bin/env bash
# Back up NON-CHECKPOINT result artifacts to a uniquely-named, dotsync-synced tarball so
# results survive a devserver loss. The git bundle already covers code/notes/splits/
# notebooks; this adds the gitignored RESULTS: per-step metrics (curves), CSVs, run
# configs, and the SFT corpora. EXCLUDES checkpoints_lora (~20G) + generations.jsonl
# (~88M) as re-runnable — pass --with-gens to include generations too.
#
# Output: ~/.claude/act-prm-backups/artifacts-<host>-<UTCstamp>.tgz  (dotsync-synced;
# unique per host+run so multiple boxes/runs never collide). Keeps the last 5 per host.
#
# Usage:  ./scripts/backup_artifacts.sh [--with-gens]
set -uo pipefail
cd "$(dirname "$0")/.."
WITH_GENS=0; [ "${1:-}" = "--with-gens" ] && WITH_GENS=1

BK="$HOME/.claude/act-prm-backups"; mkdir -p "$BK"
HOST="$(hostname -s)"
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"
OUT="$BK/artifacts-${HOST}-${STAMP}.tgz"
MAN="$(mktemp)"

# results under logs/: per-step metrics (the curves), summary CSVs, resolved configs
find logs -type f \( -name 'metrics.jsonl' -o -name '*.csv' -o -name 'config.json' \) 2>/dev/null >> "$MAN"
[ "$WITH_GENS" = 1 ] && find logs -type f -name 'generations.jsonl' 2>/dev/null >> "$MAN"
# committed-but-include-for-self-containment: notes, splits (map), notebooks (with figures)
find notes data/splits notebooks -type f 2>/dev/null >> "$MAN"
# derived SFT corpora (small, saves a relabel to reconstruct)
find data/sft_corpus -type f 2>/dev/null >> "$MAN"

N=$(wc -l < "$MAN")
tar -czf "$OUT" -T "$MAN" 2>/dev/null
rm -f "$MAN"
echo "artifact backup -> $OUT  ($(du -h "$OUT" | cut -f1), $N files, host=$HOST)"

# prune: keep the last 5 per host so dotsync doesn't bloat
ls -t "$BK"/artifacts-"${HOST}"-*.tgz 2>/dev/null | tail -n +6 | xargs -r rm -f
echo "kept $(ls "$BK"/artifacts-"${HOST}"-*.tgz 2>/dev/null | wc -l) snapshot(s) for $HOST in $BK"
