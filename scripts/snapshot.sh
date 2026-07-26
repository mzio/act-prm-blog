#!/usr/bin/env bash
# Commit all current changes and refresh the dotsync-backed git bundle, so work
# survives a devserver loss even though GitHub is unreachable from here and
# ~/projects is not in dotsync.
#
# Durability: the bundle is written under ~/.claude/, which IS synced by
# dotsync-home (unlike ~/projects). Restore instructions live next to it in
# ~/.claude/act-prm-backups/RESTORE.md.
#
# Usage:
#   ./scripts/snapshot.sh "message describing the change"
#   ./scripts/snapshot.sh            # uses a default WIP message
set -euo pipefail
cd "$(dirname "$0")/.."

MSG="${*:-wip snapshot}"
git add -A
if git diff --cached --quiet; then
  echo "(no changes to commit)"
else
  git commit -q -m "$MSG"
  echo "committed: $(git log --oneline -1)"
fi

BK="$HOME/.claude/act-prm-backups"
mkdir -p "$BK"
# PER-HOST bundle name: dotsync syncs ~/.claude across ALL boxes, so a fixed name would
# be clobbered by whichever box snapshots last. Per-host names let each box's bundle
# coexist — another box pulls this one's commits via
#   git fetch <this bundle> 'refs/heads/*:refs/remotes/<host>/*'
HOST="$(hostname -s)"
BUNDLE="$BK/act-prm-blog-$HOST.bundle"
git bundle create "$BUNDLE" --all >/dev/null
git bundle verify "$BUNDLE" >/dev/null
echo "bundle refreshed -> $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"
# Keep the legacy fixed-name bundle too (back-compat for the old RESTORE.md path),
# but the per-host one is the cross-box source of truth.
cp -f "$BUNDLE" "$BK/act-prm-blog.bundle" 2>/dev/null || true
