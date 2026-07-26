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
git bundle create "$BK/act-prm-blog.bundle" --all >/dev/null
git bundle verify "$BK/act-prm-blog.bundle" >/dev/null
echo "bundle refreshed -> $BK/act-prm-blog.bundle ($(du -h "$BK/act-prm-blog.bundle" | cut -f1))"
