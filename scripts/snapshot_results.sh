#!/usr/bin/env bash
# Periodic results snapshot: refresh the per-dataset SFT notes/CSVs, regenerate the
# curve figures, append the arm table to the running investigation note, and back the
# metrics up into dotsync. Idempotent; cron every 20 min.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 0
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
G=/tmp/aprm; mkdir -p "$G"
exec 8>"$G/snapshot.lock" || exit 0
flock -n 8 || exit 0

for e in act_prm/tau2_retail act_prm/tau2_airline act_prm/snorkel_finance_split; do
  uv run --no-project python scripts/analyze_sft.py "$e" >/dev/null 2>&1 || true
done
uv run --with matplotlib --no-project python scripts/plot_sft_curves.py --span subspan \
  --match _lr1e_3 --suffix _lr1e3 >/dev/null 2>&1 || true
uv run --with matplotlib --no-project python scripts/plot_lr_comparison.py >/dev/null 2>&1 || true
cp -f notebooks/figs_sft/*.png "$HOME/.claude/act-prm-figs/" 2>/dev/null || true

# Refresh the live results table inside the investigation note (between markers).
TBL=$(uv run --no-project python scripts/report_sft_sweep.py --match _lr1e_3 2>/dev/null)
N=notes/cc-5.0-sft-lr-investigation.md
if [ -f "$N" ] && [ -n "$TBL" ]; then
  uv run --no-project python - "$N" <<PY 2>/dev/null || true
import sys,re
p=sys.argv[1]; s=open(p).read()
tbl='''$TBL'''
new="<!--RESULTS-->\n\`\`\`\n"+tbl+"\n\`\`\`\n_last refreshed: $(date '+%Y-%m-%d %H:%M')_\n<!--/RESULTS-->"
s=re.sub(r"<!--RESULTS-->.*?<!--/RESULTS-->", new, s, flags=re.S)
open(p,"w").write(s)
PY
fi
./scripts/backup_artifacts.sh >/dev/null 2>&1 || true
./scripts/snapshot.sh "auto: SFT LR sweep results snapshot" >/dev/null 2>&1 || true
echo "[$(date '+%m-%d %H:%M:%S')] snapshot refreshed" >> "$G/snapshot.log"
