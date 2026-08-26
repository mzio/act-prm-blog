#!/usr/bin/env bash
# Sourceable W&B setup + reachability preflight. `source scripts/wandb_preflight.sh`
#
# Why a preflight rather than just enabling W&B: api.wandb.ai is not currently reachable
# through Meta's fwdproxy (curl exits 56 -- the CONNECT tunnel to fwdproxy opens, then the
# tunnel to api.wandb.ai:443 is reset), the same treatment github gets. Without a check, a
# run either blocks on a doomed wandb.init or silently drops metrics. With one, an
# unreachable endpoint degrades to WANDB_MODE=offline: the run still writes a complete
# wandb/offline-run-* dir that `wandb sync` can upload later from a machine with access.
#
# Also note main_pytorch gates W&B on WANDB_API_KEY being an ENV VAR ("WARNING:
# WANDB_API_KEY environment variable not set. Skipping W&B logging."), so a key sitting in
# ~/.netrc or .env is not enough on its own -- it has to be exported.
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,.facebook.com,.fbcdn.net,.thefacebook.com}"
# The default 90s is tight through a proxy; a slow handshake should not cost the whole run.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"

# Pull the key from .env if it is not already exported.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .env ]; then
  _k="$(grep -E '^\s*WANDB_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')"
  [ -n "$_k" ] && export WANDB_API_KEY="$_k"
  unset _k
fi

echo "== preflight: can we reach api.wandb.ai? =="
# ANY http code means the proxy CONNECT tunnel opened and the host answered -- 404 is the
# normal reply from api.wandb.ai's root and is a PASS. Only 000 (no response at all) is a
# failure; a proxy-level block shows up as 000 because curl reports the proxy's refusal, not
# the host's. NOTE: do NOT append `|| echo 000` -- curl already prints 000 on failure, and
# the fallback concatenates into "000000", which then compares unequal to "000" and reports
# a false PASS. (That bit me on 08-25.)
WANDB_CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 https://api.wandb.ai/)"
if [ "${WANDB_CODE}" != "000" ] && [ -n "${WANDB_CODE}" ]; then
  echo "   reachable (HTTP ${WANDB_CODE}) -> W&B online, run_url will be populated"
  unset WANDB_MODE
else
  echo "   NOT reachable. Falling back to WANDB_MODE=offline; 'wandb sync wandb/offline-run-*'"
  echo "   from a host with access uploads it later. (Or get api.wandb.ai allowlisted.)"
  export WANDB_MODE=offline
fi
[ -n "${WANDB_API_KEY:-}" ] \
  && echo "   WANDB_API_KEY exported (${#WANDB_API_KEY} chars)" \
  || echo "   WANDB_API_KEY NOT set -- main_pytorch will skip W&B entirely"
