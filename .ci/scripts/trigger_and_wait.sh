#!/usr/bin/env bash
# Trigger the generic lyris-exec pipeline, poll to terminal, fetch artifacts.
# Retries on infra-class outcomes (exit 90 / runner offline / queue timeout).
set -uo pipefail
API="https://gitlab-master.nvidia.com/api/v4/projects/231686"
REF=""; PART=""; IMAGE=""; TCF=""; RETRIES=2; ALLOW_FAIL=""
while [ $# -gt 0 ]; do case "$1" in
  --ref) REF=$2; shift 2;; --partition) PART=$2; shift 2;;
  --image) IMAGE=$2; shift 2;; --test-cmds-file) TCF=$2; shift 2;;
  --retries) RETRIES=$2; shift 2;;
  --allow-fail) ALLOW_FAIL=$2; shift 2;;   # space-separated test names to quarantine (non-gating)
  *) echo "bad arg $1" >&2; exit 2;; esac; done
: "${TRIGGER_TOKEN:?}"; : "${GITLAB_TOKEN:?}"
: "${REF:?}"; : "${PART:?}"; : "${IMAGE:?}"; : "${TCF:?}"
TC=$(base64 -w0 < "$TCF" 2>/dev/null || base64 < "$TCF" | tr -d '\n')

attempt=0
while :; do
  attempt=$((attempt+1))
  PID=$(curl -sS -X POST -F "token=$TRIGGER_TOKEN" -F "ref=$REF" \
    -F "variables[PARTITION]=$PART" -F "variables[IMAGE]=$IMAGE" \
    -F "variables[TEST_CMDS]=$TC" -F "variables[ALLOW_FAIL]=$ALLOW_FAIL" \
    "$API/trigger/pipeline" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
  echo "pipeline=$PID attempt=$attempt"
  # poll
  while :; do
    ST=$(curl -sS -H "PRIVATE-TOKEN: $GITLAB_TOKEN" "$API/pipelines/$PID" \
      | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
    case "$ST" in success|failed|canceled|skipped) break;; esac
    sleep 30
  done
  JID=$(curl -sS -H "PRIVATE-TOKEN: $GITLAB_TOKEN" "$API/pipelines/$PID/jobs" \
    | python3 -c "import sys,json;[print(j['id']) for j in json.load(sys.stdin) if j['name']=='lyris-exec']" | head -1)
  rm -rf lyris-artifacts && mkdir -p lyris-artifacts
  curl -sS -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
    "$API/jobs/$JID/artifacts/results/summary.txt" -o lyris-artifacts/summary.txt || true
  if grep -q '^INFRA_FAILURE' lyris-artifacts/summary.txt 2>/dev/null || [ "$ST" = "canceled" ]; then
    if [ "$attempt" -le "$RETRIES" ]; then echo "infra failure, retrying"; continue; fi
    echo "infra failure after retries"; exit 90
  fi
  fails=$(awk '/^TEST_FAILS=/{split($0,a,"=");print a[2]}' lyris-artifacts/summary.txt)
  [ "${fails:-1}" = "0" ] && exit 0 || exit 1
done
