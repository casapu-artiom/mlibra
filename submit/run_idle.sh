#!/usr/bin/env bash
# Idle "keep-alive" pod: just sits there with `sleep infinity` so you can
# `runai exec`/`kubectl exec` into it for an interactive shell. Does no work --
# handy for poking at the image, the S3 mounts, or debugging interactively.
#
# Exec in once it's running:
#   runai training exec <job_name> -it -- bash
#
# Preview the runai command without submitting:  DRY_RUN=1 ./submit/run_idle.sh

set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() { if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

CPU=${CPU:-2}
MEM=${MEM:-8G}
IMAGE=${IMAGE:-artiomartiom/sdsc:withslepc}
JOB_NAME=${JOB_NAME:-idle-$(date +'%y%m%d-%H%M')}

echo ">>> $JOB_NAME  (image=$IMAGE, cpu=$CPU, mem=$MEM) -- sleeps forever"
run_or_echo runai workspace submit "$JOB_NAME" \
    -i "$IMAGE" \
    --cpu-core-limit "$CPU"   --cpu-core-request   "$CPU" \
    --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
    -c -- sleep infinity

echo "Submitted $JOB_NAME. Exec in with:  runai workspace exec $JOB_NAME -it -- bash"
