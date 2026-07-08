#!/usr/bin/env bash
# Idle "keep-alive" dev pod for SSH / VS Code Remote / interactive debugging.
#
# Submitted with NO command, so the image's /entrypoint.sh runs interactively:
# it installs your ssh key (from /myhome/.ssh, for appuser + root) and execs
# `sshd -D` (foreground = keep-alive). Passing a command instead drops it into
# job mode and sshd never starts.
#
# Prereq: your pubkey must be at /myhome/.ssh/authorized_keys (persistent) so the
# entrypoint can install it. It's already there; a fresh volume needs it seeded
# once (append your ~/.ssh/*.pub via `runai workspace exec <job> -it -- bash`).
#
# Connect:  runai workspace port-forward <job> --port 2222:2222 &
#           ssh -p 2222 appuser@localhost      # or root@localhost
# Or:       runai workspace exec <job> -it -- bash
#
# Preview without submitting:  DRY_RUN=1 ./submit/run_idle.sh

set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() { if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"

CPU=${CPU:-2}
MEM=${MEM:-8G}
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_all_latest}
JOB_NAME=${JOB_NAME:-idle-$(date +'%y%m%d-%H%M')}
# The tag isn't literally `:latest`, so k8s would default to IfNotPresent and
# reuse a cached image -- Always forces a re-pull of each rebuilt push.
PULL_POLICY=${PULL_POLICY:-Always}

echo ">>> $JOB_NAME  (image=$IMAGE, cpu=$CPU, mem=$MEM, pull=$PULL_POLICY) -- sshd on :2222"
run_or_echo runai workspace submit "$JOB_NAME" \
    -i "$IMAGE" --image-pull-policy "$PULL_POLICY" \
    --cpu-core-limit "$CPU"   --cpu-core-request   "$CPU" \
    --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM"

cat <<EOF
Submitted $JOB_NAME.
  Shell:  runai workspace exec $JOB_NAME -it -- bash
  SSH:    runai workspace port-forward $JOB_NAME --port 2222:2222
          ssh -p 2222 appuser@localhost      # or root@localhost
EOF
