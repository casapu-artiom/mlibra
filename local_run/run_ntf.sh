#!/usr/bin/env sh
# NTF (Neural Transcriptomic Field) on MALDI -- thin wrapper over run_sota.sh
# pinning MODEL=ntf. All NTF_* and shared env vars documented in run_sota.sh
# still apply. Whole-brain renders + per-lipid true-vs-pred scatterplots +
# value-distribution diagnostics are produced exactly like the other
# experiments (run_sota.sh sets --reconstruct whole_brain).
#
#   N_EPOCHS=30 NTF_MAX_RES=1024 ./local_run/run_ntf.sh
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export MODEL=ntf
exec "$SCRIPT_DIR/run_sota.sh" "$@"
