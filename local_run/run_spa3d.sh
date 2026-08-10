#!/usr/bin/env sh
# Spa3D (spatial-pattern-enhanced GCN) on MALDI -- thin wrapper over run_sota.sh
# pinning MODEL=spa3d. All SPA3D_* and shared env vars documented in run_sota.sh
# still apply. Whole-brain renders + per-lipid true-vs-pred scatterplots +
# value-distribution diagnostics are produced exactly like the other
# experiments (run_sota.sh sets --reconstruct whole_brain).
#
#   SPA3D_SPE=alft SPA3D_Z_WEIGHT=0.5 BATCH_SIZE=4096 ./local_run/run_spa3d.sh
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export MODEL=spa3d
exec "$SCRIPT_DIR/run_sota.sh" "$@"
