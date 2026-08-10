#!/usr/bin/env sh
# FAITHFUL DeepSpatial (official GiT flow-matching + UOT + probability-flow ODE,
# within-specimen slice interpolation) on MALDI -- thin wrapper pinning
# MODEL=deepspatial. run_sota.sh handles this model inline (its DS_* knobs --
# DS_HIDDEN_SIZE, DS_DEPTH, DS_STEPS, DS_THICKNESS, DS_MAX_CELLS, ... -- and shared
# I/O env vars apply). Trains on the train-fold mice and reconstructs each
# held-out test mouse's full brain volume (per-lipid volumes + renders) plus
# leave-one-section-out metrics.
#
#   N_EPOCHS=100 DS_THICKNESS=0.02 ./local_run/run_deepspatial.sh
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export MODEL=deepspatial
exec "$SCRIPT_DIR/run_sota.sh" "$@"
