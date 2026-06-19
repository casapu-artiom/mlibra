#!/usr/bin/env bash
# Submit a single runai job that runs the spectral distance sweep
# (maldi/spectral_distance_sweep.sh) against the cloud (S3) data paths.
#
# Like submit/run_manifold_batch.sh, but the sweep grid lives inside the
# Python/bash driver itself, so this only ever submits ONE job. The sweep
# is resumable (each config writes OUT_DIR/rows/<slug>.csv), so a preempted
# job picks up where it stopped.
#
# Override anything by exporting before calling, e.g.
#     KNN_KS="60 120" BANDWIDTHS="0.05 0.1" ./run_spectral_distance_sweep.sh
#     DRY_RUN=1 ./run_spectral_distance_sweep.sh        # print, don't submit
# Extra flags pass straight through to spectral_distance_sweep.sh:
#     ./run_spectral_distance_sweep.sh --force
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}

run_or_echo() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] $*"
    else
        "$@"
    fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

MEM=${MEM:-64G}
CPU=${CPU:-4}
GPU=${GPU:-1.0}

# -------------------------------------------------------------------------
# Cloud (S3 / myhome) paths consumed by spectral_distance_sweep.sh
# -------------------------------------------------------------------------
SRC_PATH="/myhome/mlibra"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/spectral_sweep"
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/artiom/eigenvectors"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_TEMPLATE_NAME="reference"
S3_LIPIDS_FILE="/myhome/mlibra/maldi/data/lipid_subset.txt"

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"
JOB_NAME=${JOB_NAME:-"spectral-sweep-${EXP_SUFFIX}"}

echo ">>> Submitting $JOB_NAME"
run_or_echo runai training submit "$JOB_NAME" \
    -i artiomartiom/sdsc:maldi_manifold_latest \
    --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
    --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
    --gpu-request-type portion --gpu-portion-request "$GPU" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e SRC_PATH="$SRC_PATH" \
    -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
    -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
    -e MALDI_FILE="$S3_MALDI_FILE" \
    -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
    -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
    -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
    -e TEMPLATE_NAME="$S3_TEMPLATE_NAME" \
    -e LIPIDS_FILE="$S3_LIPIDS_FILE" \
    -- ./maldi/spectral_distance_sweep.sh "$@"
