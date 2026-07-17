#!/usr/bin/env bash
# Submit the FAITHFUL DeepSpatial (transport mode) experiment to run:ai, one job
# per fold. Uses the SEPARATE deepspatial image (Dockerfile.deepspatial) because
# it needs Lightning/POT/torchdiffeq/scanpy/anndata/pyvista -- NOT the main
# pure-torch image. Each job trains on the fold's train mice and reconstructs the
# held-out mice's full brains (renders + LOSO metrics).
#
#   ./submit/run_deepspatial_transport.sh
#   FOLDS="fold-2 fold-3" ./submit/run_deepspatial_transport.sh
#   DRY_RUN=1 ./submit/run_deepspatial_transport.sh
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() { if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
elif [ "$DRY_RUN" = "1" ]; then
    echo "[DRY] .env not found; using placeholder WANDB_API_KEY" >&2
    WANDB_API_KEY=${WANDB_API_KEY:-DRY_RUN_PLACEHOLDER}
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2; exit 1
fi

# Separate image built from Dockerfile.deepspatial.
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_deepspatial_latest}
MEM=${MEM:-64G}; CPU=${CPU:-6}; GPU=${GPU:-0.5}
N_EPOCHS=${N_EPOCHS:-100}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-sota_maldi}

S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_OUTPUT_DIR=${S3_OUTPUT_DIR:-"/s3/mlibra/mlibra-data/artiom/deepspatial_transport"}
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
SRC_PATH="/myhome/mlibra"
EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

submit() {
    local job_name=$1 slices=$2 prefix=$3; shift 3
    echo ">>> Submitting $job_name"
    runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e WANDB="$WANDB" -e WANDB_PROJECT="$WANDB_PROJECT" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slices" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="reference" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e SRC_PATH="$SRC_PATH" \
        -e N_EPOCHS="$N_EPOCHS" \
        -- ./sota/deepspatial_transport/run_deepspatial_transport.sh "$@"
}

FOLDS=(${FOLDS:-"fold-3"})
for fold in "${FOLDS[@]}"; do
    fold_upper=${fold^^}; fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"
    run_or_echo submit "deepspatial-tr-${fold}-${EXP_SUFFIX}" \
        "$SLICES_DATASET_FILE" "${fold_upper}" "$@"
done
