#!/usr/bin/env bash
# Submit the SOTA 3D-reconstruction papers (NTF / Spa3D / DeepSpatial) on MALDI
# to run:ai, one job per (model, fold). Mirrors run_submit_baselines.sh.
#
# The runner (sota/run_sota.sh) already specifies every input/output path with a
# LOCAL default; this script just overrides the I/O env vars (-e DATA_PATH,
# OUTPUT_DIR, MALDI_FILE, ...) to point at the S3-mounted dirs. Each job does
# whole-brain reconstruction + renders + per-lipid true-vs-pred scatterplots
# (RECONSTRUCT=whole_brain), so outputs are comparable to the manifold/baseline
# runs.
#
#   ./submit/run_sota_batch.sh                    # all models, default fold(s)
#   MODELS="ntf" ./submit/run_sota_batch.sh       # just NTF
#   FOLDS="fold-1 fold-2" ./submit/run_sota_batch.sh
#   DRY_RUN=1 ./submit/run_sota_batch.sh          # print the runai commands only
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
elif [ "$DRY_RUN" = "1" ]; then
    echo "[DRY] .env not found; continuing with a placeholder WANDB_API_KEY" >&2
    WANDB_API_KEY=${WANDB_API_KEY:-DRY_RUN_PLACEHOLDER}
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

MEM=48G
CPU=4
GPU=0.5

# W&B on by default for cluster runs (WANDB_API_KEY comes from .env). WANDB=0 disables.
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-sota_maldi}

# DeepSpatial (generative flow) converges slowest; give it more epochs.
N_EPOCHS=${N_EPOCHS:-30}
N_EPOCHS_DEEPSPATIAL=${N_EPOCHS_DEEPSPATIAL:-60}
BATCH_SIZE=${BATCH_SIZE:-16384}
# Spa3D uses per-batch KNN (O(batch^2) cdist), so it wants a smaller batch.
BATCH_SIZE_SPA3D=${BATCH_SIZE_SPA3D:-4096}

# ---- S3-mounted I/O overrides (the runner defaults are LOCAL) --------------
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_OUTPUT_DIR=${S3_OUTPUT_DIR:-"/s3/mlibra/mlibra-data/artiom/sota_batch"}
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
SRC_PATH="/myhome/mlibra"
EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

submit() {
    local job_name=$1 model=$2 slices=$3 prefix=$4 epochs=$5 batch=$6
    shift 6
    local extra_args=("$@")
    echo ">>> Submitting $job_name (model=$model, epochs=$epochs, batch=$batch)"
    runai training submit "$job_name" \
        -i artiomartiom/sdsc:maldi_manifold_latest \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e MODEL="$model" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slices" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="reference" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e SRC_PATH="$SRC_PATH" \
        -e N_EPOCHS="$epochs" \
        -e BATCH_SIZE="$batch" \
        -e WANDB="$WANDB" \
        -e WANDB_PROJECT="$WANDB_PROJECT" \
        -- ./sota/run_sota.sh "${extra_args[@]}"
}

MODELS=${MODELS:-"ntf spa3d deepspatial"}
FOLDS=(${FOLDS:-"fold-3"})           # lowercase, dashed; e.g. FOLDS="fold-1 fold-2"

for fold in "${FOLDS[@]}"; do
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"
    for model in $MODELS; do
        epochs=$N_EPOCHS
        batch=$BATCH_SIZE
        [ "$model" = "deepspatial" ] && epochs=$N_EPOCHS_DEEPSPATIAL
        [ "$model" = "spa3d" ] && batch=$BATCH_SIZE_SPA3D
        job="sota-${model}-${fold}-${EXP_SUFFIX}"
        run_or_echo submit "$job" "$model" "$SLICES_DATASET_FILE" \
            "${fold_upper}" "$epochs" "$batch" "$@"
    done
done
