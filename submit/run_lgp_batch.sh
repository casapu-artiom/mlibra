#!/usr/bin/env bash
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

MEM=48G
CPU=4
GPU=0.5

N_EPOCHS=10
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/experiment_batch_16"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_SLICES_DATASET_FILE_FOLD_3="/myhome/mlibra/maldi/data/splits/fold_3.json"
S3_SLICES_DATASET_FILE_DIFFICULT="/myhome/mlibra/maldi/data/splits/fold_3.json"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
SRC_PATH="/myhome/mlibra"

submit() {
    local job_name=$1 slices=$2 prefix=$3 norsample=$4 inducing_source=$5
	shift 5
	local extra_args=("$@")    # everything remaining goes here
    echo ">>> Submitting $job_name"
    runai training submit "$job_name" \
        -i artiomartiom/sdsc:withfaiss \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slices" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="unspecified" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e SRC_PATH="$SRC_PATH" \
        -e N_EPOCHS="$N_EPOCHS" \
        -e LATENT_DIM=5 \
        -e NUM_INDUCING_POINTS=2000 \
        -e INDUCING_SOURCE="$inducing_source" \
        -e NUM_MODES=2000 \
        -e NO_RSAMPLE="$norsample" \
        -e LEARN_INDUCING="${LEARN_INDUCING:-false}" \
        -e ARD="${ARD:-false}" \
        -- ./maldi/run_final.sh "${extra_args[@]}"
}

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

#FOLDS=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8" "difficult")           # lowercase, dashed
FOLDS=("fold-2")           # lowercase, dashed
NO_RSAMPLES=("true" "false")
LOG_TRANSFORM=("" "--log-transform")
IND_SOURCES=("reference")
LEARN_INDUCINGS=("true")   # sweep: fixed vs learned inducing locations
ARDS=("true")              # sweep: isotropic vs per-axis ARD lengthscales
exp_num=1
for fold in "${FOLDS[@]}"; do
    for norsample in "${NO_RSAMPLES[@]}"; do
        for log_transform in "${LOG_TRANSFORM[@]}"; do
            for ind_src in "${IND_SOURCES[@]}"; do
                for learn_inducing in "${LEARN_INDUCINGS[@]}"; do
                    for ard in "${ARDS[@]}"; do
                        # fold-3  -> FOLD-3  (used as wandb EXP_PREFIX)
                        # fold-3  -> fold_3  (used in the splits filename)
                        fold_upper=${fold^^}
                        fold_file=${fold//-/_}
                        SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"
                        # submit() forwards LEARN_INDUCING / ARD as -e env vars to run_final.sh.
                        LEARN_INDUCING="$learn_inducing"
                        ARD="$ard"
                        run_or_echo submit "gp-lgp-${EXP_SUFFIX}-${exp_num}" "${SLICES_DATASET_FILE}" "${fold_upper}" "${norsample}" "${ind_src}" "${log_transform}"
                        exp_num=$((exp_num + 1))
                    done
                done
            done
        done
    done
done

#run_or_echo submit "gp-lgp-fold-3-${EXP_SUFFIX}" "${S3_SLICES_DATASET_FILE_FOLD_3}" "FOLD_3"
#run_or_echo submit "gp-lgp-fold-3-log-${EXP_SUFFIX}" "${S3_SLICES_DATASET_FILE_FOLD_3}" "FOLD_3" --log-transform
# run_or_echo submit "gp-lgp-difficult-${EXP_SUFFIX}" "${S3_SLICES_DATASET_FILE_DIFFICULT}" "DIFFICULT"
# run_or_echo submit "gp-lgp-difficult-log-${EXP_SUFFIX}" "${S3_SLICES_DATASET_FILE_DIFFICULT}" "DIFFICULT" --log-transform
