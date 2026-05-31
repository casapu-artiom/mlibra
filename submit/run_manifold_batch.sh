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
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/eigenvectors"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/experiment_batch_9"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/fold_3.json"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
S3_BG_TEMPLATE_NAME="brainglobe"
S3_REFERENCE_FILE_BG="/s3/mlibra/mlibra-data/bg_template.npy"
S3_ANNOTATION_FILE_BG="/s3/mlibra/mlibra-data/bg_annotations.npy"
SRC_PATH="/myhome/mlibra"

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

submit() {
    local job_name=$1 template=$2 ref=$3 annot=$4 infl=$5 threshold=$6 knn=$7 knn_k=$8 laplacian_norm=$9 nu=$10 graphbandwidth=${11} bumpscale=${12} bumpdecay=${13} prefix=${14} slice=${15}
	shift 15
	local extra_args=("$@")    # everything remaining goes here
    echo ">>> Submitting $job_name"
    runai training submit "$job_name" \
        -i artiomartiom/sdsc:maldi_manifold_latest \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slice" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="$template" \
        -e REFERENCE_FILE="$ref" \
        -e ANNOTATION_FILE="$annot" \
        -e KNN_METHOD="$knn" \
        -e KNN_K="$knn_k" \
        -e LAPLACIAN_NORM="$laplacian_norm" \
        -e NU="$nu" \
		-e SRC_PATH="$SRC_PATH" \
        -e GRAPHBANDWIDTH="$graphbandwidth" \
        -e BUMP_SCALE="$bumpscale" \
        -e BUMP_DECAY="$bumpdecay" \
        -e N_EPOCHS="$N_EPOCHS" \
        -e NUM_INDUCING_POINTS=1000 \
        -e NUM_MODES=1300 \
        -e THRESHOLD="$threshold" \
        -e CROSS_REGION_INFLATION="$infl" \
        -- ./maldi/run_manifold.sh "${extra_args[@]}"
}

# FOLDS=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8" "difficult")           # lowercase, dashed
# GRAPH_BANDWIDTHS=(0.5 1.0)
# BUMP_SCALES=(1 20 80)
# BUMP_DECAYS=(0.01 0.1)

# FOLDS=("fold-3")           # lowercase, dashed
# KNN_K=(5 15)
# KNN_METHODS=("faiss" "anatomical_atlas")
# LAPLACIAN_NORMS=("symmetric" "randomwalk")
# GRAPH_BANDWIDTHS=(0.5 1.0)
# BUMP_SCALES=(1 20 80)
# BUMP_DECAYS=(0.1 1.0)

FOLDS=("fold-3")           # lowercase, dashed
KNN_K=(15 120 180)
MAN_KNN_METHODS=("faiss" "anatomical_atlas" "faiss_atlas_weighted")
MAN_INFLATIONS=(10 100)   # only used when knn_method=faiss_atlas_weighted
LAPLACIAN_NORMS=("symmetric" "randomwalk")
GRAPH_BANDWIDTHS=(0.1)
BUMP_SCALES=(20)
BUMP_DECAYS=(0.1)
# BUMP_SCALES=(1 20 80)
# BUMP_DECAYS=(0.1 1.0)
NU=(2)
THRESHOLDS=(5 40)

# Fixed across the whole sweep
TEMPLATE="reference"
REF="$S3_REFERENCE_FILE"
ANNOT="$S3_ANNOTATION_FILE"

exp_num=1
for fold in "${FOLDS[@]}"; do
    # fold-3  -> FOLD-3  (used as wandb EXP_PREFIX)
    # fold-3  -> fold_3  (used in the splits filename)
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"

    for gb in "${GRAPH_BANDWIDTHS[@]}"; do
        for bs in "${BUMP_SCALES[@]}"; do
            for bd in "${BUMP_DECAYS[@]}"; do
                for knn_k in "${KNN_K[@]}"; do
                    for knn_method in "${MAN_KNN_METHODS[@]}"; do
                        for laplacian_norm in "${LAPLACIAN_NORMS[@]}"; do
                            for nu in ${NU[@]}; do
                                for threshold in ${THRESHOLDS[@]}; do
                                    if [ "$knn_method" = "faiss_atlas_weighted" ]; then
                                        infl_list=("${MAN_INFLATIONS[@]}")
                                    else
                                        infl_list=(1)
                                    fi
                                    for infl in "${infl_list[@]}"; do

                                        job_name="gp-manifold-${EXP_SUFFIX}-${exp_num}"

                                        # Map exp_num -> config, so you can read it back from terminal/logs
                                        printf "  exp %2d: fold=%-10s gb=%-5s bs=%-4s bd=%s\n" \
                                            "$exp_num" "$fold" "$gb" "$bs" "$bd"

                                        run_or_echo submit "$job_name" "$TEMPLATE" "$REF" "$ANNOT" "$infl" "$threshold" \
                                            "$knn_method" "$knn_k" "$laplacian_norm" "$nu" "$gb" "$bs" "$bd" "$fold_upper" "$SLICES_DATASET_FILE"

                                        exp_num=$((exp_num + 1))
                                    done
                                done
                            done
                        done
                    done
                done 
            done
        done
    done
done

echo "Submitted $((exp_num - 1)) jobs."