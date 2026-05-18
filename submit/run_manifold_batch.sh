##!/usr/bin/env sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

MEM=32G
CPU=4
GPU=0.5

S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/eigenvectors"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/difficult.json"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
S3_BG_TEMPLATE_NAME="brainglobe"
S3_REFERENCE_FILE_BG="/s3/mlibra/mlibra-data/bg_template.npy"
S3_ANNOTATION_FILE_BG="/s3/mlibra/mlibra-data/bg_annotations.npy"
SRC_PATH="/myhome/mlibra"

EXP_SUFFIX="artiom-9"

submit() {
    local job_name=$1 template=$2 ref=$3 annot=$4 knn=$5 nu=$6
	shift 6
	local extra_args=("$@")    # everything remaining goes here
    echo ">>> Submitting $job_name"
    runai training submit "$job_name" \
        -i artiomartiom/sdsc:maldi_manifold_latest \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        --auto-deletion-time-after-completion 1h \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$S3_SLICES_DATASET_FILE" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="$template" \
        -e REFERENCE_FILE="$ref" \
        -e ANNOTATION_FILE="$annot" \
        -e KNN_METHOD="$knn" \
        -e NU="$nu" \
		-e SRC_PATH="$SRC_PATH" \
        -- ./maldi/run_manifold.sh "${extra_args[@]}"
}

submit "gp-exp-faiss-1-reference-${EXP_SUFFIX}"    reference   "$S3_REFERENCE_FILE"    "$S3_ANNOTATION_FILE"     faiss             1     
submit "gp-exp-atlas-1-reference-${EXP_SUFFIX}"    reference   "$S3_REFERENCE_FILE"    "$S3_ANNOTATION_FILE"     anatomical_atlas  1
submit "gp-exp-faiss-1-brainglobe-${EXP_SUFFIX}"   brainglobe  "$S3_REFERENCE_FILE_BG" "$S3_ANNOTATION_FILE_BG"  faiss             1
submit "gp-exp-atlas-1-brainglobe-${EXP_SUFFIX}"   brainglobe  "$S3_REFERENCE_FILE_BG" "$S3_ANNOTATION_FILE_BG"  anatomical_atlas  1
submit "gp-exp-faiss-2-reference-${EXP_SUFFIX}"    reference   "$S3_REFERENCE_FILE"    "$S3_ANNOTATION_FILE"     faiss             2     
submit "gp-exp-atlas-2-reference-${EXP_SUFFIX}"    reference   "$S3_REFERENCE_FILE"    "$S3_ANNOTATION_FILE"     anatomical_atlas  2
submit "gp-exp-faiss-2-brainglobe-${EXP_SUFFIX}"   brainglobe  "$S3_REFERENCE_FILE_BG" "$S3_ANNOTATION_FILE_BG"  faiss             2
submit "gp-exp-atlas-2-brainglobe-${EXP_SUFFIX}"   brainglobe  "$S3_REFERENCE_FILE_BG" "$S3_ANNOTATION_FILE_BG"  anatomical_atlas  2

# KNN_METHOD="faiss"
# NU=2
# TEMPLATE_NAME=$S3_TEMPLATE_NAME
# JOB_NAME="gp-exp-${KNN_METHOD}-${NU}-${TEMPLATE_NAME}-${EXP_SUFFIX}"
# runai training submit "$JOB_NAME" \
# 	-i artiomartiom/sdsc:maldi_manifold_latest \
# 	--cpu-core-limit $CPU \
#     --cpu-core-request $CPU \
#     --cpu-memory-limit $MEM \
#     --cpu-memory-request $MEM \
# 	--gpu-request-type portion \
# 	--gpu-portion-request 0.5 \
# 	-e WANDB_API_KEY=$WANDB_API_KEY \
# 	-e DATA_PATH=$S3_DATA_PATH \
# 	-e OUTPUT_DIR=$S3_OUTPUT_DIR \
# 	-e EIGENVECTOR_DIR=$S3_EIGENVECTOR_DIR \
# 	-e MALDI_FILE=$S3_MALDI_FILE \
# 	-e TEMPLATE_NAME=$TEMPLATE_NAME \
# 	-e REFERENCE_FILE=$S3_REFERENCE_FILE \
# 	-e ANNOTATION_FILE=$S3_ANNOTATION_FILE \
# 	-e SLICES_DATASET_FILE=$S3_SLICES_DATASET_FILE \
# 	-e AVAILABLE_LIPIDS_FILE=$S3_AVAILABLE_LIPIDS_FILE \
# 	-e KNN_METHOD=$KNN_METHOD \
# 	-e NU=$NU \
# 	-- "./maldi/run_manifold.sh"

# KNN_METHOD="anatomical_atlas"
# NU=2
# TEMPLATE_NAME=$S3_TEMPLATE_NAME
# JOB_NAME="gp-exp-anatomicalatlas-${NU}-${TEMPLATE_NAME}-${EXP_SUFFIX}"
# runai training submit "$JOB_NAME" \
# 	-i artiomartiom/sdsc:maldi_manifold_latest \
# 	--cpu-core-limit $CPU \
#     --cpu-core-request $CPU \
#     --cpu-memory-limit $MEM \
#     --cpu-memory-request $MEM \
# 	--gpu-request-type portion \
# 	--gpu-portion-request 0.5 \
# 	-e WANDB_API_KEY=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 \
# 	-e DATA_PATH=$S3_DATA_PATH \
# 	-e OUTPUT_DIR=$S3_OUTPUT_DIR \
# 	-e EIGENVECTOR_DIR=$S3_EIGENVECTOR_DIR \
# 	-e MALDI_FILE=$S3_MALDI_FILE \
# 	-e TEMPLATE_NAME=$TEMPLATE_NAME \
# 	-e REFERENCE_FILE=$S3_REFERENCE_FILE \
# 	-e ANNOTATION_FILE=$S3_ANNOTATION_FILE \
# 	-e SLICES_DATASET_FILE=$S3_SLICES_DATASET_FILE \
# 	-e AVAILABLE_LIPIDS_FILE=$S3_AVAILABLE_LIPIDS_FILE \
# 	-e KNN_METHOD=$KNN_METHOD \
# 	-e NU=$NU \
# 	-- "./maldi/run_manifold.sh"

# KNN_METHOD="faiss"
# NU=2
# TEMPLATE_NAME=$S3_BG_TEMPLATE_NAME
# JOB_NAME="gp-exp-${KNN_METHOD}-${NU}-${TEMPLATE_NAME}-${EXP_SUFFIX}"
# runai training submit "$JOB_NAME" \
# 	-i artiomartiom/sdsc:maldi_manifold_latest \
# 	--cpu-core-limit $CPU \
#     --cpu-core-request $CPU \
#     --cpu-memory-limit $MEM \
#     --cpu-memory-request $MEM \
# 	--gpu-request-type portion \
# 	--gpu-portion-request 0.5 \
# 	-e WANDB_API_KEY=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 \
# 	-e DATA_PATH=$S3_DATA_PATH \
# 	-e OUTPUT_DIR=$S3_OUTPUT_DIR \
# 	-e EIGENVECTOR_DIR=$S3_EIGENVECTOR_DIR \
# 	-e MALDI_FILE=$S3_MALDI_FILE \
# 	-e TEMPLATE_NAME=$TEMPLATE_NAME \
# 	-e REFERENCE_FILE=$S3_REFERENCE_FILE_BG \
# 	-e ANNOTATION_FILE=$S3_ANNOTATION_FILE_BG \
# 	-e SLICES_DATASET_FILE=$S3_SLICES_DATASET_FILE \
# 	-e AVAILABLE_LIPIDS_FILE=$S3_AVAILABLE_LIPIDS_FILE \
# 	-e KNN_METHOD=$KNN_METHOD \
# 	-e NU=$NU \
# 	-- "./maldi/run_manifold.sh"

# KNN_METHOD="anatomical_atlas"
# NU=2
# TEMPLATE_NAME=$S3_BG_TEMPLATE_NAME
# JOB_NAME="gp-exp-anatomicalatlas-${NU}-${TEMPLATE_NAME}-${EXP_SUFFIX}"
# runai training submit "$JOB_NAME" \
# 	-i artiomartiom/sdsc:maldi_manifold_latest \
# 	--cpu-core-limit $CPU \
#     --cpu-core-request $CPU \
#     --cpu-memory-limit $MEM \
#     --cpu-memory-request $MEM \
# 	--gpu-request-type portion \
# 	--gpu-portion-request 0.5 \
# 	-e WANDB_API_KEY=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 \
# 	-e DATA_PATH=$S3_DATA_PATH \
# 	-e OUTPUT_DIR=$S3_OUTPUT_DIR \
# 	-e EIGENVECTOR_DIR=$S3_EIGENVECTOR_DIR \
# 	-e MALDI_FILE=$S3_MALDI_FILE \
# 	-e TEMPLATE_NAME=$TEMPLATE_NAME \
# 	-e REFERENCE_FILE=$S3_REFERENCE_FILE_BG \
# 	-e ANNOTATION_FILE=$S3_ANNOTATION_FILE_BG \
# 	-e SLICES_DATASET_FILE=$S3_SLICES_DATASET_FILE \
# 	-e AVAILABLE_LIPIDS_FILE=$S3_AVAILABLE_LIPIDS_FILE \
# 	-e KNN_METHOD=$KNN_METHOD \
# 	-e NU=$NU \
# 	-- "./maldi/run_manifold.sh"
