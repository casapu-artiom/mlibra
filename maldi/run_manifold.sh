#!/usr/bin/env sh
: "${NUM_INDUCING_POINTS:=500}"
: "${NUM_MODES:=1000}"
: "${LATENT_DIM:=5}"
: "${STRIDE:=4}"
: "${BATCH_SIZE:=1000}"
: "${N_EPOCHS:=2}"
: "${LEARNING_RATE:=0.001}"
: "${GRAPHBANDWIDTH:=0.05}"
: "${NU:=1}"
: "${KNN_K:=15}"
: "${BUMP_SCALE:=20.0}"
: "${BUMP_DECAY:=0.01}"
: "${SEED:=416465}"
: "${KERNEL:=symmetric}"
: "${MODE:=lgp}"
: "${TEMPLATE_NAME:=reference}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/difficult.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${KNN_METHOD:=faiss}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=DIFFICULT}"
: "${LAPLACIAN_NORM:=symmetric}"

cd $SRC_PATH
#pip install -e .

EXP_NAME="$EXP_PREFIX-MANIFOLD-RSAMPLE-$LATENT_DIM-$STRIDE-$TEMPLATE_NAME-$NUM_INDUCING_POINTS-$BATCH_SIZE-$KNN_METHOD-$NU-$BUMP_SCALE-$BUMP_DECAY-$GRAPHBANDWIDTH"

python $SRC_PATH/maldi/lgp_manifold_experiment.py \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --output-dir $OUTPUT_DIR \
    --template-name $TEMPLATE_NAME \
    --reference-file $REFERENCE_FILE \
    --annotations-file $ANNOTATION_FILE \
    --eigenvector-dir $EIGENVECTOR_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-dim $LATENT_DIM \
    --seed $SEED \
    --slices-dataset-file $SLICES_DATASET_FILE \
    --num-inducing $NUM_INDUCING_POINTS \
    --num-modes $NUM_MODES \
    --kernel "$KERNEL" \
    --laplacian-norm "$LAPLACIAN_NORM" \
    --mode "$MODE" \
    --nu $NU \
    --bump-scale $BUMP_SCALE \
    --bump-decay $BUMP_DECAY \
    --graphbandwidth-init $GRAPHBANDWIDTH \
    --knn-method $KNN_METHOD \
    --knn-k $KNN_K \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --do-brain-reconstruction \
    --reconstruction-lipids "Hex2Cer 40:1;O2" "PA 36:1 PA 38:4" "PC 35:1 PE 38:1" \
    "$@"
