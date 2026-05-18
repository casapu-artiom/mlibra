#!/usr/bin/env sh
: "${NUM_INDUCING_POINTS:=500}"
: "${LATENT_DIM:=5}"
: "${BATCH_SIZE:=1000}"
: "${N_EPOCHS:=20}"
: "${LEARNING_RATE:=0.001}"
: "${SEED:=416465}"
: "${KERNEL:=matern}"
: "${MODE:=lgp}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/difficult.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${SRC_PATH:/home/casap/mlibra_git/}"

EXP_NAME="DIFFICULT-LGPALL-RSAMPLE-$LATENT_DIM-$NUM_INDUCING_POINTS-$BATCH_SIZE"

cd $SRC_PATH
#pip install -e .
python /home/casap/mlibra_git/maldi/lgp_experiment.py \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --template-name "reference" \
    --reference-file $REFERENCE_FILE \
    --annotations-file $ANNOTATION_FILE \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-dim $LATENT_DIM \
    --seed $SEED \
    --slices-dataset-file $SLICES_DATASET_FILE \
    --num-inducing $NUM_INDUCING_POINTS \
    --kernel "$KERNEL" \
    --mode "$MODE" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    "$@"
