#!/usr/bin/env sh
# Run the GPLFR (collapsed linear decoder) experiment on a MALDI CV fold.
# Mirrors run_baseline.sh but targets gplfr_experiment.py.
: "${BATCH_SIZE:=2000}"
: "${N_EPOCHS:=50}"
: "${LEARNING_RATE:=0.005}"
: "${SEED:=416465}"
: "${LATENT_DIM:=32}"
: "${NUM_INDUCING:=1000}"
: "${INVERSE_TEMPERATURE:=0.1}"
: "${KERNEL:=matern}"
: "${NU:=2.5}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_2.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=FOLD-2}"

cd $SRC_PATH

EXP_NAME="$EXP_PREFIX-GPLFR-d$LATENT_DIM"

python $SRC_PATH/maldi/gplfr_experiment.py \
    --exp-name "$EXP_NAME" \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --output-dir $OUTPUT_DIR \
    --template-name "reference" \
    --reference-file $REFERENCE_FILE \
    --batch-size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-dim $LATENT_DIM \
    --seed $SEED \
    --slices-dataset-file $SLICES_DATASET_FILE \
    --num-inducing $NUM_INDUCING \
    --kernel "$KERNEL" \
    --nu $NU \
    --inverse-temperature $INVERSE_TEMPERATURE \
    --mode "gplfr" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    "$@"
