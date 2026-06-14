#!/usr/bin/env sh
: "${NUM_INDUCING_POINTS:=1000}"
: "${INDUCING_SOURCE:=reference}"
: "${LATENT_DIM:=5}"
: "${BATCH_SIZE:=1000}"
: "${N_EPOCHS:=10}"
: "${LEARNING_RATE:=0.001}"
: "${SEED:=416465}"
: "${KERNEL:=matern}"
: "${MODE:=lgp}"
: "${NO_RSAMPLE:=false}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_3.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=FOLD-3}"

if [ "$NO_RSAMPLE" = "true" ] || [ "$NO_RSAMPLE" = "1" ]; then
    SAMPLING_TAG="MEAN"         # Update this to whatever you want the name to be without rsample
    SAMPLING_FLAG="--no-rsample" # The flag to pass to your python script
else
    SAMPLING_TAG="RSAMPLE"
    SAMPLING_FLAG=""             # Leave empty so Python falls back to its default
fi

EXP_NAME="$EXP_PREFIX-LGPALL-$SAMPLING_TAG-$LATENT_DIM-$INDUCING_SOURCE-$NUM_INDUCING_POINTS-$BATCH_SIZE"

cd $SRC_PATH
#pip install -e .

python $SRC_PATH/maldi/lgp_experiment.py \
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
    --inducing-source "$INDUCING_SOURCE" \
    --kernel "$KERNEL" \
    --mode "$MODE" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --do-brain-reconstruction \
    --reconstruction-lipids "Hex2Cer 40:1;O2" "PA 36:1 PA 38:4" "PC 35:1 PE 38:1" \
    $SAMPLING_FLAG "$@"
