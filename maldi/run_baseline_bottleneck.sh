#!/usr/bin/env sh
: "${BATCH_SIZE:=256}"
: "${N_EPOCHS:=20}"
: "${LEARNING_RATE:=0.001}"
: "${SEED:=416465}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/difficult.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${MODEL:=mlp}"
: "${RIDGE_ALHPA:=1.0}"
: "${MLP_HIDDEN:=256 5 256 256 128}"
: "${MLP_DROPOUT:=0.1}"
: "${XGB_N_ESTIMATORS:=400}"
: "${XGB_MAX_DEPTH:=6}"
: "${XGB_LR:=0}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=DIFFICULT}"

cd $SRC_PATH
#pip install -e .

EXP_NAME="$EXP_PREFIX-BASELINES-BOTTLENECK-MLP-$BATCH_SIZE"

python $SRC_PATH/maldi/experiment_baselines.py \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --output-dir $OUTPUT_DIR \
    --template-name "reference" \
    --reference-file $REFERENCE_FILE \
    --batch-size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-dim 5 \
    --seed $SEED \
    --slices-dataset-file $SLICES_DATASET_FILE \
    --num-inducing 200 \
    --kernel "matern" \
    --mode "lgp" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --model "$MODEL" \
    --ridge-alpha $RIDGE_ALHPA \
    --mlp-hidden $MLP_HIDDEN \
    --mlp-dropout $MLP_DROPOUT \
    --xgb-n-estimators $XGB_N_ESTIMATORS \
    --xgb-max-depth $XGB_MAX_DEPTH \
    --xgb-lr $XGB_LR \
    --reconstruct whole_brain \
    --reconstruction-lipids "Hex2Cer 40:1;O2" "PA 36:1 PA 38:4" "PC 35:1 PE 38:1" \
    "$@"

# --reconstruct whole_brain \
# --reconstruction-lipids "Hex2Cer 40:1;O2" "PA 36:1 PA 38:4" "PC 35:1 PE 38:1" \
