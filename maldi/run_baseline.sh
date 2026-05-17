#!/usr/bin/env sh
NUM_INDUCING_POINTS=100
LATENT_DIM=20
BATCH_SIZE=1000
DATA_PATH="/home/casap/mlibra/mlibra_data"
OUTPUT_DIR="/home/casap/mlibra/output"
MALDI_FILE="/home/casap/mlibra/mlibra_data/maindata_minimal.parquet"
N_EPOCHS=100
LEARNING_RATE=0.001
SEED=416465
SLICES_DATASET_FILE="/home/casap/mlibra_git/maldi/data/splits/difficult.json"
EXP_NAME="DIFFICULT-BASELINES-REGIONAL-BOTTLENECK-MLP-$LATENT_DIM-$NUM_INDUCING_POINTS-$BATCH_SIZE"
KERNEL="symmetric"
MODE="lgp"
AVAILABLE_LIPIDS_FILE="/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy"
REFERENCE_FILE="/home/casap/mlibra/mlibra_data/reference_image.npy"
ANNOTATION_FILE="/home/casap/mlibra/mlibra_data/level_15annot.npy"
MODEL="mlp"
RIDGE_ALHPA=1.0
MLP_HIDDEN="256 5 256 256 128"
MLP_DROPOUT=0.1
XGB_N_ESTIMATORS=400
XGB_MAX_DEPTH=6
XGB_LR=0
cd /home/casap/mlibra_git/
#pip install -e .
python /home/casap/mlibra_git/maldi/experiment_baselines.py \
    --exp-name $EXP_NAME \
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
    --num-inducing $NUM_INDUCING_POINTS \
    --kernel "$KERNEL" \
    --mode "$MODE" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --model "$MODEL" \
    --ridge-alpha $RIDGE_ALHPA \
    --mlp-hidden $MLP_HIDDEN \
    --mlp-dropout $MLP_DROPOUT \
    --xgb-n-estimators $XGB_N_ESTIMATORS \
    --xgb-max-depth $XGB_MAX_DEPTH \
    --xgb-lr $XGB_LR \
    --reconstruct none

#--region-bbox 375 475 125 225 175 275
#--log-transform \
#--region-bbox 200 250 150 200 200 250 \
