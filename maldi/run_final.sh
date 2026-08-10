#!/usr/bin/env sh
NUM_INDUCING_POINTS=500
LATENT_DIM=5
BATCH_SIZE=1000
DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
OUTPUT_DIR="/s3/mlibra/mlibra-data/maldi/lmmvae"
MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
N_EPOCHS=50
LEARNING_RATE=0.001
SEED=416465
SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/all.json"
EXP_NAME="LGPALL-$LATENT_DIM-$NUM_INDUCING_POINTS-$BATCH_SIZE"
KERNEL="symmetric"
MODE="lgp"
AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
cd /myhome/mlibra/
pip install -e .
python /myhome/mlibra/maldi/lgp_experiment.py \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
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
    --log-transform
