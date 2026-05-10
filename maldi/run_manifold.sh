#!/usr/bin/env sh
NUM_INDUCING_POINTS=500
NUM_MODES=1000
LATENT_DIM=5
BATCH_SIZE=1000
DATA_PATH="/home/casap/mlibra/mlibra_data"
EIGENVECTOR_DIR="/home/casap/mlibra/output/eigenvectors"
OUTPUT_DIR="/home/casap/mlibra/output"
MALDI_FILE="/home/casap/mlibra/mlibra_data/maindata_minimal.parquet"
N_EPOCHS=1
LEARNING_RATE=0.001
SEED=416465
SLICES_DATASET_FILE="/home/casap/mlibra_git/maldi/data/splits/fold_3.json"
EXP_NAME="MANIFOLD-$LATENT_DIM-$NUM_INDUCING_POINTS-$BATCH_SIZE-test-docker"
KERNEL="symmetric"
MODE="lgp"
AVAILABLE_LIPIDS_FILE="/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy"
cd /home/casap/mlibra_git/
#pip install -e .
python /home/casap/mlibra_git/maldi/lgp_manifold_experiment.py \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --output-dir $OUTPUT_DIR \
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
    --mode "$MODE" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --log-transform
