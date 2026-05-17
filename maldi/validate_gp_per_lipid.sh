NUM_INDUCING_POINTS=500
BATCH_SIZE=1000
DATA_PATH="/home/casap/mlibra/mlibra_data"
OUTPUT_DIR="/home/casap/mlibra/output"
MALDI_FILE="/home/casap/mlibra/mlibra_data/maindata_minimal.parquet"
EIGENVECTOR_DIR="/home/casap/mlibra/output/eigenvectors"
AVAILABLE_LIPIDS_FILE="/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy"
REFERENCE_FILE="/home/casap/mlibra/mlibra_data/reference_image.npy"
ANNOTATION_FILE="/home/casap/mlibra/mlibra_data/level_15annot.npy"
N_EPOCHS=20
LEARNING_RATE=0.001
SEED=416465
SLICES_DATASET_FILE="/home/casap/mlibra_git/maldi/data/splits/fold_3.json"
EXP_NAME="VALIDATE_GP-$NUM_INDUCING_POINTS-$BATCH_SIZE"
KERNEL="symmetric"
MODE="lgp"


python maldi/validate_gp_per_lipid.py \
    --template-name "reference" \
    --reference-file $REFERENCE_FILE \
    --annotations-file $ANNOTATION_FILE \
    --dataset-path=$DATA_PATH \
    --maldi-file=$MALDI_FILE \
    --exp-name=$EXP_NAME \
    --available-lipids-file=$AVAILABLE_LIPIDS_FILE \
    --output-dir=$OUTPUT_DIR \
    --eigenvector-dir $EIGENVECTOR_DIR \
    --slices-dataset-file=$SLICES_DATASET_FILE \
    --num-lipids=5 \
    --num-inducing=$NUM_INDUCING_POINTS \
    --epochs=$N_EPOCHS \
    --learning-rate=$LEARNING_RATE \
    --batch-size=$BATCH_SIZE
