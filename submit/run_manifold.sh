runai training submit \
	-i artiomartiom/sdsc:maldi_manifold_latest \
	--cpu-core-limit 1 \
    --cpu-core-request 1 \
    --cpu-memory-limit 32G \
    --cpu-memory-request 32G \
	--gpu-request-type portion \
	--gpu-portion-request 0.5 \
	-e WANDB_API_KEY=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 \
	-- "./maldi/run_manifold.sh"

DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/eigenvectors"
OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom"
MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
N_EPOCHS=20
LEARNING_RATE=0.001
SEED=416465
SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/difficult.json"
EXP_NAME="DIFFICULT-MANIFOLD-RSAMPLE-LEARNED-INDUCING-$LATENT_DIM-$NUM_INDUCING_POINTS-$BATCH_SIZE"
KERNEL="symmetric"
MODE="lgp"
AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
