#!/usr/bin/env sh
# Run the GPLFR (collapsed linear decoder) experiment on a MALDI CV fold.
# The latent GP that produces Z is selectable via BASE_GP:
#   euclidean (default) — Euclidean inducing-point IndependentMultitaskGPModel
#   riemann             — manifold inducing-point LatentRiemannGP
#   spectral            — weight-space SpectralLatentGP over the manifold spectrum
# The manifold bases (riemann|spectral) need the eigenpair pipeline; set
# EIGENVECTOR_DIR (and the graph/spectrum knobs below) for those.
: "${BATCH_SIZE:=2000}"
: "${N_EPOCHS:=2}"
: "${LEARNING_RATE:=0.005}"
: "${SEED:=416465}"
: "${LATENT_DIM:=8}"
: "${NUM_INDUCING:=256}"
: "${INVERSE_TEMPERATURE:=0.1}"
: "${KERNEL:=matern}"
: "${NU:=2.5}"
: "${BASE_GP:=spectral}"

# --- Manifold-base (riemann|spectral) knobs; ignored when BASE_GP=euclidean ----
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${NUM_MODES:=300}"
: "${STRIDE:=4}"
: "${THRESHOLD:=5}"
: "${KNN_K:=15}"
: "${LAPLACIAN_NORM:=randomwalk}"
: "${KNN_METHOD:=faiss_atlas_weighted}"
: "${CROSS_REGION_INFLATION:=10.0}"
# FAISS IVF sizing. Pass an int or 'sqrt' (nlist=sqrt(N), nprobe=sqrt(nlist)).
# 'sqrt'/8 partitions the index so per-step neighbor searches aren't exhaustive
# (nlist=1 = single cell = O(N) scan every step); mirrors run_manifold.sh.
: "${N_LIST:=sqrt}"
: "${N_PROBE:=8}"
: "${GRAPHBANDWIDTH_INIT:=0.1}"
: "${BUMP_SCALE:=1.0}"
: "${BUMP_DECAY:=0.01}"

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

EXP_NAME="$EXP_PREFIX-GPLFR-$BASE_GP-d$LATENT_DIM"

# Manifold flags are only forwarded for the manifold bases (euclidean ignores them).
MANIFOLD_ARGS=""
if [ "$BASE_GP" != "euclidean" ]; then
    MANIFOLD_ARGS="--eigenvector-dir $EIGENVECTOR_DIR \
        --num-modes $NUM_MODES \
        --stride $STRIDE \
        --threshold $THRESHOLD \
        --knn-k $KNN_K \
        --laplacian-norm $LAPLACIAN_NORM \
        --knn-method $KNN_METHOD \
        --annotations-file $ANNOTATION_FILE \
        --cross-region-inflation $CROSS_REGION_INFLATION \
        --n-list $N_LIST \
        --n-probe $N_PROBE \
        --graphbandwidth-init $GRAPHBANDWIDTH_INIT \
        --bump-scale $BUMP_SCALE \
        --bump-decay $BUMP_DECAY"
fi

# ---- reconstruction lipids from the curated subset file (mirror run_manifold) ----
# Reconstruct/render exactly the lipids listed in RECONSTRUCTION_LIPIDS_FILE.
# A tiny python parser emits one lipid per line; with IFS=newline we fold each
# line into a positional arg -- preserving spaces in names -- so
# --reconstruction-lipids (nargs='+') picks them up.
: "${RECONSTRUCTION_LIPIDS_FILE:=/home/casap/mlibra_git/maldi/data/lipid_subset.txt}"
RECON_LIPIDS_DEFAULT='PC 35:1 PE 38:1
PA 36:1
LPC 22:6
PE O-36:0 PE O-38:3
Hex2Cer 40:1;O2'
RECON_LIPIDS=$(python - "$RECONSTRUCTION_LIPIDS_FILE" <<'PY'
import sys
try:
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                print(line)
except FileNotFoundError:
    pass
PY
)
if [ -n "$RECON_LIPIDS" ]; then
    echo "Reconstruction lipids from $RECONSTRUCTION_LIPIDS_FILE"
else
    echo "NOTE: $RECONSTRUCTION_LIPIDS_FILE missing/empty;" \
         "using built-in default lipid subset."
    RECON_LIPIDS="$RECON_LIPIDS_DEFAULT"
fi
IFS='
'
set -- "$@" --reconstruction-lipids $RECON_LIPIDS
unset IFS

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
    --base-gp "$BASE_GP" \
    $MANIFOLD_ARGS \
    --mode "gplfr" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --do-brain-reconstruction \
    "$@"
