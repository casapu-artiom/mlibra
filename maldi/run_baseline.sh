#!/usr/bin/env sh
# Non-GP baselines, with the same reconstruction + render + diagnostics parity as
# run_manifold.sh. MODEL selects the baseline:
#   mean | linear | xgboost | mlp | mlp_bottleneck | mlp_eigen | gcn | gcn_faiss
#   mlp_bottleneck — MLP with a narrow middle (bottleneck) layer; a preset that
#               maps to --model mlp with MLP_HIDDEN='256 5 256 256 128'
#   mlp_eigen — MLP on [coords, points projected to the manifold eigenbasis]
#               (needs the eigenpair pipeline; set EIGENVECTOR_DIR + graph knobs)
#   gcn       — Graph Conv Net over a per-batch KNN graph of the coords
#   gcn_faiss — Graph Conv Net over the FAISS reference-node manifold graph
#               (needs the graph pipeline; set EIGENVECTOR_DIR + graph knobs)
: "${BATCH_SIZE:=256}"
: "${N_EPOCHS:=10}"
: "${LEARNING_RATE:=0.001}"
: "${SEED:=416465}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_2.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${MODEL:=gcn_faiss}"
: "${RIDGE_ALHPA:=1.0}"
# mlp_bottleneck is a run_baseline preset: an MLP with a narrow middle
# (bottleneck) layer. It maps to --model mlp with a bottleneck MLP_HIDDEN default
# (experiment_baselines.py has no separate bottleneck model class); MODEL still
# tags the exp name distinctly so it doesn't clobber a plain mlp run.
ACTUAL_MODEL="$MODEL"
if [ "$MODEL" = "mlp_bottleneck" ]; then
    ACTUAL_MODEL="mlp"
    : "${MLP_HIDDEN:=256 5 256 256 128}"
fi
: "${MLP_HIDDEN:=256 256 128}"
: "${MLP_DROPOUT:=0.1}"
: "${XGB_N_ESTIMATORS:=400}"
: "${XGB_MAX_DEPTH:=6}"
: "${XGB_LR:=0}"

# --- GCN knobs (MODEL=gcn / gcn_faiss) --------------------------------------
: "${GCN_HIDDEN:=256 256 128}"
: "${GCN_DROPOUT:=0.1}"
: "${GCN_KNN_K:=15}"
# gcn_faiss is full-graph (1 step/forward), so it needs its own iteration budget
# (not the shared N_EPOCHS, which is tuned for the minibatch models).
: "${GCN_FAISS_ITERS:=2000}"
: "${GCN_FAISS_NODE_BATCH:=65536}"

# --- Manifold graph/eigenbasis knobs (MODEL=mlp_eigen|gcn_faiss); ignored otherwise --
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${NUM_MODES:=100}"
: "${STRIDE:=4}"
: "${THRESHOLD:=5}"
: "${KNN_K:=15}"
: "${LAPLACIAN_NORM:=randomwalk}"
: "${KNN_METHOD:=faiss_atlas_weighted}"
# Cross-region edge-weight inflation; only takes effect with KNN_METHOD=faiss_atlas_weighted.
: "${CROSS_REGION_INFLATION:=10.0}"
: "${GRAPHBANDWIDTH_INIT:=0.1}"
: "${BUMP_SCALE:=1.0}"
: "${BUMP_DECAY:=0.01}"
: "${NU:=2.5}"

: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=FOLD-2}"

cd $SRC_PATH
#pip install -e .

# Encode the actual MODEL so a sweep over models gets distinct dirs instead of
# all landing in one hardcoded path and clobbering each other.
MODEL_TAG=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
EXP_NAME="$EXP_PREFIX-BASELINES-$MODEL_TAG-$BATCH_SIZE"

# ---- reconstruction lipids from the curated subset file (mirror run_manifold) ----
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
    echo "NOTE: $RECONSTRUCTION_LIPIDS_FILE missing/empty; using built-in default subset."
    RECON_LIPIDS="$RECON_LIPIDS_DEFAULT"
fi

# Manifold graph/eigenbasis flags are only forwarded for the manifold-aware
# baselines (mlp_eigen needs the eigenbasis, gcn_faiss needs the graph).
EIGEN_ARGS=""
if [ "$MODEL" = "mlp_eigen" ] || [ "$MODEL" = "gcn_faiss" ]; then
    EIGEN_ARGS="--eigenvector-dir $EIGENVECTOR_DIR \
        --num-modes $NUM_MODES \
        --stride $STRIDE \
        --threshold $THRESHOLD \
        --knn-k $KNN_K \
        --laplacian-norm $LAPLACIAN_NORM \
        --knn-method $KNN_METHOD \
        --annotations-file $ANNOTATION_FILE \
        --cross-region-inflation $CROSS_REGION_INFLATION \
        --graphbandwidth-init $GRAPHBANDWIDTH_INIT \
        --bump-scale $BUMP_SCALE \
        --bump-decay $BUMP_DECAY \
        --nu $NU"
fi

IFS='
'
set -- --reconstruction-lipids $RECON_LIPIDS "$@"
unset IFS

python $SRC_PATH/maldi/experiment_baselines.py \
    --exp-name "$EXP_NAME" \
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
    --model "$ACTUAL_MODEL" \
    --ridge-alpha $RIDGE_ALHPA \
    --mlp-hidden $MLP_HIDDEN \
    --mlp-dropout $MLP_DROPOUT \
    --xgb-n-estimators $XGB_N_ESTIMATORS \
    --xgb-max-depth $XGB_MAX_DEPTH \
    --xgb-lr $XGB_LR \
    --gcn-hidden $GCN_HIDDEN \
    --gcn-dropout $GCN_DROPOUT \
    --gcn-knn-k $GCN_KNN_K \
    --gcn-faiss-iters $GCN_FAISS_ITERS \
    --gcn-faiss-node-batch $GCN_FAISS_NODE_BATCH \
    $EIGEN_ARGS \
    --reconstruct whole_brain \
    "$@"
