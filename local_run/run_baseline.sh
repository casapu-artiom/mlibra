#!/usr/bin/env sh
# Non-GP baselines, with the same reconstruction + render + diagnostics parity as
# run_manifold.sh. MODEL selects the baseline:
#   mean | linear | xgboost | mlp | mlp_bottleneck | mlp_eigen | gcn | gcn_faiss
#   | euclid
#   mlp_bottleneck — MLP with a narrow middle (bottleneck) layer; a preset that
#               maps to --model mlp with MLP_HIDDEN='256 5 256 256 128'
#   mlp_eigen — MLP on [coords, points projected to the manifold eigenbasis]
#               (needs the eigenpair pipeline; set EIGENVECTOR_DIR + graph knobs)
#   gcn       — Graph Conv Net over a per-batch KNN graph of the coords
#   gcn_faiss — Graph Conv Net over the FAISS reference-node manifold graph
#               (needs the graph pipeline; set EIGENVECTOR_DIR + graph knobs)
#   euclid    — EUCLID's anatomical_interpolation, executed as-is from the
#               EUCLID checkout (repo root: ./euclid). Nothing is trained, so
#               BATCH_SIZE / N_EPOCHS / LEARNING_RATE are inert. Only knobs:
#               EUCLID_REPO, EUCLID_W (their w, default 50), EUCLID_JOBS
#               (their kernel is ~242 s/lipid; 25-way => ~30 min for 173).
: "${BATCH_SIZE:=256}"
# Reconstruction forward pass only; NOT the training minibatch (that's BATCH_SIZE,
# which is also baked into EXP_NAME, so don't repurpose it to speed up inference).
: "${INFERENCE_BATCH_SIZE:=65536}"
: "${N_EPOCHS:=10}"
: "${LEARNING_RATE:=0.001}"
: "${SEED:=416465}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output/baseline}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_2.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${MODEL:=mlp}"
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

# Reconstruct only the voxels the composite render actually reads (slice planes +
# the 3D MIP's stride): ~5.5x fewer voxels, near-identical figure. Writes sparse
# volumes to volume_sparse/ instead of the dense volume/ that napari + the
# analysis scripts consume — so leave it 0 if you need the full 3D volumes.
: "${RENDER_VOXELS_ONLY:=1}"

# --- EUCLID knobs (MODEL=euclid); ignored otherwise -------------------------
# EUCLID's own defaults; there is nothing else to tune (grid, radius, exp(-d)
# weights, leaf gate and index map all come from their code + shipped volumes).
: "${EUCLID_REPO:=/home/casap/mlibra_git/euclid}"
: "${EUCLID_W:=50}"
: "${EUCLID_JOBS:=25}"
EUCLID_ARGS=""
if [ "$MODEL" = "euclid" ]; then
    # The EUCLID checkout is not part of this repo (it is a separate clone, and
    # untracked here), so on a fresh cluster node it has to be fetched. Its two
    # 100um .npy volumes are committed in that repo, so the clone is all we need.
    if [ ! -f "$EUCLID_REPO/src/euclid_msi/postprocessing.py" ]; then
        echo "run_baseline: EUCLID checkout missing at $EUCLID_REPO -- cloning"
        git clone --depth 1 https://github.com/lamanno-epfl/EUCLID.git "$EUCLID_REPO" || {
            echo "run_baseline: FAILED to clone EUCLID into $EUCLID_REPO" >&2; exit 1; }
    fi
    EUCLID_ARGS="--euclid-repo $EUCLID_REPO --euclid-w $EUCLID_W --euclid-jobs $EUCLID_JOBS"
    [ -n "$EUCLID_VERIFY_REDUCTION" ] && EUCLID_ARGS="$EUCLID_ARGS --euclid-verify-reduction"
fi

# --- GCN knobs (MODEL=gcn / gcn_faiss) --------------------------------------
: "${GCN_HIDDEN:=512 512 256}"
: "${GCN_DROPOUT:=0.1}"
: "${GCN_KNN_K:=15}"
# gcn_faiss is full-graph (1 step/forward), so it needs its own iteration budget
# (not the shared N_EPOCHS, which is tuned for the minibatch models).
: "${GCN_FAISS_ITERS:=2000}"
: "${GCN_FAISS_NODE_BATCH:=65536}"

# --- Manifold graph/eigenbasis knobs (MODEL=mlp_eigen|gcn_faiss); ignored otherwise --
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
# mlp_eigen stages its feature memmap here; must be LOCAL disk (mmap is
# unsupported on the S3/FUSE mounts). Empty = python falls back to TMPDIR.
: "${FEAT_SCRATCH_DIR:=}"
: "${NUM_MODES:=300}"
: "${STRIDE:=4}"
: "${THRESHOLD:=5}"
: "${KNN_K:=15}"
: "${LAPLACIAN_NORM:=randomwalk}"
: "${KNN_METHOD:=faiss_atlas_weighted}"
# Cross-region edge-weight inflation; only takes effect with KNN_METHOD=faiss_atlas_weighted.
: "${CROSS_REGION_INFLATION:=50.0}"
# Atlas root handling + label denoise + hard prune (faiss_atlas_weighted; mirror run_manifold.sh).
# ROOT_HANDLING: dissolve (default) | ignore | cross. DENOISE_LABELS: majority-vote passes (0=off).
# PRUNE_CROSS_REGION: fraction of cross-region edges to HARD-remove (0=off). Non-cross root and
# any prune change the eigvec cache key (fresh solve).
: "${ROOT_HANDLING:=dissolve}"
: "${DENOISE_LABELS:=3}"
: "${PRUNE_CROSS_REGION:=0.97}"
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

# Reflect root/denoise/prune in the exp name (only for the manifold-aware models on
# the atlas-weighted graph) so a prune-config sweep gets distinct dirs instead of
# clobbering. Mirrors run_manifold.sh; 'cross' root + zero denoise/prune stay unsuffixed
# so legacy names remain valid.
if { [ "$MODEL" = "mlp_eigen" ] || [ "$MODEL" = "gcn_faiss" ]; } \
   && [ "$KNN_METHOD" = "faiss_atlas_weighted" ]; then
    if [ "$ROOT_HANDLING" != "cross" ]; then
        EXP_NAME="$EXP_NAME-root$ROOT_HANDLING"
    fi
    if [ "${DENOISE_LABELS:-0}" -gt 0 ]; then
        EXP_NAME="$EXP_NAME-dn$DENOISE_LABELS"
    fi
    if [ "$(awk "BEGIN{print (${PRUNE_CROSS_REGION:-0}>0)?1:0}")" = "1" ]; then
        EXP_NAME="$EXP_NAME-prune$PRUNE_CROSS_REGION"
    fi
fi

# Only EUCLID_W changes the result, so it is the only thing in the name.
if [ "$MODEL" = "euclid" ] && [ "$EUCLID_W" != "50" ]; then
    EXP_NAME="$EXP_NAME-w$EUCLID_W"
fi

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
        --root-handling $ROOT_HANDLING \
        --denoise-labels $DENOISE_LABELS \
        --prune-cross-region $PRUNE_CROSS_REGION \
        --graphbandwidth-init $GRAPHBANDWIDTH_INIT \
        --bump-scale $BUMP_SCALE \
        --bump-decay $BUMP_DECAY \
        --nu $NU"
fi

# Only forwarded when set; unset lets python default to TMPDIR (an empty
# --feat-scratch-dir would swallow the next flag as its value).
SCRATCH_ARGS=""
if [ -n "$FEAT_SCRATCH_DIR" ]; then
    SCRATCH_ARGS="--feat-scratch-dir $FEAT_SCRATCH_DIR"
fi

RENDER_ARGS=""
if [ "$RENDER_VOXELS_ONLY" != "0" ]; then
    RENDER_ARGS="--render-voxels-only"
fi

IFS='
'
set -- --reconstruction-lipids $RECON_LIPIDS "$@"
unset IFS

python $SRC_PATH/baselines/experiment_baselines.py \
    --exp-name "$EXP_NAME" \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --output-dir $OUTPUT_DIR \
    --template-name "reference" \
    --reference-file $REFERENCE_FILE \
    --batch-size $BATCH_SIZE \
    --inference-batch-size $INFERENCE_BATCH_SIZE \
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
    $EUCLID_ARGS \
    $SCRATCH_ARGS \
    $RENDER_ARGS \
    --reconstruct whole_brain \
    "$@"
