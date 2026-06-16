#!/usr/bin/env bash
# =============================================================================
# Per-lipid GP hyperparameter sweep.
#
# Two modes of invocation:
#
# 1) SINGLE RUN (override knobs via env vars):
#    KERNEL_FAMILY=manifold NU=2 LENGTHSCALE=1.0 ./run_per_lipid.sh
#
# 2) SWEEP (uncomment / edit one of the SWEEP_* loops below):
#    ./run_per_lipid.sh
#
# Each run lands in $OUTPUT_DIR/<EXP_NAME>/ — EXP_NAME encodes the hypers so
# successive runs in a sweep don't clobber each other. The metrics.csv +
# summary.json files inside each run dir are what you'd point a comparison
# script at.
# =============================================================================

# ---- which kernel ----
: "${KERNEL_FAMILY:=manifold}"          # euclidean | manifold
: "${KERNEL:=matern}"                   # only used for euclidean

# ---- GP hyperparameters ----
: "${NU:=2.5}"
: "${NUM_INDUCING:=1000}"
: "${INDUCING_SOURCE:=reference}"
: "${LIPID_BATCH_SIZE:=10}"
: "${EPOCHS:=10}"
: "${LEARNING_RATE:=0.005}"
: "${BATCH_SIZE:=2048}"
: "${SEED:=42}"

# ---- Variational family -----------------------------------------------------
# Default is the analytic multitask SVGP (the original behaviour). Flip
# VARIATIONAL=nngp to use the Variational Nearest-Neighbour GP (single-lipid,
# Euclidean kernel, no O(M^3) Cholesky). The NN_* knobs below only take effect
# when VARIATIONAL=nngp.
#   VARIATIONAL=nngp ./run_lgp_per_lipid.sh                       # euclidean-NN VNNGP
#   VARIATIONAL=nngp NN_METRIC=geodesic ./run_lgp_per_lipid.sh    # shortest-path NN
: "${VARIATIONAL:=analytic}"         # analytic | nngp
: "${NN_K:=256}"                     # (nngp) conditioning neighbours per point
: "${NNGP_NUM_INDUCING:=0}"          # (nngp) 0 = all training voxels; else cap
: "${NN_METRIC:=euclidean}"          # (nngp) euclidean | geodesic
: "${GEODESIC_GRAPH_K:=16}"          # (nngp, geodesic) faiss kNN graph degree

# ---- subset for fast iteration / debugging -------------------------------
# LIMIT=N           → fit only the first N lipids (after other filtering)
# LIPIDS="a b c"    → fit ONLY these specific lipids (names OR indices).
#                     CAVEAT: shell word-splitting destroys names with
#                     spaces — e.g. "PA 36:1" becomes two tokens. Use
#                     LIPIDS_FILE for those.
# LIPIDS_FILE=path  → text file with one lipid name per line. Ignores
#                     blanks and '#' comments. The right choice for any
#                     non-trivial subset.
: "${LIMIT:=}"
: "${LIPIDS:=}"
: "${LIPIDS_FILE:=/home/casap/mlibra_git/maldi/data/lipid_subset.txt}"

# ---- Manifold-only ----
: "${NUM_MODES:=6000}"
: "${STRIDE:=8}"
: "${KNN_K:=15}"
: "${KNN_METHOD:=faiss_atlas_weighted}"
: "${CROSS_REGION_INFLATION:=10.0}"
: "${LAPLACIAN_NORM:=randomwalk}"
: "${BUMP_SCALE:=1.0}"
: "${BUMP_DECAY:=0.01}"
: "${GRAPHBANDWIDTH:=0.1}"
: "${THRESHOLD:=50}"

# ---- data paths (same as run_manifold.sh) ----
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output/per_lipid}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_3.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${TEMPLATE_NAME:=reference}"
: "${SRC_PATH:=/home/casap/mlibra_git}"

#cd $SRC_PATH

# ---- one-shot launcher (called by the sweep loops too) -----------------
run_one() {
    local FAMILY=$1; local NU_=$2; local LR_=$3; local EPS_=$4
    local INDU_=$5; local BS_=$6; local LBS_=$7
    # Manifold-only knobs:
    local NMODES_=$8; local BSCALE_=$9; local BDECAY_=${10}; local BW_=${11}
    local KNN_=${12}; local LN_=${13}; local KMETHOD_=${14}

    local TAG
    if [ "$FAMILY" = "manifold" ]; then
        TAG="manifold-product-nu${NU_}-K${NMODES_}-bs${BSCALE_}-bd${BDECAY_}-bw${BW_}-knn${KNN_}-${KMETHOD_}-${THRESHOLD}-${LN_}-ind${INDU_}-lr${LR_}-ep${EPS_}-lbs${LBS_}"
    else
        TAG="euclidean-no-ard-${KERNEL}-nu${NU_}-ind${INDU_}-${THRESHOLD}-lr${LR_}-ep${EPS_}-lbs${LBS_}"
    fi
    # VNNGP runs get their own tag suffix so they don't clobber the analytic ones.
    if [ "$VARIATIONAL" = "nngp" ]; then
        TAG="${TAG}-vnngp-${NN_METRIC}-nnk${NN_K}-ni${NNGP_NUM_INDUCING}"
        [ "$NN_METRIC" = "geodesic" ] && TAG="${TAG}-gk${GEODESIC_GRAPH_K}"
    fi
    local EXP_NAME="${TAG}"

    echo ""
    echo "================================================================"
    echo "  RUN: $EXP_NAME"
    echo "================================================================"

    local manifold_args=""
    if [ "$FAMILY" = "manifold" ]; then
        manifold_args="--eigenvector-dir $EIGENVECTOR_DIR \
            --knn-method $KMETHOD_ \
            --cross-region-inflation $CROSS_REGION_INFLATION \
            --laplacian-norm $LN_ \
            --stride $STRIDE \
            --knn-k $KNN_ \
            --bump-scale $BSCALE_ \
            --bump-decay $BDECAY_ \
            --graphbandwidth-init $BW_ \
            --num-modes $NMODES_ \
            --threshold "$THRESHOLD" \
            --lengthscale-init 8.0 \
            --lengthscale-no-decay"
    fi

    # Variational-family args. --variational is always passed (defaults to
    # 'analytic'); the NN_* knobs are only appended for the nngp path.
    local vnngp_args="--variational $VARIATIONAL"
    if [ "$VARIATIONAL" = "nngp" ]; then
        vnngp_args="$vnngp_args \
            --nn-k $NN_K \
            --nngp-num-inducing $NNGP_NUM_INDUCING \
            --nn-metric $NN_METRIC \
            --geodesic-graph-k $GEODESIC_GRAPH_K"
    fi

    # Subset args from env vars. Use --limit OR --lipids OR --lipids-file
    # (or any combination — limit is applied after the name/index filter).
    local subset_args=""
    if [ -n "$LIMIT" ]; then
        subset_args="$subset_args --limit $LIMIT"
    fi
    if [ -n "$LIPIDS" ]; then
        # Don't quote $LIPIDS — we want shell-word-splitting so each
        # token in the env var becomes its own CLI arg.
        subset_args="$subset_args --lipids $LIPIDS"
    fi
    if [ -n "$LIPIDS_FILE" ]; then
        subset_args="$subset_args --lipids-file $LIPIDS_FILE"
    fi

    python "$SRC_PATH/maldi/lgp_experiment_per_lipid.py" \
        --kernel-family "$FAMILY" \
        --exp-name "$EXP_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-path "$DATA_PATH" \
        --maldi-file "$MALDI_FILE" \
        --available-lipids-file "$AVAILABLE_LIPIDS_FILE" \
        --slices-dataset-file "$SLICES_DATASET_FILE" \
        --template-name "$TEMPLATE_NAME" \
        --reference-file "$REFERENCE_FILE" \
        --annotations-file "$ANNOTATION_FILE" \
        --kernel "$KERNEL" \
        --nu "$NU_" \
        --num-inducing "$INDU_" \
        --inducing-source "$INDUCING_SOURCE" \
        --lipid-batch-size "$LBS_" \
        --epochs "$EPS_" \
        --learning-rate "$LR_" \
        --batch-size "$BS_" \
        --seed "$SEED" \
        $vnngp_args \
        $manifold_args \
        $subset_args \
        "${@:15}"
}

# =============================================================================
# Default: ONE run with the env-vars set above (skip to the sweeps below
# by passing --sweep ... or by editing this script directly).
# =============================================================================
if [ "$1" != "--sweep" ]; then
    run_one "$KERNEL_FAMILY" "$NU" "$LEARNING_RATE" "$EPOCHS" \
            "$NUM_INDUCING" "$BATCH_SIZE" "$LIPID_BATCH_SIZE" \
            "$NUM_MODES" "$BUMP_SCALE" "$BUMP_DECAY" "$GRAPHBANDWIDTH" \
            "$KNN_K" "$LAPLACIAN_NORM" "$KNN_METHOD" "$@"
    exit $?
fi
shift  # consume --sweep

# =============================================================================
# SWEEP definitions — pick the second CLI arg
#   ./run_per_lipid.sh --sweep knn_k
#   ./run_per_lipid.sh --sweep graphbandwidth
#   ./run_per_lipid.sh --sweep laplacian_norm
#   ./run_per_lipid.sh --sweep knn_method
#   ./run_per_lipid.sh --sweep all
#
# All sweeps run the MANIFOLD kernel only — these axes have no analogue
# on the Euclidean side. Run the Euclidean baseline once separately:
#   KERNEL_FAMILY=euclidean ./run_per_lipid.sh
# =============================================================================
SWEEP_NAME="${1:-all}"
echo "Running manifold sweep: $SWEEP_NAME"

# Shorthand: one positional template; sweeps override one variable at a time.
manifold_run() {
    # Args (in order): NU LR EPS INDU BS LBS NMODES BSCALE BDECAY BW KNN LN KMETHOD
    run_one manifold "$@"
}

case "$SWEEP_NAME" in
    knn_k)
        # KNN graph degree — controls how locally connected the
        # Laplacian is. Small k → very local Laplacian, eigenmodes look
        # like tiny patches; large k → smoother manifold but heavier
        # graph and eigensolve. Sensible range for the brain atlas
        # at stride 4: ~10 to ~60.
        for knn in 10 15 30 60 120; do
            manifold_run "$NU" "$LEARNING_RATE" "$EPOCHS" \
                "$NUM_INDUCING" "$BATCH_SIZE" "$LIPID_BATCH_SIZE" \
                "$NUM_MODES" "$BUMP_SCALE" "$BUMP_DECAY" "$GRAPHBANDWIDTH" \
                "$knn" "$LAPLACIAN_NORM" "$KNN_METHOD"
        done
        ;;

    graphbandwidth)
        # Graph bandwidth — the scale parameter inside the Gaussian
        # kernel that turns kNN distances into edge weights:
        #     w_ij = exp(-||x_i - x_j||² / (2 * bw²))
        # Small bw → only the very closest neighbours contribute,
        # Laplacian is sparse-effective; large bw → all kNN edges look
        # similar, Laplacian smooths over too-wide regions. Default
        # 0.1 (z-scored coords).
        for bw in 0.05 0.1 0.2 0.5 1.0; do
            manifold_run "$NU" "$LEARNING_RATE" "$EPOCHS" \
                "$NUM_INDUCING" "$BATCH_SIZE" "$LIPID_BATCH_SIZE" \
                "$NUM_MODES" "$BUMP_SCALE" "$BUMP_DECAY" "$bw" \
                "$KNN_K" "$LAPLACIAN_NORM" "$KNN_METHOD"
        done
        ;;

    laplacian_norm)
        # Normalisation of the graph Laplacian. 'symmetric' = D^-1/2 L D^-1/2
        # (eigenvectors are orthonormal in ℓ²), 'randomwalk' = D^-1 L
        # (eigenvectors are orthonormal w.r.t. degree-weighted inner
        # product). Affects which low-frequency modes get the most weight.
        for ln in symmetric randomwalk; do
            manifold_run "$NU" "$LEARNING_RATE" "$EPOCHS" \
                "$NUM_INDUCING" "$BATCH_SIZE" "$LIPID_BATCH_SIZE" \
                "$NUM_MODES" "$BUMP_SCALE" "$BUMP_DECAY" "$GRAPHBANDWIDTH" \
                "$KNN_K" "$ln" "$KNN_METHOD"
        done
        ;;

    knn_method)
        # How the kNN graph itself is built. 'faiss' = pure Euclidean
        # kNN over standardised coordinates; 'anatomical_atlas' = kNN
        # restricted to within-region or atlas-aware edges (definition
        # in the manifold_gp utils). The latter is a much stronger
        # anatomical prior at the cost of being atlas-quality-dependent.
        for km in faiss anatomical_atlas; do
            manifold_run "$NU" "$LEARNING_RATE" "$EPOCHS" \
                "$NUM_INDUCING" "$BATCH_SIZE" "$LIPID_BATCH_SIZE" \
                "$NUM_MODES" "$BUMP_SCALE" "$BUMP_DECAY" "$GRAPHBANDWIDTH" \
                "$KNN_K" "$LAPLACIAN_NORM" "$km"
        done
        ;;

    all)
        # Cross-product of the four "important" manifold axes. With
        # 5 × 5 × 2 × 2 = 100 runs this is BIG — prune the inner lists
        # to fit the time budget. Order chosen so that knn_method and
        # laplacian_norm (the cheaper-to-vary axes) are outermost,
        # maximising graph-cache hits for the inner sweeps.
        for km in faiss anatomical_atlas; do
            for ln in symmetric randomwalk; do
                for knn in 15 30 60; do
                    for bw in 0.1 0.2 0.5; do
                        manifold_run "$NU" "$LEARNING_RATE" "$EPOCHS" \
                            "$NUM_INDUCING" "$BATCH_SIZE" "$LIPID_BATCH_SIZE" \
                            "$NUM_MODES" "$BUMP_SCALE" "$BUMP_DECAY" "$bw" \
                            "$knn" "$ln" "$km"
                    done
                done
            done
        done
        ;;

    *)
        echo "Unknown sweep: $SWEEP_NAME"
        echo "  options: knn_k | graphbandwidth | laplacian_norm | knn_method | all"
        exit 2
        ;;
esac

echo ""
echo "================================================================"
echo "Sweep '$SWEEP_NAME' complete. Results under $OUTPUT_DIR/"
echo "Aggregate with e.g.:"
echo "  python -c 'import pandas as pd, pathlib as P; \\"
echo "    print(pd.concat([pd.read_csv(d/\"metrics.csv\").assign(run=d.name) \\"
echo "      for d in P.Path(\"$OUTPUT_DIR\").iterdir() \\"
echo "      if (d/\"metrics.csv\").exists()]))'"
echo "================================================================"