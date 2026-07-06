#!/usr/bin/env bash
# =============================================================================
# Per-lipid GP hyperparameter sweep.
#
# Two modes of invocation:
#
# 1) SINGLE RUN (override knobs via env vars):
#    KERNEL_FAMILY=manifold NU=2 LENGTHSCALE=1.0 ./run_per_lipid.sh
#    KERNEL_FAMILY=eigenmap EMBED_DIM=10 EIGENVECTOR_DIR=... ./run_per_lipid.sh
#    KERNEL_FAMILY=spectral EIGENVECTOR_DIR=... ./run_per_lipid.sh
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
: "${KERNEL_FAMILY:=manifold}"          # euclidean | manifold | eigenmap | spectral
: "${KERNEL:=matern}"                   # only used for euclidean / eigenmap
# eigenmap: project coords into the leading EMBED_DIM Laplacian eigenfunctions,
# then a Euclidean ARD Matern GP over that embedding (needs EIGENVECTOR_DIR).
# spectral: weight-space SpectralLatentGP over the manifold spectrum, per lipid
# (needs EIGENVECTOR_DIR).
: "${EMBED_DIM:=300}"                     # eigenmap only

# ---- ARD (euclidean kernel only) ----
# NO_ARD=1 (default) → isotropic single shared lengthscale (passes --no-ard).
# NO_ARD=0           → per-axis ARD (ard_num_dims=3; no flag passed).
# Encoded in the euclidean run TAG (no-ard | ard). Ignored for manifold.
: "${NO_ARD:=1}"

# ---- GP hyperparameters ----
: "${NU:=2.5}"
: "${NUM_INDUCING:=1000}"
: "${INDUCING_SOURCE:=reference}"
: "${LIPID_BATCH_SIZE:=10}"
: "${EPOCHS:=2}"
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
: "${NUM_MODES:=1300}"
# Lanczos Krylov subspace floor. -1 = auto (max(1500, 3*num_modes+20)).
# At STRIDE=1 the 1500 floor blows up GPU memory; set e.g. NCV_MIN=100.
: "${NCV_MIN:=-1}"
: "${STRIDE:=4}"
: "${KNN_K:=15}"
: "${KNN_METHOD:=faiss_atlas_weighted}"
: "${CROSS_REGION_INFLATION:=50.0}"
# Template-clustering node labels (--knn-method faiss_cluster_weighted): data-driven,
# whole-brain, lipid-free labels clustered from the reference template (no ANNOTATION_FILE).
# CROSS_REGION_INFLATION is reused as the cross-cluster inflation.
: "${CLUSTER_K:=64}"
: "${CLUSTER_SPATIAL_WEIGHT:=1.0}"
: "${CLUSTER_FIT_SUBSAMPLE:=40000}"
: "${CLUSTER_SEED:=0}"
: "${LAPLACIAN_NORM:=randomwalk}"
# Cosine/correlation kernel (manifold only, 1=on): L2-normalize the Riemann
# feature rows so the prior variance is constant (diagonal=1) and the
# sqrt(degree) sampling-density artifact is quotiented out; magnitude then lives
# in the ScaleKernel outputscale. Encoded in the TAG as -cos. Off by default.
: "${NORMALIZE_FEATURES:=0}"
: "${BUMP_SCALE:=1.0}"
: "${BUMP_DECAY:=0.01}"
: "${GRAPHBANDWIDTH:=0.1}"
: "${THRESHOLD:=5}"
# Add the measured MALDI voxels to the graph node set (1=on). Works with
# KNN_METHOD=faiss and faiss_atlas_weighted (MALDI nodes inherit the region
# of their nearest atlas node); NOT anatomical_atlas (adjacency-built edges).
: "${AUGMENT_MALDI_NODES:=0}"
# Cap on MALDI voxels added (0 = all). The eigensolve workspace is ~ncv*N, so
# the full measured set (millions) can OOM the GPU — set e.g. 200000 to bound N.
: "${MAX_MALDI_NODES:=200000}"
: "${MALDI_SUBSAMPLE_METHOD:=random}"   # random | fps | kmeans_snap
# Blend inducing points (1=on): ~INDUCING_DENSITY_FRAC from the densest graph
# nodes + the rest from the measured MALDI voxels that snap onto the graph most
# cheaply. INDEPENDENT of AUGMENT_MALDI_NODES — the KNN graph is not modified,
# so you can keep the graph at stride (AUGMENT_MALDI_NODES=0) and still blend.
# Defaults to follow AUGMENT_MALDI_NODES only for backward compatibility; set it
# to 1 explicitly to blend on a strided graph.
: "${INDUCING_FROM_MALDI_NODES:=$AUGMENT_MALDI_NODES}"
# Fraction of inducing points from the densest graph nodes (rest from
# cheapest-to-snap MALDI nodes). Default 0.8 → 80/20 blend.
: "${INDUCING_DENSITY_FRAC:=0.8}"
# Learn the inducing-point LOCATIONS jointly with the rest (1=on). Default off
# (points stay anchored where initialized). For the manifold kernel this lets
# them drift off the graph nodes onto the Nyström path — enable deliberately.
: "${LEARN_INDUCING:=0}"
# Weights & Biases logging (1=on): loss / KL / hypers / noise / per-group
# gradient norms (incl. inducing points when LEARN_INDUCING=1). Off by default.
: "${WANDB:=0}"
: "${WANDB_PROJECT:=l3di_maldi_per_lipid}"

# ---- Lengthscale mode (manifold kernel) ----------------------------------
# Two modes, selected via FIXED_LENGTHSCALE:
#   1 (default) → "fixed" mode: pass --lengthscale-init $LENGTHSCALE_INIT and
#                 --lengthscale-no-decay so the kernel lengthscale is pinned.
#   0           → "learned" mode: pass neither flag, letting the GP train the
#                 lengthscale from its own default init.
# The chosen mode is encoded in the run TAG (fixls<val> | learnls) so the two
# modes never clobber each other's output dir.
: "${FIXED_LENGTHSCALE:=1}"
: "${LENGTHSCALE_INIT:=8.0}"
# Per-task lengthscale (manifold only, 1=on): each lipid in the batch gets its
# OWN learnable lengthscale (PerTaskRiemannWrapper) instead of one shared across
# the batch; the eigenpairs/graph are still shared. No effect on the euclidean
# kernel (its batched kernel is already per-task). Encoded in the TAG as -ptls.
: "${PER_TASK_LENGTHSCALE:=1}"

# ---- Diffusion scale (manifold kernel) -----------------------------------
# Multiplicative scale on the (frozen) Laplacian spectrum: lambda_k ->
# DIFFUSION_SCALE * lambda_k in the Matern spectral density. Needs NO eigenpair
# recompute (scaling an operator leaves its eigenvectors unchanged), so it is a
# cheap, learnable companion to the lengthscale (which only sets the additive
# 2*nu/l^2 floor). LEARN_DIFFUSION_SCALE=1 makes it trainable; otherwise it is
# pinned at DIFFUSION_SCALE_INIT (1.0 = identity, i.e. unchanged behaviour).
# When learned, encoded in the TAG as -learndiff.
: "${DIFFUSION_SCALE_INIT:=1.0}"
: "${LEARN_DIFFUSION_SCALE:=1}"

# ---- data paths (same as run_manifold.sh) ----
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output/per_lipid}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
# Atlas level convenience: ATLAS_LEVEL=5 or 15 selects level_${ATLAS_LEVEL}annot.npy
# under DATA_PATH. Override ANNOTATION_FILE directly to use any other volume.
: "${ATLAS_LEVEL:=15}"
: "${ANNOTATION_FILE:=${DATA_PATH}/level_${ATLAS_LEVEL}annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_2.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${TEMPLATE_NAME:=reference}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=FOLD-2}"

#cd $SRC_PATH

# ---- FAISS CPU-only flags (env -> CLI) ------------------------------------
# The submit script passes FAISS_CPU_* as env vars; here we translate them into
# the Python --faiss-cpu-* flags. Python reads ONLY the CLI, never the env.
: "${FAISS_CPU_GRAPH:=0}"
: "${FAISS_CPU_SEARCH:=0}"
: "${FAISS_CPU_RECON:=0}"
FAISS_CPU_ARGS=""
[ "$FAISS_CPU_GRAPH" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --faiss-cpu-graph"
[ "$FAISS_CPU_SEARCH" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --faiss-cpu-search"
[ "$FAISS_CPU_RECON" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --faiss-cpu-recon"

# Force a fresh KNN-graph build (bypass the cache) -- needed to actually time
# graph construction; otherwise the cached graph is just reloaded.
: "${FORCE_RECOMPUTE_GRAPH:=0}"
[ "$FORCE_RECOMPUTE_GRAPH" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --force-recompute-graph"

# FAISS IVF sizing. Pass an int or 'sqrt' (nlist=sqrt(N), nprobe=sqrt(nlist)) --
# 'sqrt' is what makes the CPU path fast at scale (see faiss_bench_report).
: "${N_LIST:=sqrt}"
: "${N_PROBE:=8}"
FAISS_CPU_ARGS="$FAISS_CPU_ARGS --n-list $N_LIST --n-probe $N_PROBE"

# ---- one-shot launcher (called by the sweep loops too) -----------------
run_one() {
    local FAMILY=$1; local NU_=$2; local LR_=$3; local EPS_=$4
    local INDU_=$5; local BS_=$6; local LBS_=$7
    # Manifold-only knobs:
    local NMODES_=$8; local BSCALE_=$9; local BDECAY_=${10}; local BW_=${11}
    local KNN_=${12}; local LN_=${13}; local KMETHOD_=${14}

    # Lengthscale-mode tag — fixed (pinned init, no decay) vs learned.
    local LS_TAG
    if [ "$FIXED_LENGTHSCALE" = "1" ]; then
        LS_TAG="fixls${LENGTHSCALE_INIT}"
    else
        LS_TAG="learnls"
    fi

    # ARD-mode tag (euclidean only) — isotropic vs per-axis.
    local ARD_TAG
    if [ "$NO_ARD" = "1" ]; then
        ARD_TAG="no-ard"
    else
        ARD_TAG="ard"
    fi

    local TAG
    if [ "$FAMILY" = "manifold" ]; then
        TAG="manifold-nu${NU_}-K${NMODES_}-stride${STRIDE}-${LS_TAG}-bs${BSCALE_}-bd${BDECAY_}-bw${BW_}-knn${KNN_}-${KMETHOD_}-${THRESHOLD}-${LN_}-ind${INDU_}-lr${LR_}-ep${EPS_}-lbs${LBS_}"
        [ "$AUGMENT_MALDI_NODES" = "1" ] && TAG="${TAG}-augmaldi${MAX_MALDI_NODES}"
        # Inducing-point blend (density + cheap-snap MALDI) vs plain k-means-snap.
        # Encode the density fraction so a sweep over it gets distinct dirs.
        [ "$INDUCING_FROM_MALDI_NODES" = "1" ] && TAG="${TAG}-blend${INDUCING_DENSITY_FRAC}"
        # Cross-region edge inflation only applies to faiss_atlas_weighted; encode
        # it there so a sweep over inflation values gets distinct output dirs.
        [ "$KMETHOD_" = "faiss_atlas_weighted" ] && TAG="${TAG}-infl${CROSS_REGION_INFLATION}"
        # Template-clustering labels: encode K / spatial-weight / seed / inflation so a
        # sweep over any of them gets distinct output dirs.
        [ "$KMETHOD_" = "faiss_cluster_weighted" ] && \
            TAG="${TAG}-clk${CLUSTER_K}-sw${CLUSTER_SPATIAL_WEIGHT}-cs${CLUSTER_SEED}-infl${CROSS_REGION_INFLATION}"
        # Per-task vs shared lengthscale → distinct output dirs.
        [ "$PER_TASK_LENGTHSCALE" = "1" ] && TAG="${TAG}-ptls"
        # Learned diffusion scale (multiplicative spectral scale) → distinct dirs.
        [ "$LEARN_DIFFUSION_SCALE" = "1" ] && TAG="${TAG}-learndiff"
        # Cosine/correlation kernel (unit-norm features) → distinct dirs.
        [ "$NORMALIZE_FEATURES" = "1" ] && TAG="${TAG}-cos"
    elif [ "$FAMILY" = "eigenmap" ]; then
        TAG="eigenmap-r${EMBED_DIM}-${ARD_TAG}-${KERNEL}-nu${NU_}-K${NMODES_}-stride${STRIDE}-knn${KNN_}-${KMETHOD_}-${THRESHOLD}-${LN_}-ind${INDU_}-lr${LR_}-ep${EPS_}-lbs${LBS_}"
    elif [ "$FAMILY" = "spectral" ]; then
        TAG="spectral-nu${NU_}-K${NMODES_}-stride${STRIDE}-bw${BW_}-knn${KNN_}-${KMETHOD_}-${THRESHOLD}-${LN_}-lr${LR_}-ep${EPS_}-lbs${LBS_}"
        [ "$LEARN_DIFFUSION_SCALE" = "1" ] && TAG="${TAG}-learndiff"
    else
        TAG="euclidean-${ARD_TAG}-${KERNEL}-nu${NU_}-ind${INDU_}-${THRESHOLD}-lr${LR_}-ep${EPS_}-lbs${LBS_}"
    fi
    # Atlas methods: encode WHICH annotation volume (level_5annot vs level_15annot) so
    # level5 vs level15 runs get distinct output dirs. Only for atlas methods.
    case "$KMETHOD_" in
        faiss_atlas_weighted|anatomical_atlas) TAG="${TAG}-$(basename "$ANNOTATION_FILE" .npy)" ;;
    esac
    # Learned vs anchored inducing points → distinct output dirs (both families).
    [ "$LEARN_INDUCING" = "1" ] && TAG="${TAG}-learnind"
    # VNNGP runs get their own tag suffix so they don't clobber the analytic ones.
    if [ "$VARIATIONAL" = "nngp" ]; then
        TAG="${TAG}-vnngp-${NN_METRIC}-nnk${NN_K}-ni${NNGP_NUM_INDUCING}"
        [ "$NN_METRIC" = "geodesic" ] && TAG="${TAG}-gk${GEODESIC_GRAPH_K}"
    fi
    local EXP_NAME="${EXP_PREFIX}-${TAG}"

    echo ""
    echo "================================================================"
    echo "  RUN: $EXP_NAME"
    echo "================================================================"

    # Graph + eigensolve args are needed by every family that consumes the
    # Laplacian spectrum: 'manifold' (Riemann kernel), 'eigenmap' (eigenfunction
    # embedding) and 'spectral' (weight-space basis). Only 'euclidean' skips them.
    local manifold_args=""
    if [ "$FAMILY" != "euclidean" ]; then
        local augment_args=""
        # Graph augmentation (adds MALDI voxels as graph nodes) — optional and
        # INDEPENDENT of the inducing blend below.
        if [ "$AUGMENT_MALDI_NODES" = "1" ]; then
            augment_args="--augment-maldi-nodes \
                --max-maldi-nodes $MAX_MALDI_NODES \
                --maldi-subsample-method $MALDI_SUBSAMPLE_METHOD"
        fi
        # Inducing-point blend over the (possibly strided, unaltered) graph.
        if [ "$INDUCING_FROM_MALDI_NODES" = "1" ]; then
            augment_args="$augment_args --inducing-from-maldi-nodes \
                --inducing-density-frac $INDUCING_DENSITY_FRAC"
        fi
        manifold_args="--eigenvector-dir $EIGENVECTOR_DIR \
            $augment_args \
            --knn-method $KMETHOD_ \
            --cross-region-inflation $CROSS_REGION_INFLATION \
            --cluster-k $CLUSTER_K \
            --cluster-spatial-weight $CLUSTER_SPATIAL_WEIGHT \
            --cluster-fit-subsample $CLUSTER_FIT_SUBSAMPLE \
            --cluster-seed $CLUSTER_SEED \
            --laplacian-norm $LN_ \
            --stride $STRIDE \
            --knn-k $KNN_ \
            --bump-scale $BSCALE_ \
            --bump-decay $BDECAY_ \
            --graphbandwidth-init $BW_ \
            --num-modes $NMODES_ \
            --ncv-min $NCV_MIN \
            --threshold "$THRESHOLD""

        # Diffusion scale is consumed at kernel-build time (Riemann spectral
        # density), so it applies to 'manifold' and 'spectral'.
        if [ "$FAMILY" = "manifold" ] || [ "$FAMILY" = "spectral" ]; then
            manifold_args="$manifold_args --diffusion-scale-init $DIFFUSION_SCALE_INIT"
            [ "$LEARN_DIFFUSION_SCALE" = "1" ] && \
                manifold_args="$manifold_args --learn-diffusion-scale"
        fi
        # Lengthscale init / per-task lengthscale are applied in the manifold
        # training branch only.
        if [ "$FAMILY" = "manifold" ]; then
            [ "$FIXED_LENGTHSCALE" = "1" ] && \
                manifold_args="$manifold_args --lengthscale-init $LENGTHSCALE_INIT --lengthscale-no-decay"
            [ "$PER_TASK_LENGTHSCALE" = "1" ] && \
                manifold_args="$manifold_args --per-task-lengthscale"
            [ "$NORMALIZE_FEATURES" = "1" ] && \
                manifold_args="$manifold_args --normalize-features"
        fi
        # Eigenmap embedding dimension.
        if [ "$FAMILY" = "eigenmap" ]; then
            manifold_args="$manifold_args --embed-dim $EMBED_DIM"
        fi
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

    # ARD args (euclidean only). --no-ard requests an isotropic lengthscale;
    # omitting it lets the kernel learn per-axis ARD. The flag is ignored by
    # the manifold path, so only pass it for the euclidean family.
    local euc_args=""
    if [ "$FAMILY" != "manifold" ] && [ "$NO_ARD" = "1" ]; then
        euc_args="--no-ard"
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

    # Learned inducing points + W&B logging (both opt-in via env vars).
    local extra_args=""
    if [ "$LEARN_INDUCING" = "1" ]; then
        extra_args="$extra_args --learn-inducing"
    fi
    if [ "$WANDB" = "1" ]; then
        extra_args="$extra_args --wandb --wandb-project $WANDB_PROJECT"
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
        $euc_args \
        $subset_args \
        $extra_args \
        $FAISS_CPU_ARGS \
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