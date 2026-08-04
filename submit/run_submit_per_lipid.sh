#!/usr/bin/env bash
# =============================================================================
# Per-lipid GP runai batch submitter.
#
# Mirrors run_manifold_batch.sh but for the per-lipid pipeline
# (./maldi/run_lgp_per_lipid.sh → lgp_experiment_per_lipid.py).
#
# What it does:
#   - Loops over a small cross-product of hyperparameters
#     (KNN_METHOD × KNN_K × INFLATION × PRUNE × …)
#   - Submits one runai job per config, each running
#     ./maldi/run_lgp_per_lipid.sh inside the standard container
#   - Each job trains only the lipids listed in $LIPIDS_FILE
#     (default: 10 lipids — fast enough for parallel sweeps)
#
# Usage:
#     DRY_RUN=1 ./run_lgp_per_lipid_batch.sh   # print commands only
#     ./run_lgp_per_lipid_batch.sh             # actually submit
# =============================================================================
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() {
    if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi
}

# ---- load secrets ---------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

# ---- resource limits ------------------------------------------------------
MEM=24G
CPU=2
GPU=0.2

# ---- FAISS CPU-only toggle ------------------------------------------------
# FAISS_CPU=1 forces all FAISS KNN work (graph build, searches, reconstruction)
# onto CPU inside the job -- for benchmarking a CPU-only run before dropping
# the GPU FAISS path. Override the three phases independently if you only want
# to pin part of the pipeline. These env vars are forwarded by run_lgp_per_lipid.sh
# into the Python --faiss-cpu-* CLI flags (Python reads only the CLI).
#   FAISS_CPU=1 ./submit/run_submit_per_lipid.sh
FAISS_CPU=${FAISS_CPU:-0}
FAISS_CPU_GRAPH=${FAISS_CPU_GRAPH:-$FAISS_CPU}
FAISS_CPU_SEARCH=${FAISS_CPU_SEARCH:-$FAISS_CPU}
FAISS_CPU_RECON=${FAISS_CPU_RECON:-$FAISS_CPU}

# FORCE_RECOMPUTE_GRAPH=1 rebuilds the KNN graph from scratch instead of loading
# the cached one -- required to actually time graph construction on CPU.
FORCE_RECOMPUTE_GRAPH=${FORCE_RECOMPUTE_GRAPH:-0}

# FAISS IVF sizing. Default = the fast, recall~1.0 CPU path: nlist=sqrt(N) with a
# small FIXED nprobe (~8). In 3D recall saturates at nprobe~8 regardless of N, so
# 'sqrt' nprobe would just over-scan. Use N_LIST=1 to force the exact flat index.
#   N_LIST=1 ./submit/run_submit_per_lipid.sh   # exact flat (old behaviour)
N_LIST=${N_LIST:-sqrt}
N_PROBE=${N_PROBE:-8}

# ---- training hyperparameters (constant across sweep) ---------------------
# Per-lipid run knobs. EPOCHS=20 is the published default; drop to 5 for
# quick smoke-tests in the early-sweep phase. LIPID_BATCH_SIZE=10 fits
# our 10-lipid subset in a single GP fit.
EPOCHS=20
LIPID_BATCH_SIZE=10
NUM_INDUCING=1000
LEARNING_RATE=0.05
BATCH_SIZE=2048
STRIDE_BUMP=20.0   # bump_scale — default; overridden per-job by MAN_BUMP_PAIRS
STRIDE_DECAY=0.01  # bump_decay — default; overridden per-job by MAN_BUMP_PAIRS

# ---- diffusion scale (manifold kernel) ------------------------------------
# Learnable multiplicative scale on the (frozen) Laplacian spectrum
# (lambda_k -> DIFFUSION_SCALE_INIT * lambda_k in the Matern spectral density).
# Needs NO eigenpair recompute, so it is a cheap, learnable companion to the
# lengthscale. LEARN_DIFFUSION_SCALE=1 trains it; otherwise it is pinned at the
# init (1.0 = identity, unchanged behaviour). Encoded in the run dir as
# -learndiff when learned.
#   LEARN_DIFFUSION_SCALE=1 ./submit/run_submit_per_lipid.sh
DIFFUSION_SCALE_INIT=${DIFFUSION_SCALE_INIT:-1.0}
LEARN_DIFFUSION_SCALE=${LEARN_DIFFUSION_SCALE:-0}

# Cosine/correlation kernel (manifold only). NORMALIZE_FEATURES=1 L2-normalizes
# the Riemann feature rows so the prior variance is constant (diagonal=1) and the
# sqrt(degree) sampling-density artifact is quotiented out; magnitude moves to the
# ScaleKernel outputscale. Encoded in the run dir as -cos. Off by default.
#   NORMALIZE_FEATURES=1 ./submit/run_submit_per_lipid.sh
NORMALIZE_FEATURES=${NORMALIZE_FEATURES:-0}

# ---- subset of lipids to train -------------------------------------------
# A small subset is what makes a 30-job sweep practical. The file lives
# inside the container at the same path the run script expects.
LIPIDS_FILE="/myhome/mlibra/maldi/data/lipid_subset.txt"

# ---- S3-mounted paths inside the container --------------------------------
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/artiom/eigenvectors"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/per_lipid_cv"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
# Atlas level: ATLAS_LEVEL=5|15 (default 15) selects the annotation volume on S3.
# Cache keys + output TAG are tagged by the file stem, so levels never collide.
# Re-invoke with a different ATLAS_LEVEL for the other level.
ATLAS_LEVEL=${ATLAS_LEVEL:-15}
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_${ATLAS_LEVEL}annot.npy"
S3_SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/fold_3.json"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
# Parcel fields live on S3 so a sweep builds each one ONCE and every later job
# reuses it. parcelgp.build verifies the cached field's recorded parameters
# before reuse and writes atomically (temp + rename), so concurrent jobs racing
# to build the same field cannot corrupt it -- the losers just rewrite identical
# bytes. At stride 4 a build is ~15s; at stride 1 it is far longer, so prebuild
# those (see PARCEL_PREBUILD_CMD printed at the end).
S3_PARCEL_DIR="${S3_PARCEL_DIR:-/s3/mlibra/mlibra-data/artiom/parcels}"
SRC_PATH="/myhome/mlibra"

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

# ---- one submission -------------------------------------------------------
# All hyperparameters get passed as -e env vars; run_lgp_per_lipid.sh
# inside the container reads them via ``: "${X:=default}"`` and forwards
# to lgp_experiment_per_lipid.py.
submit() {
    local job_name=$1
    local family=$2        # euclidean | manifold | eigenmap | spectral
    local nu=$3
    local knn_method=$4
    local knn_k=$5
    local laplacian_norm=$6
    local graphbandwidth=$7
    local inflation=$8
    local threshold=$9
    local prefix=${10}
    local slice=${11}
    local stride=${12}
    local modes=${13}
    local fixed_ls=${14}
    local no_ard=${15}
    shift 15
    local extra_args=("$@")

    echo ">>> Submitting $job_name"
    run_or_echo runai training submit "$job_name" \
        -i artiomartiom/sdsc:withfaiss \
        --image-pull-policy Always \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e SRC_PATH="$SRC_PATH" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slice" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="$S3_TEMPLATE_NAME" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e LIPIDS_FILE="$LIPIDS_FILE" \
        -e KERNEL_FAMILY="$family" \
        -e NU="$nu" \
        -e KNN_METHOD="$knn_method" \
        -e KNN_K="$knn_k" \
        -e LAPLACIAN_NORM="$laplacian_norm" \
        -e GRAPHBANDWIDTH="$graphbandwidth" \
        -e CROSS_REGION_INFLATION="$inflation" \
        -e ROOT_HANDLING="${ROOT_HANDLING:-dissolve}" \
        -e DENOISE_LABELS="${DENOISE_LABELS:-0}" \
        -e PRUNE_CROSS_REGION="${PRUNE_CROSS_REGION:-0.0}" \
        -e BUMP_SCALE="$STRIDE_BUMP" \
        -e BUMP_DECAY="$STRIDE_DECAY" \
        -e NUM_MODES="$modes" \
        -e NCV_MIN="${NCV_MIN:--1}" \
        -e STRIDE="$stride" \
        -e FIXED_LENGTHSCALE="$fixed_ls" \
        -e NO_ARD="$no_ard" \
        -e EMBED_DIM="${EMBED_DIM:-10}" \
        -e THRESHOLD="$threshold" \
        -e NUM_INDUCING="$NUM_INDUCING" \
        -e LIPID_BATCH_SIZE="$LIPID_BATCH_SIZE" \
        -e EPOCHS="$EPOCHS" \
        -e LEARNING_RATE="$LEARNING_RATE" \
        -e BATCH_SIZE="$BATCH_SIZE" \
        -e AUGMENT_MALDI_NODES="${AUGMENT_MALDI_NODES:-0}" \
        -e MAX_MALDI_NODES="${MAX_MALDI_NODES:-200000}" \
        -e MALDI_SUBSAMPLE_METHOD="${MALDI_SUBSAMPLE_METHOD:-random}" \
        -e INDUCING_FROM_MALDI_NODES="${INDUCING_FROM_MALDI_NODES:-$AUGMENT_MALDI_NODES}" \
        -e INDUCING_DENSITY_FRAC="${INDUCING_DENSITY_FRAC:-0.8}" \
        -e LEARN_INDUCING="${LEARN_INDUCING:-0}" \
        -e PER_TASK_LENGTHSCALE="${PER_TASK_LENGTHSCALE:-0}" \
        -e NORMALIZE_FEATURES="${NORMALIZE_FEATURES:-0}" \
        -e DIFFUSION_SCALE_INIT="${DIFFUSION_SCALE_INIT:-1.0}" \
        -e LEARN_DIFFUSION_SCALE="${LEARN_DIFFUSION_SCALE:-0}" \
        -e LEARN_SPECTRAL_WEIGHTS="${LEARN_SPECTRAL_WEIGHTS:-0}" \
        -e WANDB="${WANDB:-0}" \
        -e WANDB_PROJECT="${WANDB_PROJECT:-l3di_maldi_per_lipid}" \
        -e FAISS_CPU_GRAPH="$FAISS_CPU_GRAPH" \
        -e FAISS_CPU_SEARCH="$FAISS_CPU_SEARCH" \
        -e FAISS_CPU_RECON="$FAISS_CPU_RECON" \
        -e FORCE_RECOMPUTE_GRAPH="$FORCE_RECOMPUTE_GRAPH" \
        -e N_LIST="$N_LIST" \
        -e N_PROBE="$N_PROBE" \
        -e OMP_NUM_THREADS="$CPU" \
        -e OPENBLAS_NUM_THREADS="$CPU" \
        -e MKL_NUM_THREADS="$CPU" \
        -e OMP_WAIT_POLICY=passive \
        -- ./maldi/run_lgp_per_lipid.sh "${extra_args[@]}"
}

# ---- one PARCEL submission ------------------------------------------------
# Runs ./parcelgp/run_parcel.sh (per-lipid GP + reference-only parcel factor)
# rather than ./maldi/run_lgp_per_lipid.sh. The parcel factor multiplies the
# spatial kernel by exp(-||z(x)-z(x')||^2/2) with z(x)=m(x)^T B; the partition
# comes from the reference image alone and only B is learned.
#
# MODE=parcel is hardcoded: run_parcel.sh defaults to MODE=both, which would run
# an identical baseline inside EVERY job of the sweep -- N times the compute, all
# writing the same baseline directory. The baseline is an ordinary no-parcel run;
# take it from the euclidean sweep above or from an existing per_lipid_cv run with
# matching hyperparameters (parcelgp.compare pairs them per lipid).
submit_parcel() {
    local job_name=$1 prefix=$2 slice=$3
    local features=$4 n_parcels=$5 spatial_weight=$6 stride=$7
    local rank=$8 normalize_blocks=$9 init_scale=${10} shared_b=${11}
    shift 11
    local extra_args=("$@")

    echo ">>> Submitting $job_name"
    run_or_echo runai training submit "$job_name" \
        -i artiomartiom/sdsc:withfaiss \
        --image-pull-policy Always \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e SRC_PATH="$SRC_PATH" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_PARCEL_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slice" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="$S3_TEMPLATE_NAME" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e LIPIDS_FILE="$LIPIDS_FILE" \
        -e MODE="parcel" \
        -e PARCEL_DIR="$S3_PARCEL_DIR" \
        -e PARCEL_FEATURES="$features" \
        -e N_PARCELS="$n_parcels" \
        -e SPATIAL_WEIGHT="$spatial_weight" \
        -e STRIDE="$stride" \
        -e THRESHOLD="${PARCEL_THRESHOLD:-5}" \
        -e NORMALIZE_BLOCKS="$normalize_blocks" \
        -e PARCEL_RANK="$rank" \
        -e PARCEL_INIT_SCALE="$init_scale" \
        -e PARCEL_SHARED_B="$shared_b" \
        -e KERNEL_FAMILY="${PARCEL_KERNEL_FAMILY:-euclidean}" \
        -e KERNEL="matern" \
        -e NU="${PARCEL_NU:-2.5}" \
        -e NO_ARD="${PARCEL_NO_ARD:-0}" \
        -e NUM_INDUCING="$NUM_INDUCING" \
        -e INDUCING_SOURCE="${PARCEL_INDUCING_SOURCE:-data}" \
        -e LIPID_BATCH_SIZE="$LIPID_BATCH_SIZE" \
        -e EPOCHS="$EPOCHS" \
        -e LEARNING_RATE="$LEARNING_RATE" \
        -e BATCH_SIZE="$BATCH_SIZE" \
        -e SEED="${PARCEL_SEED:-42}" \
        -e DEVICE="cuda" \
        -e WANDB="${WANDB:-0}" \
        -e WANDB_PROJECT="${WANDB_PROJECT:-l3di_maldi_per_lipid}" \
        -e OMP_NUM_THREADS="$CPU" \
        -e OPENBLAS_NUM_THREADS="$CPU" \
        -e MKL_NUM_THREADS="$CPU" \
        -e OMP_WAIT_POLICY=passive \
        -- ./parcelgp/run_parcel.sh "${extra_args[@]}"
}

# =============================================================================
# Sweep definitions
#
# Two separate loops below: one for the Euclidean baseline (only the
# kernel + nu vary; no kNN / Laplacian / inflation knobs apply), and
# one for the Manifold family (all the graph-spectral knobs). This
# keeps the per-family job count honest — submitting `nu=2` for
# euclidean only submits one job, not one per (knn_k, knn_method, ...)
# permutation that the value would otherwise duplicate over.
# =============================================================================

FOLDS=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8")           # lowercase, dashed

# ---- Euclidean sweep ------------------------------------------------------
# Euclidean Matern only has nu as a meaningful hyperparameter — the GP
# learns its own lengthscale per dimension. Other knobs (kernel="matern"
# etc.) are fixed in run_lgp_per_lipid.sh.
EUC_NU=(1.5)
# ARD modes (euclidean only): 1 → isotropic (--no-ard), 0 → per-axis ARD.
# Both are submitted so isotropic vs per-axis can be compared head-to-head.
EUC_NO_ARD=(1 0)
RUN_EUCLIDEAN=0   # 0 = skip the euclidean loop entirely

# ---- Manifold sweep -------------------------------------------------------
MAN_NU=(2)
MAN_THRESHOLDS=(5)
MAN_LAPLACIAN_NORMS=("randomwalk")
MAN_GRAPH_BANDWIDTHS=(0.1)
MAN_KNN_K=(15)
MAN_KNN_METHODS=("faiss_atlas_weighted")
MAN_INFLATIONS=(50)   # only used when knn_method=faiss_atlas_weighted

# ---- hard prune of cross-region edges (weighted knn methods only) ----------
# PRUNE_CROSS_REGION = fraction of cross-region (inter-atlas-region) edges hard-
# REMOVED from the graph, on top of the soft CROSS_REGION_INFLATION down-weighting.
# 0.0 = off (inflation only); 0.95 = drop 95% of them. Each value gets its own job;
# run_lgp_per_lipid.sh appends "-prune<val>" to the TAG so dirs never collide.
# Edit the list in place to sweep, e.g. MAN_PRUNE=(0.0 0.95 0.97) — it is a bash
# array, so unlike the scalar knobs above it can NOT be set from the environment.
#
# Two gotchas this sweep is built around:
#   * The Python side gates prune to the WEIGHTED methods (faiss_atlas_weighted /
#     faiss_cluster_weighted) and so does the TAG. Under a plain faiss/anatomical
#     method every prune value would collapse to the SAME output dir, so the loop
#     below sweeps a single placeholder there instead of submitting duplicates.
#   * Prune runs AFTER the label denoise, so a large DENOISE_LABELS erases much of
#     the true region boundary before prune ever sees it — sweep prune with
#     DENOISE_LABELS=0 if you want to read the prune effect on its own.
MAN_PRUNE=(0.0 0.9 0.95 0.97)
DENOISE_LABELS=${DENOISE_LABELS:-3}
# ---- (stride : num_modes) pairs to sweep (manifold only) ------------------
# Coarser strides need fewer eigenmodes to span the graph; finer strides need
# more. Each entry is "STRIDE:NUM_MODES" and gets its own job.
MAN_STRIDE_MODES=("4:100" "4:300")

# ---- (bump_scale : bump_decay) pairs to sweep (manifold only) -------------
# The bump function shapes the kernel's local support on the graph. Each entry
# is "BUMP_SCALE:BUMP_DECAY" and gets its own job; the run TAG already encodes
# -bs<scale>-bd<decay> so output dirs never collide. Add more pairs to sweep.
MAN_BUMP_PAIRS=("1.0:0.01")

# ---- lengthscale modes (manifold only)
# 1 → fixed lengthscale (--lengthscale-init 8.0 --lengthscale-no-decay)
# 0 → learned lengthscale (no flags; GP trains it).
# Both modes are submitted so they can be compared head-to-head.
MAN_FIXED_LS=(0)

# ---- per-task lengthscale (manifold only) ---------------------------------
# 1 → each lipid gets its OWN learnable lengthscale (PerTaskRiemannWrapper);
# 0 → one lengthscale shared across the batch. List both, e.g. (0 1), to
# compare head-to-head (output dirs differ via the "-ptls" tag). Default (0).
MAN_PER_TASK_LS=(0 1)

# ---- MALDI-node augmentation (manifold only) ------------------------------
# MALDI graph augmentation: add the measured MALDI voxels to the graph NODE SET
# (changes the KNN graph / eigensolve). 0 = off (atlas-only, strided graph),
# 1 = on. Listing both submits each config twice for head-to-head comparison.
# augment=1 requires knn_method faiss / faiss_atlas_weighted (not anatomical).
MAN_AUGMALDI=(0)
MAX_MALDI_NODES=200000          # cap on MALDI voxels added (bounds eigensolve N)
MALDI_SUBSAMPLE_METHOD=random   # random | fps ------------------------------------| kmeans_snap
# Inducing-point blend — INDEPENDENT of MAN_AUGMALDI (KNN graph is NOT modified):
# ~INDUCING_DENSITY_FRAC of inducing points from the densest graph nodes, the
# rest from the measured MALDI voxels that snap onto the (strided) graph most
# cheaply. 1 = blend on, 0 = plain k-means-snap inducing. Listed so each config
# is submitted both ways for head-to-head comparison (output dirs differ via the
# "-blend" tag). Manifold only — euclidean ignores it.
MAN_INDUCING_BLEND=(1)
INDUCING_DENSITY_FRAC=0.8       # frac from densest graph nodes (rest = cheapest-to-snap MALDI)
# Learned vs anchored inducing points — listed so each config is submitted both
# ways for head-to-head comparison (output dirs differ via the "-learnind" tag).
# 0 = anchored (stay on graph nodes), 1 = optimize inducing-point locations.
MAN_LEARN_INDUCING=(0)
WANDB=1                        # log loss/KL/hypers/noise/grad-norms (incl. inducing) to W&B
WANDB_PROJECT=l3di_maldi_per_lipid
RUN_MANIFOLD=0

# ---- Eigenmap sweep -------------------------------------------------------
# 'eigenmap' projects coordinates into the leading EIGENMAP_EMBED_DIMS Laplacian
# eigenfunctions and fits a Euclidean ARD Matern GP over that embedding. It
# consumes the SAME graph/eigensolve stack as the manifold family, so it reuses
# the manifold graph knobs (knn_method / knn_k / laplacian_norm / graphbandwidth
# / threshold) and the first MAN_STRIDE_MODES entry. The euclidean-side knobs
# (nu / ard) are what actually vary here.
EIGENMAP_NU=(1.5)
EIGENMAP_EMBED_DIMS=(10)
# ARD modes over the eigenfunction embedding: 1 → isotropic (--no-ard),
# 0 → per-axis ARD (one lengthscale per eigenfunction dim).
EIGENMAP_NO_ARD=(0)
RUN_EIGENMAP=0

# ---- Spectral sweep -------------------------------------------------------
# 'spectral' fits a weight-space SpectralLatentGP over the manifold spectrum,
# one per lipid. Like eigenmap it reuses the manifold graph/eigensolve knobs;
# nu and the diffusion scale (DIFFUSION_SCALE_INIT / LEARN_DIFFUSION_SCALE,
# shared with the manifold config above) are the meaningful axes.
SPECTRAL_NU=(2)
RUN_SPECTRAL=0


# ---- PARCEL sweep (reference-only parcellation, per-lipid) ----------------
# The cross-product below is the ablation grid. Every axis is a bash array, so
# edit in place to sweep; a single-element array pins that axis.
#
# What each axis buys, and what it costs in jobs:
#   PARCEL_SW_LIST   the highest-leverage build knob. 3 features vs 16 means
#                    sw=1 gives geometry only 16% of the k-means distance and
#                    sw=3 gives it 63%; sw~2.3 is the 50/50 point.
#   PARCEL_FEAT_LIST 'full' vs 'spatial' is THE control arm — if 'spatial'
#                    (no appearance features at all) wins equally, the gain is
#                    soft spatial partitioning and not the reference image.
#   PARCEL_NB_LIST   0/1 = weight each feature vs each descriptor block equally.
#   PARCEL_RANK_LIST width of the learned parcel embedding. r=2 is the
#                    interpretable one (the 128 parcels plot in 2-D).
#
# Job count = folds x features x K x sw x stride x rank x nb, PLUS one baseline
# per fold. Keep it honest: the default grid below is already 2x2x2 = 8 per fold.
PARCEL_FEAT_LIST=("full" "spatial")
PARCEL_K_LIST=(128 192)
PARCEL_SW_LIST=(3.0)
PARCEL_STRIDE_LIST=(2 4)
PARCEL_RANK_LIST=(8)
PARCEL_NB_LIST=(0 1)
PARCEL_INIT_SCALE=${PARCEL_INIT_SCALE:-0.05}
PARCEL_SHARED_B=${PARCEL_SHARED_B:-0}
RUN_PARCEL=${RUN_PARCEL:-1}
S3_PARCEL_OUTPUT_DIR="${S3_PARCEL_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/parcel_per_lipid_cv}"

exp_num=1
for fold in "${FOLDS[@]}"; do
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"

    # ---- EUCLIDEAN loop ---------------------------------------------------
    if [ "$RUN_EUCLIDEAN" = "1" ]; then
        for nu in "${EUC_NU[@]}"; do
          for no_ard in "${EUC_NO_ARD[@]}"; do
            for learn_ind in "${MAN_LEARN_INDUCING[@]}"; do
            AUGMENT_MALDI_NODES=0   # euclidean ignores it; reset for clean logs
            INDUCING_FROM_MALDI_NODES=0   # manifold-only; reset for clean logs
            PER_TASK_LENGTHSCALE=0  # manifold-only; reset for clean logs
            PRUNE_CROSS_REGION=0.0  # graph-only; clear whatever the manifold loop left
            LEARN_INDUCING=$learn_ind   # sweep anchored vs learned inducing
            job_name="gp-perlipid-${EXP_SUFFIX}-${exp_num}"
            if [ "$no_ard" = "1" ]; then ard_tag="no-ard"; else ard_tag="ard"; fi
            if [ "$learn_ind" = "1" ]; then li_tag="learn"; else li_tag="anchor"; fi
            printf "  exp %2d: %-22s nu=%-4s ard=%s ind=%s fold=%s\n" \
                "$exp_num" "euclidean" "$nu" "$ard_tag" "$li_tag" "$fold"
            # Euclidean ignores knn_method / laplacian / inflation /
            # threshold / graphbandwidth, but the submit() function takes
            # them positionally — pass dummy values that the python side
            # will silently ignore for kernel_family=euclidean.
            # stride/modes/fixed_ls are manifold-only; pass placeholders.
            submit "$job_name" "euclidean" "$nu" \
                "faiss" "15" "randomwalk" "0.1" "1" "5" \
                "$fold_upper" "$SLICES_DATASET_FILE" "4" "1300" "1" "$no_ard"
            exp_num=$((exp_num + 1))
            done
          done
        done
    fi

    # ---- MANIFOLD loop ----------------------------------------------------
    if [ "$RUN_MANIFOLD" = "1" ]; then
        for nu in "${MAN_NU[@]}"; do
            for threshold in "${MAN_THRESHOLDS[@]}"; do
                for ln in "${MAN_LAPLACIAN_NORMS[@]}"; do
                    for gb in "${MAN_GRAPH_BANDWIDTHS[@]}"; do
                        for knn in "${MAN_KNN_K[@]}"; do
                            for km in "${MAN_KNN_METHODS[@]}"; do
                                # Inflation is only meaningful for
                                # faiss_atlas_weighted; for the other
                                # methods, sweep just one placeholder so
                                # we don't submit redundant jobs.
                                if [ "$km" = "faiss_atlas_weighted" ]; then
                                    infl_list=("${MAN_INFLATIONS[@]}")
                                else
                                    infl_list=(1)
                                fi
                                # Prune is gated to the weighted methods on the
                                # Python side AND in the TAG, so under any other
                                # method every prune value would land in the SAME
                                # output dir. Sweep one placeholder there.
                                case "$km" in
                                    faiss_atlas_weighted|faiss_cluster_weighted)
                                        prune_list=("${MAN_PRUNE[@]}") ;;
                                    *)
                                        prune_list=(0.0) ;;
                                esac
                                for infl in "${infl_list[@]}"; do
                                 for prune in "${prune_list[@]}"; do
                                  # Read by submit() as -e PRUNE_CROSS_REGION.
                                  PRUNE_CROSS_REGION="$prune"
                                  for sm in "${MAN_STRIDE_MODES[@]}"; do
                                    stride="${sm%%:*}"
                                    modes="${sm##*:}"
                                   for bp in "${MAN_BUMP_PAIRS[@]}"; do
                                    bump_scale="${bp%%:*}"
                                    bump_decay="${bp##*:}"
                                    # submit() reads these globals for the
                                    # -e BUMP_SCALE / -e BUMP_DECAY env vars.
                                    STRIDE_BUMP="$bump_scale"
                                    STRIDE_DECAY="$bump_decay"
                                    for fixed_ls in "${MAN_FIXED_LS[@]}"; do
                                      for augment in "${MAN_AUGMALDI[@]}"; do
                                        # augment=1 needs a coords-based graph;
                                        # anatomical_atlas builds edges from
                                        # strided-voxel adjacency, so skip it.
                                        if [ "$augment" = "1" ] && [ "$km" = "anatomical_atlas" ]; then
                                            continue
                                        fi
                                        # Read by submit() and forwarded as
                                        # -e AUGMENT_MALDI_NODES / -e INDUCING_
                                        # FROM_MALDI_NODES. The inducing blend is
                                        # independent of graph augmentation.
                                        AUGMENT_MALDI_NODES=$augment
                                        # Sweep per-task lengthscale, inducing-point
                                        # blend (on/off) and anchored vs learned.
                                        # PER_TASK_LENGTHSCALE / INDUCING_FROM_MALDI_
                                        # NODES / LEARN_INDUCING are forwarded as -e
                                        # env vars; the "-ptls" / "-blend" /
                                        # "-learnind" TAG suffixes keep dirs apart.
                                        for ptls in "${MAN_PER_TASK_LS[@]}"; do
                                        PER_TASK_LENGTHSCALE=$ptls
                                        for blend in "${MAN_INDUCING_BLEND[@]}"; do
                                        INDUCING_FROM_MALDI_NODES=$blend
                                        for learn_ind in "${MAN_LEARN_INDUCING[@]}"; do
                                        LEARN_INDUCING=$learn_ind
                                        job_name="gp-perlipid-${EXP_SUFFIX}-${exp_num}"
                                        # Fold the lengthscale mode into the
                                        # prefix so the two modes (and their
                                        # output dirs) never collide.
                                        if [ "$fixed_ls" = "1" ]; then
                                            ls_tag="fixls"
                                        else
                                            ls_tag="learnls"
                                        fi
                                        # augment=1 already gets a "-augmaldi<N>"
                                        # suffix in the run TAG, so on/off output
                                        # dirs never collide; tag is just for logs.
                                        if [ "$augment" = "1" ]; then aug_tag="augmaldi"; else aug_tag="atlas"; fi
                                        if [ "$learn_ind" = "1" ]; then li_tag="learn"; else li_tag="anchor"; fi
                                        if [ "$blend" = "1" ]; then blend_tag="blend"; else blend_tag="kmeans"; fi
                                        if [ "$ptls" = "1" ]; then ptls_tag="ptls"; else ptls_tag="shared"; fi
                                        prefix="${fold_upper}-${ls_tag}"
                                        printf "  exp %2d: %-22s nu=%s knn_k=%-3s ln=%s gb=%s infl=%s prune=%-4s stride=%s modes=%s bs=%s bd=%s ls=%s aug=%s ind=%s blend=%s ptls=%s\n" \
                                            "$exp_num" "$km" "$nu" "$knn" "$ln" "$gb" "$infl" "$prune" "$stride" "$modes" "$bump_scale" "$bump_decay" "$ls_tag" "$aug_tag" "$li_tag" "$blend_tag" "$ptls_tag"
                                        # no_ard is euclidean-only; pass a
                                        # placeholder the manifold path ignores.
                                        submit "$job_name" "manifold" "$nu" \
                                            "$km" "$knn" "$ln" "$gb" "$infl" "$threshold" \
                                            "$prefix" "$SLICES_DATASET_FILE" \
                                            "$stride" "$modes" "$fixed_ls" "1"
                                        exp_num=$((exp_num + 1))
                                        done
                                        done
                                        done
                                      done
                                    done
                                   done
                                  done
                                 done
                                done
                            done
                        done
                    done
                done
            done
        done
    fi

    # ---- EIGENMAP loop ----------------------------------------------------
    # Reuses the manifold graph/eigensolve knobs (first entry of each MAN_*
    # array); only nu / embed_dim / ard vary. fixed_ls is manifold-only, so a
    # placeholder is passed.
    if [ "$RUN_EIGENMAP" = "1" ]; then
        em_sm="${MAN_STRIDE_MODES[0]}"
        em_stride="${em_sm%%:*}"; em_modes="${em_sm##*:}"
        em_km="${MAN_KNN_METHODS[0]}"
        if [ "$em_km" = "faiss_atlas_weighted" ]; then em_infl="${MAN_INFLATIONS[0]}"; else em_infl=1; fi
        AUGMENT_MALDI_NODES=0          # manifold-only; reset for clean logs
        INDUCING_FROM_MALDI_NODES=0    # manifold-only; reset for clean logs
        PER_TASK_LENGTHSCALE=0         # manifold-only; reset for clean logs
        LEARN_INDUCING=0
        # Eigenmap consumes the same graph, so it takes the first prune value
        # (like the other MAN_* knobs) rather than sweeping it.
        case "$em_km" in
            faiss_atlas_weighted|faiss_cluster_weighted)
                PRUNE_CROSS_REGION="${MAN_PRUNE[0]}" ;;
            *)
                PRUNE_CROSS_REGION=0.0 ;;
        esac
        for nu in "${EIGENMAP_NU[@]}"; do
          for ed in "${EIGENMAP_EMBED_DIMS[@]}"; do
            for no_ard in "${EIGENMAP_NO_ARD[@]}"; do
              EMBED_DIM=$ed   # read by submit() and forwarded as -e EMBED_DIM
              job_name="gp-perlipid-${EXP_SUFFIX}-${exp_num}"
              if [ "$no_ard" = "1" ]; then ard_tag="no-ard"; else ard_tag="ard"; fi
              printf "  exp %2d: %-22s nu=%-4s r=%s ard=%s stride=%s modes=%s fold=%s\n" \
                  "$exp_num" "eigenmap" "$nu" "$ed" "$ard_tag" "$em_stride" "$em_modes" "$fold"
              submit "$job_name" "eigenmap" "$nu" \
                  "$em_km" "${MAN_KNN_K[0]}" "${MAN_LAPLACIAN_NORMS[0]}" "${MAN_GRAPH_BANDWIDTHS[0]}" \
                  "$em_infl" "${MAN_THRESHOLDS[0]}" \
                  "$fold_upper" "$SLICES_DATASET_FILE" "$em_stride" "$em_modes" "0" "$no_ard"
              exp_num=$((exp_num + 1))
            done
          done
        done
    fi

    # ---- SPECTRAL loop ----------------------------------------------------
    # Weight-space SpectralLatentGP over the manifold spectrum. Reuses the same
    # manifold graph/eigensolve knobs; nu and the diffusion scale (shared with
    # the manifold config) are the meaningful axes. fixed_ls / no_ard are not
    # used by the spectral path — placeholders are passed.
    if [ "$RUN_SPECTRAL" = "1" ]; then
        sp_sm="${MAN_STRIDE_MODES[0]}"
        sp_stride="${sp_sm%%:*}"; sp_modes="${sp_sm##*:}"
        sp_km="${MAN_KNN_METHODS[0]}"
        if [ "$sp_km" = "faiss_atlas_weighted" ]; then sp_infl="${MAN_INFLATIONS[0]}"; else sp_infl=1; fi
        AUGMENT_MALDI_NODES=0          # manifold-only; reset for clean logs
        INDUCING_FROM_MALDI_NODES=0    # manifold-only; reset for clean logs
        PER_TASK_LENGTHSCALE=0         # manifold-only; reset for clean logs
        LEARN_INDUCING=0
        # Spectral consumes the same graph, so it takes the first prune value
        # (like the other MAN_* knobs) rather than sweeping it.
        case "$sp_km" in
            faiss_atlas_weighted|faiss_cluster_weighted)
                PRUNE_CROSS_REGION="${MAN_PRUNE[0]}" ;;
            *)
                PRUNE_CROSS_REGION=0.0 ;;
        esac
        for nu in "${SPECTRAL_NU[@]}"; do
            job_name="gp-perlipid-${EXP_SUFFIX}-${exp_num}"
            printf "  exp %2d: %-22s nu=%-4s stride=%s modes=%s fold=%s\n" \
                "$exp_num" "spectral" "$nu" "$sp_stride" "$sp_modes" "$fold"
            submit "$job_name" "spectral" "$nu" \
                "$sp_km" "${MAN_KNN_K[0]}" "${MAN_LAPLACIAN_NORMS[0]}" "${MAN_GRAPH_BANDWIDTHS[0]}" \
                "$sp_infl" "${MAN_THRESHOLDS[0]}" \
                "$fold_upper" "$SLICES_DATASET_FILE" "$sp_stride" "$sp_modes" "0" "1"
            exp_num=$((exp_num + 1))
        done
    fi

    # ---- PARCEL loop ------------------------------------------------------
    if [ "$RUN_PARCEL" = "1" ]; then
        for feat in "${PARCEL_FEAT_LIST[@]}"; do
          for k in "${PARCEL_K_LIST[@]}"; do
            for sw in "${PARCEL_SW_LIST[@]}"; do
              for stride in "${PARCEL_STRIDE_LIST[@]}"; do
                for rank in "${PARCEL_RANK_LIST[@]}"; do
                  for nb in "${PARCEL_NB_LIST[@]}"; do
                    job_name="gp-parcel-${EXP_SUFFIX}-${exp_num}"
                    printf "  exp %2d: %-22s feat=%-7s K=%-4s sw=%-4s stride=%s r=%-2s nb=%s fold=%s\n" \
                        "$exp_num" "parcel" "$feat" "$k" "$sw" "$stride" "$rank" "$nb" "$fold"
                    submit_parcel "$job_name" "$fold_upper" "$SLICES_DATASET_FILE" \
                        "$feat" "$k" "$sw" "$stride" "$rank" "$nb" \
                        "$PARCEL_INIT_SCALE" "$PARCEL_SHARED_B"
                    exp_num=$((exp_num + 1))
                  done
                done
              done
            done
          done
        done
    fi
done

echo "Submitted $((exp_num - 1)) jobs (DRY_RUN=$DRY_RUN)."
echo "Results land under $S3_OUTPUT_DIR/<EXP_NAME>/"
echo ""
echo "To visualise once done:"
echo "  RUN_DIR=$S3_OUTPUT_DIR/<one-run-name> ./maldi/visualize_lipid_gp.sh"
echo ""
echo "To aggregate metrics across runs:"
echo "  python -c 'import pandas as pd, pathlib as P; \\"
echo "    print(pd.concat([pd.read_csv(d/\"metrics.csv\").assign(run=d.name) \\"
echo "      for d in P.Path(\"$S3_OUTPUT_DIR\").iterdir() \\"
echo "      if (d/\"metrics.csv\").exists()]).groupby(\"run\")[[\"test_corr\", \"test_rmse_z\"]].mean())'"