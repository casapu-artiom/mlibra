#!/usr/bin/env bash
# =============================================================================
# Per-lipid GP runai batch submitter.
#
# Mirrors run_manifold_batch.sh but for the per-lipid pipeline
# (./maldi/run_lgp_per_lipid.sh → lgp_experiment_per_lipid.py).
#
# What it does:
#   - Loops over a small cross-product of hyperparameters
#     (KNN_METHOD × KNN_K × INFLATION × …)
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
MEM=48G
CPU=4
GPU=0.5

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

# FAISS IVF sizing. Default 1 = exact flat (current behaviour). For the fast,
# recall~1.0 CPU path use nlist=sqrt(N) with a small FIXED nprobe (~8): in 3D
# recall saturates at nprobe~8 regardless of N, so 'sqrt' nprobe just over-scans.
#   N_LIST=sqrt N_PROBE=8 FAISS_CPU=1 ./submit/run_submit_per_lipid.sh
N_LIST=${N_LIST:-sqrt}
N_PROBE=${N_PROBE:-8}

# ---- training hyperparameters (constant across sweep) ---------------------
# Per-lipid run knobs. EPOCHS=20 is the published default; drop to 5 for
# quick smoke-tests in the early-sweep phase. LIPID_BATCH_SIZE=10 fits
# our 10-lipid subset in a single GP fit.
EPOCHS=10
LIPID_BATCH_SIZE=10
NUM_INDUCING=1000
LEARNING_RATE=0.005
BATCH_SIZE=2048
STRIDE_BUMP=20.0   # bump_scale — default; overridden per-job by MAN_BUMP_PAIRS
STRIDE_DECAY=0.01  # bump_decay — default; overridden per-job by MAN_BUMP_PAIRS

# ---- subset of lipids to train -------------------------------------------
# A small subset is what makes a 30-job sweep practical. The file lives
# inside the container at the same path the run script expects.
LIPIDS_FILE="/myhome/mlibra/maldi/data/lipid_subset.txt"

# ---- S3-mounted paths inside the container --------------------------------
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/artiom/eigenvectors"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/per_lipid_batch_10"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/fold_3.json"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
SRC_PATH="/myhome/mlibra"

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

# ---- one submission -------------------------------------------------------
# All hyperparameters get passed as -e env vars; run_lgp_per_lipid.sh
# inside the container reads them via ``: "${X:=default}"`` and forwards
# to lgp_experiment_per_lipid.py.
submit() {
    local job_name=$1
    local family=$2        # euclidean | manifold
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
        -e BUMP_SCALE="$STRIDE_BUMP" \
        -e BUMP_DECAY="$STRIDE_DECAY" \
        -e NUM_MODES="$modes" \
        -e NCV_MIN="${NCV_MIN:--1}" \
        -e STRIDE="$stride" \
        -e FIXED_LENGTHSCALE="$fixed_ls" \
        -e NO_ARD="$no_ard" \
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

FOLDS=("fold-2")

# ---- Euclidean sweep ------------------------------------------------------
# Euclidean Matern only has nu as a meaningful hyperparameter — the GP
# learns its own lengthscale per dimension. Other knobs (kernel="matern"
# etc.) are fixed in run_lgp_per_lipid.sh.
EUC_NU=(1.5)
# ARD modes (euclidean only): 1 → isotropic (--no-ard), 0 → per-axis ARD.
# Both are submitted so isotropic vs per-axis can be compared head-to-head.
EUC_NO_ARD=(1 0)
RUN_EUCLIDEAN=1   # 0 = skip the euclidean loop entirely

# ---- Manifold sweep -------------------------------------------------------
MAN_NU=(2)
MAN_THRESHOLDS=(5 40)
MAN_LAPLACIAN_NORMS=("randomwalk")
MAN_GRAPH_BANDWIDTHS=(0.1)
MAN_KNN_K=(15)
MAN_KNN_METHODS=("faiss_atlas_weighted")
MAN_INFLATIONS=(10 50)   # only used when knn_method=faiss_atlas_weighted
# ---- (stride : num_modes) pairs to sweep (manifold only) ------------------
# Coarser strides need fewer eigenmodes to span the graph; finer strides need
# more. Each entry is "STRIDE:NUM_MODES" and gets its own job.
MAN_STRIDE_MODES=("4:1300")

# ---- (bump_scale : bump_decay) pairs to sweep (manifold only) -------------
# The bump function shapes the kernel's local support on the graph. Each entry
# is "BUMP_SCALE:BUMP_DECAY" and gets its own job; the run TAG already encodes
# -bs<scale>-bd<decay> so output dirs never collide. Add more pairs to sweep.
MAN_BUMP_PAIRS=("1.0:0.01")

# ---- lengthscale modes (manifold only) 
# 1 → fixed lengthscale (--lengthscale-init 8.0 --lengthscale-no-decay)
# 0 → learned lengthscale (no flags; GP trains it).
# Both modes are submitted so they can be compared head-to-head.
MAN_FIXED_LS=(1)

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
# cheaply. 1 = on, 0 = plain k-means-snap inducing.
MAN_INDUCING_BLEND=1
INDUCING_DENSITY_FRAC=0.8       # frac from densest graph nodes (rest = cheapest-to-snap MALDI)
# Learned vs anchored inducing points — listed so each config is submitted both
# ways for head-to-head comparison (output dirs differ via the "-learnind" tag).
# 0 = anchored (stay on graph nodes), 1 = optimize inducing-point locations.
MAN_LEARN_INDUCING=(0 1)
WANDB=1                        # log loss/KL/hypers/noise/grad-norms (incl. inducing) to W&B
WANDB_PROJECT=l3di_maldi_per_lipid
RUN_MANIFOLD=1


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
                                for infl in "${infl_list[@]}"; do
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
                                        INDUCING_FROM_MALDI_NODES=$MAN_INDUCING_BLEND
                                        # Sweep anchored vs learned inducing
                                        # points. LEARN_INDUCING is forwarded as
                                        # -e LEARN_INDUCING; the "-learnind" TAG
                                        # suffix keeps the two output dirs apart.
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
                                        prefix="${fold_upper}-${ls_tag}"
                                        printf "  exp %2d: %-22s nu=%s knn_k=%-3s ln=%s gb=%s infl=%s stride=%s modes=%s bs=%s bd=%s ls=%s aug=%s ind=%s\n" \
                                            "$exp_num" "$km" "$nu" "$knn" "$ln" "$gb" "$infl" "$stride" "$modes" "$bump_scale" "$bump_decay" "$ls_tag" "$aug_tag" "$li_tag"
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