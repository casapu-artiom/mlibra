#!/usr/bin/env bash
set -euo pipefail

# Spectral twin of run_manifold_batch.sh. Submits the weight-space spectral
# Riemann runner (maldi/run_spectral_manifold.sh -> spectral_lgp_manifold_experiment.py).
# Differences from run_manifold_batch.sh: the spectral GP has NO inducing points
# and no product-ARD ScaleKernel composition, so the IND_SOURCES and PRODUCT_ARD
# sweeps (and their env vars) are gone here. A LENGTHSCALE_INIT sweep is added in
# their place (the spectral kernel's lengthscale reshapes the Matern spectral
# density). Everything graph/Laplacian/eigenpair-related is identical, so the
# graph/eigvec caches are shared with the inducing runs.

DRY_RUN=${DRY_RUN:-0}

run_or_echo() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] $*"
    else
        "$@"
    fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

MEM=48G
CPU=4
GPU=0.5

# ---- FAISS CPU-only toggle ------------------------------------------------
# FAISS_CPU=1 forces all FAISS KNN work (graph build, searches, whole-brain
# reconstruction) onto CPU inside the job -- for benchmarking a CPU-only run
# before dropping the GPU FAISS path. Override the three phases independently
# if you only want to pin part of the pipeline. These env vars are forwarded by
# run_spectral_manifold.sh into the Python --faiss-cpu-* CLI flags (Python reads
# only the CLI).  FAISS_CPU=1 ./submit/run_spectral_manifold_batch.sh
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
#   N_LIST=sqrt N_PROBE=8 FAISS_CPU=1 ./submit/run_spectral_manifold_batch.sh
N_LIST=${N_LIST:-sqrt}
N_PROBE=${N_PROBE:-8}

N_EPOCHS=100
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/artiom/eigenvectors"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/experiment_batch_cv_prune"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
# Atlas level: ATLAS_LEVEL=5|15 (default 15) selects the annotation volume on S3.
# Cache keys + output dir are tagged by the file stem, so levels never collide.
# Re-invoke with a different ATLAS_LEVEL for the other level.
ATLAS_LEVEL=${ATLAS_LEVEL:-5}
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_${ATLAS_LEVEL}annot.npy"
S3_SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/fold_3.json"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
S3_BG_TEMPLATE_NAME="brainglobe"
S3_REFERENCE_FILE_BG="/s3/mlibra/mlibra-data/bg_template.npy"
S3_ANNOTATION_FILE_BG="/s3/mlibra/mlibra-data/bg_annotations.npy"
SRC_PATH="/myhome/mlibra"
# Curated lipid subset that run_spectral_manifold.sh reconstructs/renders. Lives
# in the repo (mounted at $SRC_PATH), not on S3. Override at submit time with
# RECONSTRUCTION_LIPIDS_FILE=... ./submit/run_spectral_manifold_batch.sh
RECON_LIPIDS_FILE="${RECONSTRUCTION_LIPIDS_FILE:-$SRC_PATH/maldi/data/lipid_subset.txt}"

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')-2"

submit() {
    local job_name=$1 template=$2 ref=$3 annot=$4 infl=$5 threshold=$6 knn=$7 knn_k=$8 laplacian_norm=$9 nu=${10} graphbandwidth=${11} bumpscale=${12} bumpdecay=${13} prefix=${14} slice=${15} stride=${16} num_modes=${17} diffusion_init=${18} learn_diffusion=${19} lengthscale=${20}
	shift 20
	local extra_args=("$@")    # everything remaining goes here
    echo ">>> Submitting $job_name"
    runai training submit "$job_name" \
        -i artiomartiom/sdsc:maldi_manifold_all_latest \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slice" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e RECONSTRUCTION_LIPIDS_FILE="$RECON_LIPIDS_FILE" \
        -e TEMPLATE_NAME="$template" \
        -e REFERENCE_FILE="$ref" \
        -e ANNOTATION_FILE="$annot" \
        -e KNN_METHOD="$knn" \
        -e KNN_K="$knn_k" \
        -e LAPLACIAN_NORM="$laplacian_norm" \
        -e NU="$nu" \
		-e SRC_PATH="$SRC_PATH" \
        -e GRAPHBANDWIDTH="$graphbandwidth" \
        -e DIFFUSION_SCALE_INIT="$diffusion_init" \
        -e LEARN_DIFFUSION_SCALE="$learn_diffusion" \
        -e LEARN_SPECTRAL_WEIGHTS="${LEARN_SPECTRAL_WEIGHTS:-1}" \
        -e LENGTHSCALE_INIT="$lengthscale" \
        -e BUMP_SCALE="$bumpscale" \
        -e BUMP_DECAY="$bumpdecay" \
        -e N_EPOCHS="$N_EPOCHS" \
        -e NUM_MODES="$num_modes" \
        -e NCV_MIN="${NCV_MIN:--1}" \
        -e THRESHOLD="$threshold" \
        -e STRIDE="$stride" \
        -e CROSS_REGION_INFLATION="$infl" \
        -e CLUSTER_K="${CLUSTER_K:-64}" \
        -e CLUSTER_SPATIAL_WEIGHT="${CLUSTER_SPATIAL_WEIGHT:-1.0}" \
        -e CLUSTER_FIT_SUBSAMPLE="${CLUSTER_FIT_SUBSAMPLE:-40000}" \
        -e CLUSTER_SEED="${CLUSTER_SEED:-0}" \
        -e ROOT_HANDLING="${ROOT_HANDLING:-dissolve}" \
        -e DENOISE_LABELS="${DENOISE_LABELS:-3}" \
        -e PRUNE_CROSS_REGION="${PRUNE_CROSS_REGION:-0.0}" \
        -e FAISS_CPU_GRAPH="$FAISS_CPU_GRAPH" \
        -e FAISS_CPU_SEARCH="$FAISS_CPU_SEARCH" \
        -e FAISS_CPU_RECON="$FAISS_CPU_RECON" \
        -e FORCE_RECOMPUTE_GRAPH="$FORCE_RECOMPUTE_GRAPH" \
        -e N_LIST="$N_LIST" \
        -e N_PROBE="$N_PROBE" \
        -- ./maldi/run_spectral_manifold.sh "${extra_args[@]}"
}

FOLDS=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8")           # lowercase, dashed
KNN_K=(15)
# Graph methods to sweep. Override at submit time (space-separated), e.g.
#   MAN_KNN_METHODS_STR="faiss faiss_cluster_weighted" ATLAS_LEVEL=5 ./submit/run_spectral_manifold_batch.sh
#   'faiss'                  = plain kNN, no prior
#   'faiss_atlas_weighted'   = atlas-region inflation (uses ATLAS_LEVEL's annotation volume)
#   'faiss_cluster_weighted' = template-clustering inflation (no atlas; CLUSTER_* knobs)
MAN_KNN_METHODS=(faiss_atlas_weighted)
# Cross-region inflation, used by faiss_atlas_weighted AND faiss_cluster_weighted.
MAN_INFLATIONS=(50)
# ---- graph-refinement sweeps (weighted methods only) ----------------------
# Swept like MAN_INFLATIONS; non-weighted methods collapse to the no-op value so
# they aren't submitted redundantly. Each distinct value gets its own -root/-dn/
# -prune dir tag, so a sweep never clobbers. ROOT_HANDLINGS is atlas-only.
ROOT_HANDLINGS=("dissolve")
MAN_DENOISE_LABELS=(3)
MAN_PRUNE_CROSS_REGIONS=(0.97)
LAPLACIAN_NORMS=("randomwalk")
GRAPH_BANDWIDTHS=(0.1)
BUMP_SCALES=(1.0)
BUMP_DECAYS=(0.01)
NU=(2)
THRESHOLDS=(5)

# Lengthscale-init sweep (spectral kernel). The Matern lengthscale reshapes the
# spectral density Phi = (2 nu / ell^2 + lambda_k)^(-nu): smaller ell raises the
# floor 2 nu/ell^2 (flatter Phi, more high-mode power -> finer / less smooth
# field), larger ell concentrates power on the low modes (smoother field). Each
# value lands in a distinct EXP_NAME dir (LENGTHSCALE_INIT is in the dir tag).
LENGTHSCALES=(1.0)

# Diffusion-scale sweep (manifold kernel). Each entry is "LEARN:INIT":
#   LEARN = 0/1   -> is the multiplicative spectral scale learnable?
#   INIT  = float -> its initial value (1.0 = identity; no eigenpair recompute).
# LEARN=1 runs get a -learndiff suffix in EXP_NAME, so learned vs frozen don't
# clobber each other. e.g. ("0:1.0" "1:1.0" "1:2.0") = baseline + two learned starts.
DIFFUSION_SCALES=("1:1.0")

# (stride, num_modes) pairs swept together:
#   (1) stride=4, num_modes=1300
#   (2) stride=8, num_modes=6000
#STRIDE_NUM_MODES=("4:1300" "8:6000")
STRIDE_NUM_MODES=("4:100" "4:300" "4:1300")

# Fixed across the whole sweep
TEMPLATE="reference"
REF="$S3_REFERENCE_FILE"
ANNOT="$S3_ANNOTATION_FILE"

exp_num=1
for fold in "${FOLDS[@]}"; do
    # fold-3  -> FOLD-3  (used as wandb EXP_PREFIX)
    # fold-3  -> fold_3  (used in the splits filename)
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"

    for stride_modes in "${STRIDE_NUM_MODES[@]}"; do
        stride=${stride_modes%%:*}
        num_modes=${stride_modes##*:}

        for gb in "${GRAPH_BANDWIDTHS[@]}"; do
            for bs in "${BUMP_SCALES[@]}"; do
                for bd in "${BUMP_DECAYS[@]}"; do
                    for knn_k in "${KNN_K[@]}"; do
                        for knn_method in "${MAN_KNN_METHODS[@]}"; do
                            for laplacian_norm in "${LAPLACIAN_NORMS[@]}"; do
                                for nu in ${NU[@]}; do
                                    for threshold in ${THRESHOLDS[@]}; do
                                        # Graph-prior lists apply only to weighted methods;
                                        # plain faiss collapses each to its no-op value.
                                        if [ "$knn_method" = "faiss_atlas_weighted" ] || [ "$knn_method" = "faiss_cluster_weighted" ]; then
                                            infl_list=("${MAN_INFLATIONS[@]}")
                                            denoise_list=("${MAN_DENOISE_LABELS[@]}")
                                            prune_list=("${MAN_PRUNE_CROSS_REGIONS[@]}")
                                        else
                                            infl_list=(1)
                                            denoise_list=(0)
                                            prune_list=(0.0)
                                        fi
                                        if [ "$knn_method" = "faiss_atlas_weighted" ]; then
                                            root_list=("${ROOT_HANDLINGS[@]}")
                                        else
                                            root_list=(dissolve)
                                        fi
                                        for infl in "${infl_list[@]}"; do
                                          for ls in "${LENGTHSCALES[@]}"; do
                                           for diff in "${DIFFUSION_SCALES[@]}"; do
                                            learn_diffusion=${diff%%:*}
                                            diffusion_init=${diff##*:}
                                            for ROOT_HANDLING in "${root_list[@]}"; do
                                             for DENOISE_LABELS in "${denoise_list[@]}"; do
                                              for PRUNE_CROSS_REGION in "${prune_list[@]}"; do
                                                # read by submit() via its -e ${VAR:-…} lines.

                                                job_name="gp-spectral-${EXP_SUFFIX}-${exp_num}"

                                                # Map exp_num -> config, so you can read it back from terminal/logs
                                                printf "  exp %2d: fold=%-10s stride=%s modes=%s ls=%-5s gb=%-5s bs=%-4s bd=%s diff=%s(learn=%s) root=%s dn=%s prune=%s\n" \
                                                    "$exp_num" "$fold" "$stride" "$num_modes" "$ls" "$gb" "$bs" "$bd" "$diffusion_init" "$learn_diffusion" "$ROOT_HANDLING" "$DENOISE_LABELS" "$PRUNE_CROSS_REGION"

                                                run_or_echo submit "$job_name" "$TEMPLATE" "$REF" "$ANNOT" "$infl" "$threshold" \
                                                    "$knn_method" "$knn_k" "$laplacian_norm" "$nu" "$gb" "$bs" "$bd" "$fold_upper" "$SLICES_DATASET_FILE" "$stride" "$num_modes" "$diffusion_init" "$learn_diffusion" "$ls"

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
done

echo "Submitted $((exp_num - 1)) jobs."
