#!/usr/bin/env bash
# Submit GPLFR (collapsed-decoder) runs to runai, sweeping:
#
#   BASE_GP_LIST   — the latent-GP "kernel": riemann (manifold inducing-point
#                    RiemannMaternKernel) and spectral (weight-space
#                    SpectralLatentGP over the manifold spectrum). Both use the
#                    manifold graph + eigenbasis; euclidean does not, so it's not
#                    in the sweep.
#   MODES_LIST     — NUM_MODES (eigenbasis size). The primary sweep.
#   inflation      — always KNN_METHOD=faiss_atlas_weighted, at CROSS_REGION
#                    inflation 10 AND 50.
#
# => one job per (base_gp, modes, inflation).
#
# run_gplfr.sh's EXP_NAME encodes BASE_GP + LATENT_DIM but NOT modes/inflation, so
# we fold those into EXP_PREFIX to keep every run's output dir distinct.
# Preview without submitting:  DRY_RUN=1 ./submit/run_submit_gplfr_sweep.sh
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() { if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env (need WANDB_API_KEY)" >&2
    exit 1
fi

# --- Cluster resources ------------------------------------------------------
IMAGE=${IMAGE:-artiomartiom/sdsc:withfaiss}
CPU=${CPU:-4}
MEM=${MEM:-48G}
GPU=${GPU:-0.5}

# --- Sweep values -----------------------------------------------------------
BASE_GP_LIST=(${BASE_GP_LIST:-riemann spectral})            # the "kernels"
MODES_LIST=(${MODES_LIST:-100 200 300})                     # NUM_MODES
CROSS_REGION_INFLATION_LIST=(${CROSS_REGION_INFLATION_LIST:-10.0 50.0})

# --- Run config -------------------------------------------------------------
LATENT_DIM=${LATENT_DIM:-8}
N_EPOCHS=${N_EPOCHS:-50}
NU=${NU:-2.5}
LAPLACIAN_NORM=${LAPLACIAN_NORM:-randomwalk}
GRAPHBANDWIDTH_INIT=${GRAPHBANDWIDTH_INIT:-0.1}
N_LIST=${N_LIST:-sqrt}
N_PROBE=${N_PROBE:-8}
FOLD=${FOLD:-fold-2}

# --- Paths (S3 mounts on the cluster) --------------------------------------
S3_DATA_PATH=${S3_DATA_PATH:-/s3/mlibra/mlibra-data/maldi/}
S3_EIGENVECTOR_DIR=${S3_EIGENVECTOR_DIR:-/s3/mlibra/mlibra-data/artiom/eigenvectors}
S3_OUTPUT_DIR=${S3_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/gplfr_sweep}
S3_MALDI_FILE=${S3_MALDI_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet}
S3_REFERENCE_FILE=${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}
S3_ANNOTATION_FILE=${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}
S3_AVAILABLE_LIPIDS_FILE=${S3_AVAILABLE_LIPIDS_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy}
SRC_PATH=${SRC_PATH:-/myhome/mlibra}

fold_file=${FOLD//-/_}
SLICES_DATASET_FILE=${SLICES_DATASET_FILE:-/myhome/mlibra/maldi/data/splits/${fold_file}.json}
FOLD_UPPER=${FOLD^^}
# Compact date (no 'artiom-' / extra dashes): runai caps workload names at 50
# chars, and the longest job name here (spectral) would overflow otherwise.
EXP_SUFFIX=${EXP_SUFFIX:-$(date +'%y%m%d-%H%M')}

slug() { echo "$1" | sed 's/\./p/g'; }

# --- submit_gplfr <base_gp> <modes> <inflation> -----------------------------
n_submitted=0
submit_gplfr() {
    local base_gp=$1 modes=$2 infl=$3
    local infl_slug; infl_slug=$(slug "$infl")
    # riemann is finite-rank = num_modes: inducing points > modes make K_uu
    # singular (see the inducing-points-capped-by-num-modes note), so cap
    # NUM_INDUCING at modes. spectral ignores NUM_INDUCING.
    local num_inducing=$modes
    local prefix="${FOLD_UPPER}-NM${modes}-ATLASx${infl_slug}"
    local job_name="gplfr-${base_gp}-nm${modes}-atlasx${infl_slug}-${EXP_SUFFIX}"
    n_submitted=$((n_submitted + 1))
    echo ">>> [$n_submitted] $job_name  (base=$base_gp modes=$modes infl=$infl)"
    run_or_echo runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e EXP_PREFIX="$prefix" \
        -e BASE_GP="$base_gp" \
        -e NUM_MODES="$modes" \
        -e NUM_INDUCING="$num_inducing" \
        -e LATENT_DIM="$LATENT_DIM" \
        -e KNN_METHOD="faiss_atlas_weighted" \
        -e CROSS_REGION_INFLATION="$infl" \
        -e LAPLACIAN_NORM="$LAPLACIAN_NORM" \
        -e GRAPHBANDWIDTH_INIT="$GRAPHBANDWIDTH_INIT" \
        -e NU="$NU" \
        -e N_LIST="$N_LIST" \
        -e N_PROBE="$N_PROBE" \
        -e N_EPOCHS="$N_EPOCHS" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$SLICES_DATASET_FILE" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="reference" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e SRC_PATH="$SRC_PATH" \
        -- ./maldi/run_gplfr.sh
}

for base_gp in "${BASE_GP_LIST[@]}"; do
    for modes in "${MODES_LIST[@]}"; do
        for infl in "${CROSS_REGION_INFLATION_LIST[@]}"; do
            submit_gplfr "$base_gp" "$modes" "$infl"
        done
    done
done

echo "Submitted $n_submitted jobs. Fold: $FOLD  Suffix: $EXP_SUFFIX  Output: $S3_OUTPUT_DIR"
