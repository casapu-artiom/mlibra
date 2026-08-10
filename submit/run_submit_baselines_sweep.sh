#!/usr/bin/env bash
# Submit the non-GP baselines to runai, sweeping the params that matter per model:
#
#   mean / linear / mlp   — no manifold params: one job each.
#   xgboost               — no manifold params either, but its hyperparams
#                           (XGB_LR / XGB_N_ESTIMATORS / XGB_MAX_DEPTH) are
#                           passed explicitly: run_baseline.sh's XGB_LR default
#                           of 0 trains a no-op model.
#   mlp_eigen             — sweep NUM_MODES (the eigenbasis dimensionality; more
#                           modes = richer per-point eigen features). Each modes
#                           value triggers an eigensolve (cached by modes).
#   gcn_faiss             — sweep the GRAPH: plain `faiss` vs `faiss_atlas_weighted`,
#                           and for the atlas graph two CROSS_REGION_INFLATION
#                           values. (gcn_faiss uses graph TOPOLOGY only, so
#                           NUM_MODES is inert for it — the graph is what matters.)
#
# The whole grid above is repeated for every fold in FOLDS_LIST.
#
# run_baseline.sh's EXP_NAME does NOT encode the swept value, so we fold it into
# EXP_PREFIX to keep each run's output dir distinct (otherwise a sweep clobbers
# itself). Preview without submitting:  DRY_RUN=1 ./submit/run_submit_baselines_sweep.sh
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

# --- Cluster resources 
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_all_latest}
# CPU is both the cgroup quota AND what OMP/BLAS are pinned to in submit_baseline.
# torch/OpenMP otherwise size their thread pools from the NODE's core count, which
# the cgroup then throttles to $CPU — the threads spin-wait and thrash (measured
# ~5x slower at 32 threads on 2 cores). OMP_WAIT_POLICY=passive stops the spinning.
# Mirrors run_submit_per_lipid.sh, which already does this.
CPU=${CPU:-2}
# Plain baselines (mean/linear/mlp------------------------------------------------------) are light; the manifold-aware ones
# (mlp_eigen/gcn_faiss) also run a GPU eigensolve / full-graph GCN, so give them
# more RAM + a whole GPU.
MEM=${MEM:-32G}
GPU=${GPU:-0.2}
MEM_MANIFOLD=${MEM_MANIFOLD:-48G}
GPU_MANIFOLD=${GPU_MANIFOLD:-0.5}

# --- Sweep values -----------------------------------------------------------
# Space-separated; override from the environment, e.g. MODES_LIST="100 300".
MODES_LIST=(${MODES_LIST:-1300})                  # mlp_eigen: eigenbasis size
CROSS_REGION_INFLATION_LIST=(${CROSS_REGION_INFLATION_LIST:-50.0})  # gcn_faiss atlas graph
# Baselines with no manifold params: one job each. Not swept, but xgboost's
# hyperparams are passed explicitly below (see XGB knobs).
PLAIN_MODELS=(${PLAIN_MODELS:-mean linear mlp xgboost})
#PLAIN_MODELS=(${PLAIN_MODELS:-mlp})

# --- Run config -------------------------------------------------------------
N_EPOCHS=${N_EPOCHS:-100}
GCN_FAISS_ITERS=${GCN_FAISS_ITERS:-20000}
STRIDE=${STRIDE:-4}

# --- XGBoost knobs (MODEL=xgboost) ------------------------------------------
# run_baseline.sh defaults XGB_LR to 0, and a zero learning rate makes every
# boosted tree contribute nothing — the model never leaves its base score and
# xgboost silently degenerates into the mean baseline. Always pass a real LR.
XGB_LR=${XGB_LR:-0.05}
XGB_N_ESTIMATORS=${XGB_N_ESTIMATORS:-400}
XGB_MAX_DEPTH=${XGB_MAX_DEPTH:-6}
# Folds to sweep (space-separated); override e.g. FOLDS_LIST="fold-0 fold-1 fold-2".
FOLDS_LIST=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8")           # lowercase, dashed

# --- Paths (S3 mounts on the cluster) --------------------------------------
S3_DATA_PATH=${S3_DATA_PATH:-/s3/mlibra/mlibra-data/maldi/}
S3_EIGENVECTOR_DIR=${S3_EIGENVECTOR_DIR:-/s3/mlibra/mlibra-data/artiom/eigenvectors}
S3_OUTPUT_DIR=${S3_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/baselines_cv}
S3_MALDI_FILE=${S3_MALDI_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet}
S3_REFERENCE_FILE=${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}
S3_ANNOTATION_FILE=${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}
S3_AVAILABLE_LIPIDS_FILE=${S3_AVAILABLE_LIPIDS_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy}
SRC_PATH=${SRC_PATH:-/myhome/mlibra}
# mlp_eigen stages an N*(3+NUM_MODES) float32 feature memmap on disk. It must NOT
# land on the S3 output mount: mmap there fails with OSError 95, and the random
# per-epoch reads would be network-bound anyway. Point it at local/fast storage.
FEAT_SCRATCH_DIR=${FEAT_SCRATCH_DIR:-/mydata/mlibra/artiom/scratch}

# Each fold's split file is <SPLITS_DIR>/<fold_with_underscores>.json.
SPLITS_DIR=${SPLITS_DIR:-/myhome/mlibra/maldi/data/splits}
EXP_SUFFIX=${EXP_SUFFIX:-$(date +'%y%m%d-%H-%M')}

slug() { echo "$1" | sed 's/\./p/g'; }

# --- submit_baseline <job_name> <model> <mem> <gpu> [extra -e args...] -------
n_submitted=0
submit_baseline() {
    local job_name=$1 model=$2 mem=$3 gpu=$4
    shift 4
    local extra_env=("$@")
    n_submitted=$((n_submitted + 1))
    echo ">>> [$n_submitted] $job_name  (model=$model, mem=$mem, gpu=$gpu)"
    run_or_echo runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$mem" --cpu-memory-request "$mem" \
        --gpu-request-type portion --gpu-portion-request "$gpu" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e MODEL="$model" \
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
        -e N_EPOCHS="$N_EPOCHS" \
        -e STRIDE="$STRIDE" \
        -e OMP_NUM_THREADS="$CPU" \
        -e OPENBLAS_NUM_THREADS="$CPU" \
        -e MKL_NUM_THREADS="$CPU" \
        -e OMP_WAIT_POLICY=passive \
        "${extra_env[@]}" \
        -- ./local_run/run_baseline.sh
}

for FOLD in "${FOLDS_LIST[@]}"; do
    SLICES_DATASET_FILE="${SPLITS_DIR}/${FOLD//-/_}.json"
    FOLD_UPPER=${FOLD^^}
    # runai job names must be lowercase DNS-safe: fold-2 -> f2.
    fold_slug=$(slug "${FOLD//fold-/f}")
    echo "=== Fold: $FOLD  (split: $SLICES_DATASET_FILE)"

    # --- mlp_eigen: sweep NUM_MODES (fold modes into EXP_PREFIX so dirs are distinct)
    for modes in "${MODES_LIST[@]}"; do
        submit_baseline "base-mlpeigen-nm${modes}-${fold_slug}-${EXP_SUFFIX}" "mlp_eigen" "$MEM_MANIFOLD" "$GPU_MANIFOLD" \
            -e EXP_PREFIX="${FOLD_UPPER}-NM${modes}" \
            -e NUM_MODES="$modes" \
            -e FEAT_SCRATCH_DIR="$FEAT_SCRATCH_DIR"
    done

    # --- gcn_faiss: atlas-weighted graph, sweeping cross-region inflations ---
    for infl in "${CROSS_REGION_INFLATION_LIST[@]}"; do
        infl_slug=$(slug "$infl")
        submit_baseline "base-gcnfaiss-atlasx${infl_slug}-${fold_slug}-${EXP_SUFFIX}" "gcn_faiss" "$MEM_MANIFOLD" "$GPU_MANIFOLD" \
            -e EXP_PREFIX="${FOLD_UPPER}-ATLASx${infl_slug}" \
            -e KNN_METHOD="faiss_atlas_weighted" \
            -e CROSS_REGION_INFLATION="$infl" \
            -e GCN_FAISS_ITERS="$GCN_FAISS_ITERS"
    done
done

echo "Submitted $n_submitted jobs. Folds: ${FOLDS_LIST[*]}  Suffix: $EXP_SUFFIX  Output: $S3_OUTPUT_DIR"
