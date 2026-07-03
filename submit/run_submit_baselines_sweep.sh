#!/usr/bin/env bash
# Submit the non-GP baselines to runai, sweeping the params that matter per model:
#
#   mean / linear / mlp   — no manifold params: one job each.
#   mlp_eigen             — sweep NUM_MODES (the eigenbasis dimensionality; more
#                           modes = richer per-point eigen features). Each modes
#                           value triggers an eigensolve (cached by modes).
#   gcn_faiss             — sweep the GRAPH: plain `faiss` vs `faiss_atlas_weighted`,
#                           and for the atlas graph two CROSS_REGION_INFLATION
#                           values. (gcn_faiss uses graph TOPOLOGY only, so
#                           NUM_MODES is inert for it — the graph is what matters.)
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

# --- Cluster resources ------------------------------------------------------
IMAGE=${IMAGE:-artiomartiom/sdsc:withfaiss}
CPU=${CPU:-4}
# Plain baselines (mean/linear/mlp) are light; the manifold-aware ones
# (mlp_eigen/gcn_faiss) also run a GPU eigensolve / full-graph GCN, so give them
# more RAM + a whole GPU.
MEM=${MEM:-48G}
GPU=${GPU:-0.5}
MEM_MANIFOLD=${MEM_MANIFOLD:-48G}
GPU_MANIFOLD=${GPU_MANIFOLD:-0.5}

# --- Sweep values -----------------------------------------------------------
# Space-separated; override from the environment, e.g. MODES_LIST="100 300".
MODES_LIST=(${MODES_LIST:-50 100 200 300})                  # mlp_eigen: eigenbasis size
CROSS_REGION_INFLATION_LIST=(${CROSS_REGION_INFLATION_LIST:-10.0 50.0})  # gcn_faiss atlas graph
# Param-free-from-the-sweep baselines (one job each; xgboost keeps its own fixed
# n_estimators/max_depth/lr defaults from run_baseline.sh, not swept here).
PLAIN_MODELS=(${PLAIN_MODELS:-mean linear mlp xgboost})

# --- Run config -------------------------------------------------------------
N_EPOCHS=${N_EPOCHS:-100}
GCN_FAISS_ITERS=${GCN_FAISS_ITERS:-20000}
STRIDE=${STRIDE:-4}
FOLD=${FOLD:-fold-2}

# --- Paths (S3 mounts on the cluster) --------------------------------------
S3_DATA_PATH=${S3_DATA_PATH:-/s3/mlibra/mlibra-data/maldi/}
S3_EIGENVECTOR_DIR=${S3_EIGENVECTOR_DIR:-/s3/mlibra/mlibra-data/artiom/eigenvectors}
S3_OUTPUT_DIR=${S3_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/baselines_sweep}
S3_MALDI_FILE=${S3_MALDI_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet}
S3_REFERENCE_FILE=${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}
S3_ANNOTATION_FILE=${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}
S3_AVAILABLE_LIPIDS_FILE=${S3_AVAILABLE_LIPIDS_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy}
SRC_PATH=${SRC_PATH:-/myhome/mlibra}

fold_file=${FOLD//-/_}
SLICES_DATASET_FILE=${SLICES_DATASET_FILE:-/myhome/mlibra/maldi/data/splits/${fold_file}.json}
FOLD_UPPER=${FOLD^^}
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
        "${extra_env[@]}" \
        -- ./maldi/run_baseline.sh
}

# --- Param-free baselines: one job each -------------------------------------
for model in "${PLAIN_MODELS[@]}"; do
    submit_baseline "base-${model}-${EXP_SUFFIX}" "$model" "$MEM" "$GPU" \
        -e EXP_PREFIX="$FOLD_UPPER"
done

# --- mlp_eigen: sweep NUM_MODES (fold modes into EXP_PREFIX so dirs are distinct)
for modes in "${MODES_LIST[@]}"; do
    submit_baseline "base-mlpeigen-nm${modes}-${EXP_SUFFIX}" "mlp_eigen" "$MEM_MANIFOLD" "$GPU_MANIFOLD" \
        -e EXP_PREFIX="${FOLD_UPPER}-NM${modes}" \
        -e NUM_MODES="$modes"
done

# --- gcn_faiss: plain faiss graph -------------------------------------------
submit_baseline "base-gcnfaiss-faiss-${EXP_SUFFIX}" "gcn_faiss" "$MEM_MANIFOLD" "$GPU_MANIFOLD" \
    -e EXP_PREFIX="${FOLD_UPPER}-FAISS" \
    -e KNN_METHOD="faiss" \
    -e GCN_FAISS_ITERS="$GCN_FAISS_ITERS"

# --- gcn_faiss: atlas-weighted graph, sweeping two cross-region inflations ---
for infl in "${CROSS_REGION_INFLATION_LIST[@]}"; do
    infl_slug=$(slug "$infl")
    submit_baseline "base-gcnfaiss-atlasx${infl_slug}-${EXP_SUFFIX}" "gcn_faiss" "$MEM_MANIFOLD" "$GPU_MANIFOLD" \
        -e EXP_PREFIX="${FOLD_UPPER}-ATLASx${infl_slug}" \
        -e KNN_METHOD="faiss_atlas_weighted" \
        -e CROSS_REGION_INFLATION="$infl" \
        -e GCN_FAISS_ITERS="$GCN_FAISS_ITERS"
done

echo "Submitted $n_submitted jobs. Fold: $FOLD  Suffix: $EXP_SUFFIX  Output: $S3_OUTPUT_DIR"
