#!/usr/bin/env bash
# Submit eigensolver-comparison jobs to runai.
#
# Each `submit_one` builds a self-documenting output subdir from the run's
# properties (stride/threshold/k/bandwidth/norm/modes) and forwards them to
# maldi/eigensolver_compare.sh inside the container, which writes
#     $S3_OUTPUT_DIR/$EXP_SUFFIX/<run_slug>/compare.{json,png,log}
#
# Edit the handful of submit_one calls at the bottom for the configs you care
# about. Preview without submitting:
#     DRY_RUN=1 ./run_eigensolver_compare.sh

set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() { if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

# -------------------------------------------------------------------------
# Cluster resources. A full GPU by default -- eigsh / shift-invert want the
# memory for the Lanczos basis at high mode counts.
# -------------------------------------------------------------------------
MEM=${MEM:-128G}
CPU=${CPU:-4}
GPU=${GPU:-1.0}
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_latest}

# -------------------------------------------------------------------------
# Paths (S3 mounts on the cluster)
# -------------------------------------------------------------------------
S3_KNN_DIR="${S3_KNN_DIR:-/s3/mlibra/mlibra-data/artiom/eigenvectors/knn}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/eigensolver_compare}"
S3_REFERENCE_FILE="${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}"
S3_ANNOTATION_FILE="${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}"
SRC_PATH="${SRC_PATH:-/myhome/mlibra}"

# Build the graph in-job if it isn't cached yet (set BUILD_IF_MISSING=0 to
# instead fail fast when a graph is missing).
BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"

EXP_SUFFIX="${EXP_SUFFIX:-$(date +'%y%m%d-%H%M')}"

slug() { echo "$1" | sed 's/\./p/g'; }

# -------------------------------------------------------------------------
# submit_one  <stride> <threshold> <knn_k> <modes> [bandwidth] [norm] [approaches]
#
# Per-run properties are pushed in as env vars; the inner shell consumes them
# and composes both the graph cache key and the self-documenting RUN_SLUG.
# -------------------------------------------------------------------------
n_submitted=0
submit_one() {
    local stride=$1 threshold=$2 knn_k=$3 modes=$4
    local bandwidth=${5:-1.0} norm=${6:-randomwalk}
    local approaches=${7:-cupy_sa,cupy_si_cg,scipy_si,lobpcg}

    local run_slug="str${stride}_t${threshold}_k${knn_k}_faiss_bw$(slug "$bandwidth")_${norm}_nm${modes}"
    n_submitted=$((n_submitted + 1))
    local job_name="eigcmp-${EXP_SUFFIX}-$(printf '%03d' "$n_submitted")"

    echo ">>> [$n_submitted] $job_name -> $run_slug"
    run_or_echo runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$CPU"   --cpu-core-request   "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e REPO="$SRC_PATH" \
        -e KNN_DIR="$S3_KNN_DIR" \
        -e OUT_DIR="$S3_OUTPUT_DIR/$EXP_SUFFIX" \
        -e RUN_SLUG="$run_slug" \
        -e BUILD_IF_MISSING="$BUILD_IF_MISSING" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATIONS_FILE="$S3_ANNOTATION_FILE" \
        -e STRIDE="$stride" \
        -e THRESHOLD="$threshold" \
        -e KNN_K="$knn_k" \
        -e MODES="$modes" \
        -e BANDWIDTH="$bandwidth" \
        -e NORMALIZATION="$norm" \
        -e APPROACHES="$approaches" \
        -- ./maldi/eigensolver_compare.sh
}

# -------------------------------------------------------------------------
# The configs you care about -- edit these. With BUILD_IF_MISSING=1 (default)
# any graph not yet in $S3_KNN_DIR is built in-job from the reference/annotation
# volumes, then cached for reuse; set BUILD_IF_MISSING=0 to fail fast instead.
#            stride thr  k   modes  [bw]  [norm]      [approaches]
# -------------------------------------------------------------------------
submit_one      4   5  15   1300   1.0  randomwalk  cupy_sa,cupy_si_cg,scipy_si
# stride=2 is the shift-invert sweet spot (graph auto-built on first run):
submit_one      2   5  15   1300   1.0  randomwalk  cupy_sa,cupy_si_cg
submit_one      2   5  15   1300   1.0  randomwalk  cupy_sa,cupy_si_cg
submit_one      1   5  15   300    1.0  randomwalk  cupy_sa,cupy_si_cg

echo "Submitted $n_submitted jobs. Suffix: $EXP_SUFFIX"
echo "Outputs: $S3_OUTPUT_DIR/$EXP_SUFFIX/<run_slug>/compare.{json,png,log}"
