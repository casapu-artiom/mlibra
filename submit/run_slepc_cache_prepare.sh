#!/usr/bin/env bash
# Pre-compute + cache the graph-Laplacian eigenvectors on the cloud, CPU-only,
# using SLEPc SHIFT-INVERT (slepc-si) + a parallel direct solver (MUMPS).
#
# This is the "prepare the cache ahead of time" job: each submit_one launches
# ONE pod that runs `mpirun -n <CPU cores> python slepc_eigensolve.py
# --shift-invert`. PETSc/SLEPc/MUMPS distribute the Laplacian, the factor and
# the eigenvectors across the ranks, and the result lands in the shared eigvec
# cache (LaplacianEigensolver layout) so the GP / manifold code later reads it
# as a plain cache hit -- no recompute at train time.
#
# CPU-HEAVY by design: SLEPc/PETSc/MUMPS are CPU codes, so the scaling knob is
# CPU CORES (= MPI ranks), not GPUs. We request NO GPU at all -- when a graph
# isn't cached yet, rank 0 builds it with faiss on CPU (slower, but keeps the
# job 100% CPU). Pre-build graphs separately if you'd rather not pay that here.
#
# Shift-invert is the right tool for the low end of the spectrum at small stride
# (dense fill): MUMPS is a 64-bit parallel LU that factorizes where scipy's
# 32-bit SuperLU overflows. Target sits at 0.0 (bottom of the spectrum).
#
# Preview the runai commands without submitting:   DRY_RUN=1 ./submit/run_slepc_cache_prepare.sh
#
# Multi-node (matrix + factor too big for one node's RAM): switch to a
# distributed MPI workload -- see the commented submit-dist template in
# submit/run_slepc_eigensolve.sh.

set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() { if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    exit 1
fi

# -------------------------------------------------------------------------
# Cluster resources. CPU cores == MPI ranks (SLEPc is CPU-bound). Big RAM for
# the matrix + the shift-invert (MUMPS) factor, which dominates memory at small
# stride. NO GPU is requested -- this is the CPU-heavy cache-prep job.
# -------------------------------------------------------------------------
CPU=${CPU:-32}
MEM=${MEM:-384G}
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_latest}

# -------------------------------------------------------------------------
# Paths (S3 mounts on the cluster)
# -------------------------------------------------------------------------
S3_EIGENVECTOR_DIR="${S3_EIGENVECTOR_DIR:-/s3/mlibra/mlibra-data/artiom/eigenvectors}"
S3_REFERENCE_FILE="${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}"
S3_ANNOTATION_FILE="${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}"
SRC_PATH="${SRC_PATH:-/myhome/mlibra}"

BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"   # rank 0 builds the graph (CPU faiss) if absent
FACTOR_SOLVER="${FACTOR_SOLVER:-mumps}"     # parallel direct solver for the shift-invert
TARGET="${TARGET:-0.0}"                     # shift-invert target (bottom of the spectrum)
EXP_SUFFIX="${EXP_SUFFIX:-$(date +'%y%m%d-%H%M')}"

# -------------------------------------------------------------------------
# submit_one  <stride> <threshold> <knn_k> <modes> [norm] [cpu]
# Always shift-invert; never requests a GPU.
# -------------------------------------------------------------------------
n_submitted=0
submit_one() {
    local stride=$1 threshold=$2 knn_k=$3 modes=$4
    local norm=${5:-randomwalk} cpu=${6:-$CPU}

    local run_slug="str${stride}_t${threshold}_k${knn_k}_bw1p0_${norm}_nm${modes}_si"
    n_submitted=$((n_submitted + 1))
    local job_name="slepcsi-${EXP_SUFFIX}-$(printf '%03d' "$n_submitted")"

    echo ">>> [$n_submitted] $job_name -> $run_slug  (ranks=$cpu, shift-invert)"
    run_or_echo runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$cpu"   --cpu-core-request   "$cpu" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e REPO="$SRC_PATH" \
        -e NPROC="$cpu" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e BUILD_IF_MISSING="$BUILD_IF_MISSING" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATIONS_FILE="$S3_ANNOTATION_FILE" \
        -e RUN_SLUG="$run_slug" \
        -e STRIDE="$stride" \
        -e THRESHOLD="$threshold" \
        -e KNN_K="$knn_k" \
        -e MODES="$modes" \
        -e NORMALIZATION="$norm" \
        -e SHIFT_INVERT="1" \
        -e TARGET="$TARGET" \
        -e FACTOR_SOLVER="$FACTOR_SOLVER" \
        -- ./slepc/slepc_eigensolve.sh
}

# -------------------------------------------------------------------------
# Eigenvector caches to pre-compute -- edit these.
#            stride thr  k    modes  [norm]__sdfsdf
# -------------------------------------------------------------------------
submit_one      4   5   15    300    randomwalk
submit_one      4   5   15    1300   randomwalk
submit_one      4   5   15    2300   randomwalk
submit_one      4   40   15    300    randomwalk
submit_one      4   40   15    1300   randomwalk
submit_one      4   40   15    2300   randomwalk
submit_one      4   50   15    300    randomwalk
submit_one      4   50   15    1300   randomwalk
submit_one      4   40   15    2300   randomwalk

echo "Submitted $n_submitted shift-invert cache-prep jobs. Suffix: $EXP_SUFFIX"
echo "Eigvecs -> $S3_EIGENVECTOR_DIR/eigvecs/   logs -> $S3_EIGENVECTOR_DIR/slepc_logs/"
