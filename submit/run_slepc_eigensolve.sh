#!/usr/bin/env bash
# Submit MPI-parallel SLEPc eigenvector computations to runai.
#
# Each submit_one launches ONE job that runs `mpirun -n <CPU cores> python
# slepc_eigensolve.py` inside a single pod -- PETSc/SLEPc distribute the
# Laplacian and eigenvectors across the ranks. The result is saved in the
# shared eigvec cache (LaplacianEigensolver layout), so the GP code reuses it.
#
# SLEPc/PETSc/MUMPS are CPU codes, so the scaling knob is CPU CORES (= MPI
# ranks), not GPUs. We request 1 GPU only for the rank-0 faiss build when a
# graph isn't cached yet (set GPU=0 and pre-build to drop it).
#
# Multi-node: if one node's RAM can't hold the matrix + factor, switch to a
# distributed MPI workload (see the commented submit_dist template at the
# bottom) so ranks span pods. Preview without submitting:  DRY_RUN=1 ./...

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
# Cluster resources. CPU cores == MPI ranks (SLEPc is CPU). Big RAM for the
# matrix + (shift-invert) factor. 1 GPU only for the rank-0 faiss build.
# -------------------------------------------------------------------------
CPU=${CPU:-16}
MEM=${MEM:-256G}
GPU=${GPU:-1}
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_latest}

# -------------------------------------------------------------------------
# Paths (S3 mounts on the cluster)
# -------------------------------------------------------------------------
S3_EIGENVECTOR_DIR="${S3_EIGENVECTOR_DIR:-/s3/mlibra/mlibra-data/artiom/eigenvectors}"
S3_REFERENCE_FILE="${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}"
S3_ANNOTATION_FILE="${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}"
SRC_PATH="${SRC_PATH:-/myhome/mlibra}"

BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"
EXP_SUFFIX="${EXP_SUFFIX:-$(date +'%y%m%d-%H%M')}"

# BLAS/OpenMP threads per MPI rank. 1 = pure MPI (the right mapping when ranks
# == cores: NPROC ranks x 1 thread fills the cores without oversubscription).
# Raise only for a deliberate hybrid run (fewer ranks x more threads, to speed
# the dense root of the MUMPS factorization). Forwarded to all three BLAS knobs.
OMP_THREADS="${OMP_THREADS:-1}"

slug() { echo "$1" | sed 's/\./p/g'; }

# -------------------------------------------------------------------------
# submit_one  <stride> <threshold> <knn_k> <modes> [norm] [shift_invert] [cpu]
# -------------------------------------------------------------------------
n_submitted=0
submit_one() {
    local stride=$1 threshold=$2 knn_k=$3 modes=$4
    local norm=${5:-randomwalk} shift_invert=${6:-0} cpu=${7:-$CPU}

    local run_slug="str${stride}_t${threshold}_k${knn_k}_bw1p0_${norm}_nm${modes}"
    n_submitted=$((n_submitted + 1))
    local job_name="slepc-${EXP_SUFFIX}-$(printf '%03d' "$n_submitted")"

    echo ">>> [$n_submitted] $job_name -> $run_slug  (ranks=$cpu, si=$shift_invert)"
    run_or_echo runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$cpu"   --cpu-core-request   "$cpu" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e REPO="$SRC_PATH" \
        -e NPROC="$cpu" \
        -e OMP_NUM_THREADS="$OMP_THREADS" \
        -e OPENBLAS_NUM_THREADS="$OMP_THREADS" \
        -e MKL_NUM_THREADS="$OMP_THREADS" \
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
        -e SHIFT_INVERT="$shift_invert" \
        -- ./slepc/slepc_eigensolve.sh
}

# -------------------------------------------------------------------------
# The large-scale eigenvector computations you care about -- edit these.
#            stride thr  k    modes  [norm]      [shift_invert]
# -------------------------------------------------------------------------
submit_one      1   5   15    300    randomwalk  0     # Krylov-Schur, matrix-free
submit_one      2   5   15    1300   randomwalk  0
submit_one      2   5   15    1300   randomwalk  1     # shift-invert + MUMPS
submit_one      1   5   15    300    randomwalk  1

echo "Submitted $n_submitted jobs. Suffix: $EXP_SUFFIX"
echo "Eigvecs -> $S3_EIGENVECTOR_DIR/eigvecs/   logs -> $S3_EIGENVECTOR_DIR/slepc_logs/"

# -------------------------------------------------------------------------
# MULTI-NODE template (uncomment + adapt to your cluster's MPI workload).
# A distributed MPIJob spreads ranks across pods; the operator runs mpirun, so
# the worker command is the bare python (no inner mpirun wrapper). Example:
#
#   runai training submit-dist mpi "slepc-dist-001" \
#       --workers 4 --slots-per-worker "$CPU" \
#       -i "$IMAGE" \
#       --cpu-core-request "$CPU" --cpu-memory-request "$MEM" \
#       -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" -e STRIDE=1 -e MODES=300 ... \
#       -- python /myhome/mlibra/slepc/slepc_eigensolve.py \
#            --stride 1 --modes 300 --eigenvector-dir "$S3_EIGENVECTOR_DIR" ...
# -------------------------------------------------------------------------
