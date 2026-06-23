#!/bin/bash
# Entrypoint: run the MPI-parallel SLEPc eigensolver inside one pod.
#
# Spawns NPROC MPI ranks with mpirun (ranks = CPU cores by default) and computes
# the bottom-MODES eigenpairs of the graph Laplacian, saving them in the
# LaplacianEigensolver cache layout under EIGENVECTOR_DIR/eigvecs.
#
# All knobs are env-overridable so the same script runs locally and on the
# cluster. For MULTI-NODE, launch slepc_eigensolve.py directly under a
# distributed MPI workload (the MPI operator provides mpirun across pods) and
# skip this wrapper -- the Python is partition-agnostic.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- MKL workaround for conda envs ---
# A conda env can ship an inconsistent Intel MKL that dies under mpirun
# ("Cannot load libmkl_def.so: undefined symbol ..."). Preloading libmkl_rt
# forces MKL's single-dynamic-library loader and resolves the kernel-lib skew.
# Applied only when a conda MKL is present, so it's a no-op in the Docker
# (OpenBLAS) image.
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/lib/libmkl_rt.so" ]; then
    export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libmkl_rt.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# --- ranks: default to the cores available in this pod ---
NPROC="${NPROC:-$(nproc)}"

# --- graph + solver properties ---
MODES="${MODES:-300}"
STRIDE="${STRIDE:-4}"
THRESHOLD="${THRESHOLD:-5}"
KNN_K="${KNN_K:-15}"
KNN_METHOD="${KNN_METHOD:-faiss}"
NLIST="${NLIST:-1}"
BANDWIDTH="${BANDWIDTH:-1.0}"
NORMALIZATION="${NORMALIZATION:-randomwalk}"
TEMPLATE="${TEMPLATE:-reference}"
TOL="${TOL:-1e-6}"
NCV="${NCV:--1}"
SHIFT_INVERT="${SHIFT_INVERT:-0}"        # 1 -> shift-invert + MUMPS
TARGET="${TARGET:-0.0}"
FACTOR_SOLVER="${FACTOR_SOLVER:-mumps}"

# --- paths (override to S3 mounts on the cluster) ---
EIGENVECTOR_DIR="${EIGENVECTOR_DIR:-/home/casap/mlibra/output/eigenvectors}"
BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"
REFERENCE_FILE="${REFERENCE_FILE:-/home/casap/mlibra/mlibra_data/reference_image.npy}"
ANNOTATIONS_FILE="${ANNOTATIONS_FILE:-/home/casap/mlibra/mlibra_data/level_15annot.npy}"

# --- self-documenting log dir (eigvecs themselves go to the shared cache) ---
slug() { echo "$1" | sed 's/\./p/g'; }
RUN_SLUG="${RUN_SLUG:-str${STRIDE}_t${THRESHOLD}_k${KNN_K}_${KNN_METHOD}_bw$(slug "$BANDWIDTH")_${NORMALIZATION}_nm${MODES}}"
LOG_DIR="${LOG_DIR:-${EIGENVECTOR_DIR}/slepc_logs/${RUN_SLUG}}"
mkdir -p "$LOG_DIR"

opt_args=()
[ "$BUILD_IF_MISSING" = "1" ] && opt_args+=(--build-if-missing)
[ -n "$REFERENCE_FILE" ]      && opt_args+=(--reference-file "$REFERENCE_FILE")
[ -n "$ANNOTATIONS_FILE" ]    && opt_args+=(--annotations-file "$ANNOTATIONS_FILE")
[ "$SHIFT_INVERT" = "1" ]     && opt_args+=(--shift-invert --target "$TARGET" --factor-solver "$FACTOR_SOLVER")

# Resolve the MPI launcher by ABSOLUTE path. The runai job runs us via
# `gosu appuser bash -c ...` (a non-login shell, see entrypoint.sh), whose PATH
# may not include /usr/bin where openmpi-bin installs mpirun -- so a bare
# `mpirun` can fail with "command not found" even though it's in the image.
# Honor an explicit $MPIRUN, else search PATH, else fall back to /usr/bin.
MPIRUN="${MPIRUN:-$(command -v mpirun || true)}"
[ -z "$MPIRUN" ] && [ -x /usr/bin/mpirun ] && MPIRUN=/usr/bin/mpirun
if [ -z "$MPIRUN" ] || [ ! -x "$MPIRUN" ]; then
    echo "ERROR: mpirun not found (PATH=$PATH). Install openmpi-bin or set MPIRUN=/abs/path/to/mpirun." >&2
    exit 127
fi

echo "[slepc_eigensolve] NPROC=$NPROC -> $RUN_SLUG (shift_invert=$SHIFT_INVERT) mpirun=$MPIRUN"

# Oversubscribe is allowed in case the pod reports fewer slots than cores.
"$MPIRUN" --allow-run-as-root -n "$NPROC" \
    python -u "${REPO}/slepc/slepc_eigensolve.py" \
        --template "$TEMPLATE" \
        --stride "$STRIDE" \
        --threshold "$THRESHOLD" \
        --knn-k "$KNN_K" \
        --knn-method "$KNN_METHOD" \
        --nlist "$NLIST" \
        --bandwidth "$BANDWIDTH" \
        --normalization "$NORMALIZATION" \
        --modes "$MODES" \
        --tol "$TOL" \
        --ncv "$NCV" \
        --eigenvector-dir "$EIGENVECTOR_DIR" \
        "${opt_args[@]}" 2>&1 | tee "$LOG_DIR/slepc.log"

echo ""
echo "Log: $LOG_DIR/slepc.log"
echo "Eigvecs (if converged): $EIGENVECTOR_DIR/eigvecs/"
