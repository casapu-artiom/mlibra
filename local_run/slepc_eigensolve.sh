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
# This launcher lives in local_run/, so REPO above is the REPO ROOT; the solver
# itself lives under manifold/slepc/.
SLEPC_DIR="${SLEPC_DIR:-${REPO}/manifold/slepc}"

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

# --- threads per rank: pin to 1 (pure MPI) ---
# MUMPS/SLEPc parallelize via the MPI ranks. If the threaded BLAS underneath
# *also* spawns a thread per core, then NPROC ranks x (cores) threads massively
# oversubscribes the cores -- they thrash and the factorization crawls (seen as
# >100%/rank CPU and load average >> ncores). With ranks ~= cores, one BLAS
# thread per rank is the right mapping. Override (e.g. fewer ranks x more
# threads) only if you deliberately tune hybrid MPI+OpenMP for the dense root.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# --- graph + solver properties ---
MODES="${MODES:-300}"
STRIDE="${STRIDE:-4}"
THRESHOLD="${THRESHOLD:-5}"
KNN_K="${KNN_K:-15}"
KNN_METHOD="${KNN_METHOD:-faiss}"        # faiss | anatomical_atlas | faiss_atlas_weighted
CROSS_REGION_INFLATION="${CROSS_REGION_INFLATION:-10.0}"  # faiss_atlas_weighted weight
# faiss_atlas_weighted root handling + label denoise + hard prune (match the
# per-lipid training pipeline). Defaults reproduce the legacy slepc behaviour
# (root=cross, no denoise/prune) so existing cache keys are unchanged. Set
# ROOT_HANDLING=dissolve to pre-bake a cache for a DEFAULT training run.
ROOT_HANDLING="${ROOT_HANDLING:-cross}"          # dissolve | ignore | cross
DENOISE_LABELS="${DENOISE_LABELS:-0}"            # majority-vote passes (0 = off)
PRUNE_CROSS_REGION="${PRUNE_CROSS_REGION:-0.0}"  # hard-prune fraction (0 = off)
# FAISS IVF sizing -- int or 'sqrt'. Matches run_manifold.sh / run_manifold_batch.sh
# (and run_slepc_cache_prepare.sh): nlist='sqrt' = round(sqrt(N)), resolved with the
# SAME N as the training runs, so this solve is a drop-in cache hit for them. nprobe
# is a FIXED 8 -- in 3D recall saturates there regardless of N, and nprobe is never
# cache-keyed, so all producers must agree on it to bake the same graph.
# NOTE: nlist IS cache-keyed when !=1, so this changes the key vs the old NLIST=1
# default; set NLIST=1 NPROBE=1 to reproduce the legacy exact-flat behaviour.
NLIST="${NLIST:-sqrt}"                    # FAISS IVF nlist: int or 'sqrt'
NPROBE="${NPROBE:-8}"                     # FAISS IVF nprobe: int or 'sqrt'
BANDWIDTH="${BANDWIDTH:-1.0}"
NORMALIZATION="${NORMALIZATION:-randomwalk}"
TEMPLATE="${TEMPLATE:-reference}"
TOL="${TOL:-1e-6}"
NCV="${NCV:--1}"
MPD="${MPD:--1}"                         # max projected dim; cap for large MODES
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

# Pin the interpreter to an ABSOLUTE path too. mpirun (OpenMPI) prepends its own
# bindir to the children's PATH, which can shadow /opt/venv/bin/python with a
# system/conda python that lacks petsc4py/slepc4py -- ranks then die with
# "No module named 'petsc4py'" even though the venv has it. Resolving `python`
# here (post venv-activate) and passing it by full path makes every rank use the
# venv interpreter. -x forwards PATH + LD_LIBRARY_PATH so the dynamic loader
# still finds libpetsc/libslepc (+ MUMPS) at import time.
PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found (PATH=$PATH). Activate the venv or set PYTHON=/abs/path." >&2
    exit 127
fi

echo "[slepc_eigensolve] NPROC=$NPROC -> $RUN_SLUG (shift_invert=$SHIFT_INVERT) mpirun=$MPIRUN python=$PYTHON"

# Oversubscribe is allowed in case the pod reports fewer slots than cores.
"$MPIRUN" --allow-run-as-root -n "$NPROC" \
    -x PATH -x LD_LIBRARY_PATH \
    "$PYTHON" -u "${SLEPC_DIR}/slepc_eigensolve.py" \
        --template "$TEMPLATE" \
        --stride "$STRIDE" \
        --threshold "$THRESHOLD" \
        --knn-k "$KNN_K" \
        --knn-method "$KNN_METHOD" \
        --cross-region-inflation "$CROSS_REGION_INFLATION" \
        --root-handling "$ROOT_HANDLING" \
        --denoise-labels "$DENOISE_LABELS" \
        --prune-cross-region "$PRUNE_CROSS_REGION" \
        --nlist "$NLIST" \
        --nprobe "$NPROBE" \
        --bandwidth "$BANDWIDTH" \
        --normalization "$NORMALIZATION" \
        --modes "$MODES" \
        --tol "$TOL" \
        --ncv "$NCV" \
        --mpd "$MPD" \
        --eigenvector-dir "$EIGENVECTOR_DIR" \
        "${opt_args[@]}" 2>&1 | tee "$LOG_DIR/slepc.log"

echo ""
echo "Log: $LOG_DIR/slepc.log"
echo "Eigvecs (if converged): $EIGENVECTOR_DIR/eigvecs/"
