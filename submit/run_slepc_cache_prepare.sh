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
CPU=${CPU:-16}
MEM=${MEM:-144G}
IMAGE=${IMAGE:-artiomartiom/sdsc:withslepc}

# -------------------------------------------------------------------------
# Paths (S3 mounts on the cluster)
# -------------------------------------------------------------------------
S3_EIGENVECTOR_DIR="${S3_EIGENVECTOR_DIR:-/s3/mlibra/mlibra-data/artiom/eigenvectors}"
S3_REFERENCE_FILE="${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}"
# Atlas level convenience: ATLAS_LEVEL=5 or 15 selects level_${ATLAS_LEVEL}annot.npy
# under the S3 data dir. Defaults to level 15 (the historical default, which keeps
# its original un-suffixed eigvec cache keys). Override S3_ANNOTATION_FILE directly
# to use any other volume. The atlas file is only consumed by the atlas KNN methods
# (anatomical_atlas / faiss_atlas_weighted); plain faiss ignores it.
S3_DATA_DIR="${S3_DATA_DIR:-/s3/mlibra/mlibra-data}"
ATLAS_LEVEL="${ATLAS_LEVEL:-15}"
S3_ANNOTATION_FILE="${S3_ANNOTATION_FILE:-${S3_DATA_DIR}/level_${ATLAS_LEVEL}annot.npy}"
SRC_PATH="${SRC_PATH:-/myhome/mlibra}"
BANDWIDTH="${BANDWIDTH:-0.1}"

# faiss_atlas_weighted root handling + label denoise + hard prune -- applied to
# EVERY faiss_atlas_weighted job in this batch (ignored by faiss / anatomical_
# atlas). They mirror the per-lipid training pipeline so the eigvec cache is a
# drop-in hit. Defaults reproduce the legacy behaviour (root=cross, no denoise/
# prune) -> existing atlas_x<infl> cache keys are unchanged. To pre-bake caches
# for the current TRAINING default set ROOT_HANDLING=dissolve (and e.g.
# DENOISE_LABELS=3 PRUNE_CROSS_REGION=0.95 to match a pruned run). Sweep them by
# re-running the batch with different values.
ROOT_HANDLING="${ROOT_HANDLING:-cross}"          # dissolve | ignore | cross
DENOISE_LABELS="${DENOISE_LABELS:-0}"            # majority-vote passes (0 = off)
PRUNE_CROSS_REGION="${PRUNE_CROSS_REGION:-0.0}"  # hard-prune fraction (0 = off)

BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"   # rank 0 builds the graph (CPU faiss) if absent
FACTOR_SOLVER="${FACTOR_SOLVER:-mumps}"     # parallel direct solver for the shift-invert
TARGET="${TARGET:-0.0}"                     # shift-invert target (bottom of the spectrum)
# Spectral transform. 1 (default) = SHIFT-INVERT + MUMPS: fast convergence at the
# clustered bottom of the spectrum, but the direct factor DOMINATES RAM (this is
# why MEM is set so high) and its fill-in scales badly at small stride. 0 =
# matrix-free Krylov-Schur SR: NO factorization, so memory collapses to the
# sparse Laplacian + the Krylov basis -- far less RAM, and it lets you push MODES
# much higher, at the cost of SLOWER convergence (bare Lanczos on the clustered
# low spectrum). With SHIFT_INVERT=0 you can drop MEM sharply and lean on MPD to
# cap the basis; TARGET / FACTOR_SOLVER are then ignored. This is the "more modes,
# slower, less RAM" lever.
SHIFT_INVERT="${SHIFT_INVERT:-1}"
# Krylov convergence tolerance. Shift-invert converges tightly and cheaply, so
# 1e-4 is plenty; the matrix-free path (SHIFT_INVERT=0) often needs a looser tol
# to converge in reasonable time when MODES is large.
TOL="${TOL:-1e-4}"
# KNN graph method (global; one method per run). faiss_atlas_weighted builds the
# plain-faiss graph then inflates cross-region edges by the PER-JOB inflation arg
# to submit_one -- matches the lgp per-lipid runs, so the eigvec cache (keyed
# weighting=atlas_x<infl>) is a drop-in hit. The inflation arg is ignored for
# faiss / anatomical_atlas (pass 0, the default).
KNN_METHOD="${KNN_METHOD:-faiss}"           # faiss | anatomical_atlas | faiss_atlas_weighted
# FAISS IVF sizing -- int or 'sqrt'. nlist='sqrt' = round(sqrt(N)), resolved with
# the SAME N as the per-lipid runs and keyed only when nlist!=1, so this cache is
# a drop-in hit for the per-lipid pipeline (which also defaults to N_LIST=sqrt).
# nprobe is a FIXED 8: in 3D recall saturates at nprobe~8 regardless of N, and
# nprobe is never cache-keyed -- so all producers must agree on it to bake the
# same graph. Use N_LIST=1 for the exact flat index (the old behaviour).
N_LIST="${N_LIST:-sqrt}"
N_PROBE="${N_PROBE:-8}"
# Max projected dimension. Caps the Krylov working subspace so per-restart
# orthogonalization is O(MPD^2*N) instead of O(ncv^2*N) -- the lever that keeps
# the iteration phase from blowing up when MODES is large. <=0 = SLEPc default
# (~2*modes). For modes ~2300 try MPD=512; lower = cheaper restarts but more of
# them (and too low risks non-convergence -> the run fails + deletes the cache).
MPD="${MPD:-256}"
EXP_SUFFIX="${EXP_SUFFIX:-$(date +'%y%m%d-%H%M')}"

# BLAS/OpenMP threads per MPI rank. 1 = pure MPI (the right mapping when ranks
# == cores: NPROC ranks x 1 thread fills the cores without oversubscription).
# Raise only for a deliberate hybrid run (fewer ranks x more threads, to speed
# the dense root of the MUMPS factorization). Forwarded to all three BLAS knobs.
OMP_THREADS="${OMP_THREADS:-1}"

# MPI ranks (= NPROC in the pod), DECOUPLED from the core request (CPU). Default
# one rank per core (pure MPI). Set RANKS=1 with OMP_THREADS=$CPU to fall back to
# SERIAL MUMPS + threaded BLAS -- the proven path when the DISTRIBUTED (multi-
# rank) MUMPS factorization hangs/blows up on dense graphs (e.g. t5). Keep
# RANKS x OMP_THREADS ~= CPU so the cores are filled without oversubscription:
#   default fast path : (leave unset)            -> RANKS=$CPU, OMP_THREADS=1
#   serial fallback   : RANKS=1 OMP_THREADS=$CPU -> 1 rank, $CPU BLAS threads
RANKS="${RANKS:-$CPU}"

# -------------------------------------------------------------------------
# submit_one  <stride> <threshold> <knn_k> <modes> [norm] [knn_method] [inflation] [nlist] [cpu]
#   knn_method - faiss | anatomical_atlas | faiss_atlas_weighted
#                (default = global $KNN_METHOD, itself defaulting to faiss).
#   inflation  - cross_region_inflation for faiss_atlas_weighted; 0 (default)
#                for the non-weighted methods, which ignore it.
#   nlist      - FAISS IVF nlist (int or 'sqrt'); default = global $N_LIST.
#                Keyed in the eigvec cache only when != 1 (the exact flat index).
# Always shift-invert; no GPU.
# -------------------------------------------------------------------------
n_submitted=0
submit_one() {
    local stride=$1 threshold=$2 knn_k=$3 modes=$4
    local norm=${5:-randomwalk} knn_method=${6:-${KNN_METHOD:-faiss}}
    local inflation=${7:-0} nlist=${8:-$N_LIST} cpu=${9:-$CPU}

    # Guard: weighting only makes sense with inflation > 1 (it multiplies the
    # cross-region squared distance to DOWN-weight those edges). 0 would instead
    # zero the distance -> maximal cross-region coupling -- almost certainly a
    # forgotten arg, so fail fast rather than cache a nonsensical graph.
    if [ "$knn_method" = "faiss_atlas_weighted" ] && awk "BEGIN{exit !($inflation<=0)}"; then
        echo "ERROR: faiss_atlas_weighted needs an inflation arg > 0 (got $inflation)." \
             "Pass it as the 7th submit_one arg, e.g. submit_one 4 5 15 2300 randomwalk faiss_atlas_weighted 50" >&2
        exit 1
    fi

    # Method/weighting tag so log dirs + job slugs don't collide across methods.
    local mtag=""
    if [ "$knn_method" = "faiss_atlas_weighted" ]; then
        mtag="_faw${inflation/./p}"
    elif [ "$knn_method" != "faiss" ]; then
        mtag="_${knn_method}"
    fi
    # nlist tag so two calls differing only in nlist don't share a log dir/slug.
    # Untagged for the 'sqrt' default to keep existing slugs unchanged.
    local ltag=""
    if [ "$nlist" != "sqrt" ]; then
        ltag="_nl${nlist}"
    fi
    # Atlas-level tag so level_5 vs level_15 log dirs don't collide (the atlas
    # file only matters for the atlas methods). Untagged for the legacy
    # level_15annot default -> existing slugs stay unchanged.
    local atag="" atlas_stem
    atlas_stem=$(basename "$S3_ANNOTATION_FILE" .npy)
    case "$knn_method" in
        faiss_atlas_weighted|anatomical_atlas)
            [ "$atlas_stem" != "level_15annot" ] && atag="_${atlas_stem}"
            ;;
    esac
    # Transform tag (log dir / job slug only -- NOT part of the eigvec cache key,
    # which is derived from graph params in slepc_eigensolve.py). _si = shift-invert,
    # _ks = matrix-free Krylov-Schur SR, so the two paths' logs don't collide.
    local xform_tag xform_desc
    if [ "$SHIFT_INVERT" = "1" ]; then
        xform_tag="_si"; xform_desc="shift-invert"
    else
        xform_tag="_ks"; xform_desc="krylov-schur SR (matrix-free, low-RAM)"
    fi
    # Root-handling / denoise / prune tag (faiss_atlas_weighted only) so log dirs
    # for different priors don't collide. Untagged at the legacy defaults
    # (root=cross, no denoise/prune) to keep existing slugs unchanged.
    local rptag=""
    if [ "$knn_method" = "faiss_atlas_weighted" ]; then
        [ "$ROOT_HANDLING" != "cross" ] && rptag="${rptag}_rh${ROOT_HANDLING}"
        awk "BEGIN{exit !($DENOISE_LABELS>0)}"     && rptag="${rptag}_dn${DENOISE_LABELS}"
        awk "BEGIN{exit !($PRUNE_CROSS_REGION>0)}" && rptag="${rptag}_pr${PRUNE_CROSS_REGION/./p}"
    fi
    local run_slug="str${stride}_t${threshold}_k${knn_k}_bw${BANDWIDTH}p0_${norm}_nm${modes}${mtag}${atag}${ltag}${rptag}${xform_tag}"
    n_submitted=$((n_submitted + 1))
    local job_name="slepcsi-${EXP_SUFFIX}-$(printf '%03d' "$n_submitted")"

    echo ">>> [$n_submitted] $job_name -> $run_slug  (ranks=$RANKS, cores=$cpu, omp=$OMP_THREADS, $xform_desc)"
    run_or_echo runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$cpu"   --cpu-core-request   "$cpu" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e REPO="$SRC_PATH" \
        -e NPROC="$RANKS" \
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
        -e KNN_METHOD="$knn_method" \
        -e NLIST="$nlist" \
        -e NPROBE="$N_PROBE" \
        -e CROSS_REGION_INFLATION="$inflation" \
        -e ROOT_HANDLING="$ROOT_HANDLING" \
        -e DENOISE_LABELS="$DENOISE_LABELS" \
        -e PRUNE_CROSS_REGION="$PRUNE_CROSS_REGION" \
        -e MODES="$modes" \
        -e NORMALIZATION="$norm" \
        -e SHIFT_INVERT="$SHIFT_INVERT" \
        -e TARGET="$TARGET" \
        -e MPD="$MPD" \
        -e BANDWIDTH="$BANDWIDTH" \
        -e TOL="$TOL" \
        -e FACTOR_SOLVER="$FACTOR_SOLVER" \
        -- ./local_run/slepc_eigensolve.sh
}

# -------------------------------------------------------------------------
# Eigenvector caches to pre-compute -- edit these.
#
#   submit_one  <stride> <threshold> <knn_k> <modes> [norm] [knn_method] [inflation]
#
# norm defaults to randomwalk, knn_method to faiss, inflation to 0 (unused unless
# knn_method=faiss_atlas_weighted). Examples below cover every method.
# -------------------------------------------------------------------------

# # ---- plain faiss (Euclidean kNN); inflation/method args omitted -----------
# submit_one      4   5   15    2300   randomwalk
# submit_one      4   5   15    4300   randomwalk

# ---- faiss_atlas_weighted: sweep the cross-region inflation weight ---------
# (needs --reference-file + --annotations-file, both already passed). Each
# inflation gets its own weighting=atlas_x<infl> eigvec cache -> a drop-in hit
# for the matching per-lipid GP run.
# submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  10
# submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  50
# submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  100

# ---- anatomical_atlas (per-region kNN + grid-adjacent cross-region edges) --
# submit_one    4   5   15    2300   randomwalk  anatomical_atlas

# ---- other thresholds / mode counts (plain faiss) -------------------------
# submit_one    4   40  15    1300   randomwalk
# submit_one    4   50  15    1300   randomwalk

# ---- mixing methods in one batch is fine (method is per-call) -------------
submit_one    4   5   15    2300   randomwalk  faiss 0
submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  10
submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  50
submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  100
submit_one    4   40   15    2300   randomwalk  faiss 0
submit_one    4   40   15    2300   randomwalk  faiss_atlas_weighted  10
submit_one    4   40   15    2300   randomwalk  faiss_atlas_weighted  50
submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  100
submit_one    4   50   15    2300   randomwalk  faiss 0
submit_one    4   50   15    2300   randomwalk  faiss_atlas_weighted  10
submit_one    4   50   15    2300   randomwalk  faiss_atlas_weighted  50
submit_one    4   5   15    2300   randomwalk  faiss_atlas_weighted  100
# submit_one    8   5   15    6300   randomwalk  faiss 0
# submit_one    8   5   15    6300   randomwalk  faiss_atlas_weighted  10
# submit_one    8   5   15    6300   randomwalk  faiss_atlas_weighted  50
# submit_one    8   40   15    6300   randomwalk  faiss 0
# submit_one    8   40   15    6300   randomwalk  faiss_atlas_weighted  10
# submit_one    8   40   15    6300   randomwalk  faiss_atlas_weighted  50
# submit_one    8   50   15    6300   randomwalk  faiss 0
# submit_one    8   50   15    6300   randomwalk  faiss_atlas_weighted  10
# submit_one    8   50   15    6300   randomwalk  faiss_atlas_weighted  50

# submit_one    2   5   15    300   randomwalk  faiss 0
# submit_one    2   5   15    300   randomwalk  faiss_atlas_weighted  10
# submit_one    2   5   15    300   randomwalk  faiss_atlas_weighted  50
# submit_one    2   40   15    300   randomwalk  faiss 0
# submit_one    2   40   15    300   randomwalk  faiss_atlas_weighted  10
# submit_one    2   40   15    300   randomwalk  faiss_atlas_weighted  50
# submit_one    2   50   15    300   randomwalk  faiss 0
# submit_one    2   50   15    300   randomwalk  faiss_atlas_weighted  10
# submit_one    2   50   15    300   randomwalk  faiss_atlas_weighted  50

echo "Submitted $n_submitted shift-invert cache-prep jobs. Suffix: $EXP_SUFFIX"
echo "Eigvecs -> $S3_EIGENVECTOR_DIR/eigvecs/   logs -> $S3_EIGENVECTOR_DIR/slepc_logs/"
