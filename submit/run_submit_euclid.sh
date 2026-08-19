#!/usr/bin/env bash
# Submit the EUCLID baseline to runai: fit on each fold's TRAIN split, score the
# held-out mice, and reconstruct + render the curated lipid subset.
#
# What runs on the pod is EUCLID's own `anatomical_interpolation`, lifted out of
# their checkout and executed unmodified (see baselines/euclid_kernel.py). Their
# kernel is single-threaded and costs ~242 s per lipid on the 132x80x114 grid, so
# the only lever is EUCLID_JOBS: one process per lipid. 25-way takes 173 lipids
# from 11.6 h to ~30 min. Ask for CPU == EUCLID_JOBS or the cgroup throttles them.
# Cost scales with how permissive the gate is: measured 242 s/lipid on EUCLID's
# 672-label leaf annotation vs 527 s/lipid on level_15annot, whose root label
# alone covers 57% of tissue -- coarser atlas, fewer candidates rejected, more
# accumulation. Budget ~2x the wall clock for the atlas=own arm.
#
# NO GPU: nothing here touches torch beyond the harness's tensor bookkeeping.
#
# The `w` sweep is the point of running two arms:
#   w=50  EUCLID's default. It is a threshold on THEIR 0-255 rescale of the log
#         intensities, tuned for rendering on uMAIA-normalised data. On this
#         parquet every log intensity is negative, the rescale compresses ~3x,
#         and w=50 lands above the 75th percentile -- only 7.6% of measured
#         voxels survive as donors and held-out corr collapses to ~0.19.
#   w=0   filter off; 90% of donors survive (the missing 10% is
#         normalize_to_255's own bottom-decile masking, which is unavoidable
#         inside their pipeline). This is the fair arm for a corr comparison.
# Both are submitted by default; EXP_NAME tags w!=50 so they never collide.
#
# Outputs per run dir (same layout as every other baseline):
#   metrics.csv                     per-lipid held-out corr/r2/rmse
#   test/{predictions,true_values}.npy
#   volume_sparse/<lipid>_volume_sparse.npy     (RENDER_VOXELS_ONLY=1, default)
#   volume/<lipid>_volume.npy                   (RENDER_VOXELS_ONLY=0, dense ~2.5 GB)
#   renders/<lipid>_multi_panel.png, <lipid>_diagnostics.png
#   euclid_volumes/<lipid>_interpolation_log.npy   EUCLID's own 100um volumes
#
# Preview without submitting:
#   DRY_RUN=1 ./submit/run_submit_euclid.sh
# One fold, dense volumes, faithful arm only:
#   FOLDS_LIST="fold-2" RENDER_VOXELS_ONLY=0 W_LIST=50 ./submit/run_submit_euclid.sh
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
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_all_latest}
# One core per concurrent lipid. Their kernel is a single-threaded numba loop, so
# extra threads per worker buy nothing -- width is what matters.
EUCLID_JOBS=${EUCLID_JOBS:-16}
CPU=${CPU:-$EUCLID_JOBS}
# ~400 MB per worker (each holds its own reference/annotation/working volumes)
# plus ~3 GB in the parent for the 173-lipid parquet. 25 workers -> ~13 GB.
MEM=${MEM:-80G}
# CPU-only: nothing here touches CUDA. GPU=0 omits the runai gpu flags entirely
# (a portion request of 0 is not a valid value); set GPU=0.2 to attach one anyway.
GPU=${GPU:-0}
GPU_ARGS=()
if [ "$GPU" != "0" ]; then
    GPU_ARGS=(--gpu-request-type portion --gpu-portion-request "$GPU")
fi

# --- Sweep: atlas x fold x w x norm  (2 x 8 x 2 x 1 = 32 jobs by default) ---
# ATLAS_LIST picks which volumes drive their pipeline:
#   euclid -> the two 100um .npy from the EUCLID repo, staged at EUCLID_DATA_DIR.
#             Their annotation IS the Allen 672-label leaf volume (verified
#             identical to BrainGlobe allen_mouse_25um[::4] and to
#             download_bg_atlas.py --max-depth 9).
#   own    -> this repo's REFERENCE_FILE + ANNOTATION_FILE, subsampled [::4].
#             Note level_15annot's root label alone covers 57% of tissue, so the
#             structure gate is largely inert over most of the brain -- this arm
#             is close to an ungated Shepard smoother, by construction.
ATLAS_LIST=(${ATLAS_LIST:-euclid own})
W_LIST=(${W_LIST:-0 50})
# NORM_LIST picks the intensity scale their normalize_to_255 sees:
#   none   -> as-is. Our intensities are ~1e-4 so every log(x) is negative, the
#             NaN-fill 0 becomes the array MAX, and the rescale tops out at
#             27-155 instead of 255 (per lipid). w=50 then deletes 16-100% of
#             the donors depending on the lipid: ~25% of lipids come back a flat
#             constant and 85/173 score bit-identically across two different
#             atlases, i.e. the structure gate has nothing left to gate.
#   max    -> per-lipid divide by the TRAIN max, so each lipid reaches 255 and w
#             means one thing for all of them. At w=0 this is a no-op by
#             construction (a positive per-lipid multiplier, which the harness's
#             per-lipid affine absorbs), so pairing it with w=0 is the control.
#   global -> one constant making log(x) > 0. Reaches their assumed scale but
#             moves the data off the 0 their `reference < 4` sentinel sits on,
#             so it changes the w=0 arm too and makes w=50 fully inert.
NORM_LIST=(${NORM_LIST:-none})
FOLDS_LIST=(${FOLDS_LIST:-fold-1 fold-2 fold-3 fold-4 fold-5 fold-6 fold-7 fold-8})

# --- Reconstruction ---------------------------------------------------------
# 1 = reconstruct only the voxels the composite render reads (~6.1M of 33.6M,
# writes volume_sparse/). 0 = dense whole-brain volumes (~2.5 GB per run).
RENDER_VOXELS_ONLY=${RENDER_VOXELS_ONLY:-1}

# --- Paths (S3 mounts on the cluster) --------------------------------------
S3_DATA_PATH=${S3_DATA_PATH:-/s3/mlibra/mlibra-data/maldi/}
S3_OUTPUT_DIR=${S3_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/euclid_cv}
S3_MALDI_FILE=${S3_MALDI_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet}
S3_AVAILABLE_LIPIDS_FILE=${S3_AVAILABLE_LIPIDS_FILE:-/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy}
# REFERENCE_FILE always sets the reconstruction voxel set and the render
# backdrop; for ATLAS=own it ALSO drives EUCLID's `reference < 4` background
# mask. ANNOTATION_FILE is used only by ATLAS=own, as the structure gate.
# Defaults are this repo's LGP-comparable pair; set S3_REFERENCE_FILE=
# .../ccf_bg_reference.npy if you would rather reconstruct on the BrainGlobe grid.
S3_REFERENCE_FILE=${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}
S3_ANNOTATION_FILE=${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}
# EUCLID's own two 100um volumes, staged outside the code checkout.
S3_EUCLID_DATA_DIR=${S3_EUCLID_DATA_DIR:-/s3/mlibra/mlibra-data/artiom/euclid_data}
SRC_PATH=${SRC_PATH:-/myhome/mlibra}
SPLITS_DIR=${SPLITS_DIR:-/myhome/mlibra/maldi/data/splits}
# Curated 5-lipid subset that gets reconstructed + rendered. Lives in the repo
# mount, not S3.
RECON_LIPIDS_FILE=${RECONSTRUCTION_LIPIDS_FILE:-$SRC_PATH/maldi/data/lipid_subset.txt}
# EUCLID checkout. run_baseline.sh git-clones it here if absent (their two 100um
# .npy volumes are committed in that repo, so the clone is self-sufficient).
# Point it at a WRITABLE path -- not the read-only repo mount.
EUCLID_REPO=${EUCLID_REPO:-/mydata/mlibra/artiom/euclid}

EXP_SUFFIX=${EXP_SUFFIX:-$(date +'%y%m%d-%H-%M')}

n_submitted=0
submit_euclid() {
    local job_name=$1 fold_upper=$2 slices=$3 w=$4 atlas=$5 norm=$6
    n_submitted=$((n_submitted + 1))
    echo ">>> [$n_submitted] $job_name  (fold=$fold_upper atlas=$atlas w=$w norm=$norm jobs=$EUCLID_JOBS)"
    run_or_echo runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        "${GPU_ARGS[@]}" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e MODEL="euclid" \
        -e EXP_PREFIX="$fold_upper" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slices" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e RECONSTRUCTION_LIPIDS_FILE="$RECON_LIPIDS_FILE" \
        -e TEMPLATE_NAME="reference" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e EUCLID_ATLAS="$atlas" \
        -e EUCLID_DATA_DIR="$S3_EUCLID_DATA_DIR" \
        -e SRC_PATH="$SRC_PATH" \
        -e EUCLID_REPO="$EUCLID_REPO" \
        -e EUCLID_W="$w" \
        -e EUCLID_NORM="$norm" \
        -e EUCLID_JOBS="$EUCLID_JOBS" \
        -e RENDER_VOXELS_ONLY="$RENDER_VOXELS_ONLY" \
        -- ./local_run/run_baseline.sh
}

echo "image   : $IMAGE"
echo "output  : $S3_OUTPUT_DIR"
echo "folds   : ${FOLDS_LIST[*]}"
echo "atlas   : ${ATLAS_LIST[*]}   (euclid=${S3_EUCLID_DATA_DIR}, own=${S3_ANNOTATION_FILE})"
echo "w       : ${W_LIST[*]}"
echo "norm    : ${NORM_LIST[*]}"
echo "workers : $EUCLID_JOBS (cpu=$CPU mem=$MEM gpu=$GPU)"
echo "renders : $RECON_LIPIDS_FILE  (render_voxels_only=$RENDER_VOXELS_ONLY)"
echo "euclid  : $EUCLID_REPO (cloned on the pod if absent)"
echo

for FOLD in "${FOLDS_LIST[@]}"; do
    SLICES_DATASET_FILE="${SPLITS_DIR}/${FOLD//-/_}.json"
    FOLD_UPPER=${FOLD^^}
    fold_slug="${FOLD//fold-/f}"
    for atlas in "${ATLAS_LIST[@]}"; do
        for w in "${W_LIST[@]}"; do
            for norm in "${NORM_LIST[@]}"; do
                # runai job names must be lowercase and DNS-safe.
                nslug=""; [ "$norm" != "none" ] && nslug="-n${norm}"
                submit_euclid "euclid-${atlas}-w${w}${nslug}-${fold_slug}-${EXP_SUFFIX}" \
                              "$FOLD_UPPER" "$SLICES_DATASET_FILE" "$w" "$atlas" "$norm"
            done
        done
    done
done

echo
echo "Submitted $n_submitted jobs. Suffix: $EXP_SUFFIX  Output: $S3_OUTPUT_DIR"
# Wall clock = ceil(n_lipids / EUCLID_JOBS) waves x seconds-per-lipid. The
# per-lipid cost depends on how permissive the gate is: measured 242 s on
# EUCLID's 672-label leaf annotation vs 527 s on level_15annot, whose root label
# alone covers 57% of tissue (coarser atlas -> fewer candidates rejected -> more
# accumulation). Computed rather than hard-coded so it tracks EUCLID_JOBS.
est() { awk -v n=173 -v j="$EUCLID_JOBS" -v s="$1" 'BEGIN{printf "%.0f", ((int((n+j-1)/j))*s)/60}'; }
echo "Each job: interpolation (173 lipids / $EUCLID_JOBS procs) + reconstruction."
echo "  atlas=euclid ~$(est 242) min/job   (leaf gate: ~1.6% of the donor sphere passes,"
echo "                              so most candidates are rejected cheaply)"
echo "  atlas=own    ~$(est 527) min/job   (level_15annot's root label covers 57% of tissue,"
echo "                              so far more donors pass and get accumulated)"
echo "Run dirs: <FOLD>-BASELINES-EUCLID-256[-w<W>][-norm<NORM>]-<atlas>10   (all distinct)"
echo "            -w0                            intensity filter off (fair corr arm)"
echo "            (no -w)                        w=50, EUCLID's default"
echo "            -euclidatlas                   their 672-label leaf volumes"
echo "            -<annot_stem>-<ref_stem>       this repo's pair, named by file"
