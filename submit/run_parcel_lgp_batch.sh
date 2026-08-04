#!/usr/bin/env bash
# =============================================================================
# LGP + reference-only parcel factor — runai batch submitter.
#
# Mirrors run_lgp_batch.sh but submits ./parcelgp/run_latent.sh, which runs
# parcelgp/lgp_parcel_experiment.py = maldi/lgp_experiment.py with the parcel
# factor inserted into the kernel. Training, checkpoint resume, normalization,
# prediction, whole-brain reconstruction and rendering are all the EXISTING
# pipeline, so results are directly comparable to the run_lgp_batch.sh runs.
#
#   DRY_RUN=1 ./submit/run_parcel_lgp_batch.sh   # print commands only
#   ./submit/run_parcel_lgp_batch.sh             # actually submit
#
# This submits PARCEL runs only. The baseline (no parcel factor) is an ordinary
# LGP run -- take it from ./submit/run_lgp_batch.sh with matching hyperparameters.
# parcelgp.compare pairs the two per lipid, so they only need the same lipid set,
# fold and seed.
#
# Compare arms afterwards with the PAIRED per-lipid test (~60x tighter than
# comparing run means, because between-lipid variance cancels):
#   python -m parcelgp.compare --baseline base=DIR --run parcel=DIR --metric corr
# =============================================================================
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() {
    if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi
}

# ---- load secrets ---------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

# ---- resource limits ------------------------------------------------------
MEM=${MEM:-24G}
CPU=${CPU:-2}
GPU=${GPU:-0.2}

# ---- S3-mounted paths inside the container --------------------------------
# Same roots as run_lgp_batch.sh / run_submit_per_lipid.sh; only the output dir
# is parcel-specific so these runs don't mix with the plain LGP sweep.
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/parcel_lgp_cv}"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
ATLAS_LEVEL=${ATLAS_LEVEL:-15}
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_${ATLAS_LEVEL}annot.npy"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
SRC_PATH="/myhome/mlibra"
# Parcel fields are shared across every job: parcelgp.build verifies a cached
# field's recorded build parameters before reuse, and writes atomically (temp +
# rename), so jobs racing to build the same field cannot corrupt it. A stride-4
# build is ~15s; prebuild anything finer (command printed at the end).
S3_PARCEL_DIR="${S3_PARCEL_DIR:-/s3/mlibra/mlibra-data/artiom/parcels}"
RECON_LIPIDS_FILE="${RECONSTRUCTION_LIPIDS_FILE:-$SRC_PATH/maldi/data/lipid_subset.txt}"

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

# ---- one submission -------------------------------------------------------
# MODE=parcel is hardcoded: run_latent.sh defaults to MODE=both, which would
# retrain an identical baseline inside every job of the sweep.
submit() {
    local job_name=$1 prefix=$2 slices=$3
    local features=$4 n_parcels=$5 spatial_weight=$6 stride=$7
    local rank=$8 normalize_blocks=$9 init_scale=${10} shared_b=${11}
    local latent_dim=${12} norsample=${13} inducing_source=${14}
    shift 14
    local extra_args=("$@")

    echo ">>> Submitting $job_name"
    run_or_echo runai training submit "$job_name" \
        -i artiomartiom/sdsc:withfaiss \
        --image-pull-policy Always \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e SRC_PATH="$SRC_PATH" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slices" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e RECONSTRUCTION_LIPIDS_FILE="$RECON_LIPIDS_FILE" \
        -e TEMPLATE_NAME="$S3_TEMPLATE_NAME" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e MODE="parcel" \
        -e PARCEL_DIR="$S3_PARCEL_DIR" \
        -e PARCEL_FEATURES="$features" \
        -e N_PARCELS="$n_parcels" \
        -e SPATIAL_WEIGHT="$spatial_weight" \
        -e STRIDE="$stride" \
        -e THRESHOLD="${PARCEL_THRESHOLD:-5}" \
        -e NORMALIZE_BLOCKS="$normalize_blocks" \
        -e PARCEL_RANK="$rank" \
        -e PARCEL_INIT_SCALE="$init_scale" \
        -e PARCEL_SHARED_B="$shared_b" \
        -e LATENT_DIM="$latent_dim" \
        -e NUM_INDUCING_POINTS="${NUM_INDUCING_POINTS:-2000}" \
        -e INDUCING_SOURCE="$inducing_source" \
        -e KERNEL="${KERNEL:-matern}" \
        -e NU="${NU:-1.5}" \
        -e N_PIXELS="${N_PIXELS:-10}" \
        -e BATCH_SIZE="${BATCH_SIZE:-1000}" \
        -e N_EPOCHS="${N_EPOCHS:-20}" \
        -e LEARNING_RATE="${LEARNING_RATE:-0.001}" \
        -e SEED="${SEED:-416465}" \
        -e NO_RSAMPLE="$norsample" \
        -e LEARN_INDUCING="${LEARN_INDUCING:-true}" \
        -e ARD="${ARD:-true}" \
        -e LOG_TRANSFORM="${LOG_TRANSFORM:-true}" \
        -e DO_BRAIN_RECONSTRUCTION="${DO_BRAIN_RECONSTRUCTION:-0}" \
        -e RENDER_VOXELS_ONLY="${RENDER_VOXELS_ONLY:-1}" \
        -e OMP_NUM_THREADS="$CPU" \
        -e OPENBLAS_NUM_THREADS="$CPU" \
        -e MKL_NUM_THREADS="$CPU" \
        -e OMP_WAIT_POLICY=passive \
        -- ./parcelgp/run_latent.sh "${extra_args[@]}"
}

# =============================================================================
# Sweep definitions — every axis is a bash array; a one-element array pins it.
#
# Job count = folds x features x K x sw x stride x rank x nb x latent_dim x
#             norsample x inducing_source. The default grid is 2x1x2 = 4 per fold.
#
# What each axis is for:
#   PARCEL_FEAT_LIST  'full' vs 'spatial' is THE control. 'spatial' uses no
#                     appearance features at all, so if it matches 'full' the
#                     gain is soft spatial partitioning, not the reference image.
#   PARCEL_SW_LIST    highest-leverage build knob: 3 coord columns vs 16 feature
#                     columns means sw=1 gives geometry 16% of the k-means
#                     distance and sw=3 gives it 63%. sw~2.3 is the 50/50 point.
#   PARCEL_RANK_LIST  width of the learned parcel embedding. r=2 is the
#                     interpretable setting (the parcels plot in 2-D).
#   PARCEL_NB_LIST    0/1 = weight each feature vs each descriptor block equally.
# =============================================================================
FOLDS=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8")           # lowercase, dashed
PARCEL_FEAT_LIST=("full" "spatial")
PARCEL_K_LIST=(128 192)
PARCEL_SW_LIST=(1.0 3.0)
PARCEL_STRIDE_LIST=(2)
PARCEL_RANK_LIST=(8)
PARCEL_NB_LIST=(0 1)
LATENT_DIMS=(5)
NO_RSAMPLES=("true")
IND_SOURCES=("reference")
PARCEL_INIT_SCALE=${PARCEL_INIT_SCALE:-0.05}
PARCEL_SHARED_B=${PARCEL_SHARED_B:-0}

exp_num=1
for fold in "${FOLDS[@]}"; do
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"

    for d in "${LATENT_DIMS[@]}"; do
      for norsample in "${NO_RSAMPLES[@]}"; do
        for ind_src in "${IND_SOURCES[@]}"; do

          for feat in "${PARCEL_FEAT_LIST[@]}"; do
            for k in "${PARCEL_K_LIST[@]}"; do
              for sw in "${PARCEL_SW_LIST[@]}"; do
                for stride in "${PARCEL_STRIDE_LIST[@]}"; do
                  for rank in "${PARCEL_RANK_LIST[@]}"; do
                    for nb in "${PARCEL_NB_LIST[@]}"; do
                      printf "  exp %2d: %-16s feat=%-7s K=%-4s sw=%-4s stride=%s r=%-2s nb=%s d=%-2s fold=%s\n" \
                          "$exp_num" "parcel" "$feat" "$k" "$sw" "$stride" \
                          "$rank" "$nb" "$d" "$fold"
                      submit "gp-parcellgp-${EXP_SUFFIX}-${exp_num}" \
                          "$fold_upper" "$SLICES_DATASET_FILE" \
                          "$feat" "$k" "$sw" "$stride" "$rank" "$nb" \
                          "$PARCEL_INIT_SCALE" "$PARCEL_SHARED_B" \
                          "$d" "$norsample" "$ind_src"
                      exp_num=$((exp_num + 1))
                    done
                  done
                done
              done
            done
          done

        done
      done
    done
done

echo
echo "Submitted $((exp_num - 1)) jobs (DRY_RUN=$DRY_RUN)."
echo "Results land under $S3_OUTPUT_DIR/<EXP_NAME>/"
echo "Parcel fields are cached under $S3_PARCEL_DIR/"
echo
echo "Prebuild a field (recommended for STRIDE<4, where the build is slow):"
echo "  python -m parcelgp.build --reference-file $S3_REFERENCE_FILE \\"
echo "      --out $S3_PARCEL_DIR/full_k128_sw1.0_s4_t5.npz \\"
echo "      --n-parcels 128 --features full --spatial-weight 1.0 --stride 4"
echo
echo "Baseline: run ./submit/run_lgp_batch.sh with matching hyperparameters,"
echo "then compare PAIRED over lipids:"
echo "  python -m parcelgp.compare \\"
echo "      --baseline base=/s3/mlibra/mlibra-data/artiom/lgb_experiment_cv/<dir> \\"
echo "      --run parcel=$S3_OUTPUT_DIR/<parcel-dir> --metric corr"
