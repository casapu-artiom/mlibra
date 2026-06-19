#!/usr/bin/env bash
# Batch submitter for the spectral distance sweep. Loops the sweep grid and
# submits ONE runai job per configuration (knn_method, normalization, threshold,
# knn_k, bandwidth, inflation, nu, lengthscale, stride:modes, bump_scale,
# bump_decay). Each job runs maldi/spectral_distance_sweep.sh with its config's
# values pushed in as singleton env vars, and writes its own
# OUT_DIR/rows/<slug>.csv into a SHARED S3 rows/ dir.
#
# After the jobs finish, build the aggregated report (per_lipid/per_config/ranked)
# over that shared dir with submit/run_spectral_sweep_report.sh (or, locally,
# maldi/spectral_sweep_report.sh). This script does NOT aggregate — it only fans
# the grid out into jobs.
#
# Override any axis from the calling shell (space-separated lists), e.g.
#     KNN_KS="60 120" BANDWIDTHS="0.05 0.1 0.2" ./run_spectral_distance_sweep.sh
#     DRY_RUN=1 ./run_spectral_distance_sweep.sh         # print, don't submit
# Extra flags pass straight through to spectral_distance_sweep.sh:
#     ./run_spectral_distance_sweep.sh --force
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}

run_or_echo() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] $*"
    else
        "$@"
    fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

# -------------------------------------------------------------------------
# Cluster resources
# -------------------------------------------------------------------------
MEM=${MEM:-64G}
CPU=${CPU:-4}
GPU=${GPU:-1.0}
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_latest}

# -------------------------------------------------------------------------
# Cloud (S3 / myhome) paths consumed by spectral_distance_sweep.sh. OUT_DIR is
# shared across every job — each writes its own rows/<slug>.csv into it.
# -------------------------------------------------------------------------
SRC_PATH="/myhome/mlibra"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/spectral_sweep"
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/artiom/eigenvectors"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_TEMPLATE_NAME="reference"
S3_LIPIDS_FILE="/myhome/mlibra/maldi/data/lipid_subset.txt"

# A per-batch subdir keeps each fan-out's rows together (so the report aggregates
# one batch). Override EXP_SUFFIX to resume/extend an existing batch.
EXP_SUFFIX="${EXP_SUFFIX:-$(date +'%y%m%d-%H-%M')}"
S3_OUT_DIR="${S3_OUTPUT_DIR}/${EXP_SUFFIX}"

# -------------------------------------------------------------------------
# Sweep grid (space-separated lists). These are the CONFIG axes — one job per
# cell. LOW_MODES / LOW_MODE_FRAC are METRIC axes (same for every job), so they
# are forwarded whole to each job rather than iterated here.
# -------------------------------------------------------------------------
KNN_METHODS=${KNN_METHODS:-"faiss_atlas_weighted"}
NORMALIZATIONS=${NORMALIZATIONS:-"randomwalk"}
THRESHOLDS=${THRESHOLDS:-"5 40 50"}
KNN_KS=${KNN_KS:-"15 60"}
BANDWIDTHS=${BANDWIDTHS:-"0.1 0.2"}
INFLATIONS=${INFLATIONS:-"10 50"}        # only iterated for faiss_atlas_weighted
NUS=${NUS:-"2"}
LENGTHSCALES=${LENGTHSCALES:-"1.0 2.0"}
STRIDE_MODES=${STRIDE_MODES:-"4:1300 8:6000"}
FEATURE_MODE=${FEATURE_MODE:-"nystrom"}
BUMP_SCALES=${BUMP_SCALES:-"1.0 2.0"}    # only iterated in nystrom mode
BUMP_DECAYS=${BUMP_DECAYS:-"0.01"}       # only iterated in nystrom mode
LOW_MODES=${LOW_MODES:-"20 50 100 200"}  # metric axis: forwarded to every job
LOW_MODE_FRAC=${LOW_MODE_FRAC:-0.1}
NCV_MIN=${NCV_MIN:--1}

# -------------------------------------------------------------------------
# Submit one runai job for a single config. Sweep values are forwarded as
# singleton env vars; spectral_distance_sweep.sh consumes them via the
# `: "${KEY:=default}"` pattern, so each job enumerates exactly one config and
# writes its own rows/<slug>.csv (slug built by the Python from these values).
# -------------------------------------------------------------------------
submit_one() {
    local job_name=$1; shift
    local method=$1 norm=$2 thr=$3 k=$4 bw=$5 infl=$6 nu=$7 ls=$8 sm=$9
    local bs=${10} bd=${11}
    runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$CPU"     --cpu-core-request   "$CPU" \
        --cpu-memory-limit "$MEM"   --cpu-memory-request "$MEM" \
        --gpu-request-type portion  --gpu-portion-request "$GPU" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e SRC_PATH="$SRC_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e OUT_DIR="$S3_OUT_DIR" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e TEMPLATE_NAME="$S3_TEMPLATE_NAME" \
        -e LIPIDS_FILE="$S3_LIPIDS_FILE" \
        -e KNN_METHODS="$method" \
        -e NORMALIZATIONS="$norm" \
        -e THRESHOLDS="$thr" \
        -e KNN_KS="$k" \
        -e BANDWIDTHS="$bw" \
        -e INFLATIONS="$infl" \
        -e NUS="$nu" \
        -e LENGTHSCALES="$ls" \
        -e STRIDE_MODES="$sm" \
        -e FEATURE_MODE="$FEATURE_MODE" \
        -e BUMP_SCALES="$bs" \
        -e BUMP_DECAYS="$bd" \
        -e LOW_MODES="$LOW_MODES" \
        -e LOW_MODE_FRAC="$LOW_MODE_FRAC" \
        -e NCV_MIN="$NCV_MIN" \
        -- ./maldi/spectral_distance_sweep.sh "$@"
}

# -------------------------------------------------------------------------
# Sweep loop. Mirrors enumerate_configs() in spectral_distance_sweep.py:
#   * inflation only varies for faiss_atlas_weighted (single placeholder else),
#   * bump_scale/bump_decay only vary in nystrom mode (single placeholder in snap).
# -------------------------------------------------------------------------
if [ "$FEATURE_MODE" = "nystrom" ]; then
    bump_scale_list="$BUMP_SCALES"; bump_decay_list="$BUMP_DECAYS"
else
    bump_scale_list="${BUMP_SCALES%% *}"; bump_decay_list="${BUMP_DECAYS%% *}"
fi

n_submitted=0
for method in $KNN_METHODS; do
    if [ "$method" = "faiss_atlas_weighted" ]; then
        inflation_list="$INFLATIONS"
    else
        inflation_list="${INFLATIONS%% *}"   # placeholder; unused for this method
    fi
    for norm in $NORMALIZATIONS; do
    for thr in $THRESHOLDS; do
    for k in $KNN_KS; do
    for bw in $BANDWIDTHS; do
    for infl in $inflation_list; do
    for nu in $NUS; do
    for ls in $LENGTHSCALES; do
    for sm in $STRIDE_MODES; do
    for bs in $bump_scale_list; do
    for bd in $bump_decay_list; do
        n_submitted=$((n_submitted + 1))
        job_name="spec-${EXP_SUFFIX}-$(printf '%04d' "$n_submitted")"
        echo ">>> [$n_submitted] $job_name  $method/$norm t$thr k$k bw$bw i$infl nu$nu ls$ls $sm bs$bs bd$bd"
        run_or_echo submit_one "$job_name" \
            "$method" "$norm" "$thr" "$k" "$bw" "$infl" "$nu" "$ls" "$sm" "$bs" "$bd" \
            "$@"
    done; done; done; done; done; done; done; done; done; done
done

echo
echo "Submitted $n_submitted job(s). Batch suffix: $EXP_SUFFIX"
echo "Rows:    $S3_OUT_DIR/rows/*.csv"
echo "Report:  EXP_SUFFIX=$EXP_SUFFIX ./submit/run_spectral_sweep_report.sh   (after jobs finish)"
