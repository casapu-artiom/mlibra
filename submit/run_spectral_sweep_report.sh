#!/usr/bin/env bash
# Submit a single runai job that aggregates + ranks a spectral-sweep batch.
# Reads the shared S3 rows/ dir written by the fan-out jobs from
# run_spectral_distance_sweep.sh and writes per_lipid/per_config/ranked.csv
# back into the same batch dir.
#
# Point it at a batch via EXP_SUFFIX (the suffix printed by the batch submitter):
#     EXP_SUFFIX=250619-14-30 ./run_spectral_sweep_report.sh
#     EXP_SUFFIX=250619-14-30 OBJECTIVE=spectral_gain ./run_spectral_sweep_report.sh
#     DRY_RUN=1 EXP_SUFFIX=... ./run_spectral_sweep_report.sh
# Extra flags pass straight through to spectral_sweep_report.sh:  --top 30
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

MEM=${MEM:-16G}
CPU=${CPU:-2}
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_latest}

SRC_PATH="/myhome/mlibra"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/spectral_sweep"

# Target dir: S3_OUT_DIR wins (point it at any rows dir, e.g. a prior run's
#   /s3/mlibra/mlibra-data/artiom/spectral_sweep/spectral_sweep ), else derive it
# from EXP_SUFFIX (the suffix printed by run_spectral_distance_sweep.sh).
if [ -z "${S3_OUT_DIR:-}" ]; then
    if [ -z "${EXP_SUFFIX:-}" ]; then
        echo "ERROR: set S3_OUT_DIR=<rows dir> or EXP_SUFFIX=<batch suffix>" >&2
        exit 1
    fi
    S3_OUT_DIR="${S3_OUTPUT_DIR}/${EXP_SUFFIX}"
fi
# job-name component must be k8s-safe (lowercase alphanumeric + '-')
JOB_TAG=$(echo "${EXP_SUFFIX:-$(basename "$S3_OUT_DIR")}" | tr '_/.' '-')

OBJECTIVE=${OBJECTIVE:-spectral}
RANK_STAT=${RANK_STAT:-decay_strength_quantile}
TOP=${TOP:-20}

JOB_NAME=${JOB_NAME:-"spec-report-${JOB_TAG}"}

echo ">>> Submitting $JOB_NAME  (report over $S3_OUT_DIR)"
run_or_echo runai training submit "$JOB_NAME" \
    -i "$IMAGE" \
    --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
    --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e SRC_PATH="$SRC_PATH" \
    -e OUT_DIR="$S3_OUT_DIR" \
    -e OBJECTIVE="$OBJECTIVE" \
    -e RANK_STAT="$RANK_STAT" \
    -e TOP="$TOP" \
    -- ./maldi/spectral_sweep_report.sh "$@"
