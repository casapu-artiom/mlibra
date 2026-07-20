#!/bin/bash
# Benchmark the Nyström eigenvector extension by extending a COARSE-stride
# solve to a set of held-out test points and scoring it against a FINE-stride
# reference there. By default the test points are the FINE grid's "in-between"
# nodes (the strided grids are nested, coarse ⊂ fine, so those are an honest
# held-out set with a literal ground-truth eigenvector). Set MALDI_FILE to
# score against the ACTUAL measured MALDI voxel coordinates instead -- see
# nystrom_benchmark.py's docstring for what "ground truth" means in that mode.
#
# knn/ and eigvecs/ caches share keys with compute_eigenvectors.py, so an
# already-computed stride-8 / stride-4 solve is reused.
#
# All knobs are env-overridable:
#   COARSE_STRIDE=8 FINE_STRIDE=4 MODES=200 ./nystrom_benchmark.sh
#   COARSE_STRIDE=4 FINE_STRIDE=2 BANDWIDTH=1.0 ./nystrom_benchmark.sh
#   MALDI_FILE=/path/to/maldi.parquet ./nystrom_benchmark.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

COARSE_STRIDE="${COARSE_STRIDE:-8}"
FINE_STRIDE="${FINE_STRIDE:-4}"
MODES="${MODES:-1000}"
BANDWIDTH="${BANDWIDTH:-0.1}"
NORMALIZATION="${NORMALIZATION:-randomwalk}"
KNN_K="${KNN_K:-15}"
# COARSE and FINE always share a threshold (nested grids); run once per value.
THRESHOLDS="${THRESHOLDS:-5}"
TEMPLATE="${TEMPLATE:-reference}"
NCV_MIN="${NCV_MIN:--1}"
NYSTROM_K="${NYSTROM_K:--1}"
MAX_OOS="${MAX_OOS:--1}"

# Score against real MALDI voxels instead of the synthetic grid in-between
# nodes. MALDI_FILE unset (default) keeps the grid-based test. MALDI_MODE
# 'direct' scores AT the MALDI coords (no literal ground truth there);
# 'distance-match' keeps exact ground truth by subsampling grid in-between
# nodes to match the MALDI points' NN-distance-to-coarse-graph distribution.
MALDI_FILE="${MALDI_FILE:-}"
MALDI_FILTER="${MALDI_FILTER:-}"
MALDI_COORD_COLS="${MALDI_COORD_COLS:-xccf,yccf,zccf}"
MALDI_MODE="${MALDI_MODE:-distance-match}"
DISTANCE_MATCH_BINS="${DISTANCE_MATCH_BINS:-20}"

REFERENCE_FILE="${REFERENCE_FILE:-/home/casap/mlibra/mlibra_data/reference_image.npy}"
ANNOTATIONS_FILE="${ANNOTATIONS_FILE:-/home/casap/mlibra/mlibra_data/level_15annot.npy}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/casap/mlibra/output/eigenvectors}"
OUT_DIR="${OUT_DIR:-./nystrom_benchmark_report}"

slug() { echo "$1" | sed 's/\./p/g'; }
DEVICE="${DEVICE:-$(python -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')}"
export PYTHONUNBUFFERED=1

for THRESHOLD in $THRESHOLDS; do
    RUN_SLUG="c${COARSE_STRIDE}_f${FINE_STRIDE}_k${KNN_K}_bw$(slug "$BANDWIDTH")_${NORMALIZATION}_t$(slug "$THRESHOLD")_nm${MODES}"
    RUN_DIR="${OUT_DIR}/${RUN_SLUG}"
    mkdir -p "$RUN_DIR"

    echo "[nystrom_benchmark] stride ${COARSE_STRIDE} -> ${FINE_STRIDE}  thresh=${THRESHOLD}  -> $RUN_DIR  (device=$DEVICE)"

    maldi_args=()
    if [ -n "$MALDI_FILE" ]; then
        maldi_args+=(--maldi-file "$MALDI_FILE" --maldi-coord-cols "$MALDI_COORD_COLS"
                     --maldi-mode "$MALDI_MODE" --distance-match-bins "$DISTANCE_MATCH_BINS")
        [ -n "$MALDI_FILTER" ] && maldi_args+=(--maldi-filter "$MALDI_FILTER")
    fi

    python -u "${REPO}/benchmarks/nystrom_benchmark.py" \
        --reference-volume "$REFERENCE_FILE" \
        --annotations-volume "$ANNOTATIONS_FILE" \
        --output-path "$OUTPUT_PATH" \
        --coarse-stride "$COARSE_STRIDE" \
        --fine-stride "$FINE_STRIDE" \
        --template "$TEMPLATE" \
        --threshold "$THRESHOLD" \
        --k "$KNN_K" \
        --modes "$MODES" \
        --bandwidth "$BANDWIDTH" \
        --normalization "$NORMALIZATION" \
        --ncv-min="$NCV_MIN" \
        --nystrom-k="$NYSTROM_K" \
        --max-oos="$MAX_OOS" \
        --device "$DEVICE" \
        "${maldi_args[@]}" \
        --out "$RUN_DIR/nystrom" 2>&1 | tee "$RUN_DIR/nystrom.log"

    echo ""
    echo "Report in: $RUN_DIR"
    ls "$RUN_DIR"/nystrom.* 2>/dev/null || true
done
