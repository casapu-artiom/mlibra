#!/bin/bash
# Compare eigensolvers on a cached graph, with accuracy + performance
# (time, GPU util, GPU memory, host RAM) statistics.
#
# Graph is selected by the same options the lgp experiments use
# (stride / threshold / k / normalization / bandwidth); the key is composed
# automatically. Every run writes a self-documenting subdir under OUT_DIR:
#     $OUT_DIR/$RUN_SLUG/compare.{json,png,log}
#
# All knobs are env-overridable so the same script runs locally and on the
# cluster (the submit/ runner just sets KNN_DIR / OUT_DIR to S3 mounts):
#   MODES=200 STRIDE=4 THRESHOLD=50 ./eigensolver_compare.sh
#   STRIDE=1 NCV_MIN=100 APPROACHES=cupy_sa,cupy_si_cg ./eigensolver_compare.sh

set -euo pipefail

# --- repo root (resolved from this script's location; override with REPO) ---
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- graph + solver properties ---
MODES="${MODES:-1300}"
STRIDE="${STRIDE:-4}"
THRESHOLD="${THRESHOLD:-5}"
KNN_K="${KNN_K:-15}"
KNN_METHOD="${KNN_METHOD:-faiss}"
BANDWIDTH="${BANDWIDTH:-1.0}"
NORMALIZATION="${NORMALIZATION:-randomwalk}"
TEMPLATE="${TEMPLATE:-reference}"
NCV_MIN="${NCV_MIN:--1}"
TOL="${TOL:-1e-6}"
APPROACHES="${APPROACHES:-cupy_sa,cupy_si_cg,slepc_ks}"
SI_SIGMA="${SI_SIGMA:--1e-3}"
SI_CG_INNER_RTOL="${SI_CG_INNER_RTOL:-1e-6}"
REFERENCE="${REFERENCE:-slepc_si}"
MAX_DIRECT_NODES="${MAX_DIRECT_NODES:-2000000}"

# --- paths (override KNN_DIR / OUT_DIR to S3 mounts on the cluster) ---
KNN_DIR="${KNN_DIR:-/home/casap/mlibra/output/eigenvectors/knn}"
OUT_DIR="${OUT_DIR:-./eigensolver_compare_report}"

# --- build the graph on cache miss (needs the reference/annotation volumes) ---
BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"
REFERENCE_FILE="${REFERENCE_FILE:-/home/casap/mlibra/mlibra_data/reference_image.npy}"
ANNOTATIONS_FILE="${ANNOTATIONS_FILE:-/home/casap/mlibra/mlibra_data/level_15annot.npy}"

build_args=()
if [ "$BUILD_IF_MISSING" = "1" ]; then
    build_args+=(--build-if-missing)
    [ -n "$REFERENCE_FILE" ]   && build_args+=(--reference-file   "$REFERENCE_FILE")
    [ -n "$ANNOTATIONS_FILE" ] && build_args+=(--annotations-file "$ANNOTATIONS_FILE")
fi

# --- self-documenting output subdir from the run's defining properties ---
slug() { echo "$1" | sed 's/\./p/g'; }
RUN_SLUG="${RUN_SLUG:-str${STRIDE}_t${THRESHOLD}_k${KNN_K}_${KNN_METHOD}_bw$(slug "$BANDWIDTH")_${NORMALIZATION}_nm${MODES}}"
RUN_DIR="${OUT_DIR}/${RUN_SLUG}"
mkdir -p "$RUN_DIR"

DEVICE="${DEVICE:-$(python -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')}"

echo "[eigensolver_compare] -> $RUN_DIR  (device=$DEVICE, approaches=$APPROACHES)"

# Unbuffer Python: piping through tee makes stdout/stderr block-buffered, which
# otherwise hides all prints + tqdm bars until a stage finishes (looks "stuck").
export PYTHONUNBUFFERED=1
python -u "${REPO}/benchmarks/eigensolver_compare.py" \
    --knn-dir "$KNN_DIR" \
    --template "$TEMPLATE" \
    --stride "$STRIDE" \
    --threshold "$THRESHOLD" \
    --knn-k "$KNN_K" \
    --knn-method "$KNN_METHOD" \
    --bandwidth "$BANDWIDTH" \
    --normalization "$NORMALIZATION" \
    --modes "$MODES" \
    --ncv-min="$NCV_MIN" \
    --tol "$TOL" \
    --approaches "$APPROACHES" \
    --reference "$REFERENCE" \
    --max-direct-nodes "$MAX_DIRECT_NODES" \
    --si-sigma="$SI_SIGMA" \
    --si-cg-inner-rtol "$SI_CG_INNER_RTOL" \
    --lobpcg-backend auto \
    --device "$DEVICE" \
    "${build_args[@]}" \
    --out "$RUN_DIR/compare" 2>&1 | tee "$RUN_DIR/compare.log"

echo ""
echo "Report in: $RUN_DIR"
ls "$RUN_DIR"/compare.* 2>/dev/null || true
