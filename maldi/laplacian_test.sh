#!/usr/bin/env bash
# Single-config Laplacian + Matern-PSD diagnostic.
#
# Every parameter has a default below. The batch submitter sets the sweep
# axes via env vars to override; passing `--foo bar` through "$@" works too
# (the python CLI takes precedence). Run this directly on a workstation with
# no env vars set and you get a reasonable smoke-test config out of the box.

set -eu

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${TEMPLATE_NAME:=reference}"

# -------------------------------------------------------------------------
# Graph / kernel config — every one of these is a sweep dimension when
# called from run_eigenvector_sweep_batch.sh. Set any of them in the env
# to override the local default.
# -------------------------------------------------------------------------
: "${KNN_METHOD:=anatomical_atlas}"
: "${NORMALIZATION:=symmetric}"
: "${THRESHOLD:=5}"
: "${KNN_K:=120}"
: "${GRAPHBANDWIDTH:=0.1}"
: "${CROSS_REGION_INFLATION:=10.0}"
: "${NUM_MODES:=1300}"
: "${STRIDE:=4}"
: "${N_LIST:=1}"
: "${NU:=2}"
: "${LENGTHSCALE:=1.0}"
: "${BUMP_SCALE:=3.0}"
: "${BUMP_DECAY:=0.05}"
: "${N_TEST_ON:=200}"
: "${N_TEST_OFF:=200}"
: "${TEST_SEED:=42}"

# -------------------------------------------------------------------------
# Output filename. Defaults to a slug-named per-config CSV under OUTPUT_DIR
# so parallel batch jobs don't collide. The slug includes every sweep-able
# parameter so the filename alone tells you the full config. The batch
# submitter sets OUT_CSV explicitly; standalone runs use the auto-named
# default.
#
# Slug layout:  <method>-<norm>-t<thr>-k<knn_k>-bw<bw>[-i<infl>]-bs<bs>-bd<bd>-nu<nu>-ls<ls>-nm<nm>
# Floats are encoded with `.` → `p` (0.1 → 0p1). The `i<infl>` segment is
# only present for faiss_atlas_weighted; the other methods don't use it
# (and don't include it in their cache keys either), so omitting it keeps
# the filenames honest about what actually varied.
# -------------------------------------------------------------------------
slug() { echo "$1" | sed 's/\./p/g; s/-/n/g'; }

case "$KNN_METHOD" in
    faiss)                METHOD_TAG="fa" ;;
    anatomical_atlas)     METHOD_TAG="aa" ;;
    faiss_atlas_weighted) METHOD_TAG="fw" ;;
    *)                    METHOD_TAG="$KNN_METHOD" ;;
esac
case "$NORMALIZATION" in
    symmetric)  NORM_TAG="sym" ;;
    randomwalk) NORM_TAG="rw"  ;;
    *)          NORM_TAG="$NORMALIZATION" ;;
esac

CFG_SLUG="${METHOD_TAG}-${NORM_TAG}-t${THRESHOLD}-k${KNN_K}-bw$(slug "$GRAPHBANDWIDTH")"
if [ "$KNN_METHOD" = "faiss_atlas_weighted" ]; then
    CFG_SLUG="${CFG_SLUG}-i$(slug "$CROSS_REGION_INFLATION")"
fi
CFG_SLUG="${CFG_SLUG}-bs$(slug "$BUMP_SCALE")-bd$(slug "$BUMP_DECAY")-nu${NU}-ls$(slug "$LENGTHSCALE")-nm${NUM_MODES}"
: "${OUT_CSV:=${OUTPUT_DIR}/psd_sweep/${CFG_SLUG}.csv}"

# -------------------------------------------------------------------------
# Build the python arg list. faiss_atlas_weighted is the only method that
# uses cross_region_inflation, so we only pass it through for that case
# (matches the cache-key convention in the python script).
# -------------------------------------------------------------------------
EXTRA_ARGS=()
if [ "$KNN_METHOD" = "faiss_atlas_weighted" ]; then
    EXTRA_ARGS+=(--cross-region-inflation "$CROSS_REGION_INFLATION")
fi

cd "$SRC_PATH"

python "$SRC_PATH/maldi/laplacian_test.py" \
    --template-name "$TEMPLATE_NAME" \
    --reference-file "$REFERENCE_FILE" \
    --annotations-file "$ANNOTATION_FILE" \
    --eigenvector-dir "$EIGENVECTOR_DIR" \
    --stride "$STRIDE" \
    --n-list "$N_LIST" \
    --num-modes "$NUM_MODES" \
    --knn-method "$KNN_METHOD" \
    --normalization "$NORMALIZATION" \
    --threshold "$THRESHOLD" \
    --knn-k "$KNN_K" \
    --graphbandwidth "$GRAPHBANDWIDTH" \
    --nu "$NU" \
    --lengthscale "$LENGTHSCALE" \
    --bump-scale "$BUMP_SCALE" \
    --bump-decay "$BUMP_DECAY" \
    --n-test-on "$N_TEST_ON" \
    --n-test-off "$N_TEST_OFF" \
    --test-seed "$TEST_SEED" \
    --out "$OUT_CSV" \
    --skip-if-row-exists \
    "${EXTRA_ARGS[@]}" \
    "$@"