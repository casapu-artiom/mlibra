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
: "${LIPIDS_FILE:=/home/casap/mlibra_git/maldi/data/lipid_subset.txt}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_3.json}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"

# -------------------------------------------------------------------------
# Graph / kernel config — every one of these is a sweep dimension when
# called from run_eigenvector_sweep_batch.sh. Set any of them in the env
# to override the local default.
# -------------------------------------------------------------------------
: "${KNN_METHOD:=faiss_atlas_weighted}"
: "${NORMALIZATION:=randomwalk}"
: "${THRESHOLD:=5}"
: "${KNN_K:=120}"
: "${GRAPHBANDWIDTH:=0.1}"
: "${CROSS_REGION_INFLATION:=10.0}"
: "${NUM_MODES:=6000}"
# Lanczos Krylov subspace floor. -1 = auto (max(1500, 3*NUM_MODES+20)).
# At stride=1 the 1500 floor blows up GPU memory; set e.g. NCV_MIN=100.
: "${NCV_MIN:=-1}"
: "${STRIDE:=8}"
: "${N_LIST:=1}"
: "${NU:=2}"
: "${LENGTHSCALE:=1.0}"
: "${BUMP_SCALE:=1.0}"
: "${BUMP_DECAY:=0.01}"
: "${N_TEST_ON:=500}"
: "${N_TEST_OFF:=500}"
: "${TEST_SEED:=42}"
: "${GEODESIC_ANCHORS:=500}"

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
CFG_SLUG="${CFG_SLUG}-bs$(slug "$BUMP_SCALE")-bd$(slug "$BUMP_DECAY")-nu${NU}-ls$(slug "$LENGTHSCALE")-s${STRIDE}-nm${NUM_MODES}"
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
    --ncv-min "$NCV_MIN" \
    --knn-method "$KNN_METHOD" \
    --normalization "$NORMALIZATION" \
    --threshold "$THRESHOLD" \
    --knn-k "$KNN_K" \
    --graphbandwidth "$GRAPHBANDWIDTH" \
    --nu "$NU" \
    --kernel "symmetric" \
    --lengthscale "$LENGTHSCALE" \
    --bump-scale "$BUMP_SCALE" \
    --bump-decay "$BUMP_DECAY" \
    --n-test-on "$N_TEST_ON" \
    --n-test-off "$N_TEST_OFF" \
    --test-seed "$TEST_SEED" \
    --out "$OUT_CSV" \
    --maldi-file        "$MALDI_FILE" \
    --dataset-path      "$DATA_PATH" \
    --available-lipids-file "$AVAILABLE_LIPIDS_FILE" \
    --slices-dataset-file   "$SLICES_DATASET_FILE" \
    --lipids-file           "$LIPIDS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --fold-side test \
    --geodesic-anchors $GEODESIC_ANCHORS \
    "${EXTRA_ARGS[@]}" \
    "$@"