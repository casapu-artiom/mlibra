#!/usr/bin/env bash
# =============================================================================
# Euclidean vs parcel kernel, side by side, on real MALDI slices.
#
#   - pick a mouse + section: that sample's voxels, snapped to template nodes,
#     laid out flat and drawn TWICE
#   - click a voxel -> LEFT is k_base(test, .), RIGHT is the same kernel times
#     the parcel factor (or the factor alone, or the signed difference)
#   - the parcel borders are drawn on both panels, so "does the right panel stop
#     where the left one walks straight through?" is answered by looking
#
#   ./other_experiments/parcelgp/parcel_vs_euclidean_explorer.sh
#   RUN=<per-lipid run dir> RUN_LIPID="PA 36:1" ./other_experiments/parcelgp/parcel_vs_euclidean_explorer.sh
#   DUMP=/tmp/k.png SECTION=9 ./other_experiments/parcelgp/parcel_vs_euclidean_explorer.sh   # headless
#   LIST_SAMPLES=1 ./other_experiments/parcelgp/parcel_vs_euclidean_explorer.sh
#
# WHERE B COMES FROM. The factor is exp(-||m(x)^T B - m(x')^T B||^2 / 2) and B is
# the only learned part. With no RUN set, B = STRENGTH * I: the embedding is the
# membership vector itself, so the picture is the partition's own geometry and
# nothing has been fitted to lipids (STRENGTH=0 is an exact no-op). Point RUN at
# a run made by run_parcel_per_lipid.sh's parcel arm and you get that lipid's LEARNED B,
# ARD lengthscale and outputscale instead — the deployed model, not a lookalike.
# Omit RUN_LIPID once to have the run's lipid names printed.
#
# The parcel field is built/cached first with the same knobs run_parcel_per_lipid.sh uses,
# so a RUN trained by that script indexes the same parcels this shows.
# =============================================================================
set -euo pipefail

: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${MALDI_FILE:=$DATA_PATH/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=$DATA_PATH/reference_image.npy}"
: "${AVAILABLE_LIPIDS_FILE:=$DATA_PATH/maindata_minimal_available_lipids.npy}"

# ---- parcellation (keep in sync with run_parcel_per_lipid.sh) ----
: "${PARCEL_DIR:=$DATA_PATH/parcels}"
: "${N_PARCELS:=128}"
: "${PARCEL_FEATURES:=full}"
: "${SPATIAL_WEIGHT:=3.0}"
: "${STRIDE:=2}"
: "${THRESHOLD:=5}"
: "${NORMALIZE_BLOCKS:=0}"
NB_ARG=""; NB_TAG=""
[ "$NORMALIZE_BLOCKS" = "1" ] && { NB_ARG="--normalize-blocks"; NB_TAG="_nb"; }
: "${PARCEL_FIELD:=$PARCEL_DIR/${PARCEL_FEATURES}_k${N_PARCELS}_sw${SPATIAL_WEIGHT}_s${STRIDE}_t${THRESHOLD}${NB_TAG}.npz}"

# ---- which brain's slices ----
: "${SAMPLE:=ReferenceAtlas}"
: "${MAX_SNAP_MM:=0.5}"
: "${LIST_SAMPLES:=0}"

# ---- trained B (overrides the manual kernel settings when set) ----
: "${RUN:=}"
: "${RUN_LIPID:=}"

# ---- manual kernel (used when RUN is unset, and as the GUI's starting point) --
# LENGTHSCALE is in STANDARDIZED units; the field's coord_std is mm per unit
# (2.84 mm for the shipped reference, so 0.15 is about 0.43 mm).
: "${NU:=2.5}"                      # 0.5 | 1.5 | 2.5 — the MaternKernel's only values
: "${LENGTHSCALE:=0.15}"
: "${OUTPUTSCALE:=1.0}"
: "${PARCEL_STRENGTH:=1.5}"         # untrained B = STRENGTH * I; 0 = exact no-op

# ---- headless ----
: "${DUMP:=}"                       # non-empty = write this PNG instead of napari
: "${SECTION:=}"
: "${TEST_NODE:=}"
: "${LIPID:=}"
: "${DPI:=140}"

# =============================================================================
if [ "$LIST_SAMPLES" = "1" ]; then
    python -m other_experiments.parcelgp.parcel_vs_euclidean_explorer \
        --maldi-file "$MALDI_FILE" --list-samples
    exit 0
fi

mkdir -p "$PARCEL_DIR"
echo "== parcel field: $PARCEL_FIELD"
python -m other_experiments.parcelgp.build \
    --reference-file "$REFERENCE_FILE" \
    --out "$PARCEL_FIELD" \
    --n-parcels "$N_PARCELS" \
    --features "$PARCEL_FEATURES" \
    --spatial-weight "$SPATIAL_WEIGHT" \
    --stride "$STRIDE" \
    --threshold "$THRESHOLD" $NB_ARG

args=(--field "$PARCEL_FIELD"
      --maldi-file "$MALDI_FILE"
      --available-lipids-file "$AVAILABLE_LIPIDS_FILE"
      --sample "$SAMPLE"
      --max-snap-mm "$MAX_SNAP_MM"
      --nu "$NU"
      --lengthscale $LENGTHSCALE
      --outputscale "$OUTPUTSCALE"
      --parcel-strength "$PARCEL_STRENGTH")

[ -n "$RUN" ]       && args+=(--run "$RUN")
[ -n "$RUN_LIPID" ] && args+=(--run-lipid "$RUN_LIPID")
[ -n "$DUMP" ]      && args+=(--dump "$DUMP" --dpi "$DPI")
[ -n "$SECTION" ]   && args+=(--section "$SECTION")
[ -n "$TEST_NODE" ] && args+=(--test-node "$TEST_NODE")
[ -n "$LIPID" ]     && args+=(--lipid "$LIPID")

python -m other_experiments.parcelgp.parcel_vs_euclidean_explorer "${args[@]}" "$@"
