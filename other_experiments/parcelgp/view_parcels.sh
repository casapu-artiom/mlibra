#!/usr/bin/env bash
# =============================================================================
# Look at a built parcellation on the reference brain.
#
#   ./other_experiments/parcelgp/view_parcels.sh                    # napari, 2D slice view
#   NDISPLAY=3 ./other_experiments/parcelgp/view_parcels.sh         # napari, straight into 3D
#   MONTAGE=/tmp/parcels.png ./other_experiments/parcelgp/view_parcels.sh          # headless PNG
#   AXIS=2 N_SLICES=6 MONTAGE=/tmp/sag.png ./other_experiments/parcelgp/view_parcels.sh
#   STATS=1 ./other_experiments/parcelgp/view_parcels.sh            # per-parcel table, no window
#
# The field is BUILT (and cached) first with the same parameters run_parcel_per_lipid.sh
# uses, so the two scripts always look at the same parcellation — change a knob
# in one and the other follows. The build is a no-op when the cache matches.
# =============================================================================
set -euo pipefail

: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${REFERENCE_FILE:=$DATA_PATH/reference_image.npy}"

# ---- parcellation (keep in sync with run_parcel_per_lipid.sh) ----
: "${PARCEL_DIR:=$DATA_PATH/parcels}"
: "${N_PARCELS:=128}"
: "${PARCEL_FEATURES:=full}"        # full | simple | spatial
: "${SPATIAL_WEIGHT:=3.0}"
: "${STRIDE:=2}"
: "${THRESHOLD:=5}"
: "${NORMALIZE_BLOCKS:=0}"
NB_ARG=""; NB_TAG=""
[ "$NORMALIZE_BLOCKS" = "1" ] && { NB_ARG="--normalize-blocks"; NB_TAG="_nb"; }
: "${PARCEL_FIELD:=$PARCEL_DIR/${PARCEL_FEATURES}_k${N_PARCELS}_sw${SPATIAL_WEIGHT}_s${STRIDE}_t${THRESHOLD}${NB_TAG}.npz}"

# ---- what to draw ----
: "${MONTAGE:=}"                    # non-empty = write this PNG instead of napari
: "${N_SLICES:=9}"
: "${AXIS:=0}"                      # 0 = AP (coronal), 1 = DV, 2 = LR (sagittal)
: "${ALPHA:=0.55}"                  # parcel tint over the greyscale reference
: "${NCOLS:=0}"                     # 0 = square grid
: "${DPI:=140}"
: "${NDISPLAY:=2}"                  # napari: 2 = slice view, 3 = volume
: "${STATS:=0}"

# =============================================================================
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

args=(--field "$PARCEL_FIELD" --reference-file "$REFERENCE_FILE")
[ "$STATS" = "1" ] && args+=(--stats)
[ -n "$MONTAGE" ] && args+=(--montage "$MONTAGE" --n-slices "$N_SLICES"
                            --axis "$AXIS" --alpha "$ALPHA" --ncols "$NCOLS" --dpi "$DPI")
[ -z "$MONTAGE" ] && args+=(--ndisplay "$NDISPLAY")

python -m other_experiments.parcelgp.view_parcels "${args[@]}" "$@"
