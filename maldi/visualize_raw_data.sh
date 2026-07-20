#!/usr/bin/env bash
# =============================================================================
# Visualize the RAW MALDI dataset over the reference template.
#
# Pure viewer — no training, no GP. Reads the measured MALDI voxels straight
# from the parquet and scatters them in napari on top of the CCF reference
# volume. Every mouse (the Sample column) is its own napari layer, so the
# per-layer visibility checkbox in napari's layer list is exactly "a
# checkbox for each mouse". Within a mouse, points are coloured by Section
# (slice) so individual coronal slices are distinguishable.
#
# Override knobs via env vars, e.g.
#     SAMPLES="ReferenceAtlas SecondAtlas" ./visualize_raw_data.sh
#     COLOR_BY=lipid LIPID="PC 38:1" ./visualize_raw_data.sh
# =============================================================================

# ---- data ------------------------------------------------------------------
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATIONS_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"

# ---- what to show ----------------------------------------------------------
# Space-separated subset of mouse names; empty = all mice in the parquet.
: "${SAMPLES:=}"
# section (default) | sample | lipid
: "${COLOR_BY:=section}"
# Required only when COLOR_BY=lipid.
: "${LIPID:=}"
: "${POINT_SIZE:=2.0}"
# Randomly cap each mouse to this many points (0 = no cap) to keep napari
# responsive on multi-million-voxel brains.
: "${MAX_POINTS_PER_SAMPLE:=0}"
: "${GAMMA:=0.5}"

# ---- axis alignment (same convention as visualize_lipid_gp.sh) ------------
: "${AXIS_ORDER:=0 1 2}"
: "${FLIP_AXES:=}"
: "${TEMPLATE_VOXEL_SCALE:=0.025}"

# ---- build optional args --------------------------------------------------
ANN_ARG=""
if [ -n "$ANNOTATIONS_FILE" ] && [ -f "$ANNOTATIONS_FILE" ]; then
    ANN_ARG="--annotations-file $ANNOTATIONS_FILE"
fi
SAMPLES_ARG=""
if [ -n "$SAMPLES" ]; then
    SAMPLES_ARG="--samples $SAMPLES"
fi
LIPID_ARG=""
if [ -n "$LIPID" ]; then
    LIPID_ARG="--lipid $LIPID"
fi
FLIP_ARG=""
if [ -n "$FLIP_AXES" ]; then
    FLIP_ARG="--flip-axes $FLIP_AXES"
fi

# ---- sanity check ---------------------------------------------------------
if [ ! -f "$MALDI_FILE" ]; then
    echo "ERROR: MALDI_FILE $MALDI_FILE does not exist." >&2
    exit 2
fi
if [ ! -f "$REFERENCE_FILE" ]; then
    echo "ERROR: REFERENCE_FILE $REFERENCE_FILE does not exist." >&2
    exit 2
fi

echo "================================================================"
echo "Visualize raw MALDI data"
echo "  MALDI_FILE = $MALDI_FILE"
echo "  COLOR_BY   = $COLOR_BY"
[ -n "$SAMPLES" ] && echo "  SAMPLES    = $SAMPLES"
[ -n "$LIPID" ]   && echo "  LIPID      = $LIPID"
echo "================================================================"

python maldi/visualize_raw_data.py \
    --maldi-file "$MALDI_FILE" \
    --reference-file "$REFERENCE_FILE" \
    $ANN_ARG \
    $SAMPLES_ARG \
    --color-by "$COLOR_BY" \
    $LIPID_ARG \
    --point-size "$POINT_SIZE" \
    --max-points-per-sample "$MAX_POINTS_PER_SAMPLE" \
    --gamma "$GAMMA" \
    --axis-order $AXIS_ORDER \
    $FLIP_ARG \
    --template-voxel-scale "$TEMPLATE_VOXEL_SCALE" \
    "$@"
