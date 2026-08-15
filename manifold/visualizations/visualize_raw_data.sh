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
# Optional: a precomputed dense lipid volume (.npy on the template grid, e.g. a
# GP prediction '<lipid>_volume.npy') to overlay as an extra Image layer.
: "${LIPID_VOLUME:=/home/casap/mlibra/output/lgp/FOLD-2-LGPALL-MEAN-5-reference-1000-1000-learnind-ard10/volume/Hex2Cer 40:1;O2_volume.npy}"
: "${LIPID_VOLUME_NAME:=Hex2Cer 40:1;O2_volume.npy}"
# volume (default, FAST — baked & cached 3D volumes) | points (SLOW scatter).
: "${RENDER:=volume}"
# Volume mode: integer shrink factor (2 cuts memory ~8x). 1 = full res.
: "${VOLUME_DOWNSAMPLE:=1}"
# Volume mode: cache dir for baked volumes (empty = <maldi dir>/.rawviz_cache).
: "${CACHE_DIR:=}"
# Set to 1 to bypass the baked-volume cache entirely (volume mode).
: "${NO_CACHE:=}"
# points mode only.
: "${POINT_SIZE:=2.0}"
# Randomly cap each mouse to this many points/voxels (0 = no cap).
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
CACHE_ARG=""
if [ -n "$CACHE_DIR" ]; then
    CACHE_ARG="--cache-dir $CACHE_DIR"
fi
NO_CACHE_ARG=""
if [ -n "$NO_CACHE" ]; then
    NO_CACHE_ARG="--no-cache"
fi
LIPID_VOLUME_ARG=()
if [ -n "$LIPID_VOLUME" ]; then
    LIPID_VOLUME_ARG=(--lipid-volume "$LIPID_VOLUME")
    [ -n "$LIPID_VOLUME_NAME" ] && \
        LIPID_VOLUME_ARG+=(--lipid-volume-name "$LIPID_VOLUME_NAME")
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

python "$(dirname "${BASH_SOURCE[0]}")/visualize_raw_data.py" \
    --maldi-file "$MALDI_FILE" \
    --reference-file "$REFERENCE_FILE" \
    $ANN_ARG \
    $SAMPLES_ARG \
    --color-by "$COLOR_BY" \
    $LIPID_ARG \
    --render "$RENDER" \
    --volume-downsample "$VOLUME_DOWNSAMPLE" \
    $CACHE_ARG \
    $NO_CACHE_ARG \
    "${LIPID_VOLUME_ARG[@]}" \
    --point-size "$POINT_SIZE" \
    --max-points-per-sample "$MAX_POINTS_PER_SAMPLE" \
    --gamma "$GAMMA" \
    --axis-order $AXIS_ORDER \
    $FLIP_ARG \
    --template-voxel-scale "$TEMPLATE_VOXEL_SCALE" \
    "$@"
