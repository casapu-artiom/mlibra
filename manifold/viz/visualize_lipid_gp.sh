#!/usr/bin/env bash
# =============================================================================
# Visualize per-lipid GP predictions from a completed training run.
#
# This script is PURELY a viewer — no training, no kernel construction, no
# graph/eigendecomposition. All it does is load .npy files from a run
# directory (produced by per_lipid_gp_experiment.py) and render them in
# napari. So the only paths it needs are: the run dir and the atlas /
# annotations volumes used as background.
#
# Override knobs via env vars before invoking, e.g.
#     RUN_DIR=/path/to/some_other_run ./visualize_lipid_gp.sh
#     COMPARE=/path/to/baseline ./visualize_lipid_gp.sh
#     INITIAL_LIPID="PA 36:1 PA 38:4" ./visualize_lipid_gp.sh
# =============================================================================

# ---- which run to visualize ------------------------------------------------
# IMPORTANT: --run-dir must point to a SPECIFIC run subdirectory containing
# config.json + lipid_names.json + predictions/  — i.e. one of the
# subdirectories under /home/casap/mlibra/output/per_lipid/, NOT that
# parent dir itself.
: "${RUN_DIR:=/home/casap/mlibra/output/per_lipid_batch_3/manifold-nu2-K1300-bs20.0-bd0.01-bw0.1-knn120-faiss-40-randomwalk-ind1000-lr0.005-ep10-lbs1010}"

# ---- optional: a second run for side-by-side comparison --------------------
# Leave empty (default) to skip. When set, the viewer adds an "A: …" /
# "B: …" prefixed layer set and shows metrics for both runs.
: "${COMPARE:=}"

# ---- anatomical background ------------------------------------------------
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATIONS_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"

# ---- initial UI state -----------------------------------------------------
: "${INITIAL_LIPID:=PA 36:1 PA 38:4}"   # selected lipid on startup (name)
: "${INITIAL_LIPID_IDX:=0}"             # fallback if name not found
: "${GAMMA:=0.5}"                       # display gamma
: "${POINT_SIZE:=2.5}"                  # test-point markers
: "${BRAIN_POINT_SIZE:=1.0}"            # whole-brain reconstruction markers
# How many brain-graph nodes to skip when rendering. 1 = every node
# (default). 2-8 are sensible speedups for runs with >100K brain nodes.
: "${BRAIN_RENDER_STRIDE:=8}"
# Colour range mode: "per_layer" (default — each layer normalises to
# its own p1-p99 range; best for spotting structure) or "shared" (all
# pred layers share one range; best for quantitative comparison).
: "${COLOR_RANGE_MODE:=per_layer}"

# ---- axis alignment ------------------------------------------------------
# Permutation of (xccf, yccf, zccf) → (napari axis 0, 1, 2). The default
# "0 1 2" (identity) is right for the standard Allen CCF convention where
# xccf=AP, yccf=DV, zccf=LR and the template's shape is (AP, DV, LR).
# Sanity-check by running:
#   python -c "import pandas as pd, numpy as np;
#              df = pd.read_parquet('<MALDI_FILE>', columns=['xccf','yccf','zccf']);
#              print({c: (df[c].min(), df[c].max()) for c in ['xccf','yccf','zccf']});
#              print('template:', np.load('<REFERENCE_FILE>').shape)"
# and match max(parquet column) to (template_axis * 0.025 mm).
: "${AXIS_ORDER:=0 1 2}"
# Optional mirror after permutation, e.g. "2" to flip the last axis only.
: "${FLIP_AXES:=}"
# Voxel scale (mm) of the displayed reference volume. Default 0.025 matches
# the Allen 25 µm atlas. Distinct from any training-time stride.
: "${TEMPLATE_VOXEL_SCALE:=0.025}"

# ---- build optional args --------------------------------------------------
COMPARE_ARG=""
if [ -n "$COMPARE" ]; then
    COMPARE_ARG="--compare $COMPARE"
fi
ANN_ARG=""
if [ -n "$ANNOTATIONS_FILE" ] && [ -f "$ANNOTATIONS_FILE" ]; then
    ANN_ARG="--annotations-file $ANNOTATIONS_FILE"
fi
FLIP_ARG=""
if [ -n "$FLIP_AXES" ]; then
    FLIP_ARG="--flip-axes $FLIP_AXES"
fi

# ---- sanity check ---------------------------------------------------------
# The Python viewer auto-discovers all usable sibling runs in the parent
# directory and lets you cycle through them — so RUN_DIR doesn't have to
# be a valid run dir itself, only its parent has to contain at least one
# usable run. Here we check that the PARENT exists; the Python side
# handles "this seed is invalid, fall back to the first usable sibling".
PARENT=$(dirname "$RUN_DIR")
if [ ! -d "$PARENT" ]; then
    echo "ERROR: parent directory $PARENT does not exist."
    echo "Set RUN_DIR to (a subdirectory of) somewhere a training run"
    echo "actually wrote results."
    exit 2
fi
if [ ! -f "$RUN_DIR/config.json" ]; then
    echo "NOTE: $RUN_DIR/config.json does not exist."
    echo "  The viewer will auto-pick the first usable sibling under"
    echo "  $PARENT and let you navigate via the dropdown / prev / next."
    echo ""
fi

echo "================================================================"
echo "Visualize per-lipid GP predictions"
echo "  RUN_DIR       = $RUN_DIR"
[ -n "$COMPARE" ] && echo "  COMPARE       = $COMPARE"
echo "  INITIAL_LIPID = $INITIAL_LIPID"
echo "================================================================"

python manifold/viz/visualize_lipid_gp.py \
    --run-dir "$RUN_DIR" \
    $COMPARE_ARG \
    --reference-file "$REFERENCE_FILE" \
    $ANN_ARG \
    --initial-lipid-name "$INITIAL_LIPID" \
    --initial-lipid-idx "$INITIAL_LIPID_IDX" \
    --gamma "$GAMMA" \
    --point-size "$POINT_SIZE" \
    --brain-point-size "$BRAIN_POINT_SIZE" \
    --axis-order $AXIS_ORDER \
    $FLIP_ARG \
    --template-voxel-scale "$TEMPLATE_VOXEL_SCALE" \
    --brain-render-stride "$BRAIN_RENDER_STRIDE" \
    --color-range-mode "$COLOR_RANGE_MODE" \
    "$@"