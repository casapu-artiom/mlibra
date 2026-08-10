#!/usr/bin/env bash
# =============================================================================
# Reference vs graph Laplacian vs its eigenvector reconstruction.
#
# Three things on one grid: the reference template, the EXACT graph Laplacian
# applied to it (L·f, sparse matvec), and the same field rebuilt from the first
# N eigenmodes (L_N·f) — plus the residual, i.e. what the truncated basis
# misses. Dropdowns pick the graph (faiss vs anatomically weighted) and N;
# nothing recomputes until you press "⟳ Re-render".
#
# Defaults below are chosen to HIT the existing eigvec caches for BOTH graphs
# (stride4 / thresh5 / knn15 / nlist=sqrt=729 / bw0.1 / randomwalk / infl50 /
# root dissolve / modes1000), so switching the graph dropdown is a disk load,
# not an eigensolve.
#
# Eigensolving is OFF by default (the viewer is cache-first): a combination
# with no usable cache downgrades MODES to the largest cached count, or aborts
# listing what IS cached. Set ALLOW_EIGENSOLVE=1 to let it solve — that costs
# minutes and writes ~2 GB per graph at stride 4.
#
# Env overrides, e.g.
#     ./manifold/viz/visualize_laplacian_simple.sh --cache-report   # what can load?
#     MODES=1300 BW=0.05 ./manifold/viz/visualize_laplacian_simple.sh
#     GRAPHS="faiss faiss_atlas_weighted faiss_cluster_weighted" ./...sh
#     ./manifold/viz/visualize_laplacian_simple.sh --no-launch   # headless table
#     ALLOW_EIGENSOLVE=1 BW=0.05 ./manifold/viz/visualize_laplacian_simple.sh
# =============================================================================

# ---- data ------------------------------------------------------------------
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATIONS_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"

# ---- node grid -------------------------------------------------------------
: "${STRIDE:=4}"
: "${THRESHOLD:=5}"

# ---- graphs offered in the dropdown ---------------------------------------
# faiss | faiss_atlas_weighted | anatomical_atlas | faiss_cluster_weighted
: "${GRAPHS:=faiss faiss_atlas_weighted}"
: "${KNN_K:=15}"
: "${N_LIST:=sqrt}"
: "${N_PROBE:=8}"
: "${INFLATION:=50}"
: "${ROOT_HANDLING:=dissolve}"
# Label denoise / hard prune are off by default: they change the graph (prune)
# and erode thin real boundaries (denoise) — see the flags' help text.
: "${DENOISE_LABELS:=0}"
: "${PRUNE_CROSS_REGION:=0}"

# ---- Laplacian / spectrum --------------------------------------------------
: "${LAPLACIAN_NORM:=randomwalk}"
: "${BW:=0.1}"
: "${MODES:=1000}"
: "${INITIAL_MODES:=$MODES}"
: "${NCV_MIN:=-1}"

# ---- field + display -------------------------------------------------------
# σ (in strided voxels) smooths the density before L is applied. σ=0 shows raw
# voxel-scale texture, which swamps L·f; σ≈2 leaves the major anatomical edges
# and puts enough of the field in the resolved band that L_N·f is comparable.
: "${SIGMA:=2}"
: "${GAMMA:=0.6}"
: "${CONTRAST_PCT:=99.5}"
: "${TEMPLATE_OPACITY:=0.08}"
# Draw a display-only sample of the retained nodes, and compare this many
# separated random pairs using geometric kNN shortest paths versus Euclidean
# straight-line distance. The calculation still uses the complete graph.
: "${GRAPH_NODE_DISPLAY_SAMPLE:=50000}"
: "${DISTANCE_PAIRS:=16}"
: "${ANATOMICAL_EDGE_DISPLAY_SAMPLE:=4000}"
# full | strided | both. 'both' adds the reference twice — native 25 µm and as
# the graph samples it — in the same coordinate frame, so toggling the two is
# the stride effect. All node-lattice layers draw nearest-neighbour, so a node
# looks like the stride³ block it is.
: "${TEMPLATE_RES:=both}"
# 3-D rendering of the reference layers (ignored in 2-D). iso | attenuated_mip |
# mip | translucent. 'iso' draws a surface at the node threshold; the MIP modes
# return the max along each ray, and a solid brain saturates nearly every ray,
# so the layer fills its bounding box with white fog (the "glowing cube").
: "${TEMPLATE_RENDERING:=iso}"
# 2 = slice view (blocks are literal squares — clearest for judging stride),
# 3 = volume rendering (prettier, blurs the lattice).
: "${NDISPLAY:=2}"
: "${DEVICE:=cuda}"

ANN_ARG=""
if [ -n "$ANNOTATIONS_FILE" ] && [ -f "$ANNOTATIONS_FILE" ]; then
    ANN_ARG="--annotations-file $ANNOTATIONS_FILE"
fi
SOLVE_ARG=""
if [ -n "$ALLOW_EIGENSOLVE" ] && [ "$ALLOW_EIGENSOLVE" != "0" ]; then
    SOLVE_ARG="--allow-eigensolve"
fi

python manifold/viz/visualize_laplacian_simple.py \
    --template-name reference \
    --reference-file "$REFERENCE_FILE" \
    $ANN_ARG \
    --eigenvector-dir "$EIGENVECTOR_DIR" \
    --stride "$STRIDE" \
    --threshold "$THRESHOLD" \
    --graph-methods $GRAPHS \
    --knn-k "$KNN_K" \
    --n-list "$N_LIST" \
    --n-probe "$N_PROBE" \
    --cross-region-inflation "$INFLATION" \
    --root-handling "$ROOT_HANDLING" \
    --denoise-labels "$DENOISE_LABELS" \
    --prune-cross-region "$PRUNE_CROSS_REGION" \
    --laplacian-norm "$LAPLACIAN_NORM" \
    --graphbandwidth "$BW" \
    --num-modes "$MODES" \
    --initial-modes "$INITIAL_MODES" \
    --ncv-min "$NCV_MIN" \
    --density-smooth-sigma "$SIGMA" \
    --gamma "$GAMMA" \
    --contrast-pct "$CONTRAST_PCT" \
    --template-opacity "$TEMPLATE_OPACITY" \
    --graph-node-display-sample "$GRAPH_NODE_DISPLAY_SAMPLE" \
    --distance-pairs "$DISTANCE_PAIRS" \
    --anatomical-edge-display-sample "$ANATOMICAL_EDGE_DISPLAY_SAMPLE" \
    --template-res "$TEMPLATE_RES" \
    --template-rendering "$TEMPLATE_RENDERING" \
    --ndisplay "$NDISPLAY" \
    --device "$DEVICE" \
    $SOLVE_ARG \
    "$@"
