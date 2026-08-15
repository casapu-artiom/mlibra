#!/usr/bin/env bash
# =============================================================================
# Euclidean vs Manifold kernel playroom — hand-tune both, hit REFIT, read the
# numbers the analysis notebook prints (per-lipid held-out R², the overall
# all/interior/near-border summary, and the covariance-vs-border table).
#
#   - pick a mouse + slice: that sample's MALDI voxels, snapped to template
#     nodes, laid out flat per section
#   - click a point -> k(test, ·) for the SELECTED kernel painted over the slice
#   - extra layers: true / predicted / held-out error for one lipid, plus the
#     atlas regions and their borders
#
# The graph/spectrum defaults below match the shared eigvec cache (stride4 /
# thresh5 / knn15 / nlist=sqrt / bw0.1 / randomwalk / infl50 / root dissolve /
# 1000 modes), so a manifold refit is a disk load, not an eigensolve. Anything
# off that grid needs COMPUTE=1 (a one-time GPU solve, minutes + ~GBs).
#
# Env overrides, e.g.
#     ./manifold/viz/manifold_vs_euclidean_explorer.sh --self-test   # headless report
#     MAN_MODES=300 MAN_LS=8 ./manifold/viz/manifold_vs_euclidean_explorer.sh
#     ATLAS=d7 ANCHORS=5000 ./manifold/viz/manifold_vs_euclidean_explorer.sh
#     SAMPLE=ReferenceAtlas EU_NU=1.5 EU_LS=1.0 ./...sh
#     COMPUTE=1 MAN_BW=0.05 ./manifold/viz/manifold_vs_euclidean_explorer.sh
#
# To reproduce TRAINED kernels verbatim (config + LEARNED lengthscale /
# outputscale / noise) instead of the manual specs, point at run dirs. Per-lipid
# runs (checkpoints/batch_*.pt) also need RUN_LIPID — omit it once to get the
# list of lipid names in the error:
#     MANIFOLD_RUN=/home/casap/mlibra/output/per_lipid/FOLD-2-learnls-manifold-nu2-K2300-stride4-learnls-bs1.0-bd0.01-bw0.1-knn15-faiss_atlas_weighted-5-randomwalk-ind2000-lr0.05-ep10-lbs10-blend0.8-infl50-learndiff10 \
#     EUCLIDEAN_RUN=/home/casap/mlibra/output/per_lipid/FOLD-2-euclidean-ard-matern-nu1.5-ind2000-5-lr0.05-ep10-lbs1010 \
#     RUN_LIPID="PC 35:1 PE 38:1" ./manifold/viz/manifold_vs_euclidean_explorer.sh
# (ind2000, not ind1000, is the right euclidean baseline for these comparisons.)
# =============================================================================

# ---- pooling / analysis ----------------------------------------------------
# 5/15 = LBAE levels (root catch-all is 72%/57% of the brain);
# d5/d7 = CCF depth-cuts, 175/476 regions, no root catch-all.
: "${ATLAS:=15}"
: "${ANCHORS:=3000}"
: "${CELL:=2}"

# ---- which brain's MALDI slices the GUI shows ------------------------------
: "${SAMPLE:=ReferenceAtlas}"

# ---- euclidean kernel (manual) ---------------------------------------------
: "${EU_NU:=0.5}"
: "${EU_LS:=2.0}"
: "${EU_OS:=1.0}"
: "${EU_BORDER:=0.0}"
# ABSOLUTE observation-noise variance (z-units²); 0 = auto-tune per lipid.
: "${EU_NOISE:=0.0}"

# ---- manifold kernel (manual) ----------------------------------------------
: "${MAN_METHOD:=faiss_atlas_weighted}"
: "${MAN_INFL:=50}"
: "${MAN_ROOT:=dissolve}"
# Label denoise / hard prune are off by default: denoise erases much of the true
# boundary before a prune can see it, and the prune shatters the graph.
: "${MAN_DENOISE:=0}"
: "${MAN_PRUNE:=0}"
: "${MAN_KNN_K:=15}"
: "${MAN_N_LIST:=sqrt}"
: "${MAN_N_PROBE:=8}"
: "${MAN_STRIDE:=4}"
: "${MAN_THRESHOLD:=5}"
: "${MAN_NORM:=randomwalk}"
# bw 0.1 hits the cache but makes the graph nearly unweighted (affinity contrast
# ~1.03) and throttles the atlas prior; bw≈0.05 is the honest setting — it needs
# COMPUTE=1 the first time.
: "${MAN_BW:=0.1}"
: "${MAN_MODES:=1000}"
: "${MAN_NU:=2}"
: "${MAN_LS:=5}"
: "${MAN_OS:=1.0}"
: "${MAN_NOISE:=0.0}"

# ---- trained runs (override the manual specs when set) ---------------------
: "${MANIFOLD_RUN:=}"
: "${EUCLIDEAN_RUN:=}"
: "${RUN_LIPID:=}"

# ---- solve missing eigenpairs instead of stopping? -------------------------
: "${COMPUTE:=0}"

# The atlas-weighted graph should be built from the SAME annotation the analysis
# pools/borders use, otherwise the first (pre-REFIT) report mixes two atlases —
# the GUI re-syncs this on every REFIT, this just makes the startup state match.
case "$ATLAS" in
    5)  MAN_ANNOT=/home/casap/mlibra/mlibra_data/level_5annot.npy ;;
    15) MAN_ANNOT=/home/casap/mlibra/mlibra_data/level_15annot.npy ;;
    d5) MAN_ANNOT=/home/casap/mlibra/mlibra_data/ccf_depth5annot.npy ;;
    d7) MAN_ANNOT=/home/casap/mlibra/mlibra_data/ccf_depth7annot.npy ;;
    *)  echo "unknown ATLAS=$ATLAS (expected 5 | 15 | d5 | d7)" >&2; exit 1 ;;
esac

EU_ARG=(--euclidean "nu=$EU_NU,lengthscale=$EU_LS,outputscale=$EU_OS,border_downweight=$EU_BORDER,noise=$EU_NOISE")
MAN_ARG=(--manifold "kind=manifold,knn_method=$MAN_METHOD,inflation=$MAN_INFL,root_handling=$MAN_ROOT,denoise=$MAN_DENOISE,prune=$MAN_PRUNE,knn_k=$MAN_KNN_K,n_list=$MAN_N_LIST,n_probe=$MAN_N_PROBE,stride=$MAN_STRIDE,threshold=$MAN_THRESHOLD,annotations=$MAN_ANNOT,laplacian_norm=$MAN_NORM,bandwidth=$MAN_BW,num_modes=$MAN_MODES,nu=$MAN_NU,lengthscale=$MAN_LS,outputscale=$MAN_OS,noise=$MAN_NOISE")

[ -n "$MANIFOLD_RUN" ]  && MAN_ARG=(--manifold-run "$MANIFOLD_RUN")
[ -n "$EUCLIDEAN_RUN" ] && EU_ARG=(--euclidean-run "$EUCLIDEAN_RUN")

RUN_LIPID_ARG=()
[ -n "$RUN_LIPID" ] && RUN_LIPID_ARG=(--run-lipid "$RUN_LIPID")

COMPUTE_ARG=()
[ "$COMPUTE" != "0" ] && COMPUTE_ARG=(--compute-if-missing)

python "$(dirname "${BASH_SOURCE[0]}")/manifold_vs_euclidean_explorer.py" \
    "${EU_ARG[@]}" \
    "${MAN_ARG[@]}" \
    "${RUN_LIPID_ARG[@]}" \
    --atlas "$ATLAS" \
    --anchors "$ANCHORS" \
    --cell "$CELL" \
    --sample "$SAMPLE" \
    "${COMPUTE_ARG[@]}" \
    "$@"
