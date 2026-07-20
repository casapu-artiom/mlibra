#!/usr/bin/env python
"""Interactive MALDI ↔ kernel correlation explorer.

Pick a mouse + lipid, scroll through that mouse's MALDI slices rendered
flat in 2D, drop a test point, and see three notions of "which points
are taken into account" side by side:

  * eigenvalue (Riemann/manifold) kernel   k_riem(test, ·)
  * plain Matérn (Euclidean) kernel         k_matern(test, ·)
  * diffusion-distance covariance           exp(-D_t(test, ·)²/2σ²)

…all overlaid on the same slices as toggleable colour layers, next to a
live scatter of kernel-weight vs measured lipid intensity (with Pearson /
Spearman r per kernel) so you can quantify which kernel's neighbourhood
actually tracks the lipid.

Design choices (see module for the why):
  (domain)  The graph + eigendecomposition live on the reference-TEMPLATE
            voxels (reused from the cached eigvecs). Each MALDI voxel is
            snapped to its nearest template node; the selected lipid is
            averaged onto nodes. Kernels and lipid therefore share the
            node domain and are directly correlatable.
  (slices)  MALDI `Section`s are not axis-aligned planes, so a slice is a
            *membership* set, not a z=const plane. Each section's points
            are projected onto their own best-fit plane (PCA) → a clean
            flat 2D image, tilt removed.
  (borders) For non-faiss graphs (anatomical_atlas / faiss_atlas_weighted)
            the atlas labels give a border overlay: nodes whose graph
            neighbours cross a region boundary.

Headless check (no display needed):
    python maldi/maldi_kernel_explorer.py ... --self-test
runs the whole numeric pipeline (snap, kernels, correlation) and prints a
report without opening napari.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key, resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
    labels_for_nodes_from_template_clustering, dissolve_root_labels,
    prune_cross_region_edges,
)
from utils import (
    crop_or_stride_volume, reference_ccf_from_subvolume, coord_norm_from_reference,
)


log = logging.getLogger("maldi_kernel_explorer")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- template / graph -------------------------------------------------
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None,
                   help="Atlas labels (.npy). Needed for anatomical/weighted "
                        "knn and for the border overlay.")
    p.add_argument("--eigenvector-dir", required=True,
                   help="Cache dir with knn/ and eigvecs/ subfolders.")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--knn-method",
                   choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted",
                            "faiss_cluster_weighted"],
                   default="anatomical_atlas")
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--n-list", default="sqrt",
                   help="FAISS IVF nlist: an int, or 'sqrt' (default) for "
                        "round(sqrt(N)), matching the training pipeline.")
    p.add_argument("--n-probe", dest="n_probe", default="8",
                   help="FAISS IVF nprobe: an int, or 'sqrt'. Default 8 (recall "
                        "~1.0 in 3D). MUST be > 1 when nlist > 1 -- nprobe=1 with an "
                        "IVF index builds a FRAGMENTED graph (~nlist components).")
    p.add_argument("--cross-region-inflation", dest="cross_region_inflation",
                   type=float, default=10.0,
                   help="Soft prior for faiss_atlas_weighted / "
                        "faiss_cluster_weighted (inflate cross-region edges).")
    p.add_argument("--root-handling", dest="root_handling",
                   choices=["dissolve", "ignore", "cross"], default="dissolve",
                   help="faiss_atlas_weighted only — how to treat the atlas's "
                        "label-0 'root' catch-all tissue (~6%% of nodes, threaded "
                        "throughout). 'dissolve' (default): reassign each root node "
                        "to its nearest real region before inflating, so root "
                        "tissue smooths inside a region instead of spraying the "
                        "prior through interiors (frees ~44%% of inflated edges; "
                        "real region↔region confinement preserved). 'ignore': keep "
                        "root but don't treat root-touching edges as cross. 'cross': "
                        "legacy — inflate every root-touching edge "
                        "(treat_zero_as_cross=True); reuses old eigvec caches.")
    # faiss_cluster_weighted: data-driven, lipid-free template clustering as the
    # label source instead of the atlas.
    p.add_argument("--cluster-k", dest="cluster_k", type=int, default=64,
                   help="faiss_cluster_weighted: number of template clusters.")
    p.add_argument("--cluster-spatial-weight", dest="cluster_spatial_weight",
                   type=float, default=1.0,
                   help="faiss_cluster_weighted: weight on coords (higher = "
                        "more spatially contiguous regions; 0 = speckly).")
    p.add_argument("--cluster-seed", dest="cluster_seed", type=int, default=0)
    p.add_argument("--cluster-fit-subsample", dest="cluster_fit_subsample",
                   type=int, default=40000)
    # ---- label denoise + hard prune (weighted methods) --------------------
    # Both act on the region labels the graph weighting is built on (atlas or
    # clusters), AFTER inflation — they refine the HARD topology + the border/
    # purity diagnostics, not the soft inflation. Prune changes edges, so it
    # (and the denoise that shapes it) go into the eigvec cache key.
    p.add_argument("--denoise-labels", dest="denoise_labels", type=int, default=0,
                   help="Majority-vote label smoothing passes over the graph: each "
                        "node adopts the most common label among its neighbours, so "
                        "speck / polluted nodes are absorbed into the surrounding "
                        "region. 0 = off; 2-3 cleans most specks. Feeds the border/"
                        "purity overlays and --prune-cross-region. NOTE: erodes thin "
                        "real boundaries, so it can WEAKEN region confinement — "
                        "prefer --root-handling dissolve for the atlas catch-all.")
    p.add_argument("--prune-cross-region", dest="prune_cross_region", type=float,
                   default=0.0,
                   help="Fraction of cross-region edges to REMOVE from the graph — a "
                        "HARD cut, vs --cross-region-inflation's soft down-weight. "
                        "0 = off, 0.97 = keep 3%% of crossings, 1.0 = remove all. "
                        "Connectivity-preserving: any node the prune would isolate "
                        "keeps its strongest edge, so no specks. Changes edges → "
                        "triggers a fresh eigensolve (distinct cache key).")
    p.add_argument("--laplacian-norm", choices=["symmetric", "randomwalk"],
                   default="symmetric")
    p.add_argument("--graphbandwidth", type=float, required=True)
    p.add_argument("--num-modes", type=int, default=1000)
    p.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1)
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute-eigvecs", action="store_true")

    # ---- MALDI data -------------------------------------------------------
    p.add_argument("--maldi-file", required=True,
                   help="MALDI parquet with xccf/yccf/zccf + Sample + Section "
                        "+ lipid columns.")
    p.add_argument("--coord-cols", nargs=3, default=["xccf", "yccf", "zccf"],
                   metavar=("XCOL", "YCOL", "ZCOL"))
    p.add_argument("--sample-col", default="Sample")
    p.add_argument("--section-col", default="Section")
    p.add_argument("--sample", default=None,
                   help="Initial mouse (Sample). Default: first found.")
    p.add_argument("--lipid", default=None,
                   help="Initial lipid column. Default: first numeric non-coord "
                        "column.")
    p.add_argument("--axis-order", nargs=3, type=int, default=[0, 1, 2],
                   metavar=("A0", "A1", "A2"),
                   help="Permutation applied to (xccf,yccf,zccf) before snapping "
                        "to template nodes. Default identity matches the Allen "
                        "CCF convention. Use if the snap looks mis-registered.")
    p.add_argument("--snap-max-mm", type=float, default=1.0,
                   help="Drop MALDI points whose nearest template node is "
                        "farther than this (mm); they fall outside the mask.")

    # ---- kernels ----------------------------------------------------------
    p.add_argument("--nu", type=float, default=1.5,
                   help="Smoothness ν for the Riemann spectral density "
                        "(2ν/ℓ²+λ)^(-ν).")
    p.add_argument("--matern-nu", dest="matern_nu", type=float, default=None,
                   help="Smoothness ν for the Euclidean Matérn kernel. "
                        "Defaults to --nu if unset.")
    p.add_argument("--lengthscale", type=float, default=1.0,
                   help="Matérn lengthscale (normalized-coord units).")
    p.add_argument("--outputscale", type=float, default=1.0,
                   help="ScaleKernel outputscale (signal variance) multiplying "
                        "the Riemann + Matérn covariance, i.e. k = outputscale·"
                        "k_base. Adjustable live via the slider. NOTE: this is "
                        "scale-invariant for the correlation r/ρ, the near-set "
                        "percentile threshold, region purity, and the colour "
                        "stretch — only the raw weight[min,max] magnitudes (and "
                        "the true prior self-covariance) change.")
    p.add_argument("--diffusion-time", dest="diffusion_time", type=float,
                   default=1.0, help="Heat time t for the diffusion distance.")
    p.add_argument("--diffusion-sigma", dest="diffusion_sigma", type=float,
                   default=1.0,
                   help="Covariance bandwidth as a MULTIPLE of the median "
                        "diffusion distance (1.0 = median; <1 = tighter, more "
                        "contrast). Not an absolute σ — see diffusion_cov.")

    # ---- trained models (a fold): use fitted hypers + held-out error ------
    p.add_argument("--euclidean-run", dest="euclidean_run", default=None,
                   help="Trained EUCLIDEAN per-lipid run dir (a fold). When set, "
                        "the Matérn layer uses that lipid's fitted ARD "
                        "lengthscales, and a held-out-error layer is added.")
    p.add_argument("--riemann-run", dest="riemann_run", default=None,
                   help="Trained MANIFOLD/Riemann per-lipid run dir (a fold). Its "
                        "fitted lengthscale/ν drive the Riemann layer, and a "
                        "held-out-error layer is added. Launch with the SAME "
                        "graph params (stride/knn/inflation/modes) as the run so "
                        "the eigvecs match.")
    p.add_argument("--skip-trained-validation", dest="skip_trained_validation",
                   action="store_true",
                   help="Don't abort when a trained run's graph params "
                        "(num_modes, stride, knn, inflation, …) differ from the "
                        "explorer's launch args (only warn instead).")
    p.add_argument("--kernel-normalize", action="store_true", default=True,
                   help="Cosine-normalize the Riemann kernel (self-cov=1).")
    p.add_argument("--near-kernel", dest="near_kernel",
                   choices=["riemann", "matern", "diffusion", "heat"],
                   default="riemann",
                   help="Which kernel's weight defines the highlighted "
                        "'near set' overlay.")
    p.add_argument("--heat-time", dest="heat_time", type=float, default=0.5,
                   help="Diffusion time τ for the OPERATOR heat kernel "
                        "e^{-τL} e_t (computed straight from the Laplacian, no "
                        "eigenvectors). Larger τ = more spread. NOTE: this L is "
                        "graphbandwidth-scaled (λ_max ~ 100s, not 2), so the "
                        "informative range is small τ (~0.05–1); scrub the slider. "
                        "This is the eigenvector-free control for the spectral "
                        "riemann / diffusion layers.")
    p.add_argument("--heat-steps", dest="heat_steps", type=int, default=None,
                   help="Integrator steps for the operator heat kernel "
                        "(I-τL/m)^m. Default: auto = max(40, 2τ·λ_max).")
    p.add_argument("--near-pct", dest="near_pct", type=float, default=90.0,
                   help="Percentile threshold for the 'near set': 90 = light "
                        "up the top 10%% highest-weight nodes for that kernel.")

    # ---- live held-out predictive eval (no trained run needed) ------------
    p.add_argument("--eval-holdout", dest="eval_holdout", type=int, default=10000,
                   help="Held-out prediction↔actual check on the SELECTED lipid: "
                        "randomly hold out up to this many lipid-covered nodes as "
                        "test, GP-fit the rest with the current kernels, and report "
                        "MSE + Pearson/Spearman (pred vs actual) live. Capped at "
                        "half the covered nodes. 0 disables.")
    p.add_argument("--eval-train-cap", dest="eval_train_cap", type=int, default=3000,
                   help="Max train nodes for the held-out GP (bounds the dense "
                        "Matérn solve). Larger = better fit but slower per redraw.")
    p.add_argument("--eval-noise", dest="eval_noise", type=float, default=0.1,
                   help="GP observation noise for the held-out eval, as a FRACTION "
                        "of the train-lipid variance (jitter = eval_noise·var(y)).")
    p.add_argument("--eval-seed", dest="eval_seed", type=int, default=0,
                   help="Seed for the held-out train/test split (stable split).")

    # ---- display ----------------------------------------------------------
    p.add_argument("--point-size", type=float, default=1.3,
                   help="Point size as a MULTIPLE of the section's median "
                        "inter-point spacing (auto-computed per slice). ~1.0 "
                        "tiles the slice; >1.5 starts to overlap/smear. This "
                        "is a multiplier, not an absolute mm size.")
    p.add_argument("--test-marker-scale", dest="test_marker_scale",
                   type=float, default=0.02,
                   help="Test-point star size as a fraction of the slice's "
                        "spatial extent (0.02 = 2%% of the diagonal). "
                        "Scale-robust, so it stays a small glyph on any slice.")
    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--scatter-max", type=int, default=6000,
                   help="Max points drawn in the correlation scatter.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--self-test", action="store_true",
                   help="Run the numeric pipeline headless and exit (no napari).")
    p.add_argument("-v", "--verbose", action="store_true")
    return vars(p.parse_args())


# =============================================================================
# Label denoise + hard prune (ported from visualize_kernels.py)
# =============================================================================
def denoise_labels_majority_vote(labels: np.ndarray, ei_np: np.ndarray,
                                 n_iters: int) -> np.ndarray:
    """Majority-vote label smoothing over the graph. Each node adopts the most
    common label among its graph neighbours (+ itself), iterated — so speck /
    'polluted' minority nodes are absorbed into the region they sit in and the
    partition becomes spatially coherent."""
    lab = labels.astype(np.int64).copy()
    N = lab.shape[0]
    for it in range(int(n_iters)):
        uniq, comp = np.unique(lab, return_inverse=True)      # compact ids 0..K-1
        votes = np.zeros((N, uniq.size), dtype=np.int32)
        np.add.at(votes, (ei_np[0], comp[ei_np[1]]), 1)       # neighbour votes (both dirs)
        np.add.at(votes, (ei_np[1], comp[ei_np[0]]), 1)
        np.add.at(votes, (np.arange(N), comp), 1)             # self-vote (tie -> stay)
        new = uniq[votes.argmax(1)]
        changed = int((new != lab).sum())
        lab = new
        log.info(f"denoise-labels iter {it + 1}: {changed:,} nodes relabelled")
        if changed == 0:
            break
    return lab


# =============================================================================
# Graph + eigendecomposition on the template voxels (all knn methods)
# =============================================================================
def setup_graph(args: dict) -> dict:
    device = torch.device(args["device"])
    template_full = np.load(args["reference_file"])
    annotations_full = (np.load(args["annotations_file"])
                        if args["annotations_file"] else None)

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_full, annotations_full, stride=args["stride"],
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, args["threshold"],
    )  # (N, 3) mm in (z, y, x)/template-axis order
    node_ccf = np.asarray(reference_ccf, dtype=np.float32)
    reference_nodes_mm = torch.tensor(node_ccf, dtype=torch.float32)
    # Isotropic whole-brain normalization -- the single source of truth that
    # training and SLEPc use for BOTH the graph metric AND dissolve_root_labels.
    # A per-axis std warps the metric and silently reassigns ~1000 root nodes to
    # different regions -> a different cross-region edge set -> a different prune
    # -> the eigvec-cache n_edges mismatch that stopped the SLEPc cache being
    # reused. coord_mean/coord_std are consumed only here, so this is a drop-in.
    coord_mean, coord_std = coord_norm_from_reference(template_full)
    reference_nodes = ((reference_nodes_mm - coord_mean) / coord_std).to(device)

    N = reference_nodes.shape[0]
    nlist = resolve_nlist(args["n_list"], N)   # 'sqrt' -> round(sqrt(N))
    nprobe = resolve_nprobe(args["n_probe"], nlist)
    log.info(f"{N:,} template graph nodes  (faiss nlist={nlist} nprobe={nprobe})")

    eigenvector_dir = Path(args["eigenvector_dir"])
    graphs = KnnGraphCache(cache_dir=eigenvector_dir / "knn", verbose=True)
    graph_key_parts = {
        "template": args["template_name"], "stride": args["stride"],
        "thresh": args["threshold"], "method": args["knn_method"],
        "k": args["knn_k"], "nlist": nlist, "bbox": None,
    }
    atlas_stem = (Path(args["annotations_file"]).stem
                  if args["annotations_file"] else "noatlas")
    _legacy_atlas = (atlas_stem == "level_15annot")
    if args["knn_method"] == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
    force_graph = args["force_recompute_graph"]

    # `graph_labels` = the per-node region labels the graph's weighting is built
    # on (atlas or template clusters). Drives the border overlay + purity metric
    # so they measure the SAME partition the kernel was told to respect.
    # `labels_zero_is_region` = whether label 0 is a real region (clusters) or
    # background/gap (atlas), matching the inflation convention.
    graph_labels = None
    labels_zero_is_region = False

    if args["knn_method"] == "faiss":
        graph_key = make_graph_key(graph_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss", coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe, extra=graph_key_parts,
            force_recompute=force_graph, device=device,
        )
    elif args["knn_method"] == "anatomical_atlas":
        graph_key = make_graph_key(graph_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas", volume=sub_volume,
            threshold=args["threshold"], atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts, force_recompute=force_graph, device=device,
        )
        if sub_atlas is not None:
            graph_labels = labels_for_nodes_from_sub_atlas(
                sub_volume, sub_atlas, args["threshold"])
    elif args["knn_method"] == "faiss_atlas_weighted":
        if sub_atlas is None:
            raise ValueError("faiss_atlas_weighted requires --annotations-file.")
        base_key_parts = dict(graph_key_parts, method="faiss")
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss", coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe, extra=base_key_parts,
            force_recompute=force_graph, device=device,
        )
        graph_labels = labels_for_nodes_from_sub_atlas(
            sub_volume, sub_atlas, args["threshold"])
        # Root handling: the label-0 catch-all is real tissue, not background —
        # treating it as cross (legacy) inflates ~half the edges for almost no
        # region↔region confinement gain and fragments interiors. Default is to
        # dissolve it into the nearest real region.
        root_mode = args.get("root_handling", "dissolve")
        if root_mode == "dissolve":
            graph_labels = dissolve_root_labels(
                graph_labels, reference_nodes.detach().cpu().numpy())
        treat_zero = (root_mode == "cross")
        inflation = float(args["cross_region_inflation"])
        edge_index, edge_value, info = inflate_cross_region_edges(
            edge_index, edge_value, graph_labels,
            inflation=inflation, treat_zero_as_cross=treat_zero)
        log.info(f"faiss_atlas_weighted (root={root_mode}): "
                 f"{info['n_cross']:,}/{info['n_total']:,} "
                 f"cross-region edges ×{inflation:g}")
        _base_wt = (f"atlas_x{inflation:g}" if _legacy_atlas
                    else f"{atlas_stem}_x{inflation:g}")
        # Keep legacy 'cross' key un-suffixed so existing eigvec caches still load;
        # dissolve/ignore change edge_value → must get distinct cache keys.
        graph_key_parts["weighting"] = (_base_wt if root_mode == "cross"
                                        else f"{_base_wt}_root{root_mode}")
        graph_key = make_graph_key(graph_key_parts)
    elif args["knn_method"] == "faiss_cluster_weighted":
        # Data-driven, lipid-free parcellation: cluster the reference template
        # itself. Base topology is the plain faiss graph; cluster labels only
        # reweight cross-cluster edges (same as atlas, different label source).
        base_key_parts = dict(graph_key_parts, method="faiss")
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss", coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe, extra=base_key_parts,
            force_recompute=force_graph, device=device,
        )
        cluster_k = int(args["cluster_k"])
        sw = float(args["cluster_spatial_weight"])
        cseed = int(args["cluster_seed"])
        graph_labels = labels_for_nodes_from_template_clustering(
            sub_volume, args["threshold"], n_clusters=cluster_k,
            spatial_weight=sw, fit_subsample=int(args["cluster_fit_subsample"]),
            seed=cseed)
        labels_zero_is_region = True   # cluster id 0 is a real region
        inflation = float(args["cross_region_inflation"])
        # treat_zero_as_cross=False: cluster 0 is a region, not a background gap.
        edge_index, edge_value, info = inflate_cross_region_edges(
            edge_index, edge_value, graph_labels,
            inflation=inflation, treat_zero_as_cross=False)
        log.info(f"faiss_cluster_weighted (k={cluster_k}, sw={sw:g}): "
                 f"{info['n_cross']:,}/{info['n_total']:,} cross-cluster edges "
                 f"×{inflation:g}")
        graph_key_parts["weighting"] = (
            f"tmplclust_k{cluster_k}_sw{sw:g}_s{cseed}_x{inflation:g}")
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"Unknown --knn-method {args['knn_method']!r}")

    # ---- label denoise + hard prune (after inflation) --------------------
    # These refine the HARD topology + the label-derived diagnostics (border/
    # purity), not the soft inflation, mirroring visualize_kernels.py. Denoise
    # smooths `graph_labels`; prune uses those labels to cut cross-region edges.
    # Prune changes edges → the eigvecs change, so prune (+ the denoise that
    # shaped it) go into the eigvec cache key. Denoise alone leaves edges
    # untouched, so it does NOT change the key (existing eigvec caches stay
    # valid whenever prune is off — the common case).
    n_denoise = int(args.get("denoise_labels", 0) or 0)
    if n_denoise > 0 and graph_labels is not None:
        graph_labels = denoise_labels_majority_vote(
            graph_labels, edge_index.cpu().numpy(), n_denoise)
    prune = float(args.get("prune_cross_region", 0.0) or 0.0)
    if prune > 0.0 and graph_labels is not None:
        edge_index, edge_value = prune_cross_region_edges(
            edge_index, edge_value, graph_labels, prune, labels_zero_is_region)
        graph_key_parts["prune"] = f"{prune:g}"
        if n_denoise > 0:
            graph_key_parts["denoise"] = n_denoise
        graph_key = make_graph_key(graph_key_parts)

    bw = float(args["graphbandwidth"])
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, N, torch.tensor(bw, device=device),
        args["laplacian_norm"],
    )

    eigvec_key_parts = {"graph": graph_key, "norm": args["laplacian_norm"],
                        "bw": bw, "modes": args["num_modes"]}
    eigvec_key = make_eig_key(eigvec_key_parts)
    ncv_min = resolve_ncv_min(args["num_modes"], args.get("ncv_min", -1))
    solver = LaplacianEigensolver(num_modes=args["num_modes"], backend="cupy",
                                  tol=1e-4, ncv_min=ncv_min, verbose=True)
    eigval, eigvec = solver.compute_or_load(
        laplacian_op, cache_dir=eigenvector_dir / "eigvecs", key=eigvec_key,
        graphbandwidth=bw, laplacian_normalization=args["laplacian_norm"],
        extra=eigvec_key_parts, force_recompute=args["force_recompute_eigvecs"],
        device=device,
    )
    log.info(f"Loaded {eigvec.shape[1]} eigenmodes")
    if eigvec.shape[0] != N:
        raise RuntimeError(
            f"Node-count mismatch: {N} template nodes but eigvec has "
            f"{eigvec.shape[0]} rows. The cached graph/eigvecs were built on a "
            f"different node set (stride/threshold/augment). Delete the stale "
            f"cache under {eigenvector_dir} or use --force-recompute-eigvecs.")

    # Border mask + boundary diagnostic use the SAME labels the graph weighting
    # was built on (atlas or clusters), so "does the kernel respect the border"
    # is measured against the partition it was actually told about.
    node_labels = graph_labels
    border_mask = None
    if node_labels is not None:
        node_voxel_idx = np.argwhere(sub_volume > args["threshold"])
        border_mask = border_from_node_labels(
            node_labels, node_voxel_idx, sub_volume.shape)
        log.info(f"region border nodes: {int(border_mask.sum()):,} / {N:,}")
        report_boundary_weighting(
            edge_index, edge_value, node_labels, bw,
            treat_zero_as_cross=not labels_zero_is_region)

    # Largest eigenvalue (operator power iteration) sizes the heat-kernel
    # integrator; computed once here so the interactive layer is cheap.
    lam_max = estimate_lam_max(laplacian_op, N, device)
    log.info(f"operator λ_max ≈ {lam_max:.4g} (for heat-kernel integrator)")

    return dict(
        device=device, N=N,
        node_ccf=node_ccf,                 # (N,3) mm, for MALDI snapping
        reference_nodes=reference_nodes,   # (N,3) normalized, for Matérn
        eigval=eigval, eigvec=eigvec,      # on device
        edge_index=edge_index,
        laplacian_op=laplacian_op,         # for the operator heat kernel
        lam_max=lam_max,
        node_labels=node_labels, border_mask=border_mask,
        labels_zero_is_region=labels_zero_is_region,
        knn_method=args["knn_method"],
        trained=_load_trained(args),
    )


def _load_trained(args: dict):
    """Load the optional euclidean / manifold trained runs (folds), or None."""
    if not (args.get("euclidean_run") or args.get("riemann_run")):
        return None
    tr = {
        "eu": load_trained_run(args["euclidean_run"]) if args.get("euclidean_run") else None,
        "ri": load_trained_run(args["riemann_run"]) if args.get("riemann_run") else None,
        "err_cache": {},
        "metrics_cache": {},
    }
    _validate_trained(args, tr, strict=not args.get("skip_trained_validation", False))
    return tr


def report_boundary_weighting(edge_index: torch.Tensor,
                              edge_value: torch.Tensor,
                              node_labels: np.ndarray, bw: float,
                              treat_zero_as_cross: bool = True) -> dict:
    """Empirically check whether cross-region edges are actually down-weighted
    in the loaded graph. Compares the heat-kernel weight w = exp(-d²/4bw²) of
    within-region vs cross-region edges (label 0 counts as cross only when it's
    a background/gap — atlas — not a real region — clusters).

    A large suppression ratio (within/cross ≫ 1) = the boundary is effectively
    weighted; ≈ 1 = the prior isn't biting (raise --cross-region-inflation or
    lower --graphbandwidth).
    """
    labels = torch.as_tensor(node_labels, dtype=torch.long,
                             device=edge_index.device)
    s, d = edge_index[0], edge_index[1]
    cross = labels[s] != labels[d]
    if treat_zero_as_cross:
        cross = cross | (labels[s] == 0) | (labels[d] == 0)
    w = torch.exp(-edge_value.detach().to(torch.float64) / (4.0 * bw * bw))

    def _med(x):
        return float(x.median()) if x.numel() else float("nan")

    w_in, w_cr = _med(w[~cross]), _med(w[cross])
    supp = (w_in / w_cr) if w_cr > 0 else float("inf")
    log.info(
        "boundary weighting: within-region edge w (median)=%.3g, "
        "cross-region w=%.3g  →  cross edges ~%.1f× weaker  "
        "(%.1f%% of edges are cross-region)",
        w_in, w_cr, supp, 100.0 * float(cross.float().mean()),
    )
    if supp < 2.0:
        log.warning(
            "cross-region edges are barely suppressed (%.1f×). The anatomical "
            "prior is weak here — raise --cross-region-inflation or lower "
            "--graphbandwidth for the boundary to actually bite.", supp)
    return {"w_within": w_in, "w_cross": w_cr, "suppression": supp}


def border_from_node_labels(node_labels: np.ndarray,
                            node_voxel_idx: np.ndarray,
                            shape: tuple) -> np.ndarray:
    """Border mask from per-node labels (atlas or clusters), by painting them
    into a label volume and taking 6-connected label changes. Works uniformly
    for any label source; +1 offset keeps a real label 0 distinct from the
    empty (non-tissue) background."""
    L = np.zeros(shape, dtype=np.int64)
    z, y, x = node_voxel_idx.T
    L[z, y, x] = node_labels.astype(np.int64) + 1
    return atlas_border_at_nodes(L, node_voxel_idx)


def region_color_table(node_labels: np.ndarray, zero_is_region: bool):
    """Stable one-colour-per-region-id lookup for the region overlay.

    Returns ``(code, rgba)`` where ``code`` is a (N,) int array mapping each
    node to a row of the (n_regions, 4) ``rgba`` table, so per-slice colouring
    is a pure gather ``rgba[code[nodes]]`` (same colour for a region across all
    slices). Hues are spread by the golden ratio so numerically adjacent region
    ids stay visually distinct. Label 0 renders as faint grey when it is
    background (atlas), or as a normal coloured region when it is a real cluster.
    """
    from matplotlib.colors import hsv_to_rgb
    uniq = np.unique(node_labels)
    code = np.searchsorted(uniq, node_labels).astype(np.int64)
    rgba = np.zeros((len(uniq), 4), np.float32)
    for i, lab in enumerate(uniq):
        if lab == 0 and not zero_is_region:
            rgba[i] = (0.5, 0.5, 0.5, 0.15)          # background: barely there
            continue
        h = (i * 0.6180339887) % 1.0                 # golden-ratio hue spread
        rgba[i, :3] = hsv_to_rgb((h, 0.65, 0.95))
        rgba[i, 3] = 0.9
    return code, rgba


def atlas_border_at_nodes(sub_atlas: np.ndarray,
                          node_voxel_idx: np.ndarray) -> np.ndarray:
    """True anatomical borders from the atlas VOXEL GRID: a voxel is on a
    border if any of its 6 grid neighbours has a different label. This gives
    thin region outlines, unlike a KNN-graph criterion (whose long-range,
    region-agnostic edges flag almost every node near a boundary).

    Returns a (N,) bool mask aligned to the graph nodes (tissue voxels in
    `np.where` / `np.argwhere` C-order).
    """
    A = sub_atlas
    b = np.zeros(A.shape, dtype=bool)
    b[:-1] |= A[:-1] != A[1:]
    b[1:] |= A[:-1] != A[1:]
    b[:, :-1] |= A[:, :-1] != A[:, 1:]
    b[:, 1:] |= A[:, :-1] != A[:, 1:]
    b[:, :, :-1] |= A[:, :, :-1] != A[:, :, 1:]
    b[:, :, 1:] |= A[:, :, :-1] != A[:, :, 1:]
    z, y, x = node_voxel_idx.T
    return b[z, y, x]


# =============================================================================
# MALDI → node snapping and per-node lipid aggregation
# =============================================================================
def snap_points_to_nodes(coords_mm: np.ndarray, node_ccf: np.ndarray,
                         axis_order=(0, 1, 2), max_mm: float = 1.0):
    """Nearest template node for each MALDI voxel (KD-tree in mm space).

    Returns (node_idx, valid_mask) where valid_mask drops points whose
    nearest node is farther than `max_mm` (outside the tissue mask).
    """
    from scipy.spatial import cKDTree
    n = node_ccf.shape[0]
    pts = coords_mm[:, list(axis_order)].astype(np.float32)
    tree = cKDTree(node_ccf)

    # Pre-filter non-finite query rows: some scipy versions RAISE on NaN, others
    # return the sentinel idx == n. Query only finite rows, and treat the rest
    # (plus any sentinel idx >= n) as unmatched. This keeps every returned index
    # in [0, n), so it can never index an N-sized array out of bounds — the
    # cause of "index N out of bounds for axis 0 with size N".
    finite = np.isfinite(pts).all(axis=1)
    idx = np.zeros(pts.shape[0], dtype=np.int64)
    dist = np.full(pts.shape[0], np.inf, dtype=np.float64)
    if finite.any():
        d, i = tree.query(pts[finite], k=1)
        dist[finite] = np.asarray(d)
        idx[finite] = np.asarray(i)
    oob = (~finite) | ~np.isfinite(dist) | (idx >= n) | (idx < 0)
    idx = np.where(oob, 0, idx).astype(np.int64)
    valid = np.isfinite(dist) & (dist <= max_mm) & ~oob
    if oob.any():
        log.info("snap: %d/%d query points unmatchable (non-finite coords or "
                 "no neighbour) — dropped", int(oob.sum()), oob.size)
    return idx, valid


def aggregate_lipid_to_nodes(node_idx: np.ndarray, lipid: np.ndarray,
                             N: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean lipid intensity per node + coverage count. Nodes with no MALDI
    point get NaN."""
    finite = np.isfinite(lipid)
    ni, lv = node_idx[finite], lipid[finite].astype(np.float64)
    summ = np.zeros(N); cnt = np.zeros(N)
    np.add.at(summ, ni, lv)
    np.add.at(cnt, ni, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, summ / cnt, np.nan)
    return mean.astype(np.float32), cnt.astype(np.int64)


# =============================================================================
# Kernels — covariance from one test node to all nodes
# =============================================================================
def riemann_cov(eigval, eigvec, t: int, nu: float, lengthscale: float,
                num_modes: int, normalize: bool = True) -> np.ndarray:
    """k_riem(t, j) = Σ_k s(λ_k) φ_k(t) φ_k(j),  s(λ)=(2ν/ℓ²+λ)^(-ν).

    With normalize=True returns the cosine-normalized version (self-cov=1),
    matching the trained kernel's `normalize_features` path.
    """
    K = int(min(num_modes, eigvec.shape[1]))
    lam = eigval[:K].clamp(min=0.0)
    sd = (2.0 * nu / (lengthscale ** 2) + lam).pow(-nu)     # (K,)
    phi = eigvec[:, :K]                                     # (N, K)
    w = sd * phi[t]                                         # (K,)
    cov = phi @ w                                           # (N,)
    if normalize:
        node_norm = (phi.pow(2) * sd).sum(dim=1).clamp(min=1e-12).sqrt()
        cov = cov / (node_norm * node_norm[t])
    return cov.detach().cpu().numpy()


def matern_cov(coords, t: int, lengthscale: float, nu: float) -> np.ndarray:
    """Euclidean Matérn kernel between node t and all nodes (normalized coords)."""
    d = torch.sqrt(((coords - coords[t]) ** 2).sum(dim=-1)).clamp(min=0.0)
    r = d / lengthscale
    if nu == 0.5:
        k = torch.exp(-r)
    elif nu == 1.5:
        s = (3 ** 0.5) * r
        k = (1.0 + s) * torch.exp(-s)
    elif nu == 2.5:
        s = (5 ** 0.5) * r
        k = (1.0 + s + (5.0 / 3.0) * r ** 2) * torch.exp(-s)
    else:
        k = torch.exp(-0.5 * r ** 2)
    return k.detach().cpu().numpy()


def matern_cov_ard(coords, t: int, ard_ls, nu: float) -> np.ndarray:
    """ARD Matérn correlation from node t (per-dimension lengthscales) — the
    trained euclidean kernel's form."""
    ard = torch.as_tensor(np.asarray(ard_ls, np.float32), device=coords.device)
    d = torch.sqrt((((coords - coords[t]) / ard) ** 2).sum(dim=-1)).clamp(min=0.0)
    if nu == 0.5:
        k = torch.exp(-d)
    elif nu == 1.5:
        s = (3 ** 0.5) * d; k = (1.0 + s) * torch.exp(-s)
    elif nu == 2.5:
        s = (5 ** 0.5) * d; k = (1.0 + s + (5.0 / 3.0) * d ** 2) * torch.exp(-s)
    else:
        k = torch.exp(-0.5 * d ** 2)
    return k.detach().cpu().numpy()


# =============================================================================
# Trained-model support — use fitted hyperparameters + held-out fold error
# instead of the hand-set kernels.
# =============================================================================
def _safe_slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).replace(":", "-")).strip("_")


def _constraint_transform(ms, raw_key):
    """Recover a constrained gpytorch parameter (lengthscale) from its raw value
    + stored constraint bounds (Interval / GreaterThan / LessThan / softplus)."""
    import gpytorch
    raw = ms[raw_key]
    lb = ms.get(raw_key + "_constraint.lower_bound")
    ub = ms.get(raw_key + "_constraint.upper_bound")
    fin = lambda t: t is not None and bool(torch.isfinite(torch.as_tensor(t)).all())
    if fin(lb) and fin(ub):
        return gpytorch.constraints.Interval(lb, ub).transform(raw)
    if fin(lb):
        return gpytorch.constraints.GreaterThan(lb).transform(raw)
    if fin(ub):
        return gpytorch.constraints.LessThan(ub).transform(raw)
    return torch.nn.functional.softplus(raw)


def load_trained_run(run_dir: str) -> dict:
    """Load a trained per-lipid run (one fold): fitted kernel hypers per lipid +
    the directory (for held-out predictions). Family read from config.json."""
    import json
    run = Path(run_dir)
    cfg = json.load(open(run / "config.json"))
    family = cfg.get("kernel_family") or cfg.get("kernel") or "manifold"
    per = {}
    for ck in sorted((run / "checkpoints").glob("batch_*.pt")):
        d = torch.load(ck, map_location="cpu", weights_only=False)
        ms = d["model_state"]
        nu = float(d.get("args", {}).get("nu", cfg.get("nu", 1.5)))
        names = [str(n) for n in d["lipid_names"]]
        if family == "euclidean":
            ls = _constraint_transform(
                ms, "covar_module.base_kernel.raw_lengthscale").detach().numpy()
            for k, nm in enumerate(names):
                per[nm] = dict(ard=ls[k, 0, :].astype(np.float64), nu=nu)
        else:
            key = "covar_module.base_kernel.base_kernel.raw_lengthscale"
            if key not in ms:
                key = "covar_module.base_kernel.kernels.0.base_kernel.raw_lengthscale"
            ls = _constraint_transform(ms, key).detach().numpy().ravel()
            for k, nm in enumerate(names):
                per[nm] = dict(ls=float(ls[k] if ls.size > k else ls[0]), nu=nu)
    log.info("trained %s run: %d lipids from %s", family, len(per), run.name)
    return dict(family=family, per=per, dir=run, cfg=cfg)


def _validate_trained(args: dict, tr: dict, strict: bool = True):
    """Abort (or warn) if a trained run's graph/coordinate params differ from
    the explorer's launch args — a mismatch means the loaded eigvecs/coord frame
    aren't the ones the kernel was fit on, so the trained kernel is meaningless.
    Manifold checks the full graph+mode stack; euclidean only the coord frame
    (stride/threshold)."""
    def _mismatch(cfg, keys, argmap):
        out = []
        for k in keys:
            if k not in cfg or argmap.get(k) is None:
                continue
            rv, av = cfg[k], argmap[k]
            same = (abs(float(rv) - float(av)) < 1e-6
                    if isinstance(rv, (int, float)) and isinstance(av, (int, float))
                    else str(rv) == str(av))
            if not same:
                out.append((k, av, rv))
        return out

    problems = {}
    ri = tr.get("ri")
    if ri is not None:
        cfg = dict(ri["cfg"])
        gb = cfg.get("graphbandwidth", cfg.get("graphbandwidth_init"))
        if gb is not None:
            cfg["graphbandwidth"] = gb
        keys = ["num_modes", "stride", "threshold", "knn_method", "knn_k",
                "laplacian_norm", "cross_region_inflation", "graphbandwidth"]
        argmap = dict(num_modes=args["num_modes"], stride=args["stride"],
                      threshold=args["threshold"], knn_method=args["knn_method"],
                      knn_k=args["knn_k"], laplacian_norm=args["laplacian_norm"],
                      cross_region_inflation=args["cross_region_inflation"],
                      graphbandwidth=args["graphbandwidth"])
        # cluster-weighted graphs also depend on the clustering params
        if str(cfg.get("knn_method")) == "faiss_cluster_weighted":
            keys += ["cluster_k", "cluster_spatial_weight", "cluster_seed"]
            argmap.update(cluster_k=args.get("cluster_k"),
                          cluster_spatial_weight=args.get("cluster_spatial_weight"),
                          cluster_seed=args.get("cluster_seed"))
        m = _mismatch(cfg, keys, argmap)
        if m:
            problems["riemann"] = m
    eu = tr.get("eu")
    if eu is not None:
        m = _mismatch(eu["cfg"], ["stride", "threshold"],
                      dict(stride=args["stride"], threshold=args["threshold"]))
        if m:
            problems["euclidean"] = m
    if not problems:
        return
    lines = [f"  {fam}: " + ", ".join(f"{k}: explorer={a!r} vs run={r!r}"
                                      for k, a, r in iss)
             for fam, iss in problems.items()]
    msg = ("Trained-run params don't match the explorer's launch args — the "
           "eigvecs / coordinate frame the kernel was fit on differ from what's "
           "loaded, so the trained kernel would be meaningless:\n" + "\n".join(lines)
           + "\nRelaunch with matching params, or pass --skip-trained-validation "
           "to override.")
    if strict:
        raise RuntimeError(msg)
    log.warning(msg)


def test_error_field(run_dir, lipid: str, node_ccf: np.ndarray, N: int,
                     max_mm: float = 1.0):
    """Per-node held-out |error| for one lipid: snap the fold's test voxels to
    nodes and average |pred − true| there. NaN where no test voxel landed;
    None if the lipid has no predictions in this run."""
    from scipy.spatial import cKDTree
    pdir = Path(run_dir) / "predictions" / _safe_slug(lipid)
    if not pdir.exists():
        return None
    coords = np.load(pdir / "test_coords_mm.npy").astype(np.float32)
    err = np.abs(np.load(pdir / "test_pred_z.npy")
                 - np.load(pdir / "test_true_z.npy")).astype(np.float64)
    dist, idx = cKDTree(node_ccf).query(coords, k=1)
    ok = np.isfinite(dist) & (dist <= max_mm) & (idx < N)
    summ = np.zeros(N); cnt = np.zeros(N)
    np.add.at(summ, idx[ok], err[ok]); np.add.at(cnt, idx[ok], 1.0)
    with np.errstate(invalid="ignore"):
        out = np.where(cnt > 0, summ / cnt, np.nan)
    return out.astype(np.float32)


def test_prediction_metrics(run_dir, lipid: str, max_points: int = 10000):
    """Held-out prediction quality for one lipid: MSE + Pearson/Spearman of
    predicted vs true (both z-scored) over the fold's test voxels — a genuine
    prediction↔actual score, unlike the kernel-weight↔lipid correlation the
    scatter panels show. Subsampled DETERMINISTICALLY to at most `max_points`
    (rng seed 0) so the number is stable across redraws. None if the lipid has
    no predictions in this run."""
    pdir = Path(run_dir) / "predictions" / _safe_slug(lipid)
    if not pdir.exists():
        return None
    pred = np.load(pdir / "test_pred_z.npy").astype(np.float64).ravel()
    true = np.load(pdir / "test_true_z.npy").astype(np.float64).ravel()
    m = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[m], true[m]
    n_all = int(pred.shape[0])
    if n_all == 0:
        return None
    if n_all > max_points:
        sel = np.random.default_rng(0).choice(n_all, max_points, replace=False)
        pred, true = pred[sel], true[sel]
    r = correlation(pred, true)   # finite-safe Pearson + Spearman
    return {"n": int(pred.shape[0]), "n_all": n_all,
            "mse": float(np.mean((pred - true) ** 2)),
            "pearson": r["pearson"], "spearman": r["spearman"]}


# =============================================================================
# Live leave-out predictive check — GP fit on random points of the SELECTED
# lipid, no trained run needed. Same subset-GP recipe for both kernels so the
# manifold-vs-euclidean comparison is apples-to-apples, and it updates live as
# you tune ν / ℓ / modes.
# =============================================================================
def _matern_pair(A, B, nu: float, ls: float, ard=None):
    """Dense Matérn cross-covariance between two coord sets (normalized coords).
    If `ard` is given (per-dim lengthscales), scales coords by it instead of ls
    — matching matern_cov_ard."""
    if ard is not None:
        ard_t = torch.as_tensor(np.asarray(ard, np.float32), device=A.device)
        d = torch.cdist(A / ard_t, B / ard_t)
    else:
        d = torch.cdist(A, B) / ls
    if nu == 0.5:
        return torch.exp(-d)
    if nu == 1.5:
        s = (3 ** 0.5) * d
        return (1.0 + s) * torch.exp(-s)
    if nu == 2.5:
        s = (5 ** 0.5) * d
        return (1.0 + s + (5.0 / 3.0) * d ** 2) * torch.exp(-s)
    return torch.exp(-0.5 * d ** 2)


def _pred_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """MSE + Pearson/Spearman of predicted vs actual over finite pairs."""
    m = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[m], true[m]
    if p.size < 3:
        return {"n": int(p.size), "mse": float("nan"),
                "pearson": float("nan"), "spearman": float("nan")}
    r = correlation(p, t)      # finite-safe Pearson + Spearman
    return {"n": int(p.size), "mse": float(np.mean((p - t) ** 2)),
            "pearson": r["pearson"], "spearman": r["spearman"]}


def _subset_gp_predict(Ktt: torch.Tensor, Kxt: torch.Tensor,
                       y_train: torch.Tensor, jitter: float) -> np.ndarray:
    """Exact GP posterior mean at test nodes: ŷ = K_xt (K_tt + jitter·I)⁻¹ (y-ȳ)+ȳ.
    Zero-mean prior, so centre y on the train mean. Returns NaN on a failed solve
    (singular K); the caller reports n/a."""
    n = Ktt.shape[0]
    ymean = y_train.mean()
    A = Ktt + jitter * torch.eye(n, device=Ktt.device, dtype=Ktt.dtype)
    try:
        alpha = torch.linalg.solve(A, (y_train - ymean).unsqueeze(1)).squeeze(1)
    except Exception:  # noqa: BLE001 — singular / non-finite kernel
        return np.full(Kxt.shape[0], np.nan, np.float64)
    pred = (Kxt @ alpha + ymean)
    return pred.detach().cpu().numpy().astype(np.float64)


def holdout_kernel_eval(ctx: dict, node_lipid: np.ndarray, covered: np.ndarray,
                        *, nu: float, lengthscale: float, num_modes: int,
                        matern_nu: float, eu_ard, n_test: int, train_cap: int,
                        eval_noise: float, seed: int):
    """Held-out prediction↔actual for the selected lipid with the CURRENT kernels.

    Randomly split the lipid-covered nodes into ≤`n_test` test + (≤`train_cap`)
    train, run an exact subset-of-data GP with the Riemann/manifold kernel
    (low-rank via the eigenbasis) and the Euclidean Matérn kernel, and score
    both. Returns ``{'riemann':m, 'matern':m, 'n_test':…, 'n_train':…}`` (each
    ``m`` = {n, mse, pearson, spearman}), or None if too few covered nodes.
    Deterministic given `seed` so the split is stable across redraws.
    """
    device = ctx["device"]
    idx = np.flatnonzero(covered & np.isfinite(node_lipid))
    if idx.size < 50:
        return None
    n_test = int(min(n_test, idx.size // 2))
    if n_test < 10:
        return None
    perm = np.random.default_rng(seed).permutation(idx)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    if train_idx.size > train_cap:
        train_idx = train_idx[:train_cap]          # already shuffled → random subset

    tr = torch.as_tensor(train_idx, device=device)
    te = torch.as_tensor(test_idx, device=device)
    y_train = torch.as_tensor(node_lipid[train_idx], dtype=torch.float32,
                              device=device)
    y_true = node_lipid[test_idx].astype(np.float64)
    jitter = float(eval_noise) * float(y_train.var().item() + 1e-12) + 1e-6

    out = {"n_test": int(n_test), "n_train": int(train_idx.size)}

    # ---- Riemann (manifold), low-rank K = Φ diag(s) Φᵀ over `num_modes` ----
    eigval, eigvec = ctx["eigval"], ctx["eigvec"]
    Kd = int(min(num_modes, eigvec.shape[1]))
    s = (2.0 * nu / (lengthscale ** 2) + eigval[:Kd].clamp(min=0.0)).pow(-nu)
    Ptr, Pte = eigvec[tr][:, :Kd], eigvec[te][:, :Kd]
    sPtrT = s.unsqueeze(1) * Ptr.t()                # (Kd, ntr)
    Ktt = Ptr @ sPtrT                               # (ntr, ntr)
    Kxt = Pte @ sPtrT                               # (nte, ntr)
    out["riemann"] = _pred_metrics(
        _subset_gp_predict(Ktt, Kxt, y_train, jitter), y_true)

    # ---- Matérn (Euclidean), dense on the train subset --------------------
    coords = ctx["reference_nodes"]
    c_tr, c_te = coords[tr], coords[te]
    Ktt_m = _matern_pair(c_tr, c_tr, matern_nu, lengthscale, eu_ard)
    Kxt_m = _matern_pair(c_te, c_tr, matern_nu, lengthscale, eu_ard)
    out["matern"] = _pred_metrics(
        _subset_gp_predict(Ktt_m, Kxt_m, y_train, jitter), y_true)
    return out


def diffusion_cov(eigval, eigvec, t: int, tau: float, sigma: float,
                  num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion distance D_t(t, ·) and its Gaussian covariance exp(-D²/2σ²).

    `sigma` is a MULTIPLE of the median (positive) diffusion distance, not an
    absolute bandwidth. An absolute σ is unusable here: because e^{-2λτ}
    collapses onto the lowest modes, D is tiny, so a fixed σ≈1 saturates the
    covariance to ~1 everywhere (the "monotonic" look). Referencing σ to the
    median D keeps the Gaussian in its informative range for any τ / bandwidth.
    Returns (cov, distance).
    """
    K = int(min(num_modes, eigvec.shape[1]))
    lam = eigval[:K].clamp(min=0.0)
    wt = torch.exp(-2.0 * lam * float(tau))                # (K,)
    dphi = eigvec[:, :K] - eigvec[t, :K]                   # (N, K)
    d2 = (wt * dphi.pow(2)).sum(dim=1).clamp(min=0.0)      # (N,)
    dist = d2.sqrt()
    pos = dist[dist > 0]
    ref = torch.median(pos) if pos.numel() else torch.ones((), device=dist.device)
    s = float(sigma) * float(ref) + 1e-12
    cov = torch.exp(-d2 / (2.0 * s * s))
    return cov.detach().cpu().numpy(), dist.detach().cpu().numpy()


# =============================================================================
# Operator heat kernel  e^{-tL} e_t  — NO eigenvectors, all modes
# =============================================================================
def estimate_lam_max(laplacian_op, n: int, device, iters: int = 30) -> float:
    """Largest eigenvalue of L via power iteration (operator matvecs only)."""
    v = torch.randn(n, 1, device=device)
    for _ in range(iters):
        v = laplacian_op._matmul(v)
        v = v / (v.norm() + 1e-12)
    return float((v * laplacian_op._matmul(v)).sum())


def heat_kernel_operator(laplacian_op, t: int, tau: float, n: int, device,
                         lam_max: float, n_steps: int = None) -> np.ndarray:
    """Heat kernel column  e^{-τL} e_t  straight from the Laplacian OPERATOR —
    all modes, NO eigendecomposition — via the matvec exponential integrator
    (I - (τ/m)L)^m e_t → e^{-τL} e_t.

    This is the eigenvector-FREE counterpart to the Riemann / diffusion layers
    (which are both built from the truncated eigenbasis). If this full-rank
    operator heat kernel ALSO bleeds across a boundary, the boundary genuinely
    isn't a bottleneck on this graph (not a num_modes truncation artifact); if
    it confines while the spectral layers ring, the failure is truncation.

    Returns a length-N heat profile (unnormalized; use percentile stretch for
    display / percentile threshold for the near set).
    """
    m = int(n_steps or max(40, int(2.0 * float(tau) * max(lam_max, 1e-6)) + 1))
    dt = float(tau) / m
    u = torch.zeros(n, 1, device=device, dtype=torch.float32)
    u[int(t), 0] = 1.0
    for _ in range(m):
        u = u - dt * laplacian_op._matmul(u)          # (I - dt·L) u ≈ e^{-dt·L} u
    return u.clamp(min=0.0).squeeze(1).float().detach().cpu().numpy()


# =============================================================================
# Per-section PCA projection to a flat 2D plane
# =============================================================================
def project_section_2d(coords_mm: np.ndarray) -> np.ndarray:
    """Project a (tilted) section's mm coords onto their best-fit plane.
    Returns (M, 2) in-plane coordinates (mm)."""
    c = coords_mm - coords_mm.mean(axis=0, keepdims=True)
    if c.shape[0] < 3:
        return c[:, :2].copy()
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


# =============================================================================
# Correlation report (shared by --self-test and the live scatter)
# =============================================================================
def correlation(kernel_vals: np.ndarray, lipid_vals: np.ndarray) -> dict:
    """Pearson + Spearman of a kernel weight vs lipid, over covered nodes."""
    m = np.isfinite(kernel_vals) & np.isfinite(lipid_vals)
    x, y = kernel_vals[m], lipid_vals[m]
    out = {"n": int(m.sum()), "pearson": np.nan, "spearman": np.nan}
    if out["n"] >= 3 and x.std() > 0 and y.std() > 0:
        out["pearson"] = float(np.corrcoef(x, y)[0, 1])
        try:
            from scipy.stats import spearmanr
            out["spearman"] = float(spearmanr(x, y).statistic)
        except Exception:  # noqa: BLE001
            pass
    return out


# =============================================================================
# MALDI dataframe helpers
# =============================================================================
def load_maldi(args: dict):
    import pandas as pd
    xcol, ycol, zcol = args["coord_cols"]
    meta_cols = [xcol, ycol, zcol, args["sample_col"], args["section_col"]]
    # Read the schema first so we can discover lipid columns without loading all.
    import pyarrow.parquet as pq
    schema = pq.read_schema(args["maldi_file"])
    all_cols = list(schema.names)
    lipid_cols = [c for c in all_cols if c not in meta_cols]
    log.info(f"{len(lipid_cols)} candidate lipid columns; "
             f"{len(all_cols)} total columns")
    return pd, meta_cols, lipid_cols


# =============================================================================
# Headless self-test — exercise the whole pipeline without napari
# =============================================================================
def run_self_test(args: dict, ctx: dict):
    pd, meta_cols, lipid_cols = load_maldi(args)
    sample = args["sample"]
    lipid = args["lipid"] or lipid_cols[0]
    xcol, ycol, zcol = args["coord_cols"]
    cols = meta_cols + [lipid]
    df = pd.read_parquet(args["maldi_file"], columns=cols)
    if sample is None:
        sample = df[args["sample_col"]].iloc[0]
    sub = df[df[args["sample_col"]] == sample]
    log.info(f"self-test: mouse={sample!r}  lipid={lipid!r}  {len(sub):,} voxels")

    coords = sub[[xcol, ycol, zcol]].to_numpy(np.float32)
    node_idx, valid = snap_points_to_nodes(
        coords, ctx["node_ccf"], tuple(args["axis_order"]), args["snap_max_mm"])
    log.info(f"snapped: {valid.sum():,}/{len(valid):,} within "
             f"{args['snap_max_mm']} mm")
    lipid_v = sub[lipid].to_numpy(np.float32)
    node_lipid, cover = aggregate_lipid_to_nodes(
        node_idx[valid], lipid_v[valid], ctx["N"])
    covered = cover > 0
    log.info(f"nodes with lipid coverage: {int(covered.sum()):,}")

    # Test node = highest-lipid covered node.
    lip = np.where(covered, node_lipid, -np.inf)
    t = int(np.nanargmax(lip))
    log.info(f"test node = {t} (lipid={node_lipid[t]:.4g}, "
             f"coverage={cover[t]})")

    os_ = float(args["outputscale"])
    kr = os_ * riemann_cov(ctx["eigval"], ctx["eigvec"], t, args["nu"],
                           args["lengthscale"], args["num_modes"],
                           args["kernel_normalize"])
    km = os_ * matern_cov(ctx["reference_nodes"], t, args["lengthscale"],
                          args["matern_nu"])
    kd, dd = diffusion_cov(ctx["eigval"], ctx["eigvec"], t,
                           args["diffusion_time"], args["diffusion_sigma"],
                           args["num_modes"])
    kh = heat_kernel_operator(ctx["laplacian_op"], t, args["heat_time"],
                              ctx["N"], ctx["device"], ctx["lam_max"],
                              args["heat_steps"])

    ln = np.where(covered, node_lipid, np.nan)
    print("\n" + "=" * 66)
    print(f"Correlation of kernel weight vs lipid '{lipid}' over "
          f"{int(covered.sum()):,} covered nodes:")
    for name, kv in [("riemann  ", kr), ("matern   ", km), ("diffusion", kd),
                     ("heat(op) ", kh)]:
        r = correlation(kv, ln)
        print(f"  {name}: pearson={r['pearson']:+.3f}  "
              f"spearman={r['spearman']:+.3f}  (n={r['n']:,})  "
              f"weight[min={np.nanmin(kv):.3g}, max={np.nanmax(kv):.3g}]")
    print(f"  diffusion distance range: [{dd.min():.3g}, {dd.max():.3g}]")
    if ctx["border_mask"] is not None:
        print(f"  border nodes: {int(ctx['border_mask'].sum()):,} / {ctx['N']:,}")

    # Live held-out prediction ↔ actual with the CURRENT kernels (no trained run).
    if int(args["eval_holdout"]) > 0:
        ev = holdout_kernel_eval(
            ctx, node_lipid, covered,
            nu=args["nu"], lengthscale=args["lengthscale"],
            num_modes=args["num_modes"], matern_nu=args["matern_nu"],
            eu_ard=None, n_test=int(args["eval_holdout"]),
            train_cap=int(args["eval_train_cap"]),
            eval_noise=float(args["eval_noise"]), seed=int(args["eval_seed"]))
        if ev is None:
            print(f"\nHeld-out GP eval: too few covered nodes "
                  f"({int(covered.sum()):,}) — skipped.")
        else:
            print(f"\nHeld-out prediction vs actual for '{lipid}' "
                  f"(live GP; {ev['n_test']:,} test / {ev['n_train']:,} train "
                  f"nodes):")
            for name, nm in [("riemann", "manifold "), ("matern", "euclidean")]:
                m = ev[name]
                if np.isfinite(m["mse"]):
                    print(f"  {nm}: MSE={m['mse']:.4f}  pearson={m['pearson']:+.3f}"
                          f"  spearman={m['spearman']:+.3f}  (n={m['n']:,})")
                else:
                    print(f"  {nm}: n/a (singular kernel)")

    # Held-out prediction ↔ actual (per trained model), if runs were provided.
    tr = ctx["trained"]
    if tr is not None:
        print(f"\nHeld-out pred vs actual for '{lipid}' (≤10k test points):")
        for tag, nm in [("eu", "euclid  "), ("ri", "manifold")]:
            run = tr.get(tag)
            mm = (test_prediction_metrics(run["dir"], lipid, max_points=10000)
                  if run is not None else None)
            if mm is None:
                print(f"  {nm}: no predictions for this lipid")
            else:
                print(f"  {nm}: MSE={mm['mse']:.4f}  pearson={mm['pearson']:+.3f}  "
                      f"spearman={mm['spearman']:+.3f}  "
                      f"(n={mm['n']:,} of {mm['n_all']:,})")
    print("=" * 66 + "\n")
    log.info("self-test OK")


# =============================================================================
# Interactive napari viewer
# =============================================================================
def run_viewer(args: dict, ctx: dict):
    import napari
    import pandas as pd
    from magicgui import magicgui
    import matplotlib
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QLabel

    xcol, ycol, zcol = args["coord_cols"]
    sample_col, section_col = args["sample_col"], args["section_col"]
    _, meta_cols, lipid_cols = load_maldi(args)

    # ---- read metadata ONCE; cache lipid columns + per-mouse snaps -------
    # The parquet is multi-GB; re-reading it per mouse switch was the main
    # slowness. Metadata (coords + Sample + Section) is read once into RAM;
    # lipid columns and per-mouse KD-tree snaps are memoized.
    base = pd.read_parquet(args["maldi_file"], columns=meta_cols)
    coords_all = base[[xcol, ycol, zcol]].to_numpy(np.float32)
    sample_all = base[sample_col].to_numpy().astype(str)
    section_all = base[section_col].to_numpy()
    all_samples = sorted(set(sample_all.tolist()))
    init_sample = str(args["sample"]) if str(args["sample"]) in all_samples \
        else all_samples[0]
    init_lipid = args["lipid"] if args["lipid"] in lipid_cols else lipid_cols[0]

    # Lipids with a TRAINED + TESTED model in every provided run (∩ parquet cols):
    # what the dropdown is limited to when "use trained hypers" is on.
    trained_lipids = []
    if ctx["trained"] is not None:
        avail = [{lp for lp in run["per"]
                  if (run["dir"] / "predictions" / _safe_slug(lp)).exists()}
                 for run in (ctx["trained"]["ri"], ctx["trained"]["eu"]) if run is not None]
        common = set.intersection(*avail) if avail else set()
        trained_lipids = [c for c in lipid_cols if c in common]
        if not trained_lipids:
            log.warning("no trained lipids match the parquet columns — trained-"
                        "hypers lipid limiting disabled (names likely differ).")
        else:
            log.info("trained+tested lipids (dropdown limited to these when "
                     "'use trained hypers' is on): %s", trained_lipids)
            if init_lipid not in trained_lipids:
                init_lipid = trained_lipids[0]

    viewer = napari.Viewer(title="MALDI kernel explorer", ndisplay=2)

    # ---- mutable session state -------------------------------------------
    st = dict(
        sample=init_sample, lipid=init_lipid, section=None,
        nu=float(args["nu"]), matern_nu=float(args["matern_nu"]),
        lengthscale=float(args["lengthscale"]),
        outputscale=float(args["outputscale"]),
        tau=float(args["diffusion_time"]), sigma=float(args["diffusion_sigma"]),
        heat_time=float(args["heat_time"]), heat_steps=args["heat_steps"],
        num_modes=int(args["num_modes"]), gamma=float(args["gamma"]),
        normalize=bool(args["kernel_normalize"]),
        near_kernel=str(args["near_kernel"]), near_pct=float(args["near_pct"]),
        rows=None, node_idx=None, valid=None, lip_rows=None,
        node_lipid=None, covered=None, node_cover=None,
        sections=None, test_node=None,
        _xy=None, _nodes=None, _busy=False, _near_count=0, _near_thr=0.0,
        use_trained=(ctx["trained"] is not None), trained_lipid=False,
        eu_ard=None, err_eu=None, err_ri=None,
        metrics_eu=None, metrics_ri=None,
        live_eval=None, _eval_key=None,
    )
    layers = {"lipid": None, "riemann": None, "matern": None,
              "diffusion": None, "heat": None, "border": None, "regions": None,
              "near": None, "test": None, "err_eu": None, "err_ri": None}

    # Stable per-region-id colour lookup for the region overlay (built once).
    region_code, region_rgba = (None, None)
    if ctx["node_labels"] is not None:
        region_code, region_rgba = region_color_table(
            ctx["node_labels"], ctx.get("labels_zero_is_region", False))

    # ---- matplotlib correlation panel ------------------------------------
    fig = Figure(figsize=(3.2, 7.6), tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    axes = fig.subplots(4, 1)
    viewer.window.add_dock_widget(canvas, name="kernel ↔ lipid", area="right")

    info_label = QLabel()
    info_label.setWordWrap(True)
    info_label.setStyleSheet("QLabel { font-size: 11px; }")
    viewer.window.add_dock_widget(
        info_label, name="test point · how to read", area="right")

    def _sequential(vals, cmap="magma", gamma=None, vmin=None, vmax=None):
        g = st["gamma"] if gamma is None else gamma
        v = np.asarray(vals, np.float32)
        has = np.isfinite(v).any()
        lo = (np.nanpercentile(v, 1) if vmin is None else vmin) if has else 0.0
        hi = (np.nanpercentile(v, 99) if vmax is None else vmax) if has else 1.0
        if hi <= lo:
            hi = lo + 1e-6
        n = np.nan_to_num(np.clip((v - lo) / (hi - lo), 0, 1) ** g, nan=0.0)
        return matplotlib.colormaps[cmap](n)

    # ---- memoized data access --------------------------------------------
    lipid_cache = {}

    def get_lipid(name):
        if name not in lipid_cache:
            log.info(f"reading lipid column {name!r} …")
            lipid_cache[name] = pd.read_parquet(
                args["maldi_file"], columns=[name])[name].to_numpy(np.float32)
        return lipid_cache[name]

    mouse_cache = {}

    def get_mouse(sample):
        if sample not in mouse_cache:
            rows = np.where(sample_all == sample)[0]
            nidx, valid = snap_points_to_nodes(
                coords_all[rows], ctx["node_ccf"], tuple(args["axis_order"]),
                args["snap_max_mm"])
            mouse_cache[sample] = (rows, nidx, valid)
            log.info(f"snapped mouse {sample!r}: {int(valid.sum()):,}/"
                     f"{len(rows):,} within {args['snap_max_mm']} mm")
        return mouse_cache[sample]

    def reload_mouse():
        rows, nidx, valid = get_mouse(st["sample"])
        lip_rows = get_lipid(st["lipid"])[rows]
        node_lipid, cover = aggregate_lipid_to_nodes(
            nidx[valid], lip_rows[valid], ctx["N"])
        secs = sorted(np.unique(section_all[rows]).tolist())
        st.update(rows=rows, node_idx=nidx, valid=valid, lip_rows=lip_rows,
                  node_lipid=node_lipid, covered=(cover > 0),
                  node_cover=cover, sections=secs)
        if st["section"] not in secs:
            st["section"] = secs[len(secs) // 2]
        lip = np.where(st["covered"], node_lipid, -np.inf)
        st["test_node"] = int(np.argmax(lip)) if np.isfinite(lip).any() else 0
        apply_trained()
        log.info(f"mouse {st['sample']!r}: {len(rows):,} voxels, "
                 f"{len(secs)} sections, lipid {st['lipid']!r}, "
                 f"{int(st['covered'].sum()):,} covered nodes")

    def apply_trained():
        """On lipid change: swap kernel hypers to the trained values for this
        lipid and load each fold's held-out error field (both cached)."""
        st["eu_ard"] = None; st["err_eu"] = None; st["err_ri"] = None
        st["metrics_eu"] = None; st["metrics_ri"] = None
        st["trained_lipid"] = False
        tr = ctx["trained"]
        if tr is None or not st["use_trained"]:
            return
        lip = st["lipid"]
        ri, eu = tr["ri"], tr["eu"]
        if ri is not None and lip in ri["per"]:
            st["nu"] = float(ri["per"][lip]["nu"])
            st["lengthscale"] = float(ri["per"][lip]["ls"])
            st["trained_lipid"] = True
        if eu is not None and lip in eu["per"]:
            st["eu_ard"] = np.asarray(eu["per"][lip]["ard"], np.float64)
            st["matern_nu"] = float(eu["per"][lip]["nu"])
        for tag, run in [("eu", eu), ("ri", ri)]:
            if run is None:
                continue
            key = (tag, lip)
            if key not in tr["err_cache"]:
                tr["err_cache"][key] = test_error_field(
                    run["dir"], lip, ctx["node_ccf"], ctx["N"], args["snap_max_mm"])
            st["err_" + tag] = tr["err_cache"][key]
            if key not in tr["metrics_cache"]:
                tr["metrics_cache"][key] = test_prediction_metrics(
                    run["dir"], lip, max_points=10000)
            st["metrics_" + tag] = tr["metrics_cache"][key]

    # ---- current-section geometry (PCA-projected 2D) ---------------------
    def section_frame():
        rows = st["rows"]
        in_sec = (section_all[rows] == st["section"]) & st["valid"]
        sel = np.where(in_sec)[0]
        xy = (project_section_2d(coords_all[rows[sel]]).astype(np.float32)
              if sel.size else np.zeros((0, 2), np.float32))
        return xy, st["node_idx"][sel], st["lip_rows"][sel]

    def eval_kernels():
        t = st["test_node"]
        # ScaleKernel wrapper: k = outputscale · k_base, applied to the two GP
        # kernels (riemann + matérn). Diffusion / heat are diagnostics, left raw.
        os_ = float(st["outputscale"])
        kr = os_ * riemann_cov(ctx["eigval"], ctx["eigvec"], t, st["nu"],
                               st["lengthscale"], st["num_modes"], st["normalize"])
        if st["eu_ard"] is not None:
            km = os_ * matern_cov_ard(ctx["reference_nodes"], t, st["eu_ard"],
                                      st["matern_nu"])
        else:
            km = os_ * matern_cov(ctx["reference_nodes"], t, st["lengthscale"],
                                  st["matern_nu"])
        kd, _ = diffusion_cov(ctx["eigval"], ctx["eigvec"], t, st["tau"],
                              st["sigma"], st["num_modes"])
        kh = heat_kernel_operator(ctx["laplacian_op"], t, st["heat_time"],
                                  ctx["N"], ctx["device"], ctx["lam_max"],
                                  st["heat_steps"])
        return {"riemann": kr, "matern": km, "diffusion": kd, "heat": kh}

    def region_purity(kernels):
        """For each kernel's near set (top --near-pct weights), the fraction
        of near nodes sharing ★'s atlas region. ≫ chance = respects the
        boundary; ≈ chance = bleeds through. None if no atlas."""
        lab = ctx["node_labels"]
        if lab is None:
            return None
        tlab = int(lab[st["test_node"]])
        base = float((lab == tlab).mean())
        # label 0 is a valid region for clusters, but background for the atlas.
        tlab_valid = (tlab != 0) or ctx.get("labels_zero_is_region", False)
        per = {}
        for name, kv in kernels.items():
            thr = (float(np.nanpercentile(kv, st["near_pct"]))
                   if np.isfinite(kv).any() else 0.0)
            near = np.isfinite(kv) & (kv >= thr)
            n = int(near.sum())
            pur = (float((lab[near] == tlab).mean())
                   if n and tlab_valid else float("nan"))
            per[name] = (pur, n)
        return {"per": per, "tlab": tlab, "base": base}

    def _refresh_live_eval():
        """Recompute the held-out prediction↔actual eval when a param that
        affects it changed (mouse/lipid/ν/ℓ/modes/matérn-ν/ARD) — NOT on clicks
        (the eval is independent of the test point) or the outputscale slider
        (scale-invariant), so scrubbing the star stays cheap."""
        if int(args["eval_holdout"]) <= 0 or st["node_lipid"] is None:
            st["live_eval"] = None
            return
        key = (st["sample"], st["lipid"], round(float(st["nu"]), 4),
               round(float(st["lengthscale"]), 5), int(st["num_modes"]),
               float(st["matern_nu"]), id(st["eu_ard"]))
        if key == st.get("_eval_key"):
            return
        st["_eval_key"] = key
        st["live_eval"] = holdout_kernel_eval(
            ctx, st["node_lipid"], st["covered"],
            nu=float(st["nu"]), lengthscale=float(st["lengthscale"]),
            num_modes=int(st["num_modes"]), matern_nu=float(st["matern_nu"]),
            eu_ard=st["eu_ard"], n_test=int(args["eval_holdout"]),
            train_cap=int(args["eval_train_cap"]),
            eval_noise=float(args["eval_noise"]), seed=int(args["eval_seed"]))

    # ---- create the fixed set of layers ONCE (then update in place) ------
    def ensure_layers():
        if layers["lipid"] is not None:
            return
        d = np.zeros((1, 2), np.float32)
        dc = np.zeros((1, 4), np.float32)
        layers["lipid"] = viewer.add_points(
            d.copy(), name="lipid", size=args["point_size"],
            face_color=dc.copy(), opacity=0.9, visible=True)
        for key, nm in [("riemann", "k_riemann(test, ·)"),
                        ("matern", "k_matern(test, ·)"),
                        ("diffusion", "k_diffusion(test, ·)"),
                        ("heat", "k_heat e^{-τL}(test, ·)  [operator, no eigvecs]")]:
            layers[key] = viewer.add_points(
                d.copy(), name=nm, size=args["point_size"],
                face_color=dc.copy(), opacity=0.9, visible=False)
        if ctx["border_mask"] is not None and ctx["knn_method"] != "faiss":
            layers["border"] = viewer.add_points(
                d.copy(), name="atlas borders", size=args["point_size"] * 1.4,
                face_color=np.array([[1.0, 0.15, 0.15, 0.95]]),
                opacity=0.9, visible=False)
        if region_code is not None:
            layers["regions"] = viewer.add_points(
                d.copy(), name=f"regions ({len(region_rgba)}) · one colour / id",
                size=args["point_size"], face_color=dc.copy(),
                opacity=0.9, visible=False)
        layers["near"] = viewer.add_points(
            d.copy(), name="near set (kernel ≥ thr)", size=1.0,
            face_color=np.array([[0.1, 0.9, 1.0, 0.95]]),
            opacity=0.95, visible=True)
        if ctx["trained"] is not None:
            for tag, nm in [("eu", "held-out error: euclidean"),
                            ("ri", "held-out error: manifold")]:
                layers["err_" + tag] = viewer.add_points(
                    d.copy(), name=nm, size=args["point_size"],
                    face_color=dc.copy(), opacity=0.95, visible=False)
        layers["test"] = viewer.add_points(
            d.copy(), name="★ test point", size=1.0,
            face_color=np.array([[0.1, 1.0, 0.1, 1.0]]),
            border_color="white", border_width=0.12, symbol="star",
            opacity=1.0)

    def _auto_point_size(xy):
        """Point diameter ≈ the section's inter-point spacing × --point-size,
        so points tile the slice rather than smearing into a blob."""
        if xy.shape[0] < 3:
            return 1.0
        span = xy.max(0) - xy.min(0)
        area = float(max(span[0], 1e-6) * max(span[1], 1e-6))
        return float((area / xy.shape[0]) ** 0.5 * float(args["point_size"]))

    def _update_layer(key, xy, face, size=None):
        """Update a points layer. In-place when the point COUNT is unchanged
        (fast path: clicks / kernel-param tweaks). When the count changes
        (section / mouse switch) napari defers its _indices_view refresh, so a
        later face_color set renders with stale indices → IndexError in
        _view_data. In that case we remove + re-add the layer for a clean
        internal state."""
        old = layers.get(key)
        if xy.shape[0] == 0:
            xy = np.zeros((1, 2), np.float32)
            face = np.zeros((1, 4), np.float32)
        if old is not None and len(old.data) == xy.shape[0]:
            vis = old.visible
            old.data = xy
            old.face_color = face
            if size is not None:
                old.size = size
            old.visible = vis
            return old
        # count changed (or first real build) → rebuild the layer cleanly
        if old is not None and old in viewer.layers:
            name, vis, op = old.name, old.visible, float(old.opacity)
            blend = old.blending
            sz = size if size is not None else float(np.atleast_1d(old.size)[0])
            idx = viewer.layers.index(old)
            viewer.layers.remove(old)
        else:
            name, vis, op, blend, idx = key, False, 0.9, "translucent", None
            sz = size if size is not None else 1.0
        new = viewer.add_points(xy, name=name, size=sz, face_color=face,
                                opacity=op, blending=blend, visible=vis)
        if idx is not None and idx <= len(viewer.layers) - 1:
            viewer.layers.move(viewer.layers.index(new), idx)
        layers[key] = new
        return new

    def refresh_scatter(kernels):
        cov = st["covered"]
        ln = np.where(cov, st["node_lipid"], np.nan)
        t_lip = float(st["node_lipid"][st["test_node"]])
        stats = {}
        for ax, (name, cmap) in zip(
                axes, [("riemann", "viridis"), ("matern", "plasma"),
                       ("diffusion", "cividis"), ("heat", "inferno")]):
            ax.clear()
            kv = kernels[name]
            m = cov & np.isfinite(kv) & np.isfinite(ln)
            r = correlation(kv[m], ln[m])
            stats[name] = r
            x, y = kv[m], ln[m]
            if x.size > args["scatter_max"]:
                s = np.random.default_rng(0).choice(
                    x.size, args["scatter_max"], replace=False)
                x, y = x[s], y[s]
            ax.scatter(x, y, s=3, alpha=0.35, c=matplotlib.colormaps[cmap](0.6))
            if np.isfinite(t_lip):
                # ★'s own lipid level: points above/below this line are more/
                # less abundant than the test point.
                ax.axhline(t_lip, color="lime", lw=0.9, ls="--")
            ax.set_title(f"{name}: r={r['pearson']:+.2f} "
                         f"ρ={r['spearman']:+.2f} (n={r['n']:,})", fontsize=8)
            ax.set_xlabel("kernel weight →", fontsize=7)
            ax.set_ylabel(f"{st['lipid']}", fontsize=7)
            ax.tick_params(labelsize=6)
        canvas.draw_idle()
        return stats

    # ---- refresh every layer in place (no add/remove churn) --------------
    def update_view():
        ensure_layers()
        xy, nodes, lip = section_frame()
        st["_xy"], st["_nodes"] = xy, nodes
        psize = _auto_point_size(xy)
        kernels = eval_kernels()
        _update_layer("lipid", xy, _sequential(lip, "magma"), psize)
        layers["lipid"].name = (f"lipid {st['lipid']} · {st['sample']} / "
                                f"sec {st['section']}")
        for key, cmap in [("riemann", "viridis"), ("matern", "plasma"),
                          ("diffusion", "cividis"), ("heat", "inferno")]:
            vals = kernels[key][nodes] if nodes.size else np.zeros(0, np.float32)
            if key in ("diffusion", "heat"):
                # Percentile-stretch: diffusion covariance lives in a narrow band
                # near 1, and the operator heat profile is tiny/peaked — an
                # absolute [0,max] scale washes both out.
                face = _sequential(vals, cmap)
            else:
                allk = kernels[key]
                vmax = float(np.nanmax(allk)) if np.isfinite(allk).any() else 1.0
                face = _sequential(vals, cmap, vmin=0.0, vmax=vmax or 1.0)
            _update_layer(key, xy, face, psize)
        if layers["border"] is not None:
            # full-length + transparent for non-border nodes (keeping a constant
            # point count across layers avoids napari stale-index crashes).
            bcol = np.zeros((xy.shape[0], 4), np.float32)
            if nodes.size:
                bcol[ctx["border_mask"][nodes]] = (1.0, 0.15, 0.15, 0.95)
            _update_layer("border", xy, bcol, psize)
        if layers["regions"] is not None:
            # one stable colour per region id (transparent where no nodes).
            rcol = np.zeros((xy.shape[0], 4), np.float32)
            if nodes.size:
                rcol = region_rgba[region_code[nodes]]
            _update_layer("regions", xy, rcol, psize)
        # "near set": nodes whose chosen-kernel weight exceeds the percentile
        # threshold, drawn bright so the neighbourhood is explicit (not just a
        # colour gradient). Threshold is over ALL nodes, so it's the same set
        # across slices — scroll to see the neighbourhood extend in depth.
        nk = kernels[st["near_kernel"]]
        thr = (float(np.nanpercentile(nk, st["near_pct"]))
               if np.isfinite(nk).any() else 0.0)
        near_all = np.isfinite(nk) & (nk >= thr)
        near_sec = near_all[nodes] if nodes.size else np.zeros(0, bool)
        ncol = np.zeros((xy.shape[0], 4), np.float32)
        ncol[near_sec] = (0.1, 0.9, 1.0, 0.95)
        _update_layer("near", xy, ncol, psize * 1.5)
        layers["near"].name = (f"near set: {st['near_kernel']} ≥ "
                               f"p{st['near_pct']:g}")
        st["_near_count"], st["_near_thr"] = int(near_all.sum()), thr
        st["_near_in_slice"] = int(near_sec.sum())
        st["_purity"] = region_purity(kernels)
        _refresh_live_eval()

        # held-out error layers: full-length, coloured only at nodes that had a
        # test voxel (rest transparent) — constant point count, no stale-index.
        for tag in ("eu", "ri"):
            lyr = layers.get("err_" + tag)
            if lyr is None:
                continue
            field = st.get("err_" + tag)
            face = np.zeros((xy.shape[0], 4), np.float32)
            if field is not None and nodes.size:
                ev = field[nodes]
                m = np.isfinite(ev)
                if m.any():
                    face[m] = _sequential(ev[m], "inferno")
            _update_layer("err_" + tag, xy, face, psize * 1.3)

        # test-point star: a small fraction of the slice's extent
        span = (float(np.linalg.norm(xy.max(0) - xy.min(0)))
                if xy.shape[0] > 1 else 1.0)
        star = max(span * float(args["test_marker_scale"]), 1e-3)
        t = st["test_node"]
        hit = np.where(nodes == t)[0] if nodes.size else np.empty(0, int)
        pos = (xy[hit[0]][None, :] if hit.size
               else (xy[:1] if xy.shape[0] else np.zeros((1, 2), np.float32)))
        layers["test"].data = pos
        layers["test"].size = star
        stats = refresh_scatter(kernels)
        _update_info(stats)

    def _purity_html():
        pur = st.get("_purity")
        if not pur:
            return ""
        per, base, tlab = pur["per"], pur["base"], pur["tlab"]

        def _p(name):
            v, n = per[name]
            return f"{v:.0%}" if np.isfinite(v) else "n/a"

        return (
            "<hr><b>near-set region purity</b> "
            f"(★ region {tlab}; chance {base:.0%})<br>"
            f"riemann {_p('riemann')} · matérn {_p('matern')} · "
            f"diffusion {_p('diffusion')}<br>"
            "<span style='font-size:10px'>fraction of each near set sharing "
            "★'s atlas region. ≫ chance = respects the boundary; ≈ chance = "
            "bleeds through.</span><br>")

    def _trained_html():
        tr = ctx["trained"]
        if tr is None:
            return ""
        if not st["use_trained"]:
            return ("<br><span style='font-size:10px'>trained hypers OFF "
                    "(kernels use the sliders)</span>")
        lip = st["lipid"]; ri, eu = tr["ri"], tr["eu"]; parts = ["<br><b>trained (fold):</b> "]
        if ri is not None:
            parts.append(f"riemann ℓ={ri['per'][lip]['ls']:.3g} ν={ri['per'][lip]['nu']:g}"
                         if lip in ri["per"] else "riemann: lipid not in run")
        if eu is not None and lip in eu["per"]:
            ard = eu["per"][lip]["ard"]
            parts.append(" · euclid ARD ℓ=[" + ",".join(f"{a:.2f}" for a in ard)
                         + f"] ν={eu['per'][lip]['nu']:g}")
        html = "".join(parts) + ("<br><span style='font-size:10px'>held-out |error| "
                                 "(mean over test nodes on this slice's regions): ")
        for tag, nm in [("eu", "euclid"), ("ri", "manifold")]:
            f = st.get("err_" + tag)
            if f is not None and np.isfinite(f).any():
                html += f"{nm} {np.nanmean(f):.3f}&nbsp;&nbsp;"
        html += "</span>"
        # Held-out prediction ↔ actual quality (MSE + corr over ≤10k test pts),
        # per model. This is pred-vs-true, distinct from the kernel-weight↔lipid
        # correlation in the scatter panels.
        mlines = []
        for tag, nm in [("eu", "euclid"), ("ri", "manifold")]:
            mm = st.get("metrics_" + tag)
            if mm is not None:
                mlines.append(
                    f"{nm}: MSE={mm['mse']:.3f} r={mm['pearson']:+.2f} "
                    f"ρ={mm['spearman']:+.2f} (n={mm['n']:,})")
        if mlines:
            html += ("<br><span style='font-size:10px'><b>held-out pred vs "
                     "actual</b>: " + "; ".join(mlines) + "</span>")
        return html

    def _live_eval_html():
        ev = st.get("live_eval")
        if not ev:
            return ""

        def _fmt(name):
            m = ev.get(name)
            if not m or not np.isfinite(m["mse"]):
                return "n/a"
            return (f"MSE={m['mse']:.3f} &nbsp;r={m['pearson']:+.2f} "
                    f"ρ={m['spearman']:+.2f}")

        return (
            "<hr><b>held-out prediction ↔ actual</b> "
            f"(live GP; {ev['n_test']:,} test / {ev['n_train']:,} train nodes)<br>"
            f"manifold&nbsp;&nbsp;: {_fmt('riemann')}<br>"
            f"euclidean: {_fmt('matern')}<br>"
            "<span style='font-size:10px'>random split of THIS lipid's covered "
            "nodes; exact subset GP with the current kernel (no trained run). "
            "Tune ν / ℓ / modes and watch MSE ↓ / r ↑.</span><br>")

    def _update_info(stats):
        t = st["test_node"]
        lip = float(st["node_lipid"][t])
        cov = int(st["node_cover"][t]) if st["node_cover"] is not None else 0
        lip_s = f"{lip:.4g}" if np.isfinite(lip) else "n/a (node has no MALDI voxel)"

        def _r(name):
            s = stats[name]
            return f"r={s['pearson']:+.2f}, ρ={s['spearman']:+.2f}"

        info_label.setText(
            "<b>★ selected test point</b><br>"
            f"node #{t}<br>"
            f"{st['lipid']} = <b>{lip_s}</b> &nbsp;({cov} MALDI voxels averaged)<br>"
            f"viewing {st['sample']} · section {st['section']}<br>"
            f"{_trained_html()}"
            "<hr>"
            f"<b>near set</b> (cyan): {st['near_kernel']} ≥ "
            f"p{st['near_pct']:g} &nbsp;→&nbsp; weight ≥ {st['_near_thr']:.3g}<br>"
            f"<span style='font-size:10px'>{st['_near_count']:,} nodes total, "
            f"{st.get('_near_in_slice', 0):,} on this slice</span><br>"
            f"{_purity_html()}"
            f"{_live_eval_html()}"
            "<hr>"
            "<b>kernel weight ↔ lipid correlation</b><br>"
            f"<span style='font-size:10px'>over {stats['riemann']['n']:,} "
            "covered nodes (whole mouse)</span><br>"
            f"riemann&nbsp;&nbsp;: {_r('riemann')}<br>"
            f"matérn&nbsp;&nbsp;&nbsp;: {_r('matern')}<br>"
            f"diffusion: {_r('diffusion')}<br>"
            "<hr>"
            "<b>how to read the plots</b><br>"
            "<span style='font-size:10px'>"
            "Each panel: <b>x</b> = how strongly that kernel weights a node "
            "relative to ★, <b>y</b> = the node's measured lipid. The green "
            "dashed line is ★'s own lipid level.<br><br>"
            "• <b>r &gt; 0</b>: nodes the kernel treats as 'near' ★ tend to "
            "have <i>higher</i> lipid — the kernel's neighbourhood follows this "
            "lipid.<br>"
            "• <b>r ≈ 0</b>: the kernel's neighbourhood is unrelated to the "
            "lipid pattern.<br>"
            "• Compare the three: a manifold/diffusion kernel that respects "
            "anatomy should beat plain Matérn wherever the lipid follows "
            "regions rather than Euclidean balls.<br>"
            "• Points hugging the dashed line at high x = the kernel picks out "
            "voxels matching ★'s lipid level (what you want)."
            "</span>")

    # ---- click to pick a test point --------------------------------------
    def on_click(v, event):
        xy, nodes = st["_xy"], st["_nodes"]
        if xy is None or xy.shape[0] == 0:
            return
        pos = np.asarray(event.position[-2:], np.float32)
        j = int(np.argmin(((xy - pos) ** 2).sum(axis=1)))
        st["test_node"] = int(nodes[j])
        log.info(f"test node → {st['test_node']} "
                 f"(lipid={st['node_lipid'][st['test_node']]:.4g})")
        update_view()

    # ---- controls dock ----------------------------------------------------
    lipid_choices0 = (trained_lipids if (ctx["trained"] is not None
                      and st["use_trained"] and trained_lipids) else lipid_cols)

    @magicgui(
        call_button=False, auto_call=True,
        sample={"choices": all_samples, "label": "mouse"},
        lipid={"choices": lipid_choices0, "label": "lipid"},
        section={"widget_type": "Slider", "min": 0, "max": 0,
                 "label": "section idx"},
        num_modes={"widget_type": "SpinBox", "min": 1,
                   "max": int(ctx["eigvec"].shape[1]), "label": "num modes"},
        nu={"widget_type": "FloatSpinBox", "min": 0.1, "max": 10.0,
            "step": 0.5, "label": "ν riemann"},
        matern_nu={"choices": [0.5, 1.5, 2.5], "label": "ν matérn"},
        near_kernel={"choices": ["riemann", "matern", "diffusion", "heat"],
                     "label": "near set: kernel"},
        near_pct={"widget_type": "FloatSlider", "min": 50.0, "max": 99.5,
                  "step": 0.5, "label": "near set: percentile"},
        lengthscale={"widget_type": "FloatSpinBox", "min": 1e-3, "max": 20.0,
                     "step": 0.05, "label": "lengthscale"},
        outputscale={"widget_type": "FloatSpinBox", "min": 1e-3, "max": 100.0,
                     "step": 0.1, "label": "outputscale (ScaleKernel)"},
        tau={"widget_type": "FloatSpinBox", "min": 1e-4, "max": 100.0,
             "step": 0.05, "label": "diffusion time"},
        sigma={"widget_type": "FloatSpinBox", "min": 1e-3, "max": 50.0,
               "step": 0.05, "label": "diffusion σ"},
        heat_time={"widget_type": "FloatSpinBox", "min": 1e-3, "max": 500.0,
                   "step": 0.5, "label": "heat time τ (e^{-τL})"},
        gamma={"widget_type": "FloatSpinBox", "min": 0.1, "max": 2.0,
               "step": 0.05, "label": "color gamma"},
        normalize={"label": "cosine-normalize riemann"},
        use_trained={"label": "use trained hypers"},
    )
    def controls(sample: str = str(init_sample), lipid: str = init_lipid,
                 section: int = 0, num_modes: int = int(args["num_modes"]),
                 nu: float = float(args["nu"]),
                 matern_nu: float = float(args["matern_nu"]),
                 near_kernel: str = str(args["near_kernel"]),
                 near_pct: float = float(args["near_pct"]),
                 lengthscale: float = float(args["lengthscale"]),
                 outputscale: float = float(args["outputscale"]),
                 tau: float = float(args["diffusion_time"]),
                 sigma: float = float(args["diffusion_sigma"]),
                 heat_time: float = float(args["heat_time"]),
                 gamma: float = float(args["gamma"]),
                 normalize: bool = bool(args["kernel_normalize"]),
                 use_trained: bool = bool(ctx["trained"] is not None)):
        # Reentrancy guard: setting a widget's value below re-fires this
        # callback; the guard makes those nested calls no-ops.
        if st["_busy"]:
            return
        st["_busy"] = True
        try:
            mouse_or_lipid = (sample != st["sample"]) or (lipid != st["lipid"])
            trained_toggled = bool(use_trained) != st["use_trained"]
            st.update(sample=sample, lipid=lipid, num_modes=int(num_modes),
                      nu=float(nu), matern_nu=float(matern_nu),
                      near_kernel=str(near_kernel),
                      near_pct=float(near_pct),
                      lengthscale=float(lengthscale),
                      outputscale=float(outputscale),
                      tau=float(tau), sigma=float(sigma),
                      heat_time=float(heat_time), gamma=float(gamma),
                      normalize=bool(normalize), use_trained=bool(use_trained))
            if trained_toggled:
                _sync_lipid_choices()     # limit/expand dropdown; may switch lipid
                reload_mouse()            # re-snap + (re)apply/clear trained hypers
                _sync_section_slider()
            elif mouse_or_lipid:
                reload_mouse()            # calls apply_trained() internally
                _sync_section_slider()
            else:
                idx = min(int(section), len(st["sections"]) - 1)
                st["section"] = st["sections"][idx]
                if ctx["trained"] is not None:
                    # trained hypers are authoritative; apply_trained doesn't
                    # touch the section.
                    apply_trained()
            update_view()
        finally:
            st["_busy"] = False

    def _sync_section_slider():
        nsec = len(st["sections"])
        controls.section.max = max(nsec - 1, 0)
        mid = min(nsec // 2, nsec - 1)
        controls.section.value = mid
        st["section"] = st["sections"][mid]

    def _sync_lipid_choices():
        """Limit the lipid dropdown to trained+tested lipids when 'use trained
        hypers' is on; otherwise show all parquet lipids. Clamps the current
        selection into the allowed set."""
        allowed = (trained_lipids if (ctx["trained"] is not None
                   and st["use_trained"] and trained_lipids) else lipid_cols)
        controls.lipid.choices = allowed
        if st["lipid"] not in allowed:
            st["lipid"] = allowed[0]
            controls.lipid.value = allowed[0]

    viewer.window.add_dock_widget(controls, name="controls", area="right")
    viewer.mouse_drag_callbacks.append(on_click)

    # ---- initial render ---------------------------------------------------
    st["_busy"] = True
    try:
        _sync_lipid_choices()
        reload_mouse()
        _sync_section_slider()
    finally:
        st["_busy"] = False
    update_view()

    print("\n" + "=" * 70)
    print("MALDI kernel explorer")
    print(f"  mice        : {len(all_samples)}")
    print(f"  lipids      : {len(lipid_cols)}")
    print(f"  graph nodes : {ctx['N']:,}   modes: {ctx['eigvec'].shape[1]}")
    print(f"  knn method  : {ctx['knn_method']}  "
          f"(borders {'ON' if ctx['border_mask'] is not None and ctx['knn_method'] != 'faiss' else 'off'})")
    if ctx["node_labels"] is not None:
        print(f"  regions     : {len(np.unique(ctx['node_labels']))} "
              f"(one colour per id in the 'regions' layer)")
    print("Layers (toggle on the left): lipid / k_riemann / k_matern / "
          "k_diffusion / atlas borders / regions / ★ test point")
    print("Click anywhere on a slice to move the test point.")
    print("Right dock: mouse/lipid/section + kernel controls, and the live")
    print("kernel↔lipid correlation scatter (Pearson r, Spearman ρ).")
    print("=" * 70 + "\n")
    napari.run()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if args["matern_nu"] is None:
        args["matern_nu"] = args["nu"]
    # The Euclidean Matérn only has closed forms at half-integer ν; snap the
    # inherited/CLI value to the nearest supported one (Riemann ν stays free).
    _mv = [0.5, 1.5, 2.5]
    if args["matern_nu"] not in _mv:
        snapped = min(_mv, key=lambda v: abs(v - args["matern_nu"]))
        if snapped != args["matern_nu"]:
            logging.getLogger(__name__).info(
                "Matérn ν=%s not a supported half-integer; using %s.",
                args["matern_nu"], snapped)
        args["matern_nu"] = snapped
    logging.basicConfig(
        level=logging.DEBUG if args["verbose"] else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ctx = setup_graph(args)
    if args["self_test"]:
        run_self_test(args, ctx)
        return
    run_viewer(args, ctx)


if __name__ == "__main__":
    main()
