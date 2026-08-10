"""Standalone kernel & graph visualization for the manifold GP.

Three nested views of the manifold construction, all sharing one
coordinate frame (full-resolution template voxel coords):

  Layer A — KNN fabric
    Every graph node as a faint point, a sample of edges as faint lines.
    This is the discrete topology the Laplacian is defined on.

  Layer B — Graph Laplacian
    Nodes colored by diag(L)[i] (local connectivity strength at node i).
    Edges colored by L[i, j] (diverging: red = pulls i toward j,
    blue = pushes apart, in the symmetric normalization).

  Layer C — Eigenvector reconstruction
    Nodes colored by Σ_{k=1..K} spectral_density(λ_k) · φ_k(i)²
    (the diagonal of the reconstructed kernel). Slider over K shows
    how many eigenmodes are needed to recover the kernel's local scale.

  Per-source comparison (existing)
    KNN edges, Matern (Euclidean) kernel, Riemann (manifold) kernel —
    line segments from active source colored by kernel value.

Controls (dock widget on the right):
  - active source : pick which source's per-source layers to display
  - num_modes     : eigenmodes used for Layer C and Riemann kernel
  - lengthscale   : Matern lengthscale (used by both)
  - color_gamma   : visual gamma on edge strengths
"""
from __future__ import annotations

# --- repo path bootstrap (this file moved out of maldi/) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ = _Path(__file__).resolve().parents[2]
for _p in (str(_REPO_), str(_REPO_ / "maldi"), str(_REPO_ / "manifold"),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.cm as cm

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key, resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
    dissolve_root_labels,
)
from utils import crop_or_stride_volume, reference_ccf_from_subvolume


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Standalone kernel & graph visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=int, default=5)
    
    p.add_argument("--knn-method",
                   choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted"],
                   default="anatomical_atlas")
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--n-list", dest="n_list", default="1",
                   help="FAISS IVF nlist: an int, or 'sqrt' for nlist=round(sqrt(N)). "
                        "'sqrt' matches the batch pipeline / deployed caches (e.g. "
                        "nlist=729 at N~531k); the resolved value goes into the graph "
                        "cache key, so it must match how the cache was built.")
    p.add_argument("--n-probe", dest="n_probe", default="8",
                   help="FAISS IVF nprobe: an int, or 'sqrt'. Default 8 (recall "
                        "~1.0 in 3D). MUST be > 1 when nlist > 1 -- nprobe=1 with an "
                        "IVF index builds a FRAGMENTED graph (~nlist components).")
    p.add_argument("--cross-region-inflation", dest="cross_region_inflation",
                   type=float, default=10.0,
                   help="For --knn-method=faiss_atlas_weighted: multiply the "
                        "squared distance of every cross-region (different "
                        "atlas label, or background) edge by this factor — a "
                        "soft anatomical prior. 1.0 = off, 10 = mild, 100 = "
                        "strong. Ignored by other knn methods.")
    p.add_argument("--root-handling", dest="root_handling",
                   choices=["dissolve", "ignore", "cross"], default="dissolve",
                   help="faiss_atlas_weighted only — how to treat the atlas's "
                        "label-0 'root' catch-all tissue (~6%% of nodes, threaded "
                        "throughout). 'dissolve' (default): reassign each root node "
                        "to its nearest real region before inflating, so root "
                        "tissue smooths inside a region instead of spraying the "
                        "prior through interiors (frees ~44%% of inflated edges; "
                        "region↔region confinement preserved). 'ignore': keep root "
                        "but don't treat root-touching edges as cross. 'cross': "
                        "legacy — inflate every root-touching edge "
                        "(treat_zero_as_cross=True); reuses old eigvec caches.")
    p.add_argument("--laplacian-norm", choices=["symmetric", "randomwalk"],
                   default="symmetric")
    p.add_argument("--graphbandwidth", type=float, required=True)
    # ---- feature-augmented ("bilateral") kNN metric ------------------------
    # Build the kNN on [x, y, z, beta*feature] instead of just [x, y, z], so an
    # edge that crosses a tissue boundary (large feature gap) is far in the metric
    # and drops out of / weakens in the graph. This attacks graph CONSTRUCTION
    # (which edges exist), unlike --cross-region-inflation which only rescales
    # edges already drawn. beta=0 = today's pure-geometry graph; too large
    # fragments it (bilateral sweet spot). Lipid-free: built from the template.
    p.add_argument("--feature-weight", dest="feature_weight", type=float, default=0.0,
                   help="Weight beta on the z-scored template feature appended to the "
                        "kNN coordinates. 0 = geometry only (default).")
    p.add_argument("--feature", dest="feature_kind",
                   choices=["intensity", "gradient", "both"], default="both",
                   help="Template feature(s) for the augmented metric (used when "
                        "--feature-weight > 0).")
    p.add_argument("--prune-cross-region", dest="prune_cross_region", type=float,
                   default=0.0,
                   help="Fraction of cross-region edges to REMOVE from the graph — a hard "
                        "cut, vs --cross-region-inflation's soft down-weight. 0 = off, "
                        "0.97 = keep 3%% of crossings, 1.0 = remove all. Connectivity-"
                        "preserving: any node the prune would isolate keeps its strongest "
                        "edge, so no specks. Applied in-memory; use with --skip-eigvecs.")
    p.add_argument("--label-source", dest="label_source",
                   choices=["atlas", "kmeans"], default="atlas",
                   help="Region labels for colouring / boundary edges / prune: 'atlas' "
                        "(the annotation, incl. its ~53%% 'root' catch-all + boundary "
                        "specks) or 'kmeans' (a complete, contiguous data-driven "
                        "parcellation of the template — every node in a real cluster, no "
                        "catch-all). The prune diagnostic only confines the walk on a "
                        "clean partition, so use 'kmeans' for that.")
    p.add_argument("--cluster-k", dest="cluster_k", type=int, default=69,
                   help="Number of clusters for --label-source kmeans.")
    p.add_argument("--cluster-spatial-weight", dest="cluster_spatial_weight",
                   type=float, default=3.0,
                   help="Weight on spatial coords vs template features in the k-means "
                        "parcellation (higher = more spatially contiguous clusters).")
    p.add_argument("--denoise-labels", dest="denoise_labels", type=int, default=0,
                   help="Majority-vote label smoothing passes over the graph: each node "
                        "adopts the most common label among its neighbours, so speck / "
                        "'polluted' nodes are absorbed into the surrounding cluster "
                        "(instead of spraying cross-region edges). 0 = off; 2-3 cleans "
                        "most specks. Applied before the prune.")

    p.add_argument("--eigenvector-dir", required=True)
    p.add_argument("--num-modes", type=int, default=200)
    p.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1,
                   help="Lanczos Krylov subspace floor. <=0 (default) auto-picks "
                        "max(1500, 3*num_modes+20); set a small explicit value "
                        "(e.g. 100) to fit large-N (stride=1) eigensolves in GPU memory.")
    p.add_argument("--initial-modes", type=int, default=None)
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute-eigvecs", action="store_true")
    p.add_argument("--skip-eigvecs", dest="skip_eigvecs", action="store_true",
                   help="Skip the eigenvector computation/load entirely (the only slow "
                        "step) so you can play with just the graph Laplacian. Disables "
                        "the eigvec-based layers (Layer C recon, the spectrum random "
                        "walk, per-source Riemann/diffusion-spectral); the graph "
                        "layers — fabric, atlas regions, boundary edges, |L_ij|, and "
                        "the full-Laplacian random walk — all still work.")

    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--diffusion-time", dest="diffusion_time",
                   type=float, default=1.0,
                   help="Heat-kernel diffusion time t used for the diffusion-"
                        "distance layers (both spectral and direct routes). "
                        "Larger t = coarser / more global geometry.")
    p.add_argument("--direct-diffusion", dest="direct_diffusion",
                   action="store_true",
                   help="Also compute the DIRECT diffusion-distance layer via a "
                        "full-L matrix exponential (scipy expm_multiply on the "
                        "N×N Laplacian). This is a validation of the spectral "
                        "route but is SLOW (~minutes at N~500k) and runs on every "
                        "source/time change. OFF by default — the spectral layer "
                        "already shows the diffusion distance; enable only to check "
                        "the truncation gap.")

    # Per-source comparison view
    p.add_argument("--n-sources", type=int, default=4)
    p.add_argument("--source-seed", type=int, default=0)
    p.add_argument("--n-targets", type=int, default=50)
    p.add_argument("--target-strategy",
                   choices=["random", "stratified"], default="stratified")
    p.add_argument("--k-show", type=int, default=30)
    p.add_argument("--source-marker-size", type=float, default=6.0)

    # KNN fabric (Layer A)
    p.add_argument("--fabric-edge-sample", type=int, default=200_000,
                   help="Max number of graph edges drawn for the fabric layer.")
    p.add_argument("--fabric-node-size", type=float, default=0.6)
    p.add_argument("--fabric-edge-width", type=float, default=0.3)

    # Laplacian / reconstruction views (Layers B, C) — share an edge sample
    # to keep rendering coherent across the three layers.
    p.add_argument("--laplacian-edge-sample", type=int, default=80_000,
                   help="Edges shown in Layer B (Laplacian colored). Lower "
                        "than fabric because diverging colormap on many "
                        "thin lines is hard to read.")

    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return vars(p.parse_args())


def denoise_labels_majority_vote(labels: np.ndarray, ei_np: np.ndarray,
                                 n_iters: int, log: logging.Logger) -> np.ndarray:
    """Majority-vote label smoothing over the graph.

    Each node adopts the most common label among its graph neighbours (+ itself),
    iterated. A speck labelled X inside a sea of Y sees mostly-Y neighbours and
    flips to Y — so 'polluted' minority nodes are absorbed into the cluster they
    actually sit in, the partition becomes spatially coherent, and the leftover
    cross-region edges are a thin genuine surface (no scattered noise).
    """
    lab = labels.astype(np.int64).copy()
    N = lab.shape[0]
    for it in range(int(n_iters)):
        uniq, comp = np.unique(lab, return_inverse=True)     # compact ids 0..K-1
        votes = np.zeros((N, uniq.size), dtype=np.int32)
        np.add.at(votes, (ei_np[0], comp[ei_np[1]]), 1)      # neighbour votes (both dirs)
        np.add.at(votes, (ei_np[1], comp[ei_np[0]]), 1)
        np.add.at(votes, (np.arange(N), comp), 1)            # self-vote (breaks ties toward staying)
        new = uniq[votes.argmax(1)]
        changed = int((new != lab).sum())
        lab = new
        log.info(f"denoise-labels iter {it + 1}: {changed:,} nodes relabelled")
        if changed == 0:
            break
    return lab


# =============================================================================
# Setup
# =============================================================================
def setup(args: dict, log: logging.Logger):
    device = torch.device(args["device"])
    template_full = np.load(args["reference_file"])
    annotations_full = (np.load(args["annotations_file"])
                         if args["annotations_file"] else None)

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_full, annotations_full,
        stride=args["stride"],
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, args["threshold"],
    )
    reference_nodes_mm = torch.tensor(reference_ccf, dtype=torch.float32)
    coord_mean = reference_nodes_mm.mean(dim=0)
    coord_std = reference_nodes_mm.std(dim=0).clamp(min=1e-6)
    reference_nodes = ((reference_nodes_mm - coord_mean) / coord_std).to(device)

    node_voxel_idx = np.argwhere(sub_volume > args["threshold"]).astype(np.int32)
    assert node_voxel_idx.shape[0] == reference_nodes.shape[0]

    sv_scale = np.array(
        [args["stride"], args["stride"], args["stride"]], dtype=np.float32,
    )
    sv_translate = np.asarray(voxel_offset, dtype=np.float32)

    # ---- feature-augmented ("bilateral") kNN coordinates -------------------
    # kNN on [x, y, z, beta*z(feature)] so cross-tissue edges are far in the metric
    # (they drop out of / weaken in the graph). beta=0 -> today's geometry-only graph.
    coords_knn = reference_nodes
    feat_tag = None
    beta = float(args.get("feature_weight", 0.0) or 0.0)
    if beta > 0.0:
        vi = node_voxel_idx.astype(np.int64)
        feats = []
        if args["feature_kind"] in ("intensity", "both"):
            feats.append(sub_volume[vi[:, 0], vi[:, 1], vi[:, 2]].astype(np.float64))
        if args["feature_kind"] in ("gradient", "both"):
            gmag = np.sqrt(sum(g ** 2 for g in np.gradient(sub_volume.astype(np.float64))))
            feats.append(gmag[vi[:, 0], vi[:, 1], vi[:, 2]])
        F = np.stack(feats, axis=1)                              # (N, f)
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)                  # z-score each feature
        feat_t = torch.tensor(beta * F, dtype=torch.float32, device=device)
        coords_knn = torch.cat([reference_nodes, feat_t], dim=1)  # (N, 3+f)
        feat_tag = f"{args['feature_kind']}_b{beta:g}"
        log.info(f"feature-augmented kNN: +{F.shape[1]} dim(s) "
                 f"({args['feature_kind']}, beta={beta:g}) -> coords {tuple(coords_knn.shape)}")

    eigenvector_dir = Path(args["eigenvector_dir"])
    graphs = KnnGraphCache(cache_dir=eigenvector_dir / "knn", verbose=True)
    nlist = resolve_nlist(args["n_list"], reference_nodes.shape[0])
    nprobe = resolve_nprobe(args["n_probe"], nlist)
    log.info(f"faiss nlist = {nlist} nprobe = {nprobe}  (--n-list={args['n_list']!r}, "
             f"N={reference_nodes.shape[0]:,})")
    graph_key_parts = {
        "template": args["template_name"],
        "stride": args["stride"],
        "thresh": args["threshold"],
        "method": args["knn_method"],
        "k": args["knn_k"],
        "nlist": nlist,
        "feat": feat_tag,          # None when --feature-weight 0 (keeps existing keys)
        "bbox": None,
    }
    # Level-aware atlas id (mirrors manifold_kernel_builder) so the weighted-
    # graph eigenvector cache keys line up with the batch pipeline. The
    # historical level_15annot default keeps its original un-suffixed tag.
    atlas_stem = (Path(args["annotations_file"]).stem
                  if args["annotations_file"] else "noatlas")
    _legacy_atlas = (atlas_stem == "level_15annot")
    if args["knn_method"] == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
    force_graph = args["force_recompute_graph"]

    if args["knn_method"] == "faiss":
        graph_key = make_graph_key(graph_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss",
            coords=coords_knn, k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts,
            force_recompute=force_graph, device=device,
        )
    elif args["knn_method"] == "anatomical_atlas":
        graph_key = make_graph_key(graph_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas",
            volume=sub_volume, threshold=args["threshold"],
            atlas_volume=sub_atlas, connectivity=3,
            coords=coords_knn, k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts,
            force_recompute=force_graph, device=device,
        )
    elif args["knn_method"] == "faiss_atlas_weighted":
        # Base topology is the plain faiss graph (cache shared with
        # --knn-method=faiss); atlas labels only *reweight* cross-region edges
        # via a soft inflation prior. Topology unchanged → still one zero
        # eigenvalue, kernel stays well-conditioned.
        if sub_atlas is None:
            raise ValueError(
                "--knn-method=faiss_atlas_weighted requires --annotations-file."
            )
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss",
            coords=coords_knn, k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=base_key_parts,
            force_recompute=force_graph, device=device,
        )
        node_labels = labels_for_nodes_from_sub_atlas(
            sub_volume, sub_atlas, args["threshold"],
        )
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Label/node mismatch: {node_labels.shape[0]} labelled voxels "
                f"vs {knn.x.shape[0]} graph nodes — check that atlas and "
                f"template were cropped & strided identically."
            )
        # Root handling: the label-0 catch-all is real tissue, not background —
        # treating it as cross (legacy) inflates ~half the edges for almost no
        # region↔region confinement gain and fragments interiors. Default is to
        # dissolve it into the nearest real region before inflating.
        root_mode = args.get("root_handling", "dissolve")
        if root_mode == "dissolve":
            node_labels = dissolve_root_labels(
                node_labels, reference_nodes.detach().cpu().numpy())
        treat_zero = (root_mode == "cross")
        inflation = float(args["cross_region_inflation"])
        edge_index, edge_value, info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=treat_zero,
        )
        log.info(
            f"faiss_atlas_weighted (root={root_mode}): inflated "
            f"{info['n_cross']:,}/{info['n_total']:,} "
            f"({100 * info['frac_cross']:.1f}%) cross-region edges ×{inflation:g}"
        )
        # Weighting tag discriminates the eigenvector cache across inflations
        # (inflation=10 and inflation=50 must NOT share eigvecs). Keep legacy
        # 'cross' key un-suffixed so existing caches still load; dissolve/ignore
        # change edge_value → must get distinct keys.
        _base_wt = (f"atlas_x{inflation:g}" if _legacy_atlas
                    else f"{atlas_stem}_x{inflation:g}")
        graph_key_parts["weighting"] = (
            _base_wt if root_mode == "cross" else f"{_base_wt}_root{root_mode}"
        )
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"Unknown --knn-method {args['knn_method']!r}")

    # ---- Per-node region labels: atlas, or a data-driven k-means parcellation ----
    node_labels = None
    if args.get("label_source", "atlas") == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        vi = node_voxel_idx.astype(np.int64)
        sv64 = sub_volume.astype(np.float64)
        inten = sv64[vi[:, 0], vi[:, 1], vi[:, 2]]
        grad = np.sqrt(sum(g ** 2 for g in np.gradient(sv64)))[vi[:, 0], vi[:, 1], vi[:, 2]]
        _z = lambda x: (x - x.mean()) / (x.std() + 1e-9)   # noqa: E731
        sw = float(args.get("cluster_spatial_weight", 3.0))
        feat = np.concatenate(
            [sw * reference_nodes.cpu().numpy(), _z(inten)[:, None], _z(grad)[:, None]],
            axis=1)
        K = int(args.get("cluster_k", 69))
        node_labels = (MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=20000)
                       .fit_predict(feat) + 1).astype(np.int64)
        log.info(f"label source: k-means parcellation ({K} clusters, "
                 f"spatial_weight={sw:g}) — every node in a cluster, no catch-all")
    elif sub_atlas is not None:
        try:
            _nl = np.asarray(
                labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, args["threshold"])
            ).astype(np.int64)
            # Match the display/overlay partition to the one the kernel was
            # inflated on: if root was dissolved for the graph, dissolve it here
            # too so the region/border overlays show the same regions.
            if (args.get("knn_method") == "faiss_atlas_weighted"
                    and args.get("root_handling", "dissolve") == "dissolve"):
                _nl = dissolve_root_labels(_nl, reference_nodes.detach().cpu().numpy())
            if _nl.shape[0] == knn.x.shape[0]:
                node_labels = _nl
        except Exception as e:  # noqa: BLE001
            log.warning("could not derive node region labels: %s", e)

    # ---- denoise labels: absorb speck / polluted nodes into their surrounding cluster ----
    n_denoise = int(args.get("denoise_labels", 0) or 0)
    if n_denoise > 0 and node_labels is not None:
        node_labels = denoise_labels_majority_vote(
            node_labels, edge_index.cpu().numpy(), n_denoise, log)

    # ---- prune cross-region edges (a HARD cut, connectivity-preserving) ----
    # Remove a fraction of the edges whose endpoints are in different regions. Any node
    # the prune would ISOLATE keeps its strongest incident edge (a bridge), so the bulk
    # stays connected instead of shattering into specks. On a CLEAN partition this makes
    # the boundary a real bottleneck and the walk pools in-cluster. In-memory only.
    prune = float(args.get("prune_cross_region", 0.0) or 0.0)
    if prune > 0.0 and node_labels is not None:
        ei_np = edge_index.cpu().numpy()
        ev_np = edge_value.detach().cpu().numpy()
        Nn = knn.x.shape[0]
        cross = ((node_labels[ei_np[0]] != node_labels[ei_np[1]])
                 & (node_labels[ei_np[0]] > 0) & (node_labels[ei_np[1]] > 0))
        drop = np.zeros(cross.shape, bool)
        cidx = np.where(cross)[0]
        n_req = int(round(prune * cidx.size))
        if n_req:
            drop[np.random.default_rng(0).choice(cidx, n_req, replace=False)] = True
        # connectivity-preserving: restore one (strongest) bridge per would-be isolated node
        deg = np.zeros(Nn)
        np.add.at(deg, ei_np[0][~drop], 1); np.add.at(deg, ei_np[1][~drop], 1)
        iso = deg == 0
        n_iso0 = int(iso.sum())
        if n_iso0:
            ic = np.where((iso[ei_np[0]] | iso[ei_np[1]]) & drop)[0]  # dropped edges @ iso nodes
            owner = np.where(iso[ei_np[0][ic]], ei_np[0][ic], ei_np[1][ic])
            order = np.lexsort((ev_np[ic], owner))                   # per owner, smallest ev first
            firstm = np.ones(order.size, bool)
            firstm[1:] = owner[order][1:] != owner[order][:-1]
            drop[ic[order][firstm]] = False
        keep_mask = torch.as_tensor(~drop, device=edge_index.device)
        n_before = edge_index.shape[1]
        edge_index = edge_index[:, keep_mask]
        edge_value = edge_value[keep_mask]
        deg2 = np.zeros(Nn)
        np.add.at(deg2, ei_np[0][~drop], 1); np.add.at(deg2, ei_np[1][~drop], 1)
        log.info(f"pruned {int(drop.sum()):,}/{int(cross.sum()):,} cross-region edges "
                 f"(frac={prune:g}) — {n_before:,} -> {edge_index.shape[1]:,} edges; "
                 f"isolated nodes {n_iso0:,} -> {int((deg2 == 0).sum()):,} after bridges")

    bw = float(args["graphbandwidth"])
    bw_tensor = torch.tensor(bw, device=device)
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0], bw_tensor, args["laplacian_norm"],
    )

    if args.get("skip_eigvecs"):
        log.info("--skip-eigvecs: skipping the eigenvector computation/load. "
                 "Eigvec-based layers (Layer C, spectrum walk, per-source Riemann/"
                 "diffusion-spectral) are disabled; graph-Laplacian layers still work.")
        eigval, eigvec = None, None
    else:
        eigvec_key_parts = {
            "graph": graph_key,
            "norm": args["laplacian_norm"],
            "bw": bw,
            "modes": args["num_modes"],
        }
        eigvec_key = make_eig_key(eigvec_key_parts)
        ncv_min = resolve_ncv_min(args["num_modes"], args.get("ncv_min", -1))
        solver = LaplacianEigensolver(
            num_modes=args["num_modes"], backend="cupy",
            tol=1e-4, ncv_min=ncv_min, verbose=True,
        )
        eigval, eigvec = solver.compute_or_load(
            laplacian_op,
            cache_dir=eigenvector_dir / "eigvecs", key=eigvec_key,
            graphbandwidth=bw, laplacian_normalization=args["laplacian_norm"],
            extra=eigvec_key_parts,
            force_recompute=args["force_recompute_eigvecs"], device=device,
        )
        log.info(f"Loaded {eigvec.shape[1]} eigenmodes")

    L_sparse = build_symmetric_laplacian_sparse(laplacian_op, knn.x.shape[0])
    if args["laplacian_norm"] != "symmetric":
        log.warning(
            "Direct diffusion-distance route uses the symmetric Laplacian, but "
            "eigvecs are '%s'-normalized — the spectral and direct diffusion "
            "layers will only match exactly for --laplacian-norm symmetric.",
            args["laplacian_norm"],
        )

    # (node_labels computed above, before the prune step.)

    return dict(
        device=device,
        template_full=template_full,
        sub_volume=sub_volume,
        node_voxel_idx=node_voxel_idx,
        reference_nodes=reference_nodes,
        node_labels=node_labels,
        sv_scale=sv_scale, sv_translate=sv_translate,
        knn=knn, edge_index=edge_index, edge_value=edge_value,
        laplacian_op=laplacian_op,
        L_sparse=L_sparse,
        eigval=eigval, eigvec=eigvec,
        bw=bw,
    )


# =============================================================================
# Edge subsampling
# =============================================================================
def subsample_edges(
    edge_index: torch.Tensor,
    edge_value: torch.Tensor,
    max_edges: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (edge_pairs, edge_sq_dists, subsample_mask_indices).

    Dedupes undirected edges (i, j) ~ (j, i) by keeping i < j, then random-
    samples up to max_edges. Returns the indices into the ORIGINAL edge_index
    array so we can also fetch the Laplacian entries for the same edges.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    val = edge_value.cpu().numpy()
    keep = src < dst
    keep_idx = np.where(keep)[0]
    src, dst, val = src[keep], dst[keep], val[keep]

    if src.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        sel = rng.choice(src.shape[0], size=max_edges, replace=False)
        src, dst, val = src[sel], dst[sel], val[sel]
        keep_idx = keep_idx[sel]

    pairs = np.stack([src, dst], axis=1)
    return pairs, val, keep_idx


def make_lines_array(
    pairs: np.ndarray, node_positions: np.ndarray,
) -> np.ndarray:
    """Build (M, 2, 3) line segments from edge index pairs."""
    lines = np.zeros((pairs.shape[0], 2, 3), dtype=np.float32)
    lines[:, 0, :] = node_positions[pairs[:, 0]]
    lines[:, 1, :] = node_positions[pairs[:, 1]]
    return lines


# =============================================================================
# Laplacian queries — diag + off-diagonal entries
# =============================================================================
def laplacian_diag(laplacian_op: GraphLaplacianOperator) -> np.ndarray:
    """Returns (N,) — diagonal of L."""
    return laplacian_op.laplacian_diag.detach().cpu().numpy()


def laplacian_offdiag_at_edges(
    laplacian_op: GraphLaplacianOperator,
    edge_index_subset: np.ndarray,
    full_edge_index: torch.Tensor,
) -> np.ndarray:
    """For each edge index in `edge_index_subset` (indices into full_edge_index),
    return the corresponding L[i, j] value. The Laplacian's off-diagonal is
    stored in `laplacian_triu` ordered the same as edge_index.

    Note: L[i, j] = -laplacian_triu[edge_idx] for the symmetric / random-walk
    normalizations used here (the heat-kernel adjacency contributes negatively
    off-diagonal).
    """
    triu = laplacian_op.laplacian_triu.detach().cpu().numpy()
    return -triu[edge_index_subset]


def build_symmetric_laplacian_sparse(laplacian_op: GraphLaplacianOperator, N: int):
    """Assemble the symmetric graph Laplacian L as a scipy CSR matrix.

    L = diag(laplacian_diag) - upper - upperᵀ, where `upper` holds the stored
    off-diagonal entries at `idx` (L[i, j] = -laplacian_triu). This is exactly
    the matrix the operator applies in `_matmul` for the symmetric
    normalization, so its matrix-exponential e^{-tL} is the graph heat kernel —
    used by the *direct* diffusion-distance route (no eigendecomposition).

    Note: `laplacian_triu`/`laplacian_diag` are always the symmetric-normalized
    quantities; for --laplacian-norm randomwalk the operator's action differs,
    so the direct route here still uses the symmetric L.
    """
    import scipy.sparse as sp

    idx = laplacian_op.idx.detach().cpu().numpy()
    triu = laplacian_op.laplacian_triu.detach().cpu().numpy()   # L[i, j] = -triu
    diag = laplacian_op.laplacian_diag.detach().cpu().numpy()
    diag_idx = np.arange(N)
    rows = np.concatenate([idx[0], idx[1], diag_idx])
    cols = np.concatenate([idx[1], idx[0], diag_idx])
    data = np.concatenate([-triu, -triu, diag]).astype(np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


# =============================================================================
# Diffusion distance — two routes that agree at full rank (symmetric norm)
# =============================================================================
def diffusion_distance_spectral(
    src_idx: int,
    tgt_idxs: np.ndarray,
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    t: float,
    num_modes: int,
) -> np.ndarray:
    """D_t(s, j) = sqrt( Σ_{k<K} e^{-2 λ_k t} (φ_k(s) - φ_k(j))² ).

    The Euclidean distance in the diffusion-map embedding, truncated to K
    modes. Dominated by low eigenvalues (e^{-2λt} crushes high modes), so the
    truncation is well-conditioned — unlike geodesic recovery.
    """
    K = int(min(num_modes, eigvec.shape[1]))
    lam = eigval[:K].clamp(min=0.0)
    w = torch.exp(-2.0 * lam * float(t))                     # (K,)
    dphi = eigvec[tgt_idxs, :K] - eigvec[src_idx, :K]        # (T, K)
    d2 = (w * dphi.pow(2)).sum(dim=-1).clamp(min=0.0)        # (T,)
    return d2.sqrt().cpu().numpy()


def diffusion_distance_direct(
    src_idx: int,
    tgt_idxs: np.ndarray,
    L_sparse,
    t: float,
) -> np.ndarray:
    """D_t(s, j) = || e^{-tL} e_s - e^{-tL} e_j ||₂, straight from the matrix.

    Computes the heat profiles of source + targets in one shot via a Krylov
    matrix-exponential (scipy `expm_multiply`) — no eigendecomposition. For the
    symmetric Laplacian this equals `diffusion_distance_spectral` at full rank.
    """
    from scipy.sparse.linalg import expm_multiply

    tgt_idxs = np.asarray(tgt_idxs)
    cols = np.concatenate([[int(src_idx)], tgt_idxs.astype(np.int64)])
    N = L_sparse.shape[0]
    E = np.zeros((N, cols.shape[0]), dtype=np.float64)
    E[cols, np.arange(cols.shape[0])] = 1.0
    heat = expm_multiply(-float(t) * L_sparse, E)            # (N, 1 + T)
    diff = heat[:, 1:] - heat[:, 0:1]                        # (N, T)
    d2 = (diff ** 2).sum(axis=0)                             # (T,)
    return np.sqrt(np.clip(d2, 0.0, None))


# =============================================================================
# Eigenvector reconstruction — kernel diagonal at each node
# =============================================================================
def kernel_diagonal_from_eigvecs(
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    nu: float,
    lengthscale: float,
    num_modes: int,
) -> np.ndarray:
    """K[i, i] = Σ_{k=1..K} spectral_density(λ_k) · φ_k(i)².

    Tells you how "well-represented" each node is by the truncated eigenbasis.
    For a complete eigenbasis (K=N) this is the kernel's actual diagonal,
    which should be roughly uniform for a sensible kernel. A K-truncated
    version shows which regions of the brain get their kernel value built
    up first as more modes are added.
    """
    K = int(min(num_modes, eigvec.shape[1]))
    safe_lam = eigval[:K].clamp(min=0.0)
    weight = (2.0 * nu / (lengthscale ** 2) + safe_lam).pow(-nu)
    phi_sq = eigvec[:, :K] ** 2          # (N, K)
    diag = (phi_sq * weight).sum(dim=-1)  # (N,)
    return diag.cpu().numpy()


def heat_walk_from_source(
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    src_idx: int,
    t: float,
    num_modes: int,
) -> np.ndarray:
    """Heat-kernel random walk from a source node, reconstructed from K modes:

        p_K(·) = Σ_{k<K} e^{-t λ_k} φ_k(src) φ_k(·)   (∝ the K-truncated e^{-tL} e_src).

    More modes → the walk localizes (converges to the true walk, which is contained
    by the graph geometry); fewer modes → it blurs out and bleeds across boundaries.
    Comparing this against the atlas-region / boundary-edge layers as K (num modes)
    and t (diffusion time) change is the direct debug of the num_modes limitation.
    """
    K = int(min(num_modes, eigvec.shape[1]))
    lam = eigval[:K].clamp(min=0.0)
    coeff = torch.exp(-float(t) * lam) * eigvec[int(src_idx), :K]   # (K,)
    p = (eigvec[:, :K] * coeff).sum(dim=-1)                         # (N,)
    return p.clamp(min=0.0).cpu().numpy()


def estimate_lam_max(laplacian_op, n: int, device, iters: int = 30) -> float:
    """Largest eigenvalue of L via power iteration (operator matvecs only)."""
    v = torch.randn(n, 1, device=device)
    for _ in range(iters):
        v = laplacian_op._matmul(v)
        v = v / (v.norm() + 1e-12)
    return float((v * laplacian_op._matmul(v)).sum())


def heat_walk_from_laplacian(laplacian_op, src_idx: int, t: float, n: int, device,
                             lam_max: float, n_steps: int = None) -> np.ndarray:
    """Full heat-kernel walk e^{-tL} e_s straight from the Laplacian OPERATOR — all modes,
    NO eigendecomposition — via a matvec exponential integrator (I - (t/m)L)^m e_s → e^{-tL}e_s.

    This is the reference the K-mode `heat_walk_from_source` (spectrum) converges to as K → all.
    Both are e^{-tL}e_s on the same operator, so (after per-layer normalisation) their SHAPES
    match at full rank and diverge exactly by the mode-truncation error.
    """
    m = int(n_steps or max(40, int(2.0 * float(t) * max(lam_max, 1e-6)) + 1))
    dt = float(t) / m
    u = torch.zeros(n, 1, device=device, dtype=torch.float32)
    u[int(src_idx), 0] = 1.0
    for _ in range(m):
        u = u - dt * laplacian_op._matmul(u)          # (I - dt·L) u  ≈  e^{-dt·L} u
    return u.clamp(min=0.0).squeeze(1).float().cpu().numpy()


# =============================================================================
# Per-source kernel functions (for the on-top comparison layers)
# =============================================================================
def matern_euclidean_kernel(
    src_coord: torch.Tensor,
    tgt_coords: torch.Tensor,
    lengthscale: float,
    nu: float = 1.0,
) -> np.ndarray:
    d = torch.sqrt(((tgt_coords - src_coord) ** 2).sum(dim=-1))
    r = d / lengthscale
    if nu == 0.5:
        k = torch.exp(-r)
    elif nu == 1.5:
        sqrt3_r = (3 ** 0.5) * r
        k = (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)
    elif nu == 2.5:
        sqrt5_r = (5 ** 0.5) * r
        k = (1.0 + sqrt5_r + (5.0 / 3.0) * r ** 2) * torch.exp(-sqrt5_r)
    else:
        k = torch.exp(-0.5 * r ** 2)
    return k.cpu().numpy()


def riemann_manifold_kernel(
    src_idx: int,
    tgt_idxs: np.ndarray,
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    nu: float,
    lengthscale: float,
    num_modes: int,
) -> np.ndarray:
    K = int(min(num_modes, eigvec.shape[1]))
    safe_lam = eigval[:K].clamp(min=0.0)
    weight = (2.0 * nu / (lengthscale ** 2) + safe_lam).pow(-nu)
    src_phi = eigvec[src_idx, :K]
    tgt_phi = eigvec[tgt_idxs, :K]
    return (weight * src_phi * tgt_phi).sum(dim=-1).cpu().numpy()


def heat_strength(sq_dist: np.ndarray, bw: float) -> np.ndarray:
    return np.exp(-sq_dist / (4.0 * bw * bw))


def pick_target_nodes(
    src_idx: int,
    reference_nodes: torch.Tensor,
    n_targets: int,
    strategy: str,
    seed: int,
) -> np.ndarray:
    N = reference_nodes.shape[0]
    rng = np.random.default_rng(seed + src_idx)
    if strategy == "random":
        idxs = rng.choice(N, size=min(n_targets, N), replace=False)
        return idxs.astype(np.int64)
    src_coord = reference_nodes[src_idx]
    sq_dist = ((reference_nodes - src_coord) ** 2).sum(dim=-1).cpu().numpy()
    order = np.argsort(sq_dist)
    order = order[order != src_idx]
    sample_positions = np.linspace(0, len(order) - 1, n_targets).astype(np.int64)
    return order[sample_positions]


def knn_neighbors_of(
    src_idx: int,
    edge_index: torch.Tensor,
    edge_value: torch.Tensor,
    k_show: int,
) -> tuple[np.ndarray, np.ndarray]:
    src_eq = (edge_index[0] == src_idx)
    dst_eq = (edge_index[1] == src_idx)
    nbrs = torch.cat([edge_index[1, src_eq], edge_index[0, dst_eq]]).cpu().numpy()
    dists = torch.cat([edge_value[src_eq], edge_value[dst_eq]]).cpu().numpy()
    nbrs_uniq, first = np.unique(nbrs, return_index=True)
    dists_uniq = dists[first]
    order = np.argsort(dists_uniq)[:k_show]
    return nbrs_uniq[order], dists_uniq[order]


def make_lines(
    src_voxel: np.ndarray, nbr_voxels: np.ndarray,
    sv_scale: np.ndarray, sv_translate: np.ndarray,
) -> np.ndarray:
    src_full = src_voxel.astype(np.float32) * sv_scale + sv_translate
    nbr_full = nbr_voxels.astype(np.float32) * sv_scale + sv_translate
    n = nbr_voxels.shape[0]
    lines = np.zeros((n, 2, 3), dtype=np.float32)
    lines[:, 0, :] = src_full
    lines[:, 1, :] = nbr_full
    return lines


def colors_widths(
    strengths: np.ndarray, cmap_name: str = "viridis", gamma: float = 0.5,
    min_width: float = 0.3, max_width: float = 2.5,
    ref_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(strengths, dtype=np.float32)
    has_negative = bool((s < 0).any())
    abs_s = np.abs(s)
    # ref_max lets several layers share one normalization so equal underlying
    # values map to equal colors/widths (used to compare the two diffusion
    # layers, which should agree).
    abs_max = ref_max if ref_max is not None else (
        abs_s.max() if abs_s.size else 1.0)
    if abs_max == 0:
        abs_max = 1.0
    if has_negative:
        norm = s / abs_max
        cmap = cm.get_cmap("RdBu_r")
        colors = cmap(0.5 + 0.5 * norm)
    else:
        norm = (abs_s / abs_max) ** gamma
        cmap = cm.get_cmap(cmap_name)
        colors = cmap(norm)
    widths = min_width + (max_width - min_width) * (abs_s / abs_max) ** gamma
    return colors, widths


def build_lines_for_source(
    src_idx: int, ctx: dict,
    num_modes: int, lengthscale: float, gamma: float,
    k_show: int, n_targets: int, target_strategy: str,
    source_seed: int, nu: float, diffusion_time: float,
) -> dict:
    src_voxel = ctx["node_voxel_idx"][src_idx]
    knn_idxs, knn_sq_dists = knn_neighbors_of(
        int(src_idx), ctx["edge_index"], ctx["edge_value"], k_show,
    )
    knn_strengths = heat_strength(knn_sq_dists, ctx["bw"])
    knn_lines = make_lines(
        src_voxel, ctx["node_voxel_idx"][knn_idxs],
        ctx["sv_scale"], ctx["sv_translate"],
    )
    knn_colors, knn_widths = colors_widths(knn_strengths, "viridis", gamma)

    tgt_idxs = pick_target_nodes(
        int(src_idx), ctx["reference_nodes"], n_targets, target_strategy,
        source_seed,
    )
    tgt_voxels = ctx["node_voxel_idx"][tgt_idxs]
    src_coord = ctx["reference_nodes"][src_idx]
    tgt_coords = ctx["reference_nodes"][tgt_idxs]
    matern_strengths = matern_euclidean_kernel(
        src_coord, tgt_coords, lengthscale, nu,
    )
    matern_lines = make_lines(
        src_voxel, tgt_voxels, ctx["sv_scale"], ctx["sv_translate"],
    )
    matern_colors, matern_widths = colors_widths(matern_strengths, "plasma", gamma)

    # Riemann (manifold) kernel + spectral diffusion distance both need the
    # eigenbasis; when it was skipped (--skip-eigvecs) return zeros so these layers
    # exist but are inert, leaving the graph-Laplacian layers fully functional.
    _has_eig = ctx.get("eigvec") is not None
    if _has_eig:
        riemann_strengths = riemann_manifold_kernel(
            int(src_idx), tgt_idxs, ctx["eigval"], ctx["eigvec"],
            nu=nu, lengthscale=lengthscale, num_modes=num_modes,
        )
    else:
        riemann_strengths = np.zeros(len(tgt_idxs), dtype=np.float32)
    riemann_lines = matern_lines.copy()
    riemann_colors, riemann_widths = colors_widths(riemann_strengths, "magma", gamma)

    # Diffusion distance to the same targets, computed two ways: spectrally
    # from the truncated eigenbasis, and directly from the sparse Laplacian
    # via a matrix-exponential. Shared color/width scale (ref_max) so the two
    # layers are directly comparable by eye.
    if _has_eig:
        diff_spec = diffusion_distance_spectral(
            int(src_idx), tgt_idxs, ctx["eigval"], ctx["eigvec"],
            diffusion_time, num_modes,
        )
    else:
        diff_spec = np.zeros(len(tgt_idxs), dtype=np.float32)
    # The direct route is a full-L matrix exponential (expm_multiply on the N×N
    # Laplacian): ~minutes at N~500k, and it reran on every source/time change.
    # Off by default (the spectral layer already shows the diffusion distance);
    # opt in with --direct-diffusion to check the spectral-truncation gap.
    if ctx.get("direct_diffusion", False):
        diff_direct = diffusion_distance_direct(
            int(src_idx), tgt_idxs, ctx["L_sparse"], diffusion_time,
        )
    else:
        diff_direct = np.zeros_like(diff_spec)
    diff_ref = float(max(diff_spec.max(initial=0.0),
                         diff_direct.max(initial=0.0), 1e-12))
    spec_colors, spec_widths = colors_widths(
        diff_spec, "viridis", gamma, ref_max=diff_ref)
    direct_colors, direct_widths = colors_widths(
        diff_direct, "viridis", gamma, ref_max=diff_ref)

    return {
        "knn":     dict(lines=knn_lines, colors=knn_colors,
                        widths=knn_widths, strengths=knn_strengths),
        "matern":  dict(lines=matern_lines, colors=matern_colors,
                        widths=matern_widths, strengths=matern_strengths),
        "riemann": dict(lines=riemann_lines, colors=riemann_colors,
                        widths=riemann_widths, strengths=riemann_strengths),
        "diff_spec":   dict(lines=matern_lines.copy(), colors=spec_colors,
                            widths=spec_widths, strengths=diff_spec),
        "diff_direct": dict(lines=matern_lines.copy(), colors=direct_colors,
                            widths=direct_widths, strengths=diff_direct),
    }


# =============================================================================
# Napari layer management for per-source layers
# =============================================================================
LAYER_CONFIG = {
    "knn":     dict(name="src KNN edges  exp(-d²/(4 bw²))",   visible=False),
    "matern":  dict(name="src Matern (Euclidean)",            visible=False),
    "riemann": dict(name="src Riemann (Manifold)",            visible=False),
    "diff_spec":   dict(name="src DiffDist spectral  Σe^{-2λt}(φ_s-φ_j)²",
                        visible=False),
    "diff_direct": dict(name="src DiffDist direct   ||e^{-tL}(e_s-e_j)||",
                        visible=False),
}


def replace_shapes_layer(viewer, layer_state, key, lines, colors, widths):
    old = layer_state[key]
    cfg = LAYER_CONFIG[key]
    if old is not None and old in viewer.layers:
        keep_visible = old.visible
        keep_idx = viewer.layers.index(old)
        viewer.layers.remove(old)
    else:
        keep_visible = cfg["visible"]
        keep_idx = len(viewer.layers)
    if lines.shape[0] == 0:
        lines = np.zeros((1, 2, 3), dtype=np.float32)
        colors = np.zeros((1, 4), dtype=np.float32)
        widths = np.array([0.0], dtype=np.float32)
    new_layer = viewer.add_shapes(
        [lines[i] for i in range(lines.shape[0])],
        shape_type="line",
        edge_color=colors,
        edge_width=widths.tolist(),
        name=cfg["name"],
        opacity=0.9,
    )
    new_layer.visible = keep_visible
    new_idx = viewer.layers.index(new_layer)
    if keep_idx != new_idx and keep_idx < len(viewer.layers):
        viewer.layers.move(new_idx, keep_idx)
    layer_state[key] = new_layer
    return new_layer


# =============================================================================
# Layer C: refresh helper for the eigenvector-reconstruction view
# =============================================================================
def update_recon_layer_data(layer, diag_values: np.ndarray, gamma: float):
    d = diag_values
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        norm = (d - d_min) / (d_max - d_min)
        norm = np.clip(norm, 0.0, 1.0) ** gamma
    else:
        norm = np.zeros_like(d)
    colors = cm.get_cmap("magma")(norm)
    layer.face_color = colors
    layer.border_color = colors                        # was layer.edge_color



# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args["verbose"] else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("visualize_kernels")

    ctx = setup(args, log)
    ctx["direct_diffusion"] = args["direct_diffusion"]
    if args["no_launch"]:
        log.info("--no-launch passed; precompute OK, exiting.")
        return

    import napari
    from magicgui import magicgui

    # ---- Node positions in full-res template coords ----
    all_node_positions = (
        ctx["node_voxel_idx"].astype(np.float32) * ctx["sv_scale"]
        + ctx["sv_translate"]
    )
    N = all_node_positions.shape[0]
    log.info(f"{N:,} graph nodes in full-res coords")

    # ---- Source nodes for per-source comparison view ----
    rng = np.random.default_rng(args["source_seed"])
    src_idxs = rng.choice(N, size=args["n_sources"], replace=False)
    log.info(f"Sources for kernel comparison: {src_idxs.tolist()}")

    # ---- Edge subsamples (one for each of Layer A and Layer B) ----
    log.info("Subsampling graph edges for Layer A (fabric)...")
    fabric_pairs, fabric_sq_dists, _ = subsample_edges(
        ctx["edge_index"], ctx["edge_value"],
        max_edges=args["fabric_edge_sample"], seed=args["source_seed"],
    )
    log.info(f"  fabric: {fabric_pairs.shape[0]:,} edges")

    log.info("Subsampling graph edges for Layer B (Laplacian-colored)...")
    lap_pairs, lap_sq_dists, lap_full_idx = subsample_edges(
        ctx["edge_index"], ctx["edge_value"],
        max_edges=args["laplacian_edge_sample"],
        seed=args["source_seed"] + 1,
    )
    log.info(f"  laplacian: {lap_pairs.shape[0]:,} edges")

    # ---- Precompute Laplacian quantities ----
    log.info("Computing Laplacian diagonal + edge entries...")
    lap_diag = laplacian_diag(ctx["laplacian_op"])  # (N,)
    lap_edge_vals = laplacian_offdiag_at_edges(
        ctx["laplacian_op"], lap_full_idx, ctx["edge_index"],
    )
    log.info(
        f"  diag range: [{lap_diag.min():.4g}, {lap_diag.max():.4g}], "
        f"offdiag range: [{lap_edge_vals.min():.4g}, "
        f"{lap_edge_vals.max():.4g}]"
    )

    # ============= Set up the napari viewer ============================
    viewer = napari.Viewer(title="Kernel & graph debugger")
    viewer.dims.ndisplay = 3

    # -----------------------------------------------------------------------
    # Layer A — KNN fabric
    # -----------------------------------------------------------------------
    viewer.add_points(
        all_node_positions,
        name="A1: graph nodes",
        size=float(args["fabric_node_size"]),
        face_color="white", border_color="white",     # was edge_color
        symbol="o", opacity=0.25, blending="additive",
    )
    fabric_lines = make_lines_array(fabric_pairs, all_node_positions)
    # Draw edges as a Vectors layer (a single mesh) rather than add_shapes with one
    # Shape per line: Shapes triangulates every line separately, so ~200k lines take
    # ~15s to build and leave the viewer sluggish; Vectors builds the same set in ~0.3s.
    # data is (N, 2, 3): [:,0]=origin, [:,1]=displacement (end - start); length=1 keeps it.
    fabric_vecs = np.stack(
        [fabric_lines[:, 0], fabric_lines[:, 1] - fabric_lines[:, 0]], axis=1)
    fabric_color = np.tile(
        np.array([[0.6, 0.6, 0.6, 0.35]], dtype=np.float32),
        (fabric_vecs.shape[0], 1),
    )
    log.info(f"Building fabric edge layer ({fabric_vecs.shape[0]:,} edges, Vectors)...")
    viewer.add_vectors(
        fabric_vecs, length=1.0, vector_style="line",
        edge_color=fabric_color,
        edge_width=float(args["fabric_edge_width"]),
        name="A2: KNN fabric (edges)",
        opacity=0.7, blending="translucent",
    )

    # -----------------------------------------------------------------------
    # Layer B — Graph Laplacian
    # -----------------------------------------------------------------------
    # Nodes colored by diag(L) — local connectivity strength.
    # High diag = well-connected (has many strong neighbors); low diag =
    # weakly-connected (graph fragmenting locally).
    diag_norm = (lap_diag - lap_diag.min()) / (
        max(lap_diag.max() - lap_diag.min(), 1e-12)
    )
    diag_norm = diag_norm ** float(args["gamma"])
    diag_colors = cm.get_cmap("cividis")(diag_norm)
    viewer.add_points(
        all_node_positions,
        name="B1: Laplacian diag (cividis)",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color=diag_colors,
        border_color=diag_colors,                      # was edge_color
        symbol="o", opacity=0.85, blending="translucent",
        visible=False,
    )


    # Edges colored by |L[i, j]| on a LOG scale (inferno). With cross-region
    # inflation the boundary-crossing edges are WEAKER, so the anatomical borders
    # show up as dark seams threading through the bright interior. Visible by default.
    import matplotlib.colors as _mcolors
    lap_edge_lines = make_lines_array(lap_pairs, all_node_positions)
    lap_edge_vecs = np.stack(
        [lap_edge_lines[:, 0], lap_edge_lines[:, 1] - lap_edge_lines[:, 0]], axis=1)
    lap_absL = np.abs(lap_edge_vals)
    _lo = max(float(np.percentile(lap_absL, 2)), 1e-6)
    _hi = max(float(np.percentile(lap_absL, 98)), _lo * 1.001)
    edge_colors_lap = cm.get_cmap("inferno")(
        _mcolors.LogNorm(vmin=_lo, vmax=_hi)(np.clip(lap_absL, _lo, _hi))
    ).astype(np.float32)
    log.info(f"Building Laplacian edge layer ({lap_edge_vecs.shape[0]:,} edges, Vectors)...")
    viewer.add_vectors(
        lap_edge_vecs, length=1.0, vector_style="line",
        edge_color=edge_colors_lap,
        edge_width=0.45,
        name="B2: Laplacian |L_ij| (boundaries, log)",
        opacity=0.85, blending="translucent",
        visible=True,
    )

    # -----------------------------------------------------------------------
    # Atlas regions + boundary edges (the anatomy the graph was built on)
    # -----------------------------------------------------------------------
    node_labels = ctx.get("node_labels")
    if node_labels is not None:
        # distinct color per region (background label 0 -> dark grey)
        uniq = np.unique(node_labels[node_labels > 0])
        _rng = np.random.default_rng(1)
        _pal = {int(r): _rng.random(3) * 0.65 + 0.2 for r in uniq}
        _rgb = np.array([_pal.get(int(l), (0.12, 0.12, 0.12)) for l in node_labels],
                        dtype=np.float32)
        region_colors = np.concatenate(
            [_rgb, np.full((_rgb.shape[0], 1), 0.9, np.float32)], axis=1)
        viewer.add_points(
            all_node_positions, name="atlas regions",
            size=float(args["fabric_node_size"]) * 1.5,
            face_color=region_colors, border_color=region_colors,
            symbol="o", opacity=0.9, blending="translucent", visible=True,
        )
        # the graph edges that CROSS a region boundary — they trace the borders
        _li = node_labels[lap_pairs[:, 0]]
        _lj = node_labels[lap_pairs[:, 1]]
        _xreg = (_li != _lj) & (_li > 0) & (_lj > 0)
        if _xreg.any():
            _bl = make_lines_array(lap_pairs[_xreg], all_node_positions)
            _bvecs = np.stack([_bl[:, 0], _bl[:, 1] - _bl[:, 0]], axis=1)
            viewer.add_vectors(
                _bvecs, length=1.0, vector_style="line",
                edge_color=np.tile(np.array([[1.0, 0.85, 0.1, 0.95]], np.float32),
                                   (_bvecs.shape[0], 1)),
                edge_width=0.55, name="boundary edges (cross-region)",
                opacity=0.95, blending="translucent", visible=True,
            )
        log.info(f"atlas regions: {len(uniq)} | "
                 f"boundary (cross-region) edges: {int(_xreg.sum()):,}")
    else:
        log.warning("no atlas labels available — skipping atlas-region + boundary "
                    "layers (pass --annotations-file with an atlas --knn-method).")

    # -----------------------------------------------------------------------
    # Layer C — Eigenvector reconstruction (kernel diagonal)  [needs eigvecs]
    # -----------------------------------------------------------------------
    # Slider over num_modes recomputes Σ_k spec(λ_k) φ_k(i)² and recolors the
    # nodes accordingly. Skipped entirely under --skip-eigvecs.
    has_eig = ctx["eigvec"] is not None
    if has_eig:
        initial_modes = min(args["initial_modes"] or args["num_modes"],
                            ctx["eigvec"].shape[1])
        recon_diag = kernel_diagonal_from_eigvecs(
            ctx["eigval"], ctx["eigvec"],
            nu=args["nu"], lengthscale=args["lengthscale"],
            num_modes=initial_modes,
        )
        log.info(
            f"  recon (K={initial_modes}) range: "
            f"[{recon_diag.min():.4g}, {recon_diag.max():.4g}]"
        )
        recon_layer = viewer.add_points(
            all_node_positions,
            name="C: kernel diagonal Σ_k spec(λ_k) φ_k(i)² (magma)",
            size=float(args["fabric_node_size"]) * 1.5,
            face_color="white", border_color="white",     # was edge_color
            symbol="o", opacity=0.85, blending="translucent",
            visible=False,
        )
        update_recon_layer_data(recon_layer, recon_diag, float(args["gamma"]))
    else:
        initial_modes = int(args["initial_modes"] or args["num_modes"])
        recon_layer = None

    # -----------------------------------------------------------------------
    # Random walk (heat kernel) from the active source — Laplacian vs spectrum
    # -----------------------------------------------------------------------
    # Two colourings of the SAME walk e^{-tL}e_src on the brain nodes:
    #   • "spectrum (K modes)"  = Σ_{k<K} e^{-tλ_k} φ_k(src) φ_k(·)  — the truncated eigenbasis
    #     (move the "num modes" slider: lowering K makes it blur / bleed across boundaries);
    #   • "Laplacian (full L)"  = e^{-tL}e_src straight from the operator (all modes, no
    #     eigendecomposition) — the reference the spectrum walk converges to as K → all.
    # "diffusion time t" sets the walk radius; compare against the atlas-regions / boundary
    # layers to see whether the walk respects the anatomy.
    walk_layer = None
    if has_eig:
        walk_layer = viewer.add_points(
            all_node_positions,
            name="random walk — spectrum (K modes)",
            size=float(args["fabric_node_size"]) * 1.6,
            face_color="white", border_color="white",
            symbol="o", opacity=0.9, blending="translucent",
            visible=True,
        )
    walk_full_layer = viewer.add_points(
        all_node_positions,
        name="random walk — Laplacian (full L, all modes)",
        size=float(args["fabric_node_size"]) * 1.6,
        face_color="white", border_color="white",
        symbol="o", opacity=0.9, blending="translucent",
        visible=True,
    )
    walk_lam_max = estimate_lam_max(ctx["laplacian_op"], N, ctx["device"])
    log.info(f"lambda_max(L) ~ {walk_lam_max:.1f}  (sets the full-walk integrator step count)")

    # -----------------------------------------------------------------------
    # Source markers and per-source comparison layers
    # -----------------------------------------------------------------------
    src_voxels = ctx["node_voxel_idx"][src_idxs].astype(np.float32)
    src_points = src_voxels * ctx["sv_scale"] + ctx["sv_translate"]
    viewer.add_points(
        src_points, name="source nodes",
        size=float(args["source_marker_size"]),
        face_color="red", border_color="white", symbol="o", opacity=0.95,
    )

    state = dict(
        src_pick=0,
        num_modes=initial_modes,
        lengthscale=float(args["lengthscale"]),
        diffusion_time=float(args["diffusion_time"]),
        gamma=float(args["gamma"]),
    )

    layer_state = {"knn": None, "matern": None, "riemann": None,
                   "diff_spec": None, "diff_direct": None}

    def current_src() -> int:
        return int(src_idxs[state["src_pick"] % len(src_idxs)])

    def refresh_per_source():
        s = current_src()
        data = build_lines_for_source(
            s, ctx, num_modes=state["num_modes"],
            lengthscale=state["lengthscale"], gamma=state["gamma"],
            k_show=args["k_show"], n_targets=args["n_targets"],
            target_strategy=args["target_strategy"],
            source_seed=args["source_seed"], nu=args["nu"],
            diffusion_time=state["diffusion_time"],
        )
        for key in ("knn", "matern", "riemann", "diff_spec", "diff_direct"):
            d = data[key]
            replace_shapes_layer(
                viewer, layer_state, key, d["lines"], d["colors"], d["widths"],
            )
        pts = viewer.layers["source nodes"]
        face = ["red"] * len(src_idxs)
        face[state["src_pick"] % len(src_idxs)] = "yellow"
        pts.face_color = face
        print(
            f"[src={s}]  "
            f"knn n={len(data['knn']['strengths'])} "
            f"[{data['knn']['strengths'].min():.3g}, "
            f"{data['knn']['strengths'].max():.3g}]  "
            f"matern [{data['matern']['strengths'].min():.3g}, "
            f"{data['matern']['strengths'].max():.3g}]  "
            f"riemann [{data['riemann']['strengths'].min():.3g}, "
            f"{data['riemann']['strengths'].max():.3g}]"
        )
        ds = data["diff_spec"]["strengths"]
        dd = data["diff_direct"]["strengths"]
        denom = max(float(np.abs(dd).max()), 1e-12)
        rel = float(np.abs(ds - dd).max()) / denom
        print(
            f"  diffusion dist (t={state['diffusion_time']:g}, "
            f"K={state['num_modes']}): "
            f"spectral[{ds.min():.3g}, {ds.max():.3g}]  "
            f"direct[{dd.min():.3g}, {dd.max():.3g}]  "
            f"max|Δ|/max = {rel:.2%}"
        )

    def refresh_reconstruction():
        if recon_layer is None:                       # --skip-eigvecs
            return
        d = kernel_diagonal_from_eigvecs(
            ctx["eigval"], ctx["eigvec"],
            nu=args["nu"], lengthscale=state["lengthscale"],
            num_modes=state["num_modes"],
        )
        update_recon_layer_data(recon_layer, d, state["gamma"])
        print(f"[recon K={state['num_modes']}]  diag range "
              f"[{d.min():.4g}, {d.max():.4g}]")

    def _inreg_msg(s, p):
        nl = ctx.get("node_labels")
        if nl is not None and int(nl[s]) > 0:
            tot = float(p.sum())
            if tot > 0:
                return f"  in-region mass {float(p[nl == int(nl[s])].sum() / tot):.0%}"
        return ""

    def refresh_walk():
        if walk_layer is None:                        # --skip-eigvecs
            return
        s = current_src()
        p = heat_walk_from_source(
            ctx["eigval"], ctx["eigvec"], s,
            t=state["diffusion_time"], num_modes=state["num_modes"],
        )
        update_recon_layer_data(walk_layer, p, state["gamma"])
        print(f"[walk spectrum src={s} K={state['num_modes']} "
              f"t={state['diffusion_time']:g}]{_inreg_msg(s, p)}")

    def refresh_walk_full():
        s = current_src()
        p = heat_walk_from_laplacian(
            ctx["laplacian_op"], s, t=state["diffusion_time"], n=N,
            device=ctx["device"], lam_max=walk_lam_max,
        )
        update_recon_layer_data(walk_full_layer, p, state["gamma"])
        print(f"[walk Laplacian src={s} t={state['diffusion_time']:g}]{_inreg_msg(s, p)}")

    # Initial per-source render
    refresh_per_source()
    refresh_walk()
    refresh_walk_full()

    # ---- Dock widget ----
    num_modes_max = (ctx["eigvec"].shape[1] if has_eig
                     else max(1, int(args["num_modes"])))

    @magicgui(
        auto_call=True,
        src_pick={"label": "active source",
                  "min": 0, "max": len(src_idxs) - 1, "step": 1},
        num_modes={"label": "num modes (Riemann/recon)",
                    "min": 1, "max": num_modes_max, "step": 1},
        lengthscale={"label": "lengthscale",
                      "min": 1e-3, "max": 10.0, "step": 1e-3},
        diffusion_time={"label": "diffusion time t",
                        "min": 1e-4, "max": 100.0, "step": 1e-3},
        gamma={"label": "color gamma",
                "min": 0.1, "max": 2.0, "step": 0.05},
    )
    def controls(
        src_pick: int = state["src_pick"],
        num_modes: int = state["num_modes"],
        lengthscale: float = state["lengthscale"],
        diffusion_time: float = state["diffusion_time"],
        gamma: float = state["gamma"],
    ):
        src_changed = src_pick != state["src_pick"]
        modes_or_ls_changed = (
            num_modes != state["num_modes"]
            or lengthscale != state["lengthscale"]
            or gamma != state["gamma"]
        )
        t_changed = diffusion_time != state["diffusion_time"]
        state.update(src_pick=src_pick, num_modes=num_modes,
                      lengthscale=lengthscale, diffusion_time=diffusion_time,
                      gamma=gamma)
        if src_changed or modes_or_ls_changed or t_changed:
            refresh_per_source()
        if modes_or_ls_changed:
            refresh_reconstruction()
        if src_changed or modes_or_ls_changed or t_changed:
            refresh_walk()
        if src_changed or t_changed:          # full walk is all-modes → no K dependence
            refresh_walk_full()

    viewer.window.add_dock_widget(controls, name="kernel controls", area="right")

    # ---- Banner ----
    print("\n" + "=" * 70)
    print("Kernel & graph debugger ready.")
    print(f"  graph nodes        : {N:,}")
    print(f"  eigenmodes loaded  : {num_modes_max}")
    print(f"  graphbandwidth     : {ctx['bw']:g}")
    print(f"  sources            : {src_idxs.tolist()}")
    print(f"  fabric edges       : {fabric_pairs.shape[0]:,}")
    print(f"  laplacian edges    : {lap_pairs.shape[0]:,}")
    print("Layers (toggle in left panel):")
    print("   A1: graph nodes              — every node, faint")
    print("   A2: KNN fabric (edges)       — the connectivity")
    print("   B1: Laplacian diag           — local connectivity strength")
    print("   B2: Laplacian off-diag       — edge weights (RdBu, ±)")
    print("   C : kernel diag reconstructed — Σ_k spec(λ_k) φ_k(i)²")
    print("   src KNN / Matern / Riemann   — per-source comparison")
    print("   src DiffDist spectral        — Σ e^{-2λt}(φ_s-φ_j)² (truncated)")
    print("   src DiffDist direct          — ||e^{-tL}(e_s-e_j)|| (full L)")
    print("     ↳ agree at full rank for symmetric norm; gap = truncation")
    print("Sliders (right):")
    print("   active source           — switch which source for per-source layers")
    print("   num modes (Riemann/recon)— eigenmodes used in C, Riemann, DiffDist spectral")
    print("   lengthscale             — Matern lengthscale (kernels + C)")
    print("   diffusion time t        — heat time for both DiffDist layers")
    print("   color gamma             — visual stretch on strengths")
    print("=" * 70 + "\n")

    napari.run()


if __name__ == "__main__":
    main()