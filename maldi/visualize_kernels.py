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

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.cm as cm

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
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
    p.add_argument("--region-bbox", type=int, nargs=6, default=None,
                   metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"))

    p.add_argument("--knn-method", choices=["faiss", "anatomical_atlas"],
                   default="anatomical_atlas")
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--laplacian-norm", choices=["symmetric", "randomwalk"],
                   default="symmetric")
    p.add_argument("--graphbandwidth", type=float, required=True)

    p.add_argument("--eigenvector-dir", required=True)
    p.add_argument("--num-modes", type=int, default=200)
    p.add_argument("--initial-modes", type=int, default=None)
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute-eigvecs", action="store_true")

    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--lengthscale", type=float, default=1.0)

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
        stride=args["stride"], region_bbox=args["region_bbox"],
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

    eigenvector_dir = Path(args["eigenvector_dir"])
    graphs = KnnGraphCache(cache_dir=eigenvector_dir / "knn", verbose=True)
    graph_key_parts = {
        "template": args["template_name"],
        "stride": (args["stride"] if args["region_bbox"] is None else 1),
        "thresh": args["threshold"],
        "method": args["knn_method"],
        "k": args["knn_k"],
        "nlist": args["n_list"],
        "bbox": (tuple(args["region_bbox"])
                 if args["region_bbox"] is not None else None),
    }
    if args["knn_method"] == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
    graph_key = make_graph_key(graph_key_parts)

    if args["knn_method"] == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss",
            coords=reference_nodes, k=args["knn_k"], nlist=args["n_list"],
            extra=graph_key_parts,
            force_recompute=args["force_recompute_graph"], device=device,
        )
    else:
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas",
            volume=sub_volume, threshold=args["threshold"],
            atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=args["knn_k"], nlist=args["n_list"],
            extra=graph_key_parts,
            force_recompute=args["force_recompute_graph"], device=device,
        )

    bw = float(args["graphbandwidth"])
    bw_tensor = torch.tensor(bw, device=device)
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0], bw_tensor, args["laplacian_norm"],
    )

    eigvec_key_parts = {
        "graph": graph_key,
        "norm": args["laplacian_norm"],
        "bw": bw,
        "modes": args["num_modes"],
    }
    eigvec_key = make_eig_key(eigvec_key_parts)
    ncv_min = max(1500, 3 * args["num_modes"] + 20)
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

    return dict(
        device=device,
        template_full=template_full,
        sub_volume=sub_volume,
        node_voxel_idx=node_voxel_idx,
        reference_nodes=reference_nodes,
        sv_scale=sv_scale, sv_translate=sv_translate,
        knn=knn, edge_index=edge_index, edge_value=edge_value,
        laplacian_op=laplacian_op,
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
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(strengths, dtype=np.float32)
    has_negative = bool((s < 0).any())
    abs_s = np.abs(s)
    abs_max = abs_s.max() if abs_s.size else 1.0
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
    source_seed: int, nu: float,
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

    riemann_strengths = riemann_manifold_kernel(
        int(src_idx), tgt_idxs, ctx["eigval"], ctx["eigvec"],
        nu=nu, lengthscale=lengthscale, num_modes=num_modes,
    )
    riemann_lines = matern_lines.copy()
    riemann_colors, riemann_widths = colors_widths(riemann_strengths, "magma", gamma)

    return {
        "knn":     dict(lines=knn_lines, colors=knn_colors,
                        widths=knn_widths, strengths=knn_strengths),
        "matern":  dict(lines=matern_lines, colors=matern_colors,
                        widths=matern_widths, strengths=matern_strengths),
        "riemann": dict(lines=riemann_lines, colors=riemann_colors,
                        widths=riemann_widths, strengths=riemann_strengths),
    }


# =============================================================================
# Napari layer management for per-source layers
# =============================================================================
LAYER_CONFIG = {
    "knn":     dict(name="src KNN edges  exp(-d²/(4 bw²))",   visible=False),
    "matern":  dict(name="src Matern (Euclidean)",            visible=False),
    "riemann": dict(name="src Riemann (Manifold)",            visible=False),
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
    fabric_color = np.tile(
        np.array([[0.6, 0.6, 0.6, 0.35]], dtype=np.float32),
        (fabric_lines.shape[0], 1),
    )
    viewer.add_shapes(
        [fabric_lines[i] for i in range(fabric_lines.shape[0])],
        shape_type="line",
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


    # Edges colored by L[i, j] — for symmetric normalization these are
    # negative (heat kernel adjacency); use diverging colormap centered at 0.
    lap_edge_lines = make_lines_array(lap_pairs, all_node_positions)
    abs_max = max(np.abs(lap_edge_vals).max(), 1e-12)
    edge_norm = lap_edge_vals / abs_max
    edge_colors_lap = cm.get_cmap("RdBu_r")(0.5 + 0.5 * edge_norm)
    viewer.add_shapes(
        [lap_edge_lines[i] for i in range(lap_edge_lines.shape[0])],
        shape_type="line",
        edge_color=edge_colors_lap,
        edge_width=0.4,
        name="B2: Laplacian off-diag (RdBu_r)",
        opacity=0.75, blending="translucent",
        visible=False,
    )

    # -----------------------------------------------------------------------
    # Layer C — Eigenvector reconstruction (kernel diagonal)
    # -----------------------------------------------------------------------
    # Slider over num_modes recomputes Σ_k spec(λ_k) φ_k(i)² and recolors
    # the nodes accordingly. With few modes, only the smoothest spatial
    # variation appears; with many modes, the value becomes more uniform.
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
        gamma=float(args["gamma"]),
    )

    layer_state = {"knn": None, "matern": None, "riemann": None}

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
        )
        for key in ("knn", "matern", "riemann"):
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

    def refresh_reconstruction():
        d = kernel_diagonal_from_eigvecs(
            ctx["eigval"], ctx["eigvec"],
            nu=args["nu"], lengthscale=state["lengthscale"],
            num_modes=state["num_modes"],
        )
        update_recon_layer_data(recon_layer, d, state["gamma"])
        print(f"[recon K={state['num_modes']}]  diag range "
              f"[{d.min():.4g}, {d.max():.4g}]")

    # Initial per-source render
    refresh_per_source()

    # ---- Dock widget ----
    num_modes_max = ctx["eigvec"].shape[1]

    @magicgui(
        auto_call=True,
        src_pick={"label": "active source",
                  "min": 0, "max": len(src_idxs) - 1, "step": 1},
        num_modes={"label": "num modes (Riemann/recon)",
                    "min": 1, "max": num_modes_max, "step": 1},
        lengthscale={"label": "lengthscale",
                      "min": 1e-3, "max": 10.0, "step": 1e-3},
        gamma={"label": "color gamma",
                "min": 0.1, "max": 2.0, "step": 0.05},
    )
    def controls(
        src_pick: int = state["src_pick"],
        num_modes: int = state["num_modes"],
        lengthscale: float = state["lengthscale"],
        gamma: float = state["gamma"],
    ):
        src_changed = src_pick != state["src_pick"]
        modes_or_ls_changed = (
            num_modes != state["num_modes"]
            or lengthscale != state["lengthscale"]
            or gamma != state["gamma"]
        )
        state.update(src_pick=src_pick, num_modes=num_modes,
                      lengthscale=lengthscale, gamma=gamma)
        if src_changed or modes_or_ls_changed:
            refresh_per_source()
        if modes_or_ls_changed:
            refresh_reconstruction()

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
    print("Sliders (right):")
    print("   active source           — switch which source for per-source layers")
    print("   num modes (Riemann/recon)— eigenmodes used in C and Riemann")
    print("   lengthscale             — Matern lengthscale (kernels + C)")
    print("   color gamma             — visual stretch on strengths")
    print("=" * 70 + "\n")

    napari.run()


if __name__ == "__main__":
    main()