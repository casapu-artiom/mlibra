"""Standalone kernel & graph visualization for the manifold GP.

Layers — all share the same coordinate frame (full-resolution template
voxel coords). Toggle visibility in napari's left panel.

  Layer A  — KNN graph fabric
  Layer B  — Graph Laplacian (operator-level)
  Layer C  — Kernel diagonal K(i, i) via `kernel.features()`
  Layer D  — Laplacian response: L · δ_src
  Layer D_dense — Laplacian response L · δ_src at full stride (Nystrom)
  Layer E  — Euclidean Matern K_ν,ℓ(src, ·)
  Layer F  — Manifold Matern K(src, ·) via `kernel.features()`
  Layer G  — Single eigenvector inspector: φ_k(i)
  Layer G_dense — Single eigenvector φ_k at dense full stride
  Layer H  — Weighted node degree D_i
  Layer J  — Manifold Matern at dense stride via `kernel.features()`
  Layer K  — L · density (graph Laplacian applied to reference image)
  Layer K_dense — L · density at full stride
  Layer L  — Euclidean Matern at dense stride
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.cm as cm

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges, labels_for_nodes_from_sub_atlas
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
)
from utils import crop_or_stride_volume, reference_ccf_from_subvolume

try:
    from manifold_gp.utils import bump_function as _lib_bump_function
    _USING_LIB_BUMP = True
except ImportError:
    _USING_LIB_BUMP = False


def bump_function(d, scale, decay):
    if _USING_LIB_BUMP:
        d_t = d if torch.is_tensor(d) else torch.as_tensor(d)
        scale_t = scale if torch.is_tensor(scale) else torch.as_tensor(float(scale), dtype=d_t.dtype, device=d_t.device)
        decay_t = decay if torch.is_tensor(decay) else torch.as_tensor(float(decay), dtype=d_t.dtype, device=d_t.device)
        return _lib_bump_function(d_t, scale_t, decay_t)
    d = torch.as_tensor(d) if not torch.is_tensor(d) else d
    scale = float(scale); decay = float(decay)
    out = torch.zeros_like(d)
    inside = d < scale
    if inside.any():
        u = (d[inside] / scale).clamp(0.0, 1.0 - 1e-6)
        out[inside] = torch.exp(-decay / (1.0 - u * u))
        out[inside] = out[inside] / float(np.exp(-decay))
    return out


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

    p.add_argument("--knn-method", choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted"],
                   default="anatomical_atlas")
    p.add_argument("--cross-region-inflation", type=float, default=100.0)
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

    p.add_argument("--nu", type=int, default=2)
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--diffusion-t", type=float, default=1.0)

    p.add_argument("--n-sources", type=int, default=4)
    p.add_argument("--source-seed", type=int, default=0)
    p.add_argument("--n-targets", type=int, default=50)
    p.add_argument("--target-strategy", choices=["random", "stratified"], default="stratified")
    p.add_argument("--k-show", type=int, default=30)
    p.add_argument("--source-marker-size", type=float, default=6.0)
    p.add_argument("--knn-color-by", choices=["heat", "distance"], default="heat")

    p.add_argument("--fabric-edge-sample", type=int, default=200_000)
    p.add_argument("--fabric-node-size", type=float, default=0.6)
    p.add_argument("--fabric-edge-width", type=float, default=0.3)
    p.add_argument("--laplacian-edge-sample", type=int, default=80_000)

    p.add_argument("--render-stride", type=int, default=1)
    p.add_argument("--bump-scale", type=float, default=3.0)
    p.add_argument("--bump-decay", type=float, default=0.05)
    p.add_argument("--density-smooth-sigma", type=float, default=0.0)
    p.add_argument("--nystrom-batch-size", type=int, default=20_000)
    p.add_argument("--dense-max-render-points", type=int, default=2000000)
    p.add_argument("--dense-render-threshold-frac", type=float, default=5e-3)

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
    annotations_full = np.load(args["annotations_file"]) if args["annotations_file"] else None

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

    sv_scale = np.array([args["stride"], args["stride"], args["stride"]], dtype=np.float32)
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
        "bbox": (tuple(args["region_bbox"]) if args["region_bbox"] is not None else None),
    }
    if args["knn_method"] == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
    graph_key = make_graph_key(graph_key_parts)

    if args["knn_method"] == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss", coords=reference_nodes,
            k=args["knn_k"], nlist=args["n_list"], extra=graph_key_parts,
            device=args["device"], force_recompute=args["force_recompute_graph"],
        )
    elif args["knn_method"] == "anatomical_atlas":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas", volume=sub_volume,
            threshold=args["threshold"], atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=args["knn_k"], nlist=args["n_list"],
            extra=graph_key_parts, device=args["device"], force_recompute=args["force_recompute_graph"],
        )
    elif args["knn_method"] == "faiss_atlas_weighted":
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss", coords=reference_nodes,
            k=args["knn_k"], nlist=args["n_list"], extra=base_key_parts,
            device=args["device"], force_recompute=args["force_recompute_graph"],
        )
        node_labels = labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, args["threshold"])
        inflation = float(args.get("cross_region_inflation", 10.0))
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels, inflation=inflation, treat_zero_as_cross=True,
        )
        graph_key_parts["weighting"] = f"atlas_x{inflation:g}"
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"unknown knn_method: {args['knn_method']}")

    bw = float(args["graphbandwidth"])
    bw_tensor = torch.tensor(bw, device=device)
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0], bw_tensor, args["laplacian_norm"],
    )

    eigvec_key_parts = {
        "graph": graph_key, "norm": args["laplacian_norm"], "bw": bw, "modes": args["num_modes"],
    }
    eigvec_key = make_eig_key(eigvec_key_parts)
    ncv_min = max(1500, 3 * args["num_modes"] + 20)
    solver = LaplacianEigensolver(
        num_modes=args["num_modes"], backend="cupy", tol=1e-4, ncv_min=ncv_min, verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op, cache_dir=eigenvector_dir / "eigvecs", key=eigvec_key,
        graphbandwidth=bw, laplacian_normalization=args["laplacian_norm"],
        extra=eigvec_key_parts, force_recompute=args["force_recompute_eigvecs"], device=device,
    )

    matern_kernel = RiemannMaternKernel(
        nu=args["nu"],
        lengthscale=args["lengthscale"],
        knn=knn,
        edge_index=edge_index,
        edge_value=edge_value,
        eigval=eigval,
        eigvec=eigvec,
        nearest_neighbors=args["knn_k"],
        laplacian_normalization=args["laplacian_norm"],
        num_modes=args["num_modes"],
        bump_scale=args["bump_scale"],
        bump_decay=args["bump_decay"],
        graphbandwidth_init=bw
    ).to(device)

    return dict(
        device=device, template_full=template_full, sub_volume=sub_volume,
        node_voxel_idx=node_voxel_idx, reference_nodes=reference_nodes,
        sv_scale=sv_scale, sv_translate=sv_translate,
        knn=knn, edge_index=edge_index, edge_value=edge_value,
        laplacian_op=laplacian_op, eigval=eigval, eigvec=eigvec, bw=bw,
        coord_mean=coord_mean, coord_std=coord_std,
        voxel_offset=voxel_offset, voxel_scale_mm=voxel_scale_mm,
        stride=int(args["stride"]), threshold=int(args["threshold"]),
        matern_kernel=matern_kernel
    )


# =============================================================================
# Computations and Helper Functions
# =============================================================================
def subsample_edges(edge_index: torch.Tensor, edge_value: torch.Tensor, max_edges: int, seed: int = 0):
    src, dst, val = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy(), edge_value.cpu().numpy()
    keep = src < dst
    keep_idx = np.where(keep)[0]
    src, dst, val = src[keep], dst[keep], val[keep]
    if src.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        sel = rng.choice(src.shape[0], size=max_edges, replace=False)
        src, dst, val = src[sel], dst[sel], val[sel]
        keep_idx = keep_idx[sel]
    return np.stack([src, dst], axis=1), val, keep_idx


def make_lines_array(pairs: np.ndarray, node_positions: np.ndarray) -> np.ndarray:
    lines = np.zeros((pairs.shape[0], 2, 3), dtype=np.float32)
    lines[:, 0, :] = node_positions[pairs[:, 0]]
    lines[:, 1, :] = node_positions[pairs[:, 1]]
    return lines


def _full_voxel_to_normalized(voxel_idx_full: np.ndarray, ctx: dict) -> torch.Tensor:
    mm = voxel_idx_full.astype(np.float32) * 0.025
    return (torch.from_numpy(mm) - ctx["coord_mean"]) / ctx["coord_std"]


# ---- General Out-Of-Sample Projection Helper for Non-Kernel Fields (L*delta, etc) ----
def interpolate_function_to_dense_grid(
    f_node: torch.Tensor, ctx: dict, laplacian_op: GraphLaplacianOperator,
    render_stride: int, nearest_neighbors: int = 10,
    bump_scale: float = 3.0, bump_decay: float = 0.05, batch_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree
    tmpl = ctx["template_full"]
    mask = tmpl > ctx["threshold"]
    sub_mask = mask[::render_stride, ::render_stride, ::render_stride]
    sub_idx = np.argwhere(sub_mask).astype(np.int32)
    voxel_idx = sub_idx * render_stride
    Q = voxel_idx.shape[0]

    coords_z_cpu = _full_voxel_to_normalized(voxel_idx, ctx).numpy()
    if "_query_kdt" not in ctx:
        ctx["_query_kdt"] = cKDTree(ctx["reference_nodes"].cpu().numpy())
    kdt = ctx["_query_kdt"]

    f_node = f_node.detach().to(device=laplacian_op.x.device, dtype=laplacian_op.x.dtype).view(-1, 1).contiguous()
    bump_radius = float(laplacian_op.graphbandwidth.squeeze()) * float(bump_scale)

    f_q_all = np.zeros(Q, dtype=np.float32)
    device, dtype = laplacian_op.x.device, laplacian_op.x.dtype

    for batch_start in range(0, Q, batch_size):
        batch_end = min(batch_start + batch_size, Q)
        coords_b = coords_z_cpu[batch_start:batch_end]
        dists_b, idxs_b = kdt.query(coords_b, k=nearest_neighbors, workers=-1)
        edge_value_b = torch.from_numpy((dists_b.astype(np.float32) ** 2)).to(device=device, dtype=dtype)
        edge_index_b = torch.from_numpy(idxs_b.astype(np.int64)).to(device)

        sqrt_d_nearest_b = edge_value_b[:, 0].sqrt()
        within_b = sqrt_d_nearest_b < bump_radius

        if within_b.any():
            projected_b = laplacian_op.out_of_sample(
                f_node, edge_value_b[within_b], edge_index_b[within_b],
            )
            bump_vals_b = bump_function(sqrt_d_nearest_b[within_b], bump_radius, float(bump_decay))
            vals_b = (projected_b.squeeze(-1) * bump_vals_b).cpu().numpy()
            
            within_mask_cpu = within_b.cpu().numpy()
            global_positions = np.arange(batch_start, batch_end)[within_mask_cpu]
            f_q_all[global_positions] = vals_b
            del projected_b, bump_vals_b
        del edge_value_b, edge_index_b, sqrt_d_nearest_b, within_b

    return f_q_all, voxel_idx


def laplacian_diag(laplacian_op: GraphLaplacianOperator) -> np.ndarray:
    return laplacian_op.laplacian_diag.detach().cpu().numpy()


def laplacian_offdiag_at_edges(laplacian_op: GraphLaplacianOperator, edge_index_subset: np.ndarray, full_edge_index: torch.Tensor) -> np.ndarray:
    return -laplacian_op.laplacian_triu.detach().cpu().numpy()[edge_index_subset]


def weighted_degree(laplacian_op: GraphLaplacianOperator) -> np.ndarray:
    return laplacian_op.degree_unnorm_mat.detach().cpu().numpy()


def apply_laplacian_to_delta(laplacian_op: GraphLaplacianOperator, src_idx: int) -> np.ndarray:
    N = laplacian_op.operator_dimension
    f = torch.zeros(N, 1, device=laplacian_op.x.device, dtype=laplacian_op.x.dtype)
    f[src_idx] = 1.0
    return laplacian_op._matmul(f).squeeze(-1).cpu().numpy()


def apply_laplacian_to_density(laplacian_op: GraphLaplacianOperator, sub_volume: np.ndarray, node_voxel_idx: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    vol = sub_volume.astype(np.float32, copy=False)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=float(sigma))
    density_per_node = vol[node_voxel_idx[:, 0], node_voxel_idx[:, 1], node_voxel_idx[:, 2]]
    f = torch.as_tensor(density_per_node, device=laplacian_op.x.device, dtype=laplacian_op.x.dtype).unsqueeze(-1)
    return laplacian_op._matmul(f).squeeze(-1).cpu().numpy()


# ---- DRYes Dense Kernel Interpolation using underlying `kernel.features()` ----
def kernel_at_dense_grid(
    src_idx: int, ctx: dict, render_stride: int, batch_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    matern_kernel = ctx["matern_kernel"]
    tmpl = ctx["template_full"]
    mask = tmpl > ctx["threshold"]
    sub_mask = mask[::render_stride, ::render_stride, ::render_stride]
    sub_idx = np.argwhere(sub_mask).astype(np.int32)
    voxel_idx = sub_idx * render_stride
    Q = voxel_idx.shape[0]

    coords_z_cpu = _full_voxel_to_normalized(voxel_idx, ctx)
    device = ctx["device"]
    
    # Evaluate exact features for the source node using library primitive
    src_coord = ctx["reference_nodes"][src_idx].unsqueeze(0)
    feat_src = matern_kernel.features(src_coord).squeeze(0)
    
    K_q_all = np.zeros(Q, dtype=np.float32)

    for batch_start in range(0, Q, batch_size):
        batch_end = min(batch_start + batch_size, Q)
        coords_b = coords_z_cpu[batch_start:batch_end].to(device).contiguous()
        
        # OOS Nystrom + Bump implicitly handled by library primitive!
        feat_b = matern_kernel.features(coords_b) 
        
        kvals_b = (feat_b * feat_src).sum(dim=-1).detach().cpu().numpy()
        K_q_all[batch_start:batch_end] = kvals_b

    return K_q_all, voxel_idx


def matern_euclidean_at_source(src_coord: torch.Tensor, all_coords: torch.Tensor, nu: int, lengthscale: float) -> np.ndarray:
    from scipy.special import kv, gamma as gamma_fn
    d = ((all_coords - src_coord) ** 2).sum(dim=-1).sqrt().cpu().numpy()
    nu_f, float_ls = float(nu), float(lengthscale)
    out = np.ones_like(d, dtype=np.float64)
    nz = d > 0
    if nz.any():
        z = np.sqrt(2.0 * nu_f) * d[nz] / float_ls
        out[nz] = ((2.0 ** (1.0 - nu_f)) / gamma_fn(nu_f)) * (z ** nu_f) * kv(nu_f, z)
    return out


def euclidean_kernel_at_dense_grid(src_idx: int, ctx: dict, nu: int, lengthscale: float, render_stride: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import kv, gamma as gamma_fn
    nu_f, float_ls = float(nu), float(lengthscale)
    tmpl = ctx["template_full"]
    mask = tmpl > ctx["threshold"]
    sub_mask = mask[::render_stride, ::render_stride, ::render_stride]
    sub_idx = np.argwhere(sub_mask).astype(np.int32)
    voxel_idx = sub_idx * render_stride

    coords_z = _full_voxel_to_normalized(voxel_idx, ctx).numpy()
    src_coord = ctx["reference_nodes"][src_idx].cpu().numpy()
    d = np.linalg.norm(coords_z - src_coord, axis=1).astype(np.float64)

    out = np.ones_like(d, dtype=np.float64)
    nz = d > 0
    if nz.any():
        z = np.sqrt(2.0 * nu_f) * d[nz] / float_ls
        out[nz] = ((2.0 ** (1.0 - nu_f)) / gamma_fn(nu_f)) * (z ** nu_f) * kv(nu_f, z)
    return out.astype(np.float32), voxel_idx


# =============================================================================
# Plotting and Per-Source Lines UI Builders
# =============================================================================
def make_lines_for_kernel(src_coord: torch.Tensor, tgt_coords: torch.Tensor, lengthscale: float, nu: float = 1.0) -> np.ndarray:
    d = torch.sqrt(((tgt_coords - src_coord) ** 2).sum(dim=-1))
    r = d / lengthscale
    if nu == 0.5: k = torch.exp(-r)
    elif nu == 1.5: k = (1.0 + (3**0.5)*r) * torch.exp(-(3**0.5)*r)
    elif nu == 2.5: k = (1.0 + (5**0.5)*r + (5.0/3.0)*r**2) * torch.exp(-(5**0.5)*r)
    else: k = torch.exp(-0.5 * r ** 2)
    return k.cpu().numpy()

def pick_target_nodes(src_idx: int, reference_nodes: torch.Tensor, n_targets: int, strategy: str, seed: int) -> np.ndarray:
    N = reference_nodes.shape[0]
    rng = np.random.default_rng(seed + src_idx)
    if strategy == "random": return rng.choice(N, size=min(n_targets, N), replace=False).astype(np.int64)
    sq_dist = ((reference_nodes - reference_nodes[src_idx]) ** 2).sum(dim=-1).cpu().numpy()
    order = np.argsort(sq_dist)
    order = order[order != src_idx]
    return order[np.linspace(0, len(order) - 1, n_targets).astype(np.int64)]

def knn_neighbors_of(src_idx: int, edge_index: torch.Tensor, edge_value: torch.Tensor, k_show: int) -> tuple[np.ndarray, np.ndarray]:
    src_eq, dst_eq = (edge_index[0] == src_idx), (edge_index[1] == src_idx)
    nbrs = torch.cat([edge_index[1, src_eq], edge_index[0, dst_eq]]).cpu().numpy()
    dists = torch.cat([edge_value[src_eq], edge_value[dst_eq]]).cpu().numpy()
    nbrs_uniq, first = np.unique(nbrs, return_index=True)
    order = np.argsort(dists[first])[:k_show]
    return nbrs_uniq[order], dists[first][order]

def make_lines(src_voxel: np.ndarray, nbr_voxels: np.ndarray, sv_scale: np.ndarray, sv_translate: np.ndarray) -> np.ndarray:
    src_full = src_voxel.astype(np.float32) * sv_scale + sv_translate
    nbr_full = nbr_voxels.astype(np.float32) * sv_scale + sv_translate
    lines = np.zeros((nbr_voxels.shape[0], 2, 3), dtype=np.float32)
    lines[:, 0, :] = src_full
    lines[:, 1, :] = nbr_full
    return lines

def colors_widths(strengths: np.ndarray, cmap_name: str = "viridis", gamma: float = 0.5, min_width: float = 0.3, max_width: float = 2.5):
    s = np.asarray(strengths, dtype=np.float32)
    has_negative = bool((s < 0).any())
    abs_max = np.abs(s).max() if s.size else 1.0
    if abs_max == 0: abs_max = 1.0
    if has_negative:
        colors = matplotlib.colormaps.get_cmap("RdBu_r")(0.5 + 0.5 * s / abs_max)
    else:
        colors = matplotlib.colormaps.get_cmap(cmap_name)((np.abs(s) / abs_max) ** gamma)
    widths = min_width + (max_width - min_width) * (np.abs(s) / abs_max) ** gamma
    return colors, widths

def build_lines_for_source(src_idx: int, ctx: dict, lengthscale: float, gamma: float, k_show: int, n_targets: int, target_strategy: str, source_seed: int, nu: float, knn_color_by: str) -> dict:
    src_voxel = ctx["node_voxel_idx"][src_idx]
    knn_idxs, knn_sq_dists = knn_neighbors_of(int(src_idx), ctx["edge_index"], ctx["edge_value"], k_show)
    knn_strengths = (1.0 - np.sqrt(knn_sq_dists)/max(np.sqrt(knn_sq_dists).max(), 1e-8)) if knn_color_by == "distance" else np.exp(-knn_sq_dists / (4.0 * ctx["bw"]**2))
    knn_lines = make_lines(src_voxel, ctx["node_voxel_idx"][knn_idxs], ctx["sv_scale"], ctx["sv_translate"])
    knn_colors, knn_widths = colors_widths(knn_strengths, "viridis", gamma)

    tgt_idxs = pick_target_nodes(int(src_idx), ctx["reference_nodes"], n_targets, target_strategy, source_seed)
    matern_strengths = make_lines_for_kernel(ctx["reference_nodes"][src_idx], ctx["reference_nodes"][tgt_idxs], lengthscale, nu)
    matern_lines = make_lines(src_voxel, ctx["node_voxel_idx"][tgt_idxs], ctx["sv_scale"], ctx["sv_translate"])
    matern_colors, matern_widths = colors_widths(matern_strengths, "plasma", gamma)

    # Use underlying library to compute strengths!
    matern_kernel = ctx["matern_kernel"]
    feat_src = matern_kernel.features(ctx["reference_nodes"][src_idx].unsqueeze(0)).squeeze(0)
    feat_tgt = matern_kernel.features(ctx["reference_nodes"][tgt_idxs])
    riemann_strengths = (feat_tgt * feat_src).sum(dim=-1).detach().cpu().numpy()
    
    riemann_colors, riemann_widths = colors_widths(riemann_strengths, "magma", gamma)

    return {
        "knn": dict(lines=knn_lines, colors=knn_colors, widths=knn_widths),
        "matern": dict(lines=matern_lines, colors=matern_colors, widths=matern_widths),
        "riemann": dict(lines=matern_lines, colors=riemann_colors, widths=riemann_widths),
    }


LAYER_CONFIG = {
    "knn": dict(name="src KNN edges", visible=False),
    "matern": dict(name="src Matern (Eucl.)", visible=False),
    "riemann": dict(name="src Riemann (Manif.)", visible=False),
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
    new_layer = viewer.add_shapes([lines[i] for i in range(lines.shape[0])], shape_type="line", edge_color=colors, edge_width=widths.tolist(), name=cfg["name"], opacity=0.9)
    new_layer.visible = keep_visible
    if keep_idx < len(viewer.layers): viewer.layers.move(viewer.layers.index(new_layer), keep_idx)
    layer_state[key] = new_layer


# =============================================================================
# Node and Grid Colorization Engines
# =============================================================================
# Node and Grid Colorization Engines (legacy Points helpers — removed.
# All diagnostic layers now render as Image volumes through
# _update_sparse_image_layer / _update_dense_image_layer, which give solid
# voxel rendering in 2D and 3D with a single colour/alpha curve.)


def make_layer_info_panel():
    try:
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QTextEdit
        from qtpy.QtGui import QFont
    except ImportError:
        return None, (lambda *a, **k: None), (lambda: None)
    text = QTextEdit()
    text.setReadOnly(True)
    mono = QFont("Monospace")
    mono.setStyleHint(QFont.TypeWriter)
    mono.setPointSize(9)
    text.setFont(mono)
    text.setMinimumHeight(180)
    widget = QWidget()
    layout = QVBoxLayout(widget); layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(text)
    info_dict = {}
    def render():
        lines = [f"<span>{tag:<8}</span> {info}" for tag, info in info_dict.items()]
        text.setHtml("<pre>" + "\n".join(lines) + "</pre>")
    return widget, (lambda t, i: [info_dict.update({t: i}), render()][1]), render


def fmt_info_sequential(vmin, vmax, sat, gamma, cmap):
    return f"range=[{vmin:>+10.3g}, {vmax:>+10.3g}]  sat={sat:>+10.3g}  γ={gamma:>4.2f}  cmap={cmap}"

def fmt_info_diverging(vmin, vmax, sat, gamma, cmap):
    return f"range=[{vmin:>+10.3g}, {vmax:>+10.3g}]  sat=±{sat:>9.3g}  γ={gamma:>4.2f}  cmap={cmap}"

def fmt_info_sparse_signed(vmin, vmax, pos_max, neg_min, cmap):
    return f"range=[{vmin:>+10.3g}, {vmax:>+10.3g}]  +sat={pos_max:>+9.3g}  −sat={neg_min:>+9.3g}  cmap={cmap}"


def make_bump_widget(graphbandwidth: float, initial_scale: float, initial_decay: float):
    try:
        from qtpy.QtWidgets import QWidget, QVBoxLayout
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvas
    except ImportError:
        return None, (lambda *_args, **_kw: None)
    bw = float(graphbandwidth)
    fig = Figure(figsize=(4.0, 3.0), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    d_grid = np.linspace(0.0, 5.0 * bw * max(initial_scale, 1.0), 400)
    line, = ax.plot([], [], "-", color="#3578a8", lw=1.5)
    v_bw = ax.axvline(bw, color="#888888", ls="--", lw=0.7)
    v_supp = ax.axvline(initial_scale * bw, color="#cc3344", ls="--", lw=0.7)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("distance to nearest training node")
    ax.set_ylabel("bump value")
    ax.grid(alpha=0.3)
    canvas = FigureCanvas(fig)
    def update(scale, decay):
        vals = bump_function(torch.from_numpy(d_grid).float(), float(scale) * bw, float(decay)).cpu().numpy()
        line.set_data(d_grid, vals)
        v_supp.set_xdata([float(scale) * bw, float(scale) * bw])
        ax.set_xlim(0, max(5.0 * bw, float(scale) * bw * 1.2))
        canvas.draw_idle()
    update(initial_scale, initial_decay)
    widget = QWidget(); layout = QVBoxLayout(widget); layout.setContentsMargins(2, 2, 2, 2); layout.addWidget(canvas)
    return widget, update


# =============================================================================
# Main Visualizer Loop
# =============================================================================
def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args["verbose"] else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("visualize_kernels")

    ctx = setup(args, log)
    if args["no_launch"]: return

    import napari
    from magicgui import magicgui

    all_node_positions = ctx["node_voxel_idx"].astype(np.float32) * ctx["sv_scale"] + ctx["sv_translate"]
    N = all_node_positions.shape[0]
    rng = np.random.default_rng(args["source_seed"])
    src_idxs = rng.choice(N, size=args["n_sources"], replace=False)

    fabric_pairs, fabric_sq_dists, _ = subsample_edges(ctx["edge_index"], ctx["edge_value"], max_edges=args["fabric_edge_sample"], seed=args["source_seed"])
    lap_pairs, lap_sq_dists, lap_full_idx = subsample_edges(ctx["edge_index"], ctx["edge_value"], max_edges=args["laplacian_edge_sample"], seed=args["source_seed"] + 1)
    lap_diag_vals = laplacian_diag(ctx["laplacian_op"])
    lap_edge_vals = laplacian_offdiag_at_edges(ctx["laplacian_op"], lap_full_idx, ctx["edge_index"])
    deg_vals = weighted_degree(ctx["laplacian_op"])

    viewer = napari.Viewer(title="Kernel & graph debugger")
    # NOTE: viewer.dims.ndisplay starts at 2. We deliberately stay in 2D mode
    # while constructing the Image layers — some napari/vispy versions raise
    # "Volume visual needs a 3D array" when constructing volume visuals at
    # add_image time with a tiny placeholder, even though the array IS 3D.
    # Creating in 2D mode sidesteps that path entirely; we switch to 3D right
    # before napari.run() once all layers have real sub_shape data.

    viewer.add_points(all_node_positions, name="A1: graph nodes", size=float(args["fabric_node_size"]), face_color="white", border_color="white", symbol="o", opacity=0.25, blending="additive")
    fabric_lines = make_lines_array(fabric_pairs, all_node_positions)
    viewer.add_shapes([fabric_lines[i] for i in range(fabric_lines.shape[0])], shape_type="line", edge_color=np.tile([[0.6, 0.6, 0.6, 0.35]], (fabric_lines.shape[0], 1)), edge_width=float(args["fabric_edge_width"]), name="A2: KNN fabric (edges)", opacity=0.7, blending="translucent")

    placeholder_vol = np.zeros((2, 2, 2), dtype=np.float32)
    _img_kw = dict(opacity=1.0, rendering="translucent")

    # All scalar-per-node diagnostic layers render as Image volumes built by
    # splatting per-node values into a sub_volume-shaped array via
    # node_voxel_idx. Image layers give solid voxel rendering in both 2D
    # slices and 3D volume mode, with proper depth and contrast, where the
    # earlier Points approach was sub-pixel and dim in 2D.
    #
    # Layers are created with a tiny (2,2,2) placeholder (same pattern as the
    # dense layers below) — vispy/napari objects to having a full-size
    # zero-volume forced through volume rendering at construction time. The
    # real sub_shape array is assigned by _update_sparse_image_layer, which
    # the boot sequence calls right before napari.run().
    sub_shape = ctx["sub_volume"].shape
    node_idx = ctx["node_voxel_idx"]   # (N, 3) sub-volume indices

    def _new_img(name, visible=False):
        lyr = viewer.add_image(placeholder_vol, name=name, visible=visible, **_img_kw)
        lyr.scale = ctx["sv_scale"]; lyr.translate = ctx["sv_translate"]
        return lyr

    b1_layer = _new_img("B1: Laplacian diag (often uniform)")
    lap_edge_lines = make_lines_array(lap_pairs, all_node_positions)
    viewer.add_shapes([lap_edge_lines[i] for i in range(lap_edge_lines.shape[0])], shape_type="line", edge_color=matplotlib.colormaps.get_cmap("RdBu_r")(0.5 + 0.5 * lap_edge_vals / max(np.abs(lap_edge_vals).max(), 1e-12)), edge_width=0.4, name="B2: Laplacian off-diag (edges)", opacity=0.75, blending="translucent", visible=False)

    h_layer = _new_img("H: weighted degree D_i")

    initial_modes = min(args["initial_modes"] or args["num_modes"], ctx["eigvec"].shape[1])
    c_layer = _new_img("C: kernel diag K(i, i)")
    d_layer = _new_img("D: L · δ_src (sharp, diverging)")
    e_layer = _new_img("E: Euclidean Matern K(src, ·)", visible=True)
    f_layer = _new_img("F: Manifold Matern K(src, ·)",  visible=True)
    g_layer = _new_img("G: eigenvector φ_k(i)")
    k_layer = _new_img("K: L · density")

    j_layer = viewer.add_image(placeholder_vol, name=f"J: K(src, ·) dense @ stride={args['render_stride']} (Nyström)", visible=False, **_img_kw)
    l_layer = viewer.add_image(placeholder_vol, name=f"L: Euclidean K(d) dense @ stride={args['render_stride']}", visible=False, **_img_kw)
    g_dense_layer = viewer.add_image(placeholder_vol, name=f"G_dense: eigenvector φ_k dense @ full-stride", visible=False, **_img_kw)
    d_dense_layer = viewer.add_image(placeholder_vol, name=f"D_dense: L · δ_src dense @ full-stride", visible=False, **_img_kw)
    k_dense_layer = viewer.add_image(placeholder_vol, name=f"K_dense: L · density dense @ full-stride", visible=False, **_img_kw)

    src_points = ctx["node_voxel_idx"][src_idxs].astype(np.float32) * ctx["sv_scale"] + ctx["sv_translate"]
    viewer.add_points(src_points, name="source nodes", size=float(args["source_marker_size"]), face_color="red", border_color="white", symbol="o", opacity=0.95)

    state = dict(
        src_pick=0, num_modes=initial_modes, nu=int(args["nu"]), lengthscale=float(args["lengthscale"]),
        eigvec_idx=0, gamma=float(args["gamma"]), alpha_power=3.0, render_stride=int(args["render_stride"]),
        bump_scale=float(args["bump_scale"]), bump_decay=float(args["bump_decay"]), density_smooth_sigma=float(args["density_smooth_sigma"]),
    )
    layer_state = {"knn": None, "matern": None, "riemann": None}
    current_src = lambda: int(src_idxs[state["src_pick"] % len(src_idxs)])

    info_panel, set_info, refresh_info = make_layer_info_panel()
    if info_panel is not None:
        viewer.window.add_dock_widget(info_panel, name="layer info", area="bottom")

    rng_render = np.random.default_rng(args["source_seed"])
    
    def _update_dense_image_layer(layer, K_q_flat, voxel_idx, label, tag):
        if K_q_flat.size == 0:
            set_info(tag, "(empty result)")
            return
            
        render_stride = int(state["render_stride"])
        sub_mask = (ctx["template_full"] > ctx["threshold"])[::render_stride, ::render_stride, ::render_stride]
        vol = np.zeros(sub_mask.shape, dtype=np.float32)

        # voxel_idx may arrive as float (callers cast for various reasons).
        # numpy fancy-indexing requires integer arrays — cast to int64 here
        # so refresh_J / refresh_L can pass whatever dtype is convenient.
        vox_i = np.asarray(voxel_idx, dtype=np.int64)
        sub_idx = vox_i // render_stride
        vol[sub_idx[:, 0], sub_idx[:, 1], sub_idx[:, 2]] = K_q_flat
        
        sat = float(np.percentile(np.abs(K_q_flat), 99))
        
        from napari.utils.colormaps import Colormap
        is_3d = viewer.dims.ndisplay == 3
        eff_alpha_power = float(state.get("alpha_power", 3.0)) if is_3d else 1.0
        
        if (K_q_flat < 0).any():
            t = np.linspace(0, 1, 1024)
            val = (t - 0.5) * 2.0
            mag = np.abs(val) ** state["gamma"]
            norm = 0.5 + 0.5 * np.sign(val) * mag
            colors = matplotlib.colormaps.get_cmap("RdBu_r")(norm)
            colors[:, 3] = mag ** eff_alpha_power
            
            layer.colormap = Colormap(colors, name='custom_diverging')
            layer.contrast_limits = [-sat, sat]
            info_line = fmt_info_diverging(float(K_q_flat.min()), float(K_q_flat.max()), sat, state["gamma"], "RdBu_r")
        else:
            vmin, vmax = float(K_q_flat.min()), float(K_q_flat.max())
            if vmax == vmin: vmax = vmin + 1e-6
            t = np.linspace(0, 1, 1024)
            mag = t ** state["gamma"]
            colors = matplotlib.colormaps.get_cmap("magma")(mag)
            colors[:, 3] = mag ** eff_alpha_power
            
            layer.colormap = Colormap(colors, name='custom_sequential')
            layer.contrast_limits = [vmin, vmax]
            info_line = fmt_info_sequential(float(K_q_flat.min()), float(K_q_flat.max()), sat, state["gamma"], "magma")
            
        layer.data = vol
        # Dense layers iterate over template_full[::R, ::R, ::R] and produce
        # voxel_idx = sub_idx * R (full-resolution voxel indices). To place
        # dense voxel (i, j, k) at full-res display position (i*R, j*R, k*R)
        # the scale is just (R, R, R) and translate is zero — regardless of
        # graph stride or bbox mode. Multiplying by sv_scale (= [stride]*3 in
        # the strided path) used to inflate the volume by a factor of `stride`,
        # which is why the dense brain rendered far larger than the sparse
        # training-node layers in the same scene.
        rs = float(render_stride)
        layer.scale = np.array([rs, rs, rs], dtype=np.float32)
        layer.translate = np.zeros(3, dtype=np.float32)
        
        set_info(tag, info_line + f"  ({K_q_flat.size:,} voxels)")

    def _update_sparse_image_layer(layer, values: np.ndarray, label: str, tag: str,
                                   cmap_name: str | None = None,
                                   mode: str = "auto"):
        """Splat per-node `values` into a sub_volume-shaped array at
        node_voxel_idx and update an Image layer.

        mode: "sequential" → magma, "diverging" → RdBu_r centred at 0,
              "auto" picks diverging if values have any negatives, else sequential.
        cmap_name: optional override for the colormap name.
        """
        from napari.utils.colormaps import Colormap
        if values.size == 0:
            set_info(tag, "(empty result)")
            return

        # Splat into a fresh sub-volume each time (cheap: shape = sub_volume).
        vol = np.zeros(sub_shape, dtype=np.float32)
        vol[node_idx[:, 0], node_idx[:, 1], node_idx[:, 2]] = values

        sat = float(np.percentile(np.abs(values), 99))
        is_3d = viewer.dims.ndisplay == 3
        eff_alpha_power = float(state.get("alpha_power", 3.0)) if is_3d else 1.0

        effective_mode = mode
        if effective_mode == "auto":
            effective_mode = "diverging" if (values < 0).any() else "sequential"

        if effective_mode == "diverging":
            cname = cmap_name or "RdBu_r"
            t = np.linspace(0, 1, 1024)
            val = (t - 0.5) * 2.0
            mag = np.abs(val) ** state["gamma"]
            norm = 0.5 + 0.5 * np.sign(val) * mag
            colors = matplotlib.colormaps.get_cmap(cname)(norm)
            colors[:, 3] = mag ** eff_alpha_power
            layer.colormap = Colormap(colors, name=f'sparse_div_{tag}')
            layer.contrast_limits = [-sat if sat > 0 else -1e-6, sat if sat > 0 else 1e-6]
            info_line = fmt_info_diverging(float(values.min()), float(values.max()), sat, state["gamma"], cname)
        else:
            cname = cmap_name or "magma"
            vmin, vmax = float(values.min()), float(values.max())
            if vmax == vmin: vmax = vmin + 1e-6
            t = np.linspace(0, 1, 1024)
            mag = t ** state["gamma"]
            colors = matplotlib.colormaps.get_cmap(cname)(mag)
            colors[:, 3] = mag ** eff_alpha_power
            layer.colormap = Colormap(colors, name=f'sparse_seq_{tag}')
            layer.contrast_limits = [vmin, vmax]
            info_line = fmt_info_sequential(float(values.min()), float(values.max()), sat, state["gamma"], cname)

        layer.data = vol
        # scale/translate were set at creation; no need to reapply each refresh
        set_info(tag, info_line + f"  ({values.size:,} nodes)")

    def refresh_G_dense():
        if not g_dense_layer.visible: return
        k = int(state["eigvec_idx"])
        phi_q, voxel_idx = interpolate_function_to_dense_grid(
            ctx["eigvec"][:, k], ctx, ctx["laplacian_op"], state["render_stride"],
            bump_scale=state["bump_scale"], bump_decay=state["bump_decay"], batch_size=int(args["nystrom_batch_size"])
        )
        _update_dense_image_layer(g_dense_layer, phi_q, voxel_idx, f"G_dense φ_{k}", "G_dense")

    def refresh_D_dense():
        if not d_dense_layer.visible: return
        s = current_src()
        Lf_node = torch.as_tensor(apply_laplacian_to_delta(ctx["laplacian_op"], s), device=ctx["device"])
        Lf_q, voxel_idx = interpolate_function_to_dense_grid(
            Lf_node, ctx, ctx["laplacian_op"], state["render_stride"],
            bump_scale=state["bump_scale"], bump_decay=state["bump_decay"], batch_size=int(args["nystrom_batch_size"])
        )
        _update_dense_image_layer(d_dense_layer, Lf_q, voxel_idx, f"D_dense L·δ_{s}", "D_dense")

    def refresh_K_dense():
        if not k_dense_layer.visible: return
        Lf_node = torch.as_tensor(apply_laplacian_to_density(ctx["laplacian_op"], ctx["sub_volume"], ctx["node_voxel_idx"], sigma=state["density_smooth_sigma"]), device=ctx["device"])
        Lf_q, voxel_idx = interpolate_function_to_dense_grid(
            Lf_node, ctx, ctx["laplacian_op"], state["render_stride"],
            bump_scale=state["bump_scale"], bump_decay=state["bump_decay"], batch_size=int(args["nystrom_batch_size"])
        )
        _update_dense_image_layer(k_dense_layer, Lf_q, voxel_idx, "K_dense L·density", "K_dense")

    def refresh_J():
        if j_layer is None or not j_layer.visible: return
        s = current_src()
        K_q, voxel_idx = kernel_at_dense_grid(s, ctx, state["render_stride"], batch_size=int(args["nystrom_batch_size"]))
        _update_dense_image_layer(j_layer, K_q, voxel_idx, "J Manif Kernel", "J")

    def refresh_L():
        if l_layer is None or not l_layer.visible: return
        s = current_src()
        K_q, voxel_idx = euclidean_kernel_at_dense_grid(s, ctx, state["nu"], state["lengthscale"], state["render_stride"])
        _update_dense_image_layer(l_layer, K_q, voxel_idx, "L Eucl Kernel", "L")

    def refresh_per_source():
        s = current_src()
        data = build_lines_for_source(s, ctx, state["lengthscale"], state["gamma"], args["k_show"], args["n_targets"], args["target_strategy"], args["source_seed"], state["nu"], args["knn_color_by"])
        for key in ("knn", "matern", "riemann"): replace_shapes_layer(viewer, layer_state, key, data[key]["lines"], data[key]["colors"], data[key]["widths"])

        face_colors = ["red"] * len(src_idxs)
        face_colors[state["src_pick"] % len(src_idxs)] = "yellow"
        viewer.layers["source nodes"].face_color_mode = 'direct'
        viewer.layers["source nodes"].face_color = face_colors

    def _batched_features_reduce(reduce_fn, batch_size: int):
        """Iterate matern_kernel.features over training nodes in batches and
        reduce per-batch on GPU before moving to CPU. `reduce_fn(feat_batch)`
        must return a 1D tensor of length batch_size that can be concatenated.

        Avoids the OOM in refresh_C / refresh_F where calling features() on the
        full reference_nodes set tries to materialise an (N, K_nn, num_modes)
        intermediate inside out_of_sample — ~133 GiB at N=458K, K=15, modes=1300.
        Batching keeps the intermediate to (batch, K_nn, num_modes).
        """
        nodes = ctx["reference_nodes"]
        N = nodes.shape[0]
        out = np.empty(N, dtype=np.float32)
        mk = ctx["matern_kernel"]
        for s_ in range(0, N, batch_size):
            e_ = min(s_ + batch_size, N)
            feat_b = mk.features(nodes[s_:e_])
            out[s_:e_] = reduce_fn(feat_b).detach().cpu().numpy()
            del feat_b
        return out

    def refresh_C():
        # d_i = ||features(node_i)||^2 — reduce within each batch so the full
        # features tensor never lives in memory at once.
        d = _batched_features_reduce(
            lambda fb: (fb ** 2).sum(dim=-1),
            batch_size=int(args["nystrom_batch_size"]),
        )
        _update_sparse_image_layer(c_layer, d, "C", "C", cmap_name="magma", mode="sequential")

    def refresh_D():
        s = current_src()
        Lf = apply_laplacian_to_delta(ctx["laplacian_op"], s)
        _update_sparse_image_layer(d_layer, Lf, f"D (src={s})", "D", cmap_name="RdBu_r", mode="diverging")

    def refresh_E():
        s = current_src()
        k_eu = matern_euclidean_at_source(ctx["reference_nodes"][s], ctx["reference_nodes"], state["nu"], state["lengthscale"])
        _update_sparse_image_layer(e_layer, k_eu, f"E (src={s})", "E", cmap_name="magma", mode="sequential")

    def refresh_F():
        # k_mf[i] = features(node_i) · features(src) — feat_src is a single
        # vector, computed once; the inner product reduces each batch's
        # (batch, num_modes) features to a (batch,) scalar before leaving GPU.
        s = current_src()
        matern_kernel = ctx["matern_kernel"]
        feat_src = matern_kernel.features(ctx["reference_nodes"][s].unsqueeze(0)).squeeze(0)
        k_mf = _batched_features_reduce(
            lambda fb: (fb * feat_src).sum(dim=-1),
            batch_size=int(args["nystrom_batch_size"]),
        )
        # mode="auto" → diverging if k_mf has negatives (truncation artefacts),
        # else sequential. Matches the previous behaviour.
        _update_sparse_image_layer(f_layer, k_mf, f"F (src={s})", "F", mode="auto")

    def refresh_G():
        k = int(state["eigvec_idx"])
        phi = ctx["eigvec"][:, k].cpu().numpy()
        _update_sparse_image_layer(g_layer, phi, f"G φ_{k}", "G", cmap_name="RdBu_r", mode="diverging")

    def refresh_K():
        if not k_layer.visible: return
        Lf = apply_laplacian_to_density(ctx["laplacian_op"], ctx["sub_volume"], ctx["node_voxel_idx"], sigma=state["density_smooth_sigma"])
        _update_sparse_image_layer(k_layer, Lf, "K L·density", "K", cmap_name="RdBu_r", mode="diverging")

    # Initial boot sequence (B1 and H are static — set once from precomputed
    # diag/degree values; the others depend on src / k / etc and get refreshed
    # via the controls callback below)
    _update_sparse_image_layer(b1_layer, lap_diag_vals, "B1 diag(L)", "B1", cmap_name="cividis", mode="sequential")
    _update_sparse_image_layer(h_layer,   deg_vals,      "H D_i",     "H",  cmap_name="cividis", mode="sequential")
    refresh_per_source(); refresh_C(); refresh_D(); refresh_E(); refresh_F(); refresh_G()

    bump_widget, update_bump_plot = make_bump_widget(ctx["bw"], state["bump_scale"], state["bump_decay"])
    if bump_widget is not None: viewer.window.add_dock_widget(bump_widget, name="bump function", area="left")

    @magicgui(
        auto_call=True,
        src_pick={"label": "active source", "min": 0, "max": len(src_idxs) - 1},
        num_modes={"label": "num modes (C, F, J, Riemann)", "min": 1, "max": ctx["eigvec"].shape[1]},
        nu={"label": "ν (Matern smoothness)", "min": 1, "max": 6},
        lengthscale={"label": "ℓ (Matern lengthscale)", "min": 1e-3, "max": 10.0, "step": 1e-3},
        eigvec_idx={"label": "eigenvector index (G)", "min": 0, "max": ctx["eigvec"].shape[1] - 1},
        gamma={"label": "color gamma", "min": 0.1, "max": 2.0, "step": 0.05},
        alpha_power={"label": "opacity sharpness", "min": 0.5, "max": 8.0, "step": 0.5},
        render_stride={"label": "render stride (J, L)", "min": 1, "max": 8},
        bump_scale={"label": "bump scale (× bw)", "min": 0.001, "max": 200.0, "step": 0.1},
        bump_decay={"label": "bump decay", "min": 0.001, "max": 2.0, "step": 0.001},
        density_smooth_sigma={"label": "L·density: σ (voxels)", "min": 0.0, "max": 10.0, "step": 0.1},
    )
    def controls(src_pick=state["src_pick"], num_modes=state["num_modes"], nu=state["nu"], lengthscale=state["lengthscale"], eigvec_idx=state["eigvec_idx"], gamma=state["gamma"], alpha_power=state.get("alpha_power", 3.0), render_stride=state["render_stride"], bump_scale=state["bump_scale"], bump_decay=state["bump_decay"], density_smooth_sigma=state["density_smooth_sigma"]):
        chg_src, chg_K, chg_nu, chg_ls, chg_eig, chg_gamma, chg_alpha, chg_rs, chg_bs, chg_bd, chg_sigma = src_pick != state["src_pick"], num_modes != state["num_modes"], nu != state["nu"], lengthscale != state["lengthscale"], eigvec_idx != state["eigvec_idx"], gamma != state["gamma"], alpha_power != state.get("alpha_power", 3.0), render_stride != state["render_stride"], bump_scale != state["bump_scale"], bump_decay != state["bump_decay"], density_smooth_sigma != state["density_smooth_sigma"]
        state.update(src_pick=src_pick, num_modes=num_modes, nu=nu, lengthscale=lengthscale, eigvec_idx=eigvec_idx, gamma=gamma, alpha_power=alpha_power, render_stride=render_stride, bump_scale=bump_scale, bump_decay=bump_decay, density_smooth_sigma=density_smooth_sigma)

        # Sync visualizer state back to the actual library kernel!
        kernel = ctx["matern_kernel"]
        kernel.nu = state["nu"]
        # Use torch.tensor for lengthscale to match gpytorch params
        kernel.lengthscale = torch.tensor(state["lengthscale"], device=ctx["device"])
        kernel.num_modes = state["num_modes"]
        kernel.bump_scale = state["bump_scale"]
        kernel.bump_decay = state["bump_decay"]

        if chg_src or chg_K or chg_nu or chg_ls or chg_gamma or chg_alpha: refresh_per_source()
        if chg_K or chg_nu or chg_ls or chg_gamma or chg_alpha: refresh_C()
        if chg_src: refresh_D()
        if chg_src or chg_nu or chg_ls or chg_gamma or chg_alpha: refresh_E()
        if chg_src or chg_K or chg_nu or chg_ls or chg_gamma or chg_alpha: refresh_F()
        if chg_eig or chg_gamma or chg_alpha: refresh_G()
        if chg_sigma or chg_gamma or chg_alpha: refresh_K()
        
        if chg_src or chg_K or chg_nu or chg_ls or chg_rs or chg_bs or chg_bd: refresh_J()
        if chg_src or chg_nu or chg_ls or chg_rs: refresh_L()
        if chg_eig or chg_rs or chg_bs or chg_bd or chg_gamma or chg_alpha: refresh_G_dense()
        if chg_src or chg_rs or chg_bs or chg_bd or chg_gamma or chg_alpha: refresh_D_dense()
        if chg_sigma or chg_rs or chg_bs or chg_bd or chg_gamma or chg_alpha: refresh_K_dense()
        if chg_bs or chg_bd: update_bump_plot(bump_scale, bump_decay)

    viewer.window.add_dock_widget(controls, name="kernel controls", area="right")

    k_layer.events.visible.connect(lambda e: refresh_K() if k_layer.visible else None)
    if j_layer is not None: j_layer.events.visible.connect(lambda e: refresh_J() if j_layer.visible else None)
    if l_layer is not None: l_layer.events.visible.connect(lambda e: refresh_L() if l_layer.visible else None)
    g_dense_layer.events.visible.connect(lambda e: refresh_G_dense() if g_dense_layer.visible else None)
    d_dense_layer.events.visible.connect(lambda e: refresh_D_dense() if d_dense_layer.visible else None)
    k_dense_layer.events.visible.connect(lambda e: refresh_K_dense() if k_dense_layer.visible else None)

    def force_refresh_on_dim_switch(event):
        # B1 and H are static-input but their colormap alpha curve depends
        # on ndisplay (2D vs 3D), so re-splat with the current mode too.
        if b1_layer.visible:
            _update_sparse_image_layer(b1_layer, lap_diag_vals, "B1 diag(L)", "B1", cmap_name="cividis", mode="sequential")
        if h_layer.visible:
            _update_sparse_image_layer(h_layer, deg_vals, "H D_i", "H", cmap_name="cividis", mode="sequential")
        if c_layer.visible: refresh_C()
        if d_layer.visible: refresh_D()
        if e_layer.visible: refresh_E()
        if f_layer.visible: refresh_F()
        if g_layer.visible: refresh_G()
        if k_layer.visible: refresh_K()
        if g_dense_layer.visible: refresh_G_dense()
        if d_dense_layer.visible: refresh_D_dense()
        if k_dense_layer.visible: refresh_K_dense()
        if j_layer is not None and j_layer.visible: refresh_J()
        if l_layer is not None and l_layer.visible: refresh_L()
        refresh_per_source()

    viewer.dims.events.ndisplay.connect(force_refresh_on_dim_switch)

    # All layers now have real sub_shape data (boot sequence populated B1/H
    # statically and the refresh_X functions populated C/D/E/F/G dynamically).
    # Flipping to 3D here builds volume visuals against real data, not the
    # (2,2,2) placeholder — sidestepping the vispy "Volume visual needs a 3D
    # array" error on first construction.
    viewer.dims.ndisplay = 3

    napari.run()

if __name__ == "__main__":
    main()