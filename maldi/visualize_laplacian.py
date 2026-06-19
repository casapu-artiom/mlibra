"""Standalone kernel & graph visualization for the manifold GP.

Layers — all share the same coordinate frame (full-resolution template
voxel coords). Toggle visibility in napari's left panel.

  Layer A0 — Reference template (anatomical backdrop / context)
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
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
)
from utils import crop_or_stride_volume, reference_ccf_from_subvolume, coord_norm_from_reference

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
    p.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1,
                   help="Lanczos Krylov subspace floor. <=0 (default) auto-picks "
                        "max(1500, 3*num_modes+20); set a small explicit value "
                        "(e.g. 100) to fit large-N (stride=1) eigensolves in GPU memory.")
    p.add_argument("--initial-modes", type=int, default=None)
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute-eigvecs", action="store_true")

    p.add_argument("--nu", type=int, default=2)
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--diffusion-t", type=float, default=1.0)

    p.add_argument("--n-sources", type=int, default=4)
    p.add_argument("--source-seed", type=int, default=0)
    # PSD diagnostic panel sampling — fixed at startup, doesn't sweep
    p.add_argument("--psd-n-on",  type=int, default=100,
                   help="In-sample test points for the PSD diagnostic panel.")
    p.add_argument("--psd-n-off", type=int, default=100,
                   help="Out-of-sample test points (sub-threshold voxels) for the PSD diagnostic panel.")
    p.add_argument("--psd-seed",  type=int, default=42,
                   help="RNG seed for PSD panel test-point sampling.")
    p.add_argument("--n-targets", type=int, default=50)
    p.add_argument("--target-strategy", choices=["random", "stratified"], default="stratified")
    p.add_argument("--k-show", type=int, default=30)
    p.add_argument("--source-marker-size", type=float, default=6.0)
    p.add_argument("--knn-color-by", choices=["heat", "distance"], default="heat")

    p.add_argument("--fabric-edge-sample", type=int, default=200_000)
    p.add_argument("--fabric-node-size", type=float, default=0.6)
    p.add_argument("--fabric-edge-width", type=float, default=0.3)
    p.add_argument("--fabric-color-by", choices=["flat", "distance", "affinity"],
                   default="distance",
                   help=("A2 fabric edge shading. flat = uniform gray; distance = "
                         "highlight geometrically SHORT edges (bright/opaque), fade "
                         "long ones (inflation-free); affinity = shade by the "
                         "Laplacian affinity exp(-edge_value/4bw^2) (folds in "
                         "cross-region inflation)."))
    p.add_argument("--fabric-cmap", default="plasma",
                   help="matplotlib colormap for --fabric-color-by (close=bright end).")
    p.add_argument("--laplacian-edge-sample", type=int, default=80_000)

    p.add_argument("--render-stride", type=int, default=1)
    p.add_argument("--bump-scale", type=float, default=3.0)
    p.add_argument("--bump-decay", type=float, default=0.05)
    p.add_argument("--bump-glow-edge-sample", type=int, default=6000,
                   help="How many sampled fabric edges to seed the bump glow from.")
    p.add_argument("--bump-glow-steps", type=int, default=3,
                   help="Interpolation points per edge interior for the bump glow.")
    p.add_argument("--bump-glow-opacity", type=float, default=0.35,
                   help="Per-point base opacity of the bump glow (additive).")
    p.add_argument("--bump-glow-min-size", type=float, default=3.0,
                   help="Minimum glow disc diameter (voxels) so small bump "
                        "scales stay visible instead of collapsing to nothing.")
    p.add_argument("--bump-glow-max-size", type=float, default=10.0,
                   help="Maximum glow disc diameter (voxels). Caps the DRAWN "
                        "radius so large bump scales don't balloon past the "
                        "manifold's own features; the true support radius is "
                        "still reported in the info panel. Set very large to "
                        "draw the true (uncapped) support extent.")
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
        stride=args["stride"],
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, args["threshold"],
    )
    reference_nodes_mm = torch.tensor(reference_ccf, dtype=torch.float32)
    # Single source of truth shared with training (utils.coord_norm_from_reference):
    # per-axis mean + scalar (isotropic) std from reference_image > 0. Computed
    # straight from the reference image the visualizer already loaded, so it
    # never depends on a training run having happened in this environment.
    coord_mean, coord_std = coord_norm_from_reference(template_full)
    reference_nodes = ((reference_nodes_mm - coord_mean) / coord_std).to(device)

    node_voxel_idx = np.argwhere(sub_volume > args["threshold"]).astype(np.int32)
    assert node_voxel_idx.shape[0] == reference_nodes.shape[0]

    sv_scale = np.array([args["stride"], args["stride"], args["stride"]], dtype=np.float32)
    sv_translate = np.asarray(voxel_offset, dtype=np.float32)

    eigenvector_dir = Path(args["eigenvector_dir"])
    graphs = KnnGraphCache(cache_dir=eigenvector_dir / "knn", verbose=True)
    graph_key_parts = {
        "template": args["template_name"],
        "stride": args["stride"],
        "thresh": args["threshold"],
        "method": args["knn_method"],
        "k": args["knn_k"],
        "nlist": args["n_list"],
        "bbox": None,
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
    ncv_min = resolve_ncv_min(args["num_modes"], args.get("ncv_min", -1))
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


def fabric_edge_rgba(fabric_lines: np.ndarray, edge_sq_dists: np.ndarray,
                     mode: str, bw: float, cmap_name: str = "plasma") -> np.ndarray:
    """Per-edge RGBA for the A2 KNN fabric so CLOSE edges are highlighted and
    far ones fade out (the raw fabric draws every edge identically).

      flat      -> uniform faint gray (the original look)
      distance  -> closeness from the edge's GEOMETRIC length (endpoints), so it
                   is inflation-free: short edges bright/opaque, long edges dim.
                   Robust min/max via the 5th/95th length percentiles.
      affinity  -> closeness = exp(-edge_value/4bw^2), the affinity the Laplacian
                   actually uses; this DOES fold in cross-region inflation (those
                   edges get large edge_value -> near-zero affinity -> faded).

    closeness in [0,1] (1 = close) is gamma-lifted for contrast, mapped through
    `cmap_name` (close = bright end), and also drives alpha so far edges are
    nearly transparent."""
    M = int(fabric_lines.shape[0])
    if M == 0 or mode == "flat":
        return np.tile(np.array([[0.6, 0.6, 0.6, 0.35]], np.float32), (M, 1))
    if mode == "affinity":
        d2 = np.asarray(edge_sq_dists, np.float64)
        closeness = np.exp(-d2 / max(4.0 * bw * bw, 1e-12))          # (0,1], 1=strong
    else:  # "distance": literal geometric closeness, independent of inflation
        d = np.linalg.norm(fabric_lines[:, 1, :] - fabric_lines[:, 0, :], axis=1)
        lo, hi = np.percentile(d, 5), np.percentile(d, 95)
        closeness = 1.0 - np.clip((d - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    c = np.clip(closeness, 0.0, 1.0) ** 0.5                          # lift contrast
    rgba = matplotlib.colormaps.get_cmap(cmap_name)(c).astype(np.float32)
    rgba[:, 3] = 0.06 + 0.92 * c                                     # far -> transparent
    return rgba


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


def bump_field_on_dense_grid(
    ctx: dict, laplacian_op: GraphLaplacianOperator, render_stride: int,
    bump_scale: float, bump_decay: float, batch_size: int = 50_000,
    max_points: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The bump *support cloud* around the manifold.

    Returns, for a strided grid of voxels NEAR the graph nodes (not just the
    above-threshold manifold voxels — we want the surrounding halo), the value
    b(p) = bump_function(dist(p, nearest node), alpha, beta) in [0, 1], where
    alpha = bump_scale * graphbandwidth. b = 1 on the manifold, fades to 0 at
    the support radius, and is exactly 0 (omitted) beyond it. This is the field
    that multiplies the out-of-sample features in RiemannKernel.features().

    The grid is bounded to the node bounding box dilated by alpha and, if that
    still exceeds `max_points`, the cloud stride is increased so the layer stays
    cheap. Returns (b_within, voxel_idx_within, cloud_stride)."""
    from scipy.spatial import cKDTree
    tmpl = ctx["template_full"]
    shape = np.asarray(tmpl.shape)
    bw = float(laplacian_op.graphbandwidth.squeeze())
    bump_radius = bw * float(bump_scale)  # in z-scored coordinate units

    coord_std = ctx["coord_std"].cpu().numpy().astype(np.float64)
    # full-resolution voxel positions of the graph nodes
    node_full = (ctx["node_voxel_idx"].astype(np.float64)
                 * ctx["sv_scale"] + ctx["sv_translate"])
    # alpha expressed in voxels, per axis: dz/dvoxel = 0.025 / coord_std
    alpha_vox = bump_radius * coord_std / 0.025

    lo = np.maximum(np.floor(node_full.min(0) - alpha_vox).astype(int), 0)
    hi = np.minimum(np.ceil(node_full.max(0) + alpha_vox).astype(int), shape - 1)

    # pick a cloud stride that keeps the candidate count under max_points
    cloud_stride = int(render_stride)
    def n_candidates(s):
        return int(np.prod(np.maximum((hi - lo) // s + 1, 1)))
    while n_candidates(cloud_stride) > max_points:
        cloud_stride += 1

    zs = np.arange(lo[0], hi[0] + 1, cloud_stride)
    ys = np.arange(lo[1], hi[1] + 1, cloud_stride)
    xs = np.arange(lo[2], hi[2] + 1, cloud_stride)
    grid = np.stack(np.meshgrid(zs, ys, xs, indexing="ij"), -1).reshape(-1, 3).astype(np.int32)

    if "_query_kdt" not in ctx:
        ctx["_query_kdt"] = cKDTree(ctx["reference_nodes"].cpu().numpy())
    kdt = ctx["_query_kdt"]

    coords_z = _full_voxel_to_normalized(grid, ctx).numpy()
    keep_idx, keep_b = [], []
    for s in range(0, grid.shape[0], batch_size):
        e = min(s + batch_size, grid.shape[0])
        d, _ = kdt.query(coords_z[s:e], k=1, workers=-1)   # euclidean distance (z units)
        d = d.astype(np.float32)
        within = d < bump_radius
        if within.any():
            b = bump_function(torch.from_numpy(d[within]),
                              bump_radius, float(bump_decay)).cpu().numpy()
            keep_idx.append(grid[s:e][within])
            keep_b.append(b)
    if keep_b:
        return (np.concatenate(keep_b), np.concatenate(keep_idx), cloud_stride)
    return np.empty(0, np.float32), np.empty((0, 3), np.int32), cloud_stride

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


# -----------------------------------------------------------------------------
# PSD diagnostic panel — shows two sections:
#   (1) Laplacian eigenvalue statistics (zero / near-zero / negative counts,
#       λ_min, λ_max, condition number, spectral gap, below-Matern-floor count).
#       These depend only on the loaded eigenpairs and are computed once at
#       startup — they don't change as the user tweaks nu / lengthscale /
#       num_modes / bump_scale, because the *Laplacian* spectrum is fixed.
#   (2) Kernel-Gram PSD on a small sample of in-sample (training nodes) and
#       out-of-sample (sub-threshold voxels) test points. Three Gram matrices
#       are reported: K_m (bare manifold), K_bm (bump-modulated), K_full
#       (deployed kernel including bump-Euclidean fallback). For each matrix
#       at each block (in-sample, out-of-sample), we report the smallest
#       eigenvalue / λ_max ratio and the condition number. This DOES depend
#       on the current state — the refresh callback recomputes it whenever
#       relevant params change.
# -----------------------------------------------------------------------------
def analyze_spectrum_for_panel(ev: np.ndarray, matern_floor: float = float("-inf"),
                                tol_zero: float = 1e-10, tol_neg: float = 1e-6) -> dict:
    """Lightweight reimplementation of laplacian_sweep.analyze_eigvals for the
    UI panel. Standalone so we don't need to import the sweep script."""
    ev = np.asarray(ev, dtype=np.float64).ravel()
    if ev.size == 0:
        return {}
    lam_min = float(ev.min())
    lam_max = float(ev.max())
    scale = max(abs(lam_max), 1e-30)
    pos = ev[ev > 0]
    lam_min_pos = float(pos.min()) if pos.size else float("nan")
    ev_sorted = np.sort(ev)
    spectral_gap = float(ev_sorted[1] - ev_sorted[0]) if ev.size >= 2 else float("nan")
    return {
        "n":              int(ev.size),
        "lam_min":        lam_min,
        "lam_max":        lam_max,
        "ratio":          lam_min / scale,
        "n_zero_exact":   int(np.sum(ev == 0.0)),
        "n_zero_eps":     int(np.sum(np.abs(ev) < tol_zero * scale)),
        "n_neg":          int(np.sum(ev < 0)),
        "n_neg_sig":      int(np.sum(ev < -tol_neg * scale)),
        "n_below_floor":  int(np.sum(ev < matern_floor)),
        "spectral_gap":   spectral_gap,
        "cond":           (lam_max / lam_min_pos) if pos.size else float("inf"),
        "lam_min_pos":    lam_min_pos,
    }


def classify_psd(ratio: float, tol_neg: float = 1e-6, tol_zero: float = 1e-10) -> str:
    """Three-level classification: clean / jitter / violation."""
    if ratio >= -tol_zero:
        return "clean"
    if ratio >= -tol_neg:
        return "jitter"
    return "violation"


def matern_euclidean_pairwise(coords: torch.Tensor, nu: float, lengthscale: float) -> np.ndarray:
    """N×N Euclidean Matern kernel via the scipy.special Bessel formula.
    Matches matern_euclidean_at_source's convention."""
    from scipy.special import kv, gamma as gamma_fn
    pts = coords.detach().cpu().numpy()
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    nu_f = float(nu); ls = float(lengthscale)
    K = np.ones_like(d, dtype=np.float64)
    nz = d > 0
    if nz.any():
        z = np.sqrt(2.0 * nu_f) * d[nz] / ls
        # Floor z away from zero — kv(nu, z) for tiny z is finite but the
        # multiplication z^nu * kv(nu, z) can NaN through 0*inf cancellation.
        z = np.maximum(z, 1e-12)
        K[nz] = ((2.0 ** (1.0 - nu_f)) / gamma_fn(nu_f)) * (z ** nu_f) * kv(nu_f, z)
    return K


def make_psd_info_panel(ctx: dict, n_on: int = 100, n_off: int = 100, seed: int = 42):
    """Returns (widget, refresh_callback, sample_dict). The widget can be
    docked into napari; the callback recomputes the kernel-PSD section using
    current state values (nu, lengthscale, num_modes, bump_scale, bump_decay).
    The Laplacian section is computed once and cached in sample_dict.

    sample_dict carries the precomputed test points and bookkeeping needed
    by refresh so we don't re-sample on every parameter change (re-sampling
    would make the diagnostic non-comparable across param tweaks).
    """
    try:
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
        from qtpy.QtGui import QFont
    except ImportError:
        return None, (lambda *_a, **_k: None), {}

    text = QTextEdit()
    text.setReadOnly(True)
    mono = QFont("Monospace")
    mono.setStyleHint(QFont.TypeWriter)
    mono.setPointSize(9)
    text.setFont(mono)
    text.setMinimumHeight(260)

    widget = QWidget()
    layout = QVBoxLayout(widget); layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(QLabel("Laplacian + Kernel PSD diagnostics"))
    layout.addWidget(text)

    # --- Sample test points once at startup; reuse across refreshes ---
    device = ctx["device"]
    rng = np.random.default_rng(seed)
    reference_nodes = ctx["reference_nodes"]
    N = reference_nodes.shape[0]

    n_on  = min(n_on, N)
    on_idx = rng.choice(N, size=n_on, replace=False)
    pts_on = reference_nodes[on_idx].clone()

    # Off-manifold: sub-threshold voxels (template > 0 and ≤ threshold).
    sub_volume = ctx["sub_volume"]
    threshold = int(ctx["threshold"])
    off_mask = (sub_volume > 0) & (sub_volume <= threshold)
    off_zyx = np.argwhere(off_mask)
    if off_zyx.shape[0] > 0:
        take = min(n_off, off_zyx.shape[0])
        pick = rng.choice(off_zyx.shape[0], size=take, replace=False)
        off_idx_full = off_zyx[pick].astype(np.float32)
        oz, oy, ox = ctx["voxel_offset"]
        # Same coord convention as reference_ccf_from_subvolume
        if float(ctx["voxel_scale_mm"]) == 0.025:
            off_mm = (off_idx_full + np.array([oz, oy, ox], dtype=np.float32)) * 0.025
        else:
            off_mm = off_idx_full * ctx["voxel_scale_mm"]
        off_mm_t = torch.from_numpy(off_mm).to(device)
        pts_off = (off_mm_t - ctx["coord_mean"].to(device)) / ctx["coord_std"].to(device)
    else:
        pts_off = torch.empty(0, 3, dtype=pts_on.dtype, device=device)

    if pts_off.shape[0] < n_off:
        # Fallback: jitter training nodes to fill the off sample.
        deficit = n_off - pts_off.shape[0]
        seed_idx = rng.choice(N, size=deficit, replace=True)
        jitter = torch.from_numpy(
            rng.normal(0, 0.05, size=(deficit, 3)).astype(np.float32)
        ).to(device)
        pts_off = torch.cat([pts_off, reference_nodes[seed_idx] + jitter], dim=0)

    test_pts = torch.cat([pts_on, pts_off], dim=0)
    n_on_actual = int(pts_on.shape[0])
    n_off_actual = int(test_pts.shape[0] - n_on_actual)

    sample_dict = {
        "test_pts":    test_pts,
        "n_on":        n_on_actual,
        "n_off":       n_off_actual,
        "lap_summary": None,  # filled in on first refresh
    }

    # --- Format helpers ---
    def fmt(d: dict, prefix: str = "") -> str:
        cls = classify_psd(d["ratio"])
        color = {"clean": "#2a8", "jitter": "#888", "violation": "#c44"}[cls]
        return (
            f"{prefix:<18} n={d['n']:>5d}  "
            f"λ_min={d['lam_min']:>+10.3e}  λ_max={d['lam_max']:>10.3e}  "
            f"<span style='color:{color}'>ratio={d['ratio']:>+10.3e}</span>  "
            f"n_neg={d['n_neg']:>4d}/{d['n_neg_sig']:>4d}  "
            f"n_zero={d['n_zero_eps']:>4d}  cond={d['cond']:>10.2e}"
        )

    # --- Laplacian section: computed once ---
    eigval_np = ctx["eigval"].detach().cpu().numpy()
    matern_floor = -2.0 * float(ctx.get("_nu_for_floor", 2)) / (float(ctx.get("_ls_for_floor", 1.0)) ** 2)
    lap_summary = analyze_spectrum_for_panel(eigval_np, matern_floor=matern_floor)
    sample_dict["lap_summary"] = lap_summary

    # --- Kernel-Gram section: recomputed on demand ---
    def refresh(nu: float, lengthscale: float, num_modes: int,
                bump_scale: float, bump_decay: float):
        """Recompute the kernel-Gram diagnostics with current state values.
        Uses ctx['matern_kernel'] which gets rebuilt by the surrounding code
        when num_modes / nu / lengthscale change."""
        try:
            matern_kernel = ctx["matern_kernel"]
            graphbandwidth = float(ctx["bw"])

            with torch.no_grad():
                feat = matern_kernel.features(test_pts)
                K_m = (feat @ feat.t()).detach().cpu().numpy().astype(np.float64)
                K_m = 0.5 * (K_m + K_m.T)
                edge_value, _ = matern_kernel.knn.search(test_pts, 1)
                d_nearest = edge_value.sqrt().squeeze(-1)
                b = bump_function(
                    d_nearest,
                    scale=float(bump_scale) * graphbandwidth,
                    decay=float(bump_decay),
                ).detach().cpu().numpy().astype(np.float64)

            K_eu = matern_euclidean_pairwise(test_pts, nu, lengthscale)
            K_bm = b[:, None] * K_m * b[None, :]
            one_m_b = 1.0 - b
            K_full = K_bm + (one_m_b[:, None] * one_m_b[None, :]) * K_eu

            # Recompute floor with the CURRENT nu/lengthscale (the Laplacian
            # spectrum doesn't change, but where the floor lands does).
            floor_now = -2.0 * float(nu) / (float(lengthscale) ** 2)
            n_below = int(np.sum(eigval_np < floor_now))

            blocks = [
                ("on",  slice(0, n_on_actual)),
                ("off", slice(n_on_actual, None)),
            ]
            kernel_lines: list[str] = []
            for tag, mat in [("K_m", K_m), ("K_bm", K_bm), ("K_full", K_full)]:
                for blk_name, sl in blocks:
                    sub = mat[sl, sl]
                    if sub.size == 0:
                        continue
                    try:
                        ev = np.linalg.eigvalsh(sub)
                    except np.linalg.LinAlgError as e:
                        kernel_lines.append(f"{tag:<7s} {blk_name:<3s}  EIG_FAIL: {e}")
                        continue
                    d = analyze_spectrum_for_panel(ev)
                    kernel_lines.append(fmt(d, prefix=f"{tag:<7s} {blk_name:<3s}"))

            bump_on  = float(b[:n_on_actual].mean()) if n_on_actual > 0 else float("nan")
            bump_off = float(b[n_on_actual:].mean()) if n_off_actual > 0 else float("nan")

            # --- Compose the panel HTML ---
            lap_line = fmt(lap_summary, prefix="Laplacian (full)")
            params_line = (
                f"params: ν={nu}  ℓ={lengthscale:g}  modes={num_modes}  "
                f"bw={graphbandwidth:g}  bump_scale={bump_scale:g}  bump_decay={bump_decay:g}"
            )
            floor_line = (
                f"matern floor = -2ν/ℓ² = {floor_now:>10.3g}  "
                f"n_below_floor={n_below:>5d}  "
                + ("<span style='color:#c44'>(spectral density flips sign — kernel non-PSD)</span>"
                   if n_below > 0 else "<span style='color:#2a8'>(all eigvals above floor)</span>")
            )
            bump_line = (
                f"bump: on_mean={bump_on:.4f}  off_mean={bump_off:.4f}  "
                f"n_on={n_on_actual}  n_off={n_off_actual}"
                + ("  <span style='color:#888'>(bump ≈ 1 everywhere ⇒ K_bm ≈ K_m, K_full ≈ K_m)</span>"
                   if bump_off > 0.95 else "")
            )

            html = (
                "<pre>"
                + lap_line + "\n"
                + floor_line + "\n"
                + "\n"
                + params_line + "\n"
                + bump_line + "\n"
                + "\n"
                + "Kernel Gram on test sample (in-sample = training nodes; out-of-sample = sub-threshold voxels):\n"
                + "\n".join(kernel_lines)
                + "\n"
                + "\n"
                + "<span style='color:#888'>legend: ratio = λ_min/λ_max.  green=clean (≥-1e-10), grey=Lanczos jitter (-1e-6..-1e-10), red=real violation (<-1e-6).  n_neg = total/significant.</span>"
                + "</pre>"
            )
            text.setHtml(html)
        except Exception as e:
            import traceback
            text.setHtml(f"<pre>refresh failed: {type(e).__name__}: {e}\n{traceback.format_exc()}</pre>")

    return widget, refresh, sample_dict


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

    # A0: the raw reference template as an anatomical backdrop. It is natively in
    # full-resolution voxel coords -- the shared frame -- so scale=(1,1,1) and
    # translate=(0,0,0) (default) align it with the strided Image layers (which
    # carry sv_scale/sv_translate) and the node Points. Rendered as an attenuated
    # MIP so the tissue shows without a black bounding box; sub-threshold voxels
    # sit at the low end of the contrast range. Added first => bottom of the
    # layer list / behind everything else.
    _tmpl = ctx["template_full"]
    _tmpl_pos = _tmpl[_tmpl > 0]
    _tmpl_lo = float(ctx["threshold"])
    _tmpl_hi = float(np.percentile(_tmpl_pos, 99)) if _tmpl_pos.size else float(_tmpl.max())
    _tmpl_hi = max(_tmpl_hi, _tmpl_lo + 1.0)
    viewer.add_image(
        _tmpl, name="A0: reference template", colormap="gray",
        blending="translucent", rendering="attenuated_mip",
        contrast_limits=(_tmpl_lo, _tmpl_hi), opacity=0.4, visible=True,
    )

    viewer.add_points(all_node_positions, name="A1: graph nodes", size=float(args["fabric_node_size"]), face_color="white", border_color="white", symbol="o", opacity=0.25, blending="additive")
    fabric_lines = make_lines_array(fabric_pairs, all_node_positions)
    fabric_rgba = fabric_edge_rgba(fabric_lines, fabric_sq_dists,
                                   args["fabric_color_by"], ctx["bw"],
                                   cmap_name=args["fabric_cmap"])
    viewer.add_shapes([fabric_lines[i] for i in range(fabric_lines.shape[0])], shape_type="line", edge_color=fabric_rgba, edge_width=float(args["fabric_edge_width"]), name="A2: KNN fabric (edges)", opacity=0.7, blending="translucent")

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
    bump_cloud_layer = viewer.add_points(
        np.empty((0, 3), dtype=np.float32), name="C0: bump glow (around sampled fabric)",
        size=1.0, face_color="cyan", border_width=0.0, symbol="disc",
        blending="additive", opacity=1.0, visible=False, out_of_slice_display=True,
    )

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

    # PSD diagnostic panel — Laplacian eigenvalue stats (fixed) + kernel-Gram
    # PSD on a small in-sample / out-of-sample test sample (refreshes when
    # the user changes nu, lengthscale, num_modes, bump_scale, bump_decay).
    psd_panel, refresh_psd, _psd_sample = make_psd_info_panel(
        ctx,
        n_on=int(args.get("psd_n_on", 100)),
        n_off=int(args.get("psd_n_off", 100)),
        seed=int(args.get("psd_seed", 42)),
    )
    if psd_panel is not None:
        viewer.window.add_dock_widget(psd_panel, name="PSD diagnostics", area="bottom")

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

    def refresh_bump_cloud():
        if not bump_cloud_layer.visible: return
        from scipy.spatial import cKDTree
        laplacian_op = ctx["laplacian_op"]
        # Seed ONCE from the same sampled fabric edges shown in layer A2.
        # Separate the graph NODES (which are on the manifold → bump == 1 by
        # definition, so always drawn) from the edge-interior points (off the
        # nodes → their bump is computed and they appear only once the support
        # is wide enough to reach them).
        if "_bump_glow_seed" not in ctx:
            pairs = fabric_pairs
            if pairs.shape[0] > int(args["bump_glow_edge_sample"]):
                sel = np.random.default_rng(0).choice(
                    pairs.shape[0], int(args["bump_glow_edge_sample"]), replace=False)
                pairs = pairs[sel]
            node_ids = np.unique(pairs.reshape(-1))
            node_pts = all_node_positions[node_ids].astype(np.float32)
            steps = max(0, int(args["bump_glow_steps"]))
            if steps > 0:
                a = all_node_positions[pairs[:, 0]]; b = all_node_positions[pairs[:, 1]]
                ts = np.linspace(0.0, 1.0, steps + 2)[1:-1][None, :, None]  # interior only
                edge_pts = (a[:, None, :] * (1 - ts) + b[:, None, :] * ts).reshape(-1, 3).astype(np.float32)
            else:
                edge_pts = np.empty((0, 3), dtype=np.float32)
            ctx["_bump_glow_node_pts"] = node_pts
            ctx["_bump_glow_edge_pts"] = edge_pts
            ctx["_bump_glow_edge_z"] = (_full_voxel_to_normalized(edge_pts, ctx).numpy()
                                        if edge_pts.shape[0] else np.empty((0, 3), np.float32))
            ctx["_bump_glow_n_edges"] = int(pairs.shape[0])
            ctx["_bump_glow_seed"] = True
        if "_query_kdt" not in ctx:
            ctx["_query_kdt"] = cKDTree(ctx["reference_nodes"].cpu().numpy())

        node_pts = ctx["_bump_glow_node_pts"]
        edge_pts = ctx["_bump_glow_edge_pts"]; edge_z = ctx["_bump_glow_edge_z"]
        alpha_z = float(state["bump_scale"]) * float(laplacian_op.graphbandwidth.squeeze())

        # nodes: on the manifold, bump == 1 (drawn at every bump scale)
        pts_list = [node_pts]
        b_list = [np.ones(node_pts.shape[0], dtype=np.float32)]
        # edge interior: keep only those the support actually reaches
        if edge_pts.shape[0]:
            d, _ = ctx["_query_kdt"].query(edge_z, k=1, workers=-1)
            d = d.astype(np.float32)
            eb = np.zeros(d.shape[0], dtype=np.float32)
            w = d < alpha_z
            if w.any():
                eb[w] = bump_function(torch.from_numpy(d[w]), alpha_z,
                                      float(state["bump_decay"])).cpu().numpy()
            ek = eb > 1e-3
            pts_list.append(edge_pts[ek]); b_list.append(eb[ek])

        pts = np.concatenate(pts_list, axis=0)
        bv = np.concatenate(b_list, axis=0)
        # True support diameter in voxels (coords are 0.025 mm/voxel)...
        true_diam = 2.0 * alpha_z * float(ctx["coord_std"].mean().item()) / 0.025
        # ...but the DRAWN disc is clamped so a large support doesn't balloon
        # past the manifold's own features. The true extent is reported below.
        size_vox = float(np.clip(true_diam,
                                 float(args["bump_glow_min_size"]),
                                 float(args["bump_glow_max_size"])))
        capped = true_diam > float(args["bump_glow_max_size"]) + 1e-6
        rgba = np.zeros((pts.shape[0], 4), dtype=np.float32)
        rgba[:, 0], rgba[:, 1], rgba[:, 2] = 0.45, 0.80, 1.0   # cyan
        rgba[:, 3] = np.clip(bv * float(args["bump_glow_opacity"]), 0.0, 1.0)
        n_pts = pts.shape[0]
        bump_cloud_layer.data = pts
        # Assigning an (N,4) array switches the layer to direct face-color mode
        # on its own; setting face_color_mode explicitly is unnecessary and
        # raises on some napari versions. Size as a per-point array is always
        # accepted, whereas a bare scalar can raise after a data-length change.
        bump_cloud_layer.face_color = rgba
        bump_cloud_layer.size = np.full(n_pts, size_vox, dtype=np.float32)
        radius_mm = alpha_z * float(ctx["coord_std"].mean().item())
        set_info("C0",
                 f"bump glow: {node_pts.shape[0]:,} node anchors + "
                 f"{pts.shape[0] - node_pts.shape[0]:,} edge pts  "
                 f"drawn⌀={size_vox:.0f} vox{' (CAPPED)' if capped else ''}  "
                 f"true support α={state['bump_scale']:g}×bw={alpha_z:.3g} z "
                 f"≈ {radius_mm:.2f} mm (⌀≈{true_diam:.0f} vox)")
        print(f"[C0 glow] refreshed: {pts.shape[0]:,} pts, drawn⌀={size_vox:.0f} vox, "
              f"alpha_max={float(rgba[:, 3].max()) if pts.shape[0] else 0:.2f}", flush=True)

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

    # Initial PSD panel populate — without this it shows nothing until the
    # user wiggles a slider. Uses the same state values the kernel was built
    # from, so the panel reflects the actual deployed configuration.
    if refresh_psd is not None:
        refresh_psd(
            state["nu"], state["lengthscale"], state["num_modes"],
            state["bump_scale"], state["bump_decay"],
        )

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
        if chg_bs or chg_bd or chg_rs or chg_alpha: refresh_bump_cloud()
        if chg_bs or chg_bd: update_bump_plot(bump_scale, bump_decay)
        # PSD diagnostic panel refreshes whenever kernel hyperparams change.
        # The matern_kernel object on ctx is rebuilt by the surrounding code
        # for changes to num_modes/nu/lengthscale, so by the time refresh_psd
        # runs it sees the new features() behaviour. bump_scale/bump_decay
        # are passed in directly since the kernel object reads them per-call.
        if refresh_psd is not None and (chg_K or chg_nu or chg_ls or chg_bs or chg_bd):
            refresh_psd(nu, lengthscale, num_modes, bump_scale, bump_decay)

    viewer.window.add_dock_widget(controls, name="kernel controls", area="right")

    k_layer.events.visible.connect(lambda e: refresh_K() if k_layer.visible else None)
    if j_layer is not None: j_layer.events.visible.connect(lambda e: refresh_J() if j_layer.visible else None)
    if l_layer is not None: l_layer.events.visible.connect(lambda e: refresh_L() if l_layer.visible else None)
    g_dense_layer.events.visible.connect(lambda e: refresh_G_dense() if g_dense_layer.visible else None)
    d_dense_layer.events.visible.connect(lambda e: refresh_D_dense() if d_dense_layer.visible else None)
    k_dense_layer.events.visible.connect(lambda e: refresh_K_dense() if k_dense_layer.visible else None)
    bump_cloud_layer.events.visible.connect(lambda e: refresh_bump_cloud() if bump_cloud_layer.visible else None)

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
        if bump_cloud_layer.visible: refresh_bump_cloud()
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