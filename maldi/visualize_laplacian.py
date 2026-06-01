"""Standalone kernel & graph visualization for the manifold GP.

Layers — all share the same coordinate frame (full-resolution template
voxel coords). Toggle visibility in napari's left panel.

  Layer A  — KNN graph fabric
    A1: every node as a faint point
    A2: a subsample of edges as faint gray lines

  Layer B  — Graph Laplacian (operator-level)
    B1: nodes colored by diag(L)[i] (uniform for normalized L — usually
        more informative is Layer H below)
    B2: edges colored by L[i, j] (off by default — harder to read)

  Layer C  — Kernel diagonal Σ_k w(λ_k) φ_k(i)² (prior variance per node)

  Layer D  — Laplacian response: L · δ_src
    Sharp/local; nonzero only at src and its KNN neighbors.

  Layer E  — Euclidean Matern K_ν,ℓ(src, ·)   [REPLACED]
    The covariance an *Euclidean* Matern GP would assign from src.
    Reference for comparison with the manifold version (Layer F).

  Layer F  — Manifold Matern K_ν,ℓ(src, ·)   [REPLACED]
    The covariance your library's RiemannMaternKernel actually computes,
    at the training nodes. Compare with Layer E to see what the manifold
    structure buys you.

  Layer G  — Single eigenvector inspector: φ_k(i)

  Layer H  — Weighted node degree D_i (real per-node connectivity)
    Replaces Layer B1 for normalized Laplacians where the diagonal is
    uniform. Shows where the graph is denser vs sparser.

  Layer J  — Manifold Matern at dense stride (Nyström interpolation)
    Same kernel as Layer F but rendered at a finer voxel grid
    (independent of training stride). Off by default; expensive.

  Layer K  — L · density (graph Laplacian applied to reference image)
    Edge-detector view of the operator on the image intensity values.

  Layer L  — Euclidean Matern at dense stride
    Same kernel as Layer E but rendered at a finer voxel grid.
    Toggle J ↔ L for an A/B comparison of manifold vs Euclidean.

  Per-source layers (lines, on top of the fabric):
    KNN edges, Matern (Euclidean) kernel, Riemann (manifold) kernel.

Controls (dock widget, right side):
  src_pick               — active source for D, E, F, J, L, per-source lines
  num_modes              — # eigenmodes for C, F, J, Riemann
  nu                     — Matern smoothness ν (integer; library default 2)
  lengthscale            — Matern lengthscale ℓ
  eigvec_idx             — which eigenvector G shows
  gamma                  — visual stretch on color mappings
  render_stride          — voxel stride for Layers J, L (1 = full-res, slow)
  bump_scale             — bump-function support radius (× graphbandwidth)
  bump_decay             — bump-function boundary softness
  density_smooth_sigma   — Gaussian smoothing applied before L·density
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.cm as cm

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges, labels_for_nodes_from_sub_atlas
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
)
from utils import crop_or_stride_volume, reference_ccf_from_subvolume

# Bump function — try the library; fall back to a clean stub if not exposed.
try:
    from manifold_gp.utils import bump_function as _lib_bump_function
    _USING_LIB_BUMP = True
except ImportError:
    _USING_LIB_BUMP = False


def bump_function(d, scale, decay):
    """Smooth compact-support bump: ~1 at d=0, fades to 0 at d=scale.

    Tries the library's implementation; falls back to a standard mollifier.
    `d` is distance (sqrt of squared distance), `scale` is the support
    radius, `decay` controls boundary softness (smaller → sharper).
    """
    if _USING_LIB_BUMP:
        # Library expects torch tensors for scale/decay (calls .square() etc.)
        d_t = d if torch.is_tensor(d) else torch.as_tensor(d)
        scale_t = (scale if torch.is_tensor(scale)
                   else torch.as_tensor(float(scale), dtype=d_t.dtype,
                                        device=d_t.device))
        decay_t = (decay if torch.is_tensor(decay)
                   else torch.as_tensor(float(decay), dtype=d_t.dtype,
                                        device=d_t.device))
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
    p.add_argument("--cross-region-inflation", type=float, default=10.0,
                   help=("For --knn-method=faiss_atlas_weighted only. "
                         "Multiplier applied to squared Euclidean "
                         "distance on edges that connect two atlas "
                         "regions. Default 10 = mild prior; try 100 "
                         "for a strong prior, 1 for none. The "
                         "underlying graph topology is identical to "
                         "pure faiss; only the edge weights change."))
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

    # Matern hyperparameters (live-adjustable)
    p.add_argument("--nu", type=int, default=2,
                   help="Matern smoothness ν (integer; library default 2).")
    p.add_argument("--lengthscale", type=float, default=1.0)

    # KEPT for CLI compatibility, no longer used (was Layer E/F diffusion knob)
    p.add_argument("--diffusion-t", type=float, default=1.0,
                   help="DEPRECATED. Layers E/F are now Matern kernels.")

    # Per-source comparison view
    p.add_argument("--n-sources", type=int, default=4)
    p.add_argument("--source-seed", type=int, default=0)
    p.add_argument("--n-targets", type=int, default=50)
    p.add_argument("--target-strategy",
                   choices=["random", "stratified"], default="stratified")
    p.add_argument("--k-show", type=int, default=30)
    p.add_argument("--source-marker-size", type=float, default=6.0)
    p.add_argument("--knn-color-by", choices=["heat", "distance"], default="heat")

    # Fabric (Layer A) + Laplacian edges (Layer B2)
    p.add_argument("--fabric-edge-sample", type=int, default=200_000)
    p.add_argument("--fabric-node-size", type=float, default=0.6)
    p.add_argument("--fabric-edge-width", type=float, default=0.3)
    p.add_argument("--laplacian-edge-sample", type=int, default=80_000)

    # Dense-grid kernel rendering (Layers J, L)
    p.add_argument("--render-stride", type=int, default=1,
                   help="Voxel stride for Layers J, L (Nyström dense kernel).")
    p.add_argument("--bump-scale", type=float, default=3.0,
                   help="Bump-function support radius, in units of graphbandwidth.")
    p.add_argument("--bump-decay", type=float, default=0.05,
                   help="Bump-function boundary softness (smaller = sharper).")
    p.add_argument("--density-smooth-sigma", type=float, default=0.0,
                   help="Gaussian σ (sub-volume voxels) applied to density "
                        "before computing L·density. 0 disables.")
    p.add_argument("--nystrom-batch-size", type=int, default=20_000,
                   help="Query-point batch size for Layer J's Nyström "
                        "interpolation. Reduce if you hit GPU OOM.")
    p.add_argument("--dense-max-render-points", type=int, default=300_000,
                   help="Maximum number of points rendered for Layers J / L "
                        "(dense kernels). If the dense grid has more "
                        "significant-value points, a random subsample is "
                        "drawn each refresh.")
    p.add_argument("--dense-render-threshold-frac", type=float, default=5e-3,
                   help="Threshold for which points are rendered in J / L, "
                        "expressed as a fraction of the kernel's max value. "
                        "Points below this are not drawn. 0 = render all.")

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
            coords=reference_nodes,
            k=args["knn_k"], nlist=args["n_list"],
            extra=graph_key_parts, device=args["device"],
            force_recompute=args["force_recompute_graph"],
        )
    elif args["knn_method"] == "anatomical_atlas":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas",
            volume=sub_volume, threshold=args["threshold"],
            atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes,
            k=args["knn_k"], nlist=args["n_list"],
            extra=graph_key_parts, device=args["device"],
            force_recompute=args["force_recompute_graph"],
        )
    elif args["knn_method"] == "faiss_atlas_weighted":
        # ---- 1. Get the base faiss graph ---------------------------
        # Build (or load from cache) the *exact same* faiss kNN graph
        # that --knn-method=faiss would produce, so caches are shared
        # across the two methods.
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"  # cache under the faiss key
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss",
            coords=reference_nodes,
            k=args["knn_k"], nlist=args["n_list"],
            extra=base_key_parts, device=args["device"],
            force_recompute=args["force_recompute_graph"],
        )
        # ---- 2. Look up each node's atlas region -------------------
        # Use the helper in anatomical_knn so other models can do the
        # same conversion (sub_volume + sub_atlas + threshold → labels
        # in node order).
        if sub_atlas is None:
            raise ValueError(
                "--knn-method=faiss_atlas_weighted requires "
                "--annotations-file; got none."
            )
        node_labels = labels_for_nodes_from_sub_atlas(
            sub_volume, sub_atlas, args["threshold"],
        )
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Mismatch between labelled voxels ({node_labels.shape[0]}) "
                f"and graph nodes ({knn.x.shape[0]}). The atlas and "
                f"reference template were probably cropped/strided "
                f"differently — check that they have the same shape."
            )

        # ---- 3. Inflate cross-region edges via library function -----
        # Topology is unchanged; only edge_value gets reweighted.
        inflation = float(args.get("cross_region_inflation", 10.0))
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=True,
        )

        # ---- 4. Re-key the eigenvector cache -----------------------
        # Same graph topology but different edge weights → different
        # Laplacian → different eigenmodes. Encode the inflation in
        # graph_key so the eigvec cache doesn't collide with vanilla
        # faiss runs.
        graph_key_parts["weighting"] = f"atlas_x{inflation:g}"
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"unknown knn_method: {args['knn_method']}")

    print(f"[DEBUG] operator_dimension (knn.x.shape[0]): {knn.x.shape[0]}")
    print(f"[DEBUG] edge_index min: {edge_index.min().item()}, max: {edge_index.max().item()}")

    if edge_index.max().item() >= knn.x.shape[0] or edge_index.min().item() < 0:
        raise ValueError(
            f"CRITICAL: edge_index contains out-of-bounds indices! "
            f"Valid range is [0, {knn.x.shape[0] - 1}], but edge_index ranges "
            f"from {edge_index.min().item()} to {edge_index.max().item()}."
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

    print(f"eigvec shape:          {tuple(eigvec.shape)}")  # (N_nodes, num_modes)
    print(f"eigval shape:          {tuple(eigval.shape)}")
    print(f"NaN in eigvec:         {torch.isnan(eigvec).any().item()}")
    print(f"NaN in eigval:         {torch.isnan(eigval).any().item()}")
    print(f"eigval range:          [{eigval.min().item():.4g}, {eigval.max().item():.4g}]")
    print(f"first 10 eigvals:      {eigval[:10].cpu().numpy()}")

    # Orthonormality check — eigvec.T @ eigvec should be ~I for a well-conditioned eigensolve
    inner = eigvec.T @ eigvec
    off_diag = inner - torch.eye(inner.shape[0], device=inner.device)
    print(f"|eigvec.T @ eigvec - I|_max (off-diag): {off_diag.abs().max().item():.4g}")

    # Eigval ordering — should be non-decreasing
    diffs = eigval[1:] - eigval[:-1]
    print(f"min eigval gap (should be >= 0): {diffs.min().item():.4g}")

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
        # Exposed for Nyström dense-grid queries (Layers J, L)
        coord_mean=coord_mean, coord_std=coord_std,
        voxel_offset=voxel_offset, voxel_scale_mm=voxel_scale_mm,
        stride=int(args["stride"]),
        threshold=int(args["threshold"]),
    )


# =============================================================================
# Edge subsampling + line construction
# =============================================================================
def subsample_edges(
    edge_index: torch.Tensor, edge_value: torch.Tensor,
    max_edges: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def make_lines_array(pairs: np.ndarray, node_positions: np.ndarray) -> np.ndarray:
    lines = np.zeros((pairs.shape[0], 2, 3), dtype=np.float32)
    lines[:, 0, :] = node_positions[pairs[:, 0]]
    lines[:, 1, :] = node_positions[pairs[:, 1]]
    return lines


# =============================================================================
# Laplacian + spectral computations
# =============================================================================
def laplacian_diag(laplacian_op: GraphLaplacianOperator) -> np.ndarray:
    return laplacian_op.laplacian_diag.detach().cpu().numpy()


def laplacian_offdiag_at_edges(
    laplacian_op: GraphLaplacianOperator,
    edge_index_subset: np.ndarray,
    full_edge_index: torch.Tensor,
) -> np.ndarray:
    triu = laplacian_op.laplacian_triu.detach().cpu().numpy()
    return -triu[edge_index_subset]


def weighted_degree(laplacian_op: GraphLaplacianOperator) -> np.ndarray:
    """Per-node weighted degree D_i — the diagonal of the *unnormalized* L."""
    return laplacian_op.degree_unnorm_mat.detach().cpu().numpy()


def apply_laplacian_to_delta(
    laplacian_op: GraphLaplacianOperator, src_idx: int,
) -> np.ndarray:
    """Layer D: L · δ_src.  Sparse — positive at src, negative at neighbors."""
    N = laplacian_op.operator_dimension
    f = torch.zeros(N, 1, device=laplacian_op.x.device,
                    dtype=laplacian_op.x.dtype)
    f[src_idx] = 1.0
    Lf = laplacian_op._matmul(f).squeeze(-1).cpu().numpy()
    return Lf


def apply_laplacian_to_density(
    laplacian_op: GraphLaplacianOperator,
    sub_volume: np.ndarray,
    node_voxel_idx: np.ndarray,
    sigma: float = 0.0,
) -> np.ndarray:
    """Layer K: L · density.  Density = reference-image intensity per node.
    Optional Gaussian pre-smoothing of the density (in sub-volume voxels)."""
    vol = sub_volume.astype(np.float32, copy=False)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=float(sigma))
    density_per_node = vol[
        node_voxel_idx[:, 0], node_voxel_idx[:, 1], node_voxel_idx[:, 2]
    ]
    device = laplacian_op.x.device
    dtype = laplacian_op.x.dtype
    f = torch.as_tensor(density_per_node, device=device, dtype=dtype).unsqueeze(-1)
    Lf = laplacian_op._matmul(f).squeeze(-1).cpu().numpy()
    return Lf


def kernel_diagonal_from_eigvecs(
    eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: int, lengthscale: float, num_modes: int,
) -> np.ndarray:
    """Layer C: Σ_k w(λ_k) φ_k(i)² — GP prior variance per node."""
    K = int(min(num_modes, eigvec.shape[1]))
    safe_lam = eigval[:K].clamp(min=0.0)
    weight = (2.0 * float(nu) / (float(lengthscale) ** 2) + safe_lam).pow(-float(nu))
    phi_sq = eigvec[:, :K] ** 2
    return (phi_sq * weight).sum(dim=-1).cpu().numpy()


# =============================================================================
# Kernel functions (Matern + Manifold Matern) at training nodes
# =============================================================================
def matern_euclidean_at_source(
    src_coord: torch.Tensor, all_coords: torch.Tensor,
    nu: int, lengthscale: float,
) -> np.ndarray:
    """Euclidean Matern K_ν,ℓ(src, j) at every training node j.
    Distance is in the normalized coord space (same as ctx['reference_nodes']).
    Normalized so K(d=0) = 1.
    """
    from scipy.special import kv, gamma as gamma_fn
    d = ((all_coords - src_coord) ** 2).sum(dim=-1).sqrt().cpu().numpy()
    nu_f = float(nu); ell_f = float(lengthscale)
    out = np.ones_like(d, dtype=np.float64)
    nz = d > 0
    if nz.any():
        z = np.sqrt(2.0 * nu_f) * d[nz] / ell_f
        coef = (2.0 ** (1.0 - nu_f)) / gamma_fn(nu_f)
        out[nz] = coef * (z ** nu_f) * kv(nu_f, z)
    return out


def manifold_matern_at_source(
    src_idx: int, eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: int, lengthscale: float, num_modes: int,
) -> np.ndarray:
    """Manifold Matern K_ν,ℓ(src, j) at every training node j.
    Same formula as your library's RiemannMaternKernel."""
    K = int(min(num_modes, eigvec.shape[1]))
    safe_lam = eigval[:K].clamp(min=0.0)
    weight = (2.0 * float(nu) / (float(lengthscale) ** 2) + safe_lam).pow(-float(nu))
    src_phi = eigvec[src_idx, :K]
    return (weight * src_phi * eigvec[:, :K]).sum(dim=-1).cpu().numpy()


# =============================================================================
# Dense-grid kernel functions (at arbitrary voxel positions, via Nyström)
# =============================================================================
def _full_voxel_to_normalized(
    voxel_idx_full: np.ndarray, ctx: dict,
) -> torch.Tensor:
    """Convert full-template voxel indices → normalized coords matching
    ctx['reference_nodes'].

    >>> VERIFY this against your reference_ccf_from_subvolume <<<
    Assumes its formula is roughly:
        mm(sub_voxel) = sub_voxel * voxel_scale_mm + voxel_offset
    Then full_voxel → sub_voxel via dividing by stride.
    If your version differs, adjust here.
    """
    stride = float(ctx["stride"])
    sub_voxel_frac = voxel_idx_full.astype(np.float32) / stride
    mm = sub_voxel_frac * np.asarray(ctx["voxel_scale_mm"], dtype=np.float32) \
         + np.asarray(ctx["voxel_offset"], dtype=np.float32)
    mm_t = torch.from_numpy(mm)
    return (mm_t - ctx["coord_mean"]) / ctx["coord_std"]


def kernel_at_dense_grid(
    src_idx: int,
    ctx: dict, laplacian_op,
    eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: int, lengthscale: float, num_modes: int,
    render_stride: int,
    nearest_neighbors: int = 10,
    bump_scale: float = 3.0, bump_decay: float = 0.05,
    batch_size: int = 20_000,
) -> tuple:
    """Layer J: Manifold Matern K_ν,ℓ(src, q) at all q on a stride=`render_stride`
    grid over the full template (independent of the training graph stride).

    Memory-bounded: processes query points in batches and computes the kernel
    inline rather than materializing the full (Q, K_modes) eigenvector tensor.
    Peak memory per batch ≈ batch_size · k · K_modes · 4 bytes.
    """
    from scipy.spatial import cKDTree
    K_modes = int(min(num_modes, eigvec.shape[1]))

    tmpl = ctx["template_full"]
    mask = tmpl > ctx["threshold"]
    sub_mask = mask[::render_stride, ::render_stride, ::render_stride]
    sub_idx = np.argwhere(sub_mask).astype(np.int32)
    voxel_idx = sub_idx * render_stride
    Q = voxel_idx.shape[0]

    # Normalized query coords on CPU; we'll move per-batch to GPU.
    coords_z_cpu = _full_voxel_to_normalized(voxel_idx, ctx).numpy()

    # Cache KDTree on training coords across calls
    if "_query_kdt" not in ctx:
        ctx["_query_kdt"] = cKDTree(ctx["reference_nodes"].cpu().numpy())
    kdt = ctx["_query_kdt"]

    # Precompute the "weighted source modes": v_k = w(λ_k) · φ_k(src)
    # The kernel becomes K(src, q) = Σ_k v_k · φ_k_interp(q) — one dot product
    # per query, no need to store all interpolated eigvecs.
    eigvec_K = eigvec[:, :K_modes]
    safe_lam = eigval[:K_modes].clamp(min=0.0)
    weight = (2.0 * float(nu) / (float(lengthscale) ** 2) + safe_lam).pow(-float(nu))
    v_src = (weight * eigvec[src_idx, :K_modes]).contiguous()    # (K_modes,)
    bump_radius = float(laplacian_op.graphbandwidth.squeeze()) * float(bump_scale)

    K_q_all = np.zeros(Q, dtype=np.float32)
    device = eigvec.device
    dtype = eigvec.dtype

    for batch_start in range(0, Q, batch_size):
        batch_end = min(batch_start + batch_size, Q)
        coords_b = coords_z_cpu[batch_start:batch_end]
        # KDTree query (CPU) — small allocation
        dists_b, idxs_b = kdt.query(coords_b, k=nearest_neighbors, workers=-1)
        edge_value_b = torch.from_numpy(
            (dists_b.astype(np.float32) ** 2)
        ).to(device=device, dtype=dtype)
        edge_index_b = torch.from_numpy(idxs_b.astype(np.int64)).to(device)

        sqrt_d_nearest_b = edge_value_b[:, 0].sqrt()
        within_b = sqrt_d_nearest_b < bump_radius

        if within_b.any():
            projected_b = laplacian_op.out_of_sample(
                eigvec_K, edge_value_b[within_b], edge_index_b[within_b],
            )                                                     # (M, K_modes)
            bump_vals_b = bump_function(
                sqrt_d_nearest_b[within_b], bump_radius, float(bump_decay),
            )                                                     # (M,)
            # Apply bump and contract against v_src — kernel values for batch
            kvals_b = (projected_b * bump_vals_b.unsqueeze(-1) * v_src) \
                      .sum(dim=-1).cpu().numpy()                  # (M,)
            # Scatter back into K_q_all at the right (global) positions
            within_mask_cpu = within_b.cpu().numpy()
            global_positions = np.arange(batch_start, batch_end)[within_mask_cpu]
            K_q_all[global_positions] = kvals_b
            del projected_b, bump_vals_b
        del edge_value_b, edge_index_b, sqrt_d_nearest_b, within_b

    # Place into return arrays — values per point + their voxel positions
    return K_q_all, voxel_idx


def kernel_at_dense_grid_volume(
    K_q_flat: np.ndarray, voxel_idx: np.ndarray, template_shape: tuple,
) -> np.ndarray:
    """Place flat kernel values at their voxel positions into a full-template
    volume. Retained for callers that want a 3-D ndarray; not used by the
    Points-based Layers J / L (which don't need it)."""
    K_vol = np.zeros(template_shape, dtype=np.float32)
    K_vol[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = K_q_flat
    return K_vol


def euclidean_kernel_at_dense_grid(
    src_idx: int, ctx: dict,
    nu: int, lengthscale: float,
    render_stride: int,
) -> tuple:
    """Layer L: Euclidean Matern K_ν,ℓ(d) at all voxels on a stride=`render_stride`
    grid. Distance is Euclidean in the same normalized coord space."""
    from scipy.special import kv, gamma as gamma_fn
    nu_f = float(nu); ell_f = float(lengthscale)

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
        z = np.sqrt(2.0 * nu_f) * d[nz] / ell_f
        coef = (2.0 ** (1.0 - nu_f)) / gamma_fn(nu_f)
        out[nz] = coef * (z ** nu_f) * kv(nu_f, z)

    K_vals = out.astype(np.float32)
    return K_vals, voxel_idx


# =============================================================================
# Per-source kernel functions (for the on-top line layers)
# =============================================================================
def matern_euclidean_kernel(
    src_coord: torch.Tensor, tgt_coords: torch.Tensor,
    lengthscale: float, nu: float = 1.0,
) -> np.ndarray:
    """Closed-form Euclidean Matern for the per-source line view.
    Handles ν ∈ {0.5, 1.5, 2.5, else} with closed forms; falls back to RBF."""
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
    src_idx: int, tgt_idxs: np.ndarray,
    eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: float, lengthscale: float, num_modes: int,
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
    src_idx: int, reference_nodes: torch.Tensor,
    n_targets: int, strategy: str, seed: int,
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
    src_idx: int, edge_index: torch.Tensor, edge_value: torch.Tensor,
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
    source_seed: int, nu: float, knn_color_by: str,
) -> dict:
    src_voxel = ctx["node_voxel_idx"][src_idx]
    knn_idxs, knn_sq_dists = knn_neighbors_of(
        int(src_idx), ctx["edge_index"], ctx["edge_value"], k_show,
    )
    if knn_color_by == "distance":
        knn_dist = np.sqrt(knn_sq_dists)
        dmax = max(knn_dist.max(), 1e-8)
        knn_strengths = 1.0 - knn_dist / dmax
    else:
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
# Layer management — per-source shape layers
# =============================================================================
LAYER_CONFIG = {
    "knn":     dict(name="src KNN edges",        visible=False),
    "matern":  dict(name="src Matern (Eucl.)",   visible=False),
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
# Node coloring helpers
# =============================================================================
def color_nodes_sequential(
    layer, values: np.ndarray, gamma: float, cmap_name: str = "magma",
):
    vmin, vmax = float(values.min()), float(values.max())
    if vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1) ** gamma
    else:
        norm = np.zeros_like(values, dtype=np.float32)
    colors = cm.get_cmap(cmap_name)(norm)
    layer.face_color = colors
    layer.border_color = colors


def color_nodes_diverging(
    layer, values: np.ndarray, cmap_name: str = "RdBu_r",
    pct: float = 99.0, gamma: float = 0.5,
):
    """Recolor a Points layer with a diverging colormap centered at 0.

    Uses the `pct`-th percentile of |values| for saturation (not the
    max — outliers would otherwise wash everything else to white) and
    applies `gamma` to amplify small magnitudes. With gamma < 1, weak
    signals become visible; gamma = 1 is linear.
    """
    abs_values = np.abs(values)
    amax = max(float(np.percentile(abs_values, pct)), 1e-12)
    sign = np.sign(values)
    rel = np.clip(np.abs(values) / amax, 0, 1) ** float(gamma)
    norm = 0.5 + 0.5 * sign * rel
    norm = np.clip(norm, 0, 1)
    colors = cm.get_cmap(cmap_name)(norm)
    layer.face_color = colors
    layer.border_color = colors


def color_nodes_signed_sparse(
    layer, values: np.ndarray, cmap_name: str = "RdBu_r", threshold: float = 1e-12,
):
    """Sparse-signed coloring: positive and negative get separate amax scaling,
    near-zero values become transparent. For Layer D where one large positive
    spike would otherwise wash out small negatives."""
    cmap = cm.get_cmap(cmap_name)
    pos = values > threshold
    neg = values < -threshold
    pos_max = float(values[pos].max()) if pos.any() else 1.0
    neg_min = float(values[neg].min()) if neg.any() else -1.0
    norm = np.full_like(values, 0.5, dtype=np.float64)
    if pos.any():
        norm[pos] = 0.5 + 0.5 * (values[pos] / pos_max)
    if neg.any():
        norm[neg] = 0.5 - 0.5 * (values[neg] / neg_min)
    colors = cmap(norm)
    colors[:, 3] = np.where(np.abs(values) > threshold, 1.0, 0.0)
    layer.face_color = colors
    layer.border_color = colors


# =============================================================================
# Per-layer info panel (text dock at the bottom of the viewer)
# =============================================================================
def make_layer_info_panel():
    """Return (widget, set_fn, refresh_fn) — a read-only monospaced text panel
    that shows per-layer value ranges and rendering metadata. Each layer's
    refresh function calls `set_fn(tag, info_string)`; the panel re-renders
    on each call."""
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

    # Preserve insertion order so layers appear in the dock in the same
    # order they're created in main().
    info_dict: "dict[str, str]" = {}

    def render():
        # Use HTML so we can right-align numbers and dim non-updated lines.
        lines = []
        for tag, info in info_dict.items():
            lines.append(f"<span>{tag:<6}</span> {info}")
        text.setHtml("<pre>" + "\n".join(lines) + "</pre>")

    def set_info(tag: str, info: str):
        info_dict[tag] = info
        render()

    return widget, set_info, render


def fmt_info_sequential(vmin, vmax, sat, gamma, cmap):
    return (f"range=[{vmin:>+10.3g}, {vmax:>+10.3g}]  "
            f"sat={sat:>+10.3g}  γ={gamma:>4.2f}  cmap={cmap}")


def fmt_info_diverging(vmin, vmax, sat, gamma, cmap):
    return (f"range=[{vmin:>+10.3g}, {vmax:>+10.3g}]  "
            f"sat=±{sat:>9.3g}  γ={gamma:>4.2f}  cmap={cmap}")


def fmt_info_sparse_signed(vmin, vmax, pos_max, neg_min, cmap):
    return (f"range=[{vmin:>+10.3g}, {vmax:>+10.3g}]  "
            f"+sat={pos_max:>+9.3g}  −sat={neg_min:>+9.3g}  cmap={cmap}")


# =============================================================================
# Robust Image-layer creation (handles environment-specific vispy failures)
# =============================================================================
def safe_add_image(viewer, log, name: str, data, **kwargs):
    """Wrap viewer.add_image with full cleanup on failure.

    Some napari/vispy combinations refuse to create a Volume visual even
    when the data is a valid 3-D array. When this happens, napari has
    *already* inserted the layer into viewer.layers (the model) before
    the vispy visual creation throws; the layer then becomes an orphan
    that breaks any later add_*() call via the reorder hook.

    This helper catches the exception AND removes the orphaned model
    layer, returning None on failure.
    """
    try:
        return viewer.add_image(data, name=name, **kwargs)
    except Exception as exc:
        log.warning(f"Failed to add Image layer '{name}': {exc}")
        # Clean up any model-side orphan
        for lyr in list(viewer.layers):
            if lyr.name == name:
                try:
                    viewer.layers.remove(lyr)
                    log.warning(f"  Removed orphaned model layer '{name}'.")
                except Exception as cleanup_exc:
                    log.warning(f"  Could not remove orphan: {cleanup_exc}")
        return None


# =============================================================================
# Bump-function plot widget (matplotlib in a Qt dock)
# =============================================================================
def make_bump_widget(graphbandwidth: float, initial_scale: float, initial_decay: float):
    try:
        from qtpy.QtWidgets import QWidget, QVBoxLayout
        from matplotlib.figure import Figure
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvas
        except ImportError:
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
    except ImportError:
        return None, (lambda *_args, **_kw: None)

    bw = float(graphbandwidth)
    fig = Figure(figsize=(4.0, 2.5), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    d_grid = np.linspace(0.0, 5.0 * bw * max(initial_scale, 1.0), 400)

    (line,) = ax.plot([], [], "-", color="#3578a8", lw=1.5)
    v_bw = ax.axvline(bw, color="#888888", ls="--", lw=0.7)
    v_supp = ax.axvline(initial_scale * bw, color="#cc3344", ls="--", lw=0.7)

    ax.set_xlim(0, max(5.0 * bw, initial_scale * bw * 1.2))
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("distance to nearest training node")
    ax.set_ylabel("bump value")
    ax.grid(alpha=0.3)

    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(220)

    def update(scale, decay):
        vals = bump_function(
            torch.from_numpy(d_grid).float(),
            float(scale) * bw, float(decay),
        ).cpu().numpy()
        line.set_data(d_grid, vals)
        v_supp.set_xdata([float(scale) * bw, float(scale) * bw])
        ax.set_title(
            f"bump (scale={scale:.2g}, decay={decay:.3g}), bw={bw:.3g}",
            fontsize=9,
        )
        ax.legend([v_bw, v_supp],
                  [f"graphbandwidth = {bw:.3g}",
                   f"support = {scale:.2g}·bw"],
                  fontsize=7, loc="upper right")
        ax.set_xlim(0, max(5.0 * bw, float(scale) * bw * 1.2))
        canvas.draw_idle()

    update(initial_scale, initial_decay)
    widget = QWidget()
    layout = QVBoxLayout(widget); layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(canvas)
    return widget, update


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

    all_node_positions = (
        ctx["node_voxel_idx"].astype(np.float32) * ctx["sv_scale"]
        + ctx["sv_translate"]
    )
    N = all_node_positions.shape[0]
    log.info(f"{N:,} graph nodes in full-res coords")

    rng = np.random.default_rng(args["source_seed"])
    src_idxs = rng.choice(N, size=args["n_sources"], replace=False)
    log.info(f"Sources for kernel comparison: {src_idxs.tolist()}")

    log.info("Subsampling graph edges for Layer A (fabric)...")
    fabric_pairs, fabric_sq_dists, _ = subsample_edges(
        ctx["edge_index"], ctx["edge_value"],
        max_edges=args["fabric_edge_sample"], seed=args["source_seed"],
    )
    log.info(f"  fabric: {fabric_pairs.shape[0]:,} edges")

    log.info("Subsampling graph edges for Layer B2 (Laplacian-colored)...")
    lap_pairs, lap_sq_dists, lap_full_idx = subsample_edges(
        ctx["edge_index"], ctx["edge_value"],
        max_edges=args["laplacian_edge_sample"],
        seed=args["source_seed"] + 1,
    )
    log.info(f"  laplacian: {lap_pairs.shape[0]:,} edges")

    log.info("Computing Laplacian diagonal + edge entries...")
    lap_diag_vals = laplacian_diag(ctx["laplacian_op"])
    lap_edge_vals = laplacian_offdiag_at_edges(
        ctx["laplacian_op"], lap_full_idx, ctx["edge_index"],
    )
    log.info(
        f"  diag range: [{lap_diag_vals.min():.4g}, {lap_diag_vals.max():.4g}], "
        f"offdiag range: [{lap_edge_vals.min():.4g}, {lap_edge_vals.max():.4g}]"
    )

    log.info("Computing weighted node degree (Layer H)...")
    deg_vals = weighted_degree(ctx["laplacian_op"])
    log.info(f"  degree range: [{deg_vals.min():.4g}, {deg_vals.max():.4g}]")

    # ============= Set up the napari viewer ============================
    viewer = napari.Viewer(title="Kernel & graph debugger")
    viewer.dims.ndisplay = 3

    # ---- Layer A1: graph nodes (faint fabric) -----------------------------
    viewer.add_points(
        all_node_positions,
        name="A1: graph nodes",
        size=float(args["fabric_node_size"]),
        face_color="white", border_color="white",
        symbol="o", opacity=0.25, blending="additive",
    )

    # ---- Layer A2: KNN fabric edges (faint gray) --------------------------
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

    # ---- Layer B1: Laplacian diagonal -------------------------------------
    b1_layer = viewer.add_points(
        all_node_positions,
        name="B1: Laplacian diag (often uniform)",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=False,
    )
    color_nodes_sequential(
        b1_layer, lap_diag_vals, gamma=float(args["gamma"]), cmap_name="cividis",
    )

    # ---- Layer B2: Laplacian off-diag (edges, RdBu_r) ---------------------
    lap_edge_lines = make_lines_array(lap_pairs, all_node_positions)
    amax = max(np.abs(lap_edge_vals).max(), 1e-12)
    edge_colors_lap = cm.get_cmap("RdBu_r")(0.5 + 0.5 * lap_edge_vals / amax)
    viewer.add_shapes(
        [lap_edge_lines[i] for i in range(lap_edge_lines.shape[0])],
        shape_type="line",
        edge_color=edge_colors_lap,
        edge_width=0.4,
        name="B2: Laplacian off-diag (edges)",
        opacity=0.75, blending="translucent",
        visible=False,
    )

    # ---- Layer H: weighted degree (NEW — useful per-node connectivity) ----
    h_layer = viewer.add_points(
        all_node_positions,
        name="H: weighted degree D_i",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=False,
    )
    color_nodes_sequential(
        h_layer, deg_vals, gamma=float(args["gamma"]), cmap_name="cividis",
    )

    # ---- Layer C: kernel diagonal (prior variance per node) ---------------
    initial_modes = min(args["initial_modes"] or args["num_modes"],
                        ctx["eigvec"].shape[1])
    c_layer = viewer.add_points(
        all_node_positions,
        name="C: kernel diag K(i, i)",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=False,
    )

    # ---- Layer D: L · δ_src (sharp) ---------------------------------------
    d_layer = viewer.add_points(
        all_node_positions,
        name="D: L · δ_src (sharp, diverging)",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=False,
    )

    # ---- Layer E: Euclidean Matern K_ν,ℓ(src, ·) at training nodes --------
    e_layer = viewer.add_points(
        all_node_positions,
        name="E: Euclidean Matern K(src, ·)",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=True,
    )

    # ---- Layer F: Manifold Matern K_ν,ℓ(src, ·) at training nodes ---------
    f_layer = viewer.add_points(
        all_node_positions,
        name="F: Manifold Matern K(src, ·)",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=True,
    )

    # ---- Layer G: single eigenvector --------------------------------------
    g_layer = viewer.add_points(
        all_node_positions,
        name="G: eigenvector φ_k(i)",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=False,
    )

    # ---- Layer K: L · density (off by default — independent of source) ----
    k_layer = viewer.add_points(
        all_node_positions,
        name="K: L · density",
        size=float(args["fabric_node_size"]) * 1.5,
        face_color="white", border_color="white",
        symbol="o", opacity=0.85, blending="translucent",
        visible=False,
    )

    # ---- Layers J and L: dense kernels rendered as Points (NOT Image) ----
    # The Image / vispy Volume path errors out in this napari/vispy combo
    # with "Volume needs a 3D array" even on valid 3-D data. Points-based
    # rendering bypasses the Volume visual entirely and uses the same
    # machinery that the working Points layers (A1, B1, C, D, E, F, G, H,
    # K) all use. One point per significant voxel; threshold + cap make it
    # tractable at dense strides.
    tpl_shape = tuple(ctx["template_full"].shape)
    j_layer = None
    l_layer = None
    if len(tpl_shape) == 3:
        # Placeholder positions — refresh_J/L will replace these on first
        # toggle-visible.
        placeholder_pts = np.zeros((1, 3), dtype=np.float32)
        j_layer = viewer.add_points(
            placeholder_pts,
            name=f"J: K(src, ·) dense @ stride={args['render_stride']} (Nyström)",
            size=float(args["fabric_node_size"]) * 1.2,
            face_color="white", border_color="white",
            symbol="o", opacity=0.85, blending="translucent",
            visible=False,
        )
        l_layer = viewer.add_points(
            placeholder_pts,
            name=f"L: Euclidean K(d) dense @ stride={args['render_stride']}",
            size=float(args["fabric_node_size"]) * 1.2,
            face_color="white", border_color="white",
            symbol="o", opacity=0.85, blending="translucent",
            visible=False,
        )
    else:
        log.warning(
            f"template_full has shape {tpl_shape} (ndim={len(tpl_shape)}); "
            "Layers J and L require a 3-D template. Skipping both."
        )

    # ---- Source markers ---------------------------------------------------
    src_voxels = ctx["node_voxel_idx"][src_idxs].astype(np.float32)
    src_points = src_voxels * ctx["sv_scale"] + ctx["sv_translate"]
    viewer.add_points(
        src_points, name="source nodes",
        size=float(args["source_marker_size"]),
        face_color="red", border_color="white", symbol="o", opacity=0.95,
    )

    # ===== Sliders ========================================================
    num_modes_max = ctx["eigvec"].shape[1]

    state = dict(
        src_pick=0,
        num_modes=initial_modes,
        nu=int(args["nu"]),
        lengthscale=float(args["lengthscale"]),
        eigvec_idx=0,
        gamma=float(args["gamma"]),
        render_stride=int(args["render_stride"]),
        bump_scale=float(args["bump_scale"]),
        bump_decay=float(args["bump_decay"]),
        density_smooth_sigma=float(args["density_smooth_sigma"]),
    )

    layer_state = {"knn": None, "matern": None, "riemann": None}

    def current_src() -> int:
        return int(src_idxs[state["src_pick"] % len(src_idxs)])

    # ---- Per-layer info panel (bottom dock) ------------------------------
    # Created here, BEFORE refresh_X definitions, so set_info exists in the
    # enclosing scope when refresh functions reference it.
    info_panel, set_info, refresh_info = make_layer_info_panel()
    if info_panel is not None:
        viewer.window.add_dock_widget(
            info_panel, name="layer info", area="bottom",
        )
        # Seed all layer entries so they appear in the dock from the start
        for tag, name in [
            ("A1", "graph nodes (fabric)"),
            ("A2", "KNN fabric edges"),
            ("B1", "Laplacian diag"),
            ("B2", "Laplacian off-diag (edges)"),
            ("H",  "weighted degree D_i"),
            ("C",  "kernel diag K(i, i)"),
            ("D",  "L · δ_src"),
            ("E",  "Eucl. Matern K(src, ·)"),
            ("F",  "Manif. Matern K(src, ·)"),
            ("G",  "eigenvector φ_k"),
            ("K",  "L · density"),
            ("J",  "Manif. K dense (Nyström)"),
            ("L",  "Eucl. K dense"),
        ]:
            set_info(tag, f"({name})  — not yet rendered —")
        # Pre-populate B1/B2/H/A1/A2 from startup computations
        set_info("A1", f"({all_node_positions.shape[0]:,} training nodes)")
        set_info("A2", f"({fabric_pairs.shape[0]:,} edges)")
        set_info("B1", fmt_info_sequential(
            float(lap_diag_vals.min()), float(lap_diag_vals.max()),
            float(np.percentile(np.abs(lap_diag_vals), 99)),
            float(args["gamma"]), "cividis",
        ))
        set_info("B2", fmt_info_diverging(
            float(lap_edge_vals.min()), float(lap_edge_vals.max()),
            float(np.percentile(np.abs(lap_edge_vals), 99)),
            float(args["gamma"]), "RdBu_r",
        ))
        set_info("H", fmt_info_sequential(
            float(deg_vals.min()), float(deg_vals.max()),
            float(np.percentile(np.abs(deg_vals), 99)),
            float(args["gamma"]), "cividis",
        ))

    # ---- Refresh: per-source line layers ----------------------------------
    def refresh_per_source():
        s = current_src()
        data = build_lines_for_source(
            s, ctx, num_modes=state["num_modes"],
            lengthscale=state["lengthscale"], gamma=state["gamma"],
            k_show=args["k_show"], n_targets=args["n_targets"],
            target_strategy=args["target_strategy"],
            source_seed=args["source_seed"], nu=state["nu"],
            knn_color_by=args["knn_color_by"],
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

    # ---- Refresh: Layer C -------------------------------------------------
    def refresh_C():
        d = kernel_diagonal_from_eigvecs(
            ctx["eigval"], ctx["eigvec"],
            nu=state["nu"], lengthscale=state["lengthscale"],
            num_modes=state["num_modes"],
        )
        color_nodes_sequential(c_layer, d, state["gamma"], "magma")
        sat = float(np.percentile(np.abs(d), 99))
        set_info("C", fmt_info_sequential(
            float(d.min()), float(d.max()), sat, state["gamma"], "magma",
        ))
        print(f"[C ν={state['nu']} ℓ={state['lengthscale']:.3g} "
              f"K={state['num_modes']}]  K(i,i) range "
              f"[{d.min():.4g}, {d.max():.4g}]")

    # ---- Refresh: Layer D -------------------------------------------------
    def refresh_D():
        s = current_src()
        Lf = apply_laplacian_to_delta(ctx["laplacian_op"], s)
        color_nodes_signed_sparse(d_layer, Lf, "RdBu_r")
        nz = int((np.abs(Lf) > 1e-12).sum())
        pos = Lf[Lf > 1e-12]
        neg = Lf[Lf < -1e-12]
        pos_max = float(pos.max()) if pos.size else 0.0
        neg_min = float(neg.min()) if neg.size else 0.0
        set_info("D", fmt_info_sparse_signed(
            float(Lf.min()), float(Lf.max()), pos_max, neg_min, "RdBu_r",
        ) + f"  #nonzero={nz}  (src={s})")
        print(f"[D L·δ_{s}]  range [{Lf.min():.4g}, {Lf.max():.4g}], "
              f"#nonzero={nz}")

    # ---- Refresh: Layer E (Euclidean Matern at training nodes) ------------
    def refresh_E():
        s = current_src()
        k_eu = matern_euclidean_at_source(
            ctx["reference_nodes"][s], ctx["reference_nodes"],
            nu=state["nu"], lengthscale=state["lengthscale"],
        )
        color_nodes_sequential(e_layer, k_eu, state["gamma"], "magma")
        sat = float(np.percentile(np.abs(k_eu), 99))
        set_info("E", fmt_info_sequential(
            float(k_eu.min()), float(k_eu.max()), sat, state["gamma"], "magma",
        ) + f"  (ν={state['nu']} ℓ={state['lengthscale']:.3g} src={s})")
        print(f"[E Eucl. Matern  ν={state['nu']}, ℓ={state['lengthscale']:.3g}, "
              f"src={s}]  range [{k_eu.min():.4g}, {k_eu.max():.4g}]")

    # ---- Refresh: Layer F (Manifold Matern at training nodes) -------------
    def refresh_F():
        s = current_src()
        k_mf = manifold_matern_at_source(
            s, ctx["eigval"], ctx["eigvec"],
            nu=state["nu"], lengthscale=state["lengthscale"],
            num_modes=state["num_modes"],
        )
        sat = float(np.percentile(np.abs(k_mf), 99))
        if (k_mf < 0).any():
            color_nodes_diverging(f_layer, k_mf, "RdBu_r", gamma=state["gamma"])
            set_info("F", fmt_info_diverging(
                float(k_mf.min()), float(k_mf.max()), sat, state["gamma"],
                "RdBu_r (diverging — truncation negatives)",
            ) + f"  (ν={state['nu']} ℓ={state['lengthscale']:.3g} src={s} K={state['num_modes']})")
        else:
            color_nodes_sequential(f_layer, k_mf, state["gamma"], "magma")
            set_info("F", fmt_info_sequential(
                float(k_mf.min()), float(k_mf.max()), sat, state["gamma"], "magma",
            ) + f"  (ν={state['nu']} ℓ={state['lengthscale']:.3g} src={s} K={state['num_modes']})")
        print(f"[F Manif. Matern  ν={state['nu']}, ℓ={state['lengthscale']:.3g}, "
              f"K={state['num_modes']}, src={s}]  "
              f"range [{k_mf.min():.4g}, {k_mf.max():.4g}]")

    # ---- Refresh: Layer G -------------------------------------------------
    def refresh_G():
        k = int(state["eigvec_idx"])
        phi = ctx["eigvec"][:, k].cpu().numpy()
        color_nodes_diverging(g_layer, phi, "RdBu_r", gamma=state["gamma"])
        lam = float(ctx["eigval"][k].item())
        sat = float(np.percentile(np.abs(phi), 99))
        set_info("G", fmt_info_diverging(
            float(phi.min()), float(phi.max()), sat, state["gamma"], "RdBu_r",
        ) + f"  (k={k}, λ={lam:.4g})")
        print(f"[G φ_{k}, λ={lam:.4g}]  range [{phi.min():.4g}, {phi.max():.4g}]")

    # ---- Refresh: Layer K (L · density) -----------------------------------
    def refresh_K():
        Lf = apply_laplacian_to_density(
            ctx["laplacian_op"], ctx["sub_volume"], ctx["node_voxel_idx"],
            sigma=state["density_smooth_sigma"],
        )
        color_nodes_diverging(k_layer, Lf, "RdBu_r", gamma=state["gamma"])
        sat = float(np.percentile(np.abs(Lf), 99))
        set_info("K", fmt_info_diverging(
            float(Lf.min()), float(Lf.max()), sat, state["gamma"], "RdBu_r",
        ) + f"  (σ={state['density_smooth_sigma']:.2g})")
        print(f"[K L·density σ={state['density_smooth_sigma']:.2g}]  "
              f"range [{Lf.min():.4g}, {Lf.max():.4g}], sat ±{sat:.4g}")

    # ---- Refresh: Layer J (Manifold Matern at dense grid, as Points) ------
    rng_render = np.random.default_rng(args["source_seed"])

    def _update_dense_points_layer(layer, K_q_flat, voxel_idx, label, tag):
        """Thresholds, subsamples, and colors a dense-kernel Points layer."""
        if K_q_flat.size == 0:
            set_info(tag, f"(empty result)")
            return
        vmax = float(np.abs(K_q_flat).max()) if K_q_flat.size else 0.0
        thresh = vmax * float(args["dense_render_threshold_frac"])
        mask = np.abs(K_q_flat) > max(thresh, 1e-12)
        K_keep = K_q_flat[mask]
        idx_keep = voxel_idx[mask]
        # Cap to max_render_points to keep napari responsive
        max_pts = int(args["dense_max_render_points"])
        if idx_keep.shape[0] > max_pts:
            sel = rng_render.choice(idx_keep.shape[0], size=max_pts, replace=False)
            K_keep = K_keep[sel]
            idx_keep = idx_keep[sel]
        # Place points at full-res voxel coords (same frame as A1/A2/etc.)
        positions = idx_keep.astype(np.float32)
        if positions.shape[0] == 0:
            # Hide via single placeholder if no significant points
            layer.data = np.zeros((1, 3), dtype=np.float32)
            layer.face_color = np.array([[0, 0, 0, 0]], dtype=np.float32)
            set_info(tag, f"no points above threshold ({thresh:.3g})")
            print(f"[{label}]  no points above threshold; nothing rendered")
            return
        layer.data = positions
        sat = float(np.percentile(np.abs(K_keep), 99))
        # Color sequentially by value with magma
        if (K_keep < 0).any():
            color_nodes_diverging(layer, K_keep, "RdBu_r", gamma=state["gamma"])
            info_line = fmt_info_diverging(
                float(K_q_flat.min()), float(K_q_flat.max()),
                sat, state["gamma"], "RdBu_r",
            )
        else:
            color_nodes_sequential(layer, K_keep, state["gamma"], "magma")
            info_line = fmt_info_sequential(
                float(K_q_flat.min()), float(K_q_flat.max()),
                sat, state["gamma"], "magma",
            )
        set_info(tag,
                 info_line +
                 f"  ({positions.shape[0]:,} of {K_q_flat.size:,} pts rendered, "
                 f"thresh={thresh:.3g})")
        print(f"[{label}]  rendered {positions.shape[0]:,} points  "
              f"range [{K_q_flat.min():.4g}, {K_q_flat.max():.4g}], "
              f"threshold={thresh:.4g}")

    def refresh_J():
        if j_layer is None:
            return
        s = current_src()
        K_q, voxel_idx = kernel_at_dense_grid(
            src_idx=s, ctx=ctx, laplacian_op=ctx["laplacian_op"],
            eigval=ctx["eigval"], eigvec=ctx["eigvec"],
            nu=state["nu"], lengthscale=state["lengthscale"],
            num_modes=state["num_modes"],
            render_stride=state["render_stride"],
            bump_scale=state["bump_scale"], bump_decay=state["bump_decay"],
            batch_size=int(args["nystrom_batch_size"]),
        )
        label = (f"J Manif. K(src={s},·) dense @ stride={state['render_stride']}, "
                 f"ν={state['nu']}, ℓ={state['lengthscale']:.3g}")
        _update_dense_points_layer(j_layer, K_q, voxel_idx, label, "J")

    # ---- Refresh: Layer L (Euclidean Matern at dense grid, as Points) -----
    def refresh_L():
        if l_layer is None:
            return
        s = current_src()
        K_q, voxel_idx = euclidean_kernel_at_dense_grid(
            src_idx=s, ctx=ctx,
            nu=state["nu"], lengthscale=state["lengthscale"],
            render_stride=state["render_stride"],
        )
        label = (f"L Eucl. K(src={s},·) dense @ stride={state['render_stride']}, "
                 f"ν={state['nu']}, ℓ={state['lengthscale']:.3g}")
        _update_dense_points_layer(l_layer, K_q, voxel_idx, label, "L")

    # Initial render
    refresh_per_source()
    refresh_C()
    refresh_D()
    refresh_E()
    refresh_F()
    refresh_G()
    # Layer K, J, L start hidden — don't refresh on startup
    # (J and L already rendered once above to populate)

    # ---- Bump-function widget (left dock) ---------------------------------
    bump_widget, update_bump_plot = make_bump_widget(
        ctx["bw"], state["bump_scale"], state["bump_decay"],
    )
    if bump_widget is not None:
        viewer.window.add_dock_widget(
            bump_widget, name="bump function", area="left",
        )

    # ---- Dock widget: all sliders -----------------------------------------
    @magicgui(
        auto_call=True,
        src_pick={"label": "active source",
                  "min": 0, "max": len(src_idxs) - 1, "step": 1},
        num_modes={"label": "num modes (C, F, J, Riemann)",
                   "min": 1, "max": num_modes_max, "step": 1},
        nu={"label": "ν (Matern smoothness, integer)",
            "min": 1, "max": 6, "step": 1},
        lengthscale={"label": "ℓ (Matern lengthscale)",
                     "min": 1e-3, "max": 10.0, "step": 1e-3},
        eigvec_idx={"label": "eigenvector index (G)",
                    "min": 0, "max": num_modes_max - 1, "step": 1},
        gamma={"label": "color gamma",
               "min": 0.1, "max": 2.0, "step": 0.05},
        render_stride={"label": "render stride (J, L)",
                       "min": 1, "max": 8, "step": 1},
        bump_scale={"label": "bump scale (× bw)",
                    "min": 0.001, "max": 200.0, "step": 0.1},
        bump_decay={"label": "bump decay",
                    "min": 0.001, "max": 2.0, "step": 0.001},
        density_smooth_sigma={"label": "L·density: σ (voxels)",
                              "min": 0.0, "max": 10.0, "step": 0.1},
    )
    def controls(
        src_pick: int = state["src_pick"],
        num_modes: int = state["num_modes"],
        nu: int = state["nu"],
        lengthscale: float = state["lengthscale"],
        eigvec_idx: int = state["eigvec_idx"],
        gamma: float = state["gamma"],
        render_stride: int = state["render_stride"],
        bump_scale: float = state["bump_scale"],
        bump_decay: float = state["bump_decay"],
        density_smooth_sigma: float = state["density_smooth_sigma"],
    ):
        chg_src    = src_pick     != state["src_pick"]
        chg_K      = num_modes    != state["num_modes"]
        chg_nu     = nu           != state["nu"]
        chg_ls     = lengthscale  != state["lengthscale"]
        chg_eig    = eigvec_idx   != state["eigvec_idx"]
        chg_gamma  = gamma        != state["gamma"]
        chg_rs     = render_stride != state["render_stride"]
        chg_bs     = bump_scale   != state["bump_scale"]
        chg_bd     = bump_decay   != state["bump_decay"]
        chg_sigma  = density_smooth_sigma != state["density_smooth_sigma"]

        state.update(
            src_pick=src_pick, num_modes=num_modes, nu=nu,
            lengthscale=lengthscale, eigvec_idx=eigvec_idx, gamma=gamma,
            render_stride=render_stride, bump_scale=bump_scale,
            bump_decay=bump_decay, density_smooth_sigma=density_smooth_sigma,
        )

        # Per-source lines: src, K, ν, ℓ, γ
        if chg_src or chg_K or chg_nu or chg_ls or chg_gamma:
            refresh_per_source()

        # Layer C: K, ν, ℓ, γ
        if chg_K or chg_nu or chg_ls or chg_gamma:
            refresh_C()

        # Layer D: src only
        if chg_src:
            refresh_D()

        # Layer E (Euclidean Matern at training): src, ν, ℓ, γ
        if chg_src or chg_nu or chg_ls or chg_gamma:
            refresh_E()

        # Layer F (Manifold Matern at training): src, K, ν, ℓ, γ
        if chg_src or chg_K or chg_nu or chg_ls or chg_gamma:
            refresh_F()

        # Layer G: eigvec_idx only
        if chg_eig:
            refresh_G()

        # Layer K (L·density): density_smooth_sigma only — independent of src.
        # Only refresh if the layer is visible (it's expensive).
        if k_layer.visible and chg_sigma:
            refresh_K()

        # Layer J (dense manifold): src, K, ν, ℓ, render_stride, bump_*.
        # Only refresh if layer exists and is visible (very expensive).
        if j_layer is not None and j_layer.visible and (
            chg_src or chg_K or chg_nu or chg_ls
            or chg_rs or chg_bs or chg_bd
        ):
            refresh_J()

        # Layer L (dense Euclidean): src, ν, ℓ, render_stride. No bump.
        if l_layer is not None and l_layer.visible and (
            chg_src or chg_nu or chg_ls or chg_rs
        ):
            refresh_L()

        # Bump function plot
        if chg_bs or chg_bd:
            update_bump_plot(bump_scale, bump_decay)

    viewer.window.add_dock_widget(controls, name="kernel controls", area="right")

    # ---- Layer-visibility hooks: refresh J/K/L when newly toggled ON ------
    if j_layer is not None:
        def _on_J_visibility(event):
            if j_layer.visible:
                refresh_J()
        j_layer.events.visible.connect(_on_J_visibility)

    def _on_K_visibility(event):
        if k_layer.visible:
            refresh_K()
    k_layer.events.visible.connect(_on_K_visibility)

    if l_layer is not None:
        def _on_L_visibility(event):
            if l_layer.visible:
                refresh_L()
        l_layer.events.visible.connect(_on_L_visibility)

    # ---- Banner ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("Kernel & graph debugger ready.")
    print(f"  graph nodes        : {N:,}")
    print(f"  eigenmodes loaded  : {num_modes_max}")
    print(f"  graphbandwidth     : {ctx['bw']:g}")
    print(f"  sources            : {src_idxs.tolist()}")
    print(f"  fabric edges       : {fabric_pairs.shape[0]:,}")
    print(f"  laplacian edges    : {lap_pairs.shape[0]:,}")
    print("Layers (toggle in left panel):")
    print("   A1 / A2  — graph fabric (nodes + edges)")
    print("   B1 / B2  — Laplacian diag (uniform for norm L) / off-diag (edges)")
    print("   H        — weighted degree D_i (true per-node connectivity)")
    print("   C        — kernel diagonal K(i, i) = prior variance per node")
    print("   D        — L · δ_src  (sharp, immediate neighbors)")
    print("   E        — Euclidean Matern K(src, ·) at training nodes")
    print("   F        — Manifold Matern K(src, ·) at training nodes")
    print("   G        — single eigenvector φ_k(i)")
    print("   K        — L · density (graph Laplacian on reference image)")
    print("   J        — Manifold Matern at dense voxel grid (Nyström)  [SLOW]")
    print("   L        — Euclidean Matern at dense voxel grid          [fast]")
    print("Sliders (right):")
    print(f"   active source / num modes / ν (int 1-6) / ℓ / eigvec / γ")
    print(f"   render_stride (J, L) / bump_scale, bump_decay / density σ")
    print("Tips:")
    print("   · Toggle E ↔ F to see Euclidean vs Manifold Matern at training")
    print("     nodes. Toggle J ↔ L for the same comparison at a dense grid.")
    print("   · Layer K refreshes when toggled visible. Adjust σ live.")
    print("=" * 72 + "\n")

    napari.run()


if __name__ == "__main__":
    main()