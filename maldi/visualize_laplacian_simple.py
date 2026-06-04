"""Minimal Laplacian + eigenvector visualizer.

Four volume layers, all in full-resolution template voxel coords:

  template (reference)   — the original anatomical reference volume
                           (3D Image, gray, attenuated_mip).

  L · density            — graph Laplacian applied to the reference image
                           values at each node (optionally Gaussian-blurred
                           in voxel space first via the σ slider, to
                           suppress fine intensity boundaries before L is
                           applied). Rasterized back into a template-shaped
                           volume by upsampling each sub-voxel into a
                           stride³ block. Transparent-center RdBu_r,
                           symmetric contrast at the --lap-contrast-pct
                           percentile.

  eigenvector φ_k        — kth Laplacian eigenvector, same rasterization
                           trick. Transparent-center RdBu_r with signed γ
                           baked into the colormap — γ<1 boosts low
                           magnitudes so Fiedler-type modes show their
                           interior, not just the extremes at the boundary.

  Σ φ_k  (first N modes) — pointwise sum Σ_{k=0..N−1} φ_k(i), same
                           rasterization + colormap as the single-mode
                           layer. Note: φ_k signs are arbitrary (solver
                           convention) so individual contributions may
                           cancel — interpret with that in mind.

Five sliders in the dock widget on the right:
  · density smoothing σ     — Gaussian blur on the density before L is
                              applied (suppresses voxel-scale boundaries)
  · eigenvector index k     — which single mode to display
  · modes summed N          — how many modes to sum in the Σ layer
  · γ                       — signed gamma around zero; γ<1 boosts interior
                              (shared between the φ_k and Σ φ_k layers)
  · saturation percentile   — percentile of |signal| → colormap endpoints
                              (also shared between the two layers)

The state tracker only re-rasterizes the layers whose driving slider
changed (σ → L·density, k → φ_k, N → Σ φ_k); γ / pct changes just re-bake
the colormap on φ_k and Σ φ_k.

Docked on the left: a small matplotlib widget showing the eigenvalue
spectrum λ_k vs k, linear and log-log, with a `k^(2/3)` Weyl reference
overlaid. A red vertical line tracks the current k; a green shade tracks
the sum range [0, N). Reading the spectrum: smooth power-law growth →
clean Laplacian operator; flat-then-jump near k=0 → multiple weakly-
connected components; stairstep / plateau → discrete graph clusters.
Every slider change prints λ, raw range, and p50/p99 of |signal|.

CLI is compatible with the existing visualize_laplacian.sh — extra args
(n-sources, n-targets, k-show, fabric-*, etc.) are accepted but ignored.
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
# CLI — same shape as visualize_laplacian.py so the existing .sh script works.
# Args that the simple visualizer doesn't use are accepted and ignored.
# =============================================================================
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Minimal Laplacian + eigenvector visualizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- used ---
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
    p.add_argument("--initial-eigvec", type=int, default=1,
                   help="Eigenvector index shown on first render. "
                        "Skip 0 by default (constant on connected components).")
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute-eigvecs", action="store_true")

    p.add_argument("--node-size", type=float, default=1.0)
    p.add_argument("--template-opacity", type=float, default=0.2,
                   help="Opacity of the anatomical template volume layer.")
    p.add_argument("--lap-contrast-pct", type=float, default=99.5,
                   help="Percentile of |L·density| used to set the symmetric "
                        "contrast limits of the volume layer (lower → more "
                        "saturated signal). Set to 100 for true [-amax, amax].")
    p.add_argument("--density-smooth-sigma", type=float, default=0.0,
                   help="Initial Gaussian smoothing σ applied to the density "
                        "(in sub-volume voxels) before computing L·density. "
                        "σ=0 disables. Live-adjustable via the dock-widget slider.")
    p.add_argument("--eigvec-contrast-pct", type=float, default=99.0,
                   help="Percentile of |φ_k| used to set the symmetric "
                        "saturation point for the eigenvector colormap.")
    p.add_argument("--eigvec-gamma", type=float, default=0.5,
                   help="Sign-preserving gamma applied to φ_k around zero "
                        "(γ<1 boosts low magnitudes so the interior of the "
                        "eigenvector becomes visible). γ=1 disables.")
    p.add_argument("--initial-sum-N", type=int, default=20,
                   help="Initial number of eigenvectors summed pointwise "
                        "(layer 'Σ φ_k (first N modes)'). Slider tweaks live.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")

    # --- accepted but unused (kept for CLI compatibility with the full script) ---
    p.add_argument("--initial-modes", type=int, default=None)
    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--diffusion-t", type=float, default=1.0)
    p.add_argument("--n-sources", type=int, default=4)
    p.add_argument("--source-seed", type=int, default=0)
    p.add_argument("--n-targets", type=int, default=50)
    p.add_argument("--target-strategy",
                   choices=["random", "stratified"], default="stratified")
    p.add_argument("--k-show", type=int, default=30)
    p.add_argument("--source-marker-size", type=float, default=6.0)
    p.add_argument("--knn-color-by", choices=["heat", "distance"], default="heat")
    p.add_argument("--fabric-edge-sample", type=int, default=200_000)
    p.add_argument("--fabric-node-size", type=float, default=0.6)
    p.add_argument("--fabric-edge-width", type=float, default=0.3)
    p.add_argument("--laplacian-edge-sample", type=int, default=80_000)
    p.add_argument("--gamma", type=float, default=0.5)
    return vars(p.parse_args())


# =============================================================================
# Setup — load template, build KNN graph, build Laplacian, load/compute eigvecs.
# Mirrors visualize_laplacian.py:setup() but returns a slimmer context.
# =============================================================================
def setup(args: dict, log: logging.Logger) -> dict:
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

    # ---- KNN graph -----------------------------------------------------------
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

    # ---- Laplacian -----------------------------------------------------------
    bw = float(args["graphbandwidth"])
    bw_tensor = torch.tensor(bw, device=device)
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0], bw_tensor, args["laplacian_norm"],
    )

    # ---- Eigenvectors --------------------------------------------------------
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
        sv_scale=sv_scale, sv_translate=sv_translate,
        laplacian_op=laplacian_op,
        eigval=eigval, eigvec=eigvec,
    )


# =============================================================================
# Per-node signals
# =============================================================================
def density_at_nodes(
    sub_volume: np.ndarray,
    node_voxel_idx: np.ndarray,
    sigma: float = 0.0,
) -> np.ndarray:
    """Reference-image intensity at each node, in float32.

    If `sigma>0`, the sub_volume is Gaussian-blurred in voxel space first
    (using scipy.ndimage.gaussian_filter). Larger σ suppresses fine
    intensity boundaries before sampling, so applying L afterwards reveals
    coarser anatomical structure instead of every voxel-scale edge.
    σ is in *sub-volume* voxels — at stride=4 a σ of 2 corresponds to
    roughly 8 voxels of the full-res template.
    """
    vol = sub_volume.astype(np.float32, copy=False)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=float(sigma))
    return vol[
        node_voxel_idx[:, 0], node_voxel_idx[:, 1], node_voxel_idx[:, 2]
    ]


def apply_laplacian(laplacian_op: GraphLaplacianOperator, f_np: np.ndarray) -> np.ndarray:
    """L · f, where f is a 1D numpy vector of length N."""
    device = laplacian_op.x.device
    dtype = laplacian_op.x.dtype
    f = torch.as_tensor(f_np, device=device, dtype=dtype).unsqueeze(-1)
    Lf = laplacian_op._matmul(f).squeeze(-1).cpu().numpy()
    return Lf


def eigvec_sum_at_nodes(eigvec: torch.Tensor, N: int) -> np.ndarray:
    """Pointwise sum of the first N eigenvectors at every node.

    Returns Σ_{k=0..N−1} φ_k(i), shape (num_nodes,). Note that φ_k carry
    arbitrary signs from the solver — interpret accordingly.
    """
    K_max = eigvec.shape[1]
    N = int(np.clip(N, 1, K_max))
    return eigvec[:, :N].sum(dim=1).cpu().numpy()


def rasterize_to_full_res(
    values: np.ndarray,
    node_voxel_idx: np.ndarray,
    sub_volume_shape: tuple,
    full_volume_shape: tuple,
    sv_scale: np.ndarray,
    sv_translate: np.ndarray,
) -> np.ndarray:
    """Place a per-node signal back into a full-resolution volume.

    Each sub-voxel becomes a stride³ block at the corresponding offset in the
    full-res volume, matching the placement of the Points layer (which sits at
    `node_voxel_idx * sv_scale + sv_translate`).
    """
    sz, sy, sx = sv_scale.astype(int).tolist()
    oz, oy, ox = sv_translate.astype(int).tolist()

    sub_grid = np.zeros(sub_volume_shape, dtype=np.float32)
    sub_grid[node_voxel_idx[:, 0],
             node_voxel_idx[:, 1],
             node_voxel_idx[:, 2]] = values.astype(np.float32)

    up = sub_grid
    if sz > 1: up = np.repeat(up, sz, axis=0)
    if sy > 1: up = np.repeat(up, sy, axis=1)
    if sx > 1: up = np.repeat(up, sx, axis=2)

    full = np.zeros(full_volume_shape, dtype=np.float32)
    ez = min(oz + up.shape[0], full_volume_shape[0])
    ey = min(oy + up.shape[1], full_volume_shape[1])
    ex = min(ox + up.shape[2], full_volume_shape[2])
    full[oz:ez, oy:ey, ox:ex] = up[: ez - oz, : ey - oy, : ex - ox]
    return full


# =============================================================================
# Eigenvalue spectrum widget (matplotlib in a Qt dock widget)
# =============================================================================
def make_spectrum_widget(
    eigval: torch.Tensor,
    initial_k: int = 0,
    initial_N: int = 1,
):
    """Build a docked widget showing λ_k vs k (linear + log-log).

    Returns (widget, update_k, update_N). If matplotlib's Qt backend is
    unavailable, returns (None, no-op, no-op) so the caller can degrade
    gracefully.

    The log-log panel overlays a `k^{2/3}` Weyl reference (3-D manifold):
      · close to the reference → operator behaves like a clean Laplacian
      · flat segment near k=0  → multiple weakly-connected components
      · stairstep / plateau    → discrete clusters in the graph
    The red vertical line marks the currently-displayed eigenvector index;
    the green shaded span shows the modes summed in the Σ φ_k layer.
    """
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
        return None, (lambda _k: None), (lambda _n: None)

    eigval_np = eigval.detach().cpu().numpy()
    K = int(eigval_np.shape[0])
    ks = np.arange(K)

    fig = Figure(figsize=(4.2, 5.2), tight_layout=True)
    ax_lin = fig.add_subplot(2, 1, 1)
    ax_log = fig.add_subplot(2, 1, 2)

    # ----- Linear panel ------------------------------------------------------
    ax_lin.plot(ks, eigval_np, marker=".", ms=2.5, lw=0.7, color="#3578a8")
    ax_lin.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    ax_lin.set_xlabel("k")
    ax_lin.set_ylabel(r"$\lambda_k$")
    ax_lin.set_title("λ_k vs k  (linear)", fontsize=10)
    ax_lin.grid(True, alpha=0.3)

    # ----- Log-log panel + Weyl reference ------------------------------------
    nonzero = eigval_np > 1e-12
    if nonzero.any():
        ks_pos = ks[nonzero]
        lam_pos = eigval_np[nonzero]
        ax_log.loglog(ks_pos, lam_pos, marker=".", ms=2.5, lw=0.7, color="#3578a8")
        if len(ks_pos) > 10:
            anchor = max(len(ks_pos) // 4, 1)
            anchor_k = float(ks_pos[anchor])
            anchor_lam = float(lam_pos[anchor])
            ref_lam = anchor_lam * (
                ks_pos.astype(np.float64) / anchor_k
            ) ** (2.0 / 3.0)
            ax_log.loglog(
                ks_pos, ref_lam, "--", lw=0.7, alpha=0.7, color="#cc3344",
                label=r"$k^{2/3}$  (3-D Weyl)",
            )
            ax_log.legend(fontsize=8, loc="lower right")
    ax_log.set_xlabel("k")
    ax_log.set_ylabel(r"$\lambda_k$")
    ax_log.set_title("λ_k vs k  (log-log)", fontsize=10)
    ax_log.grid(True, alpha=0.3, which="both")

    # ----- Live markers ------------------------------------------------------
    k0 = int(np.clip(initial_k, 0, max(K - 1, 0)))
    n0 = int(np.clip(initial_N, 1, K))
    marker_lin = ax_lin.axvline(k0, color="#cc3344", lw=1.2, alpha=0.85)
    marker_log = ax_log.axvline(max(k0, 1), color="#cc3344", lw=1.2, alpha=0.85)
    shade = {
        "lin": ax_lin.axvspan(0, n0, alpha=0.15, color="#3aaa54"),
        "log": ax_log.axvspan(1, max(n0, 1), alpha=0.15, color="#3aaa54"),
    }

    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(420)

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(canvas)

    def update_k(k):
        k = int(k)
        marker_lin.set_xdata([k, k])
        marker_log.set_xdata([max(k, 1), max(k, 1)])
        canvas.draw_idle()

    def update_N(N):
        N = int(N)
        shade["lin"].remove()
        shade["log"].remove()
        shade["lin"] = ax_lin.axvspan(0, max(N, 0.1), alpha=0.15, color="#3aaa54")
        shade["log"] = ax_log.axvspan(1, max(N, 1.0), alpha=0.15, color="#3aaa54")
        canvas.draw_idle()

    return widget, update_k, update_N


# =============================================================================
# Coloring — symmetric diverging RdBu_r with percentile saturation + signed γ.
# =============================================================================
def compute_amax_pct(values: np.ndarray, pct: float = 99.0) -> float:
    """`pct`-th percentile of |values| over nonzero entries — robust to outliers."""
    abs_vals = np.abs(values)
    nz = abs_vals[abs_vals > 0]
    if nz.size == 0:
        return 1.0
    return max(float(np.percentile(nz, pct)), 1e-12)


def _napari_diverging_cmap(transparent_center: bool = True, gamma: float = 1.0):
    """napari Colormap from matplotlib's RdBu_r, with optional transparent
    center and signed γ baked into the lookup.

    With transparent_center=True, alpha fades to 0 at the colormap midpoint
    so zero-valued background and near-zero signal are invisible.
    With γ<1, both color saturation and alpha get boosted for low magnitudes —
    use this to make Fiedler-type modes (mass concentrated at boundary nodes)
    readable in their interior.
    """
    from napari.utils.colormaps import Colormap
    positions = np.linspace(0.0, 1.0, 256)
    signed_mag = 2.0 * positions - 1.0                     # in [-1, 1]
    if gamma != 1.0:
        signed_mag_warped = np.sign(signed_mag) * (np.abs(signed_mag) ** gamma)
    else:
        signed_mag_warped = signed_mag
    color_pos = np.clip(0.5 + 0.5 * signed_mag_warped, 0.0, 1.0)
    rgba = cm.get_cmap("RdBu_r")(color_pos).copy()
    if transparent_center:
        rgba[:, 3] = np.abs(signed_mag_warped)
    return Colormap(colors=rgba, name=f"RdBu_r_div_g{gamma:.2g}")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args["verbose"] else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("visualize_laplacian_simple")

    ctx = setup(args, log)
    if args["no_launch"]:
        log.info("--no-launch passed; precompute OK, exiting.")
        return

    import napari
    from magicgui import magicgui

    N = ctx["node_voxel_idx"].shape[0]
    K_max = ctx["eigvec"].shape[1]
    log.info(f"{N:,} graph nodes, {K_max} eigenmodes available")

    # ---- Initial L · density (driven by σ slider below) ---------------------
    default_sigma = max(float(args["density_smooth_sigma"]), 0.0)
    lap_pct = float(np.clip(args["lap_contrast_pct"], 50.0, 100.0))

    density_np = density_at_nodes(
        ctx["sub_volume"], ctx["node_voxel_idx"], sigma=default_sigma,
    )
    L_density = apply_laplacian(ctx["laplacian_op"], density_np)
    L_density_volume = rasterize_to_full_res(
        L_density,
        ctx["node_voxel_idx"],
        ctx["sub_volume"].shape,
        ctx["template_full"].shape,
        ctx["sv_scale"],
        ctx["sv_translate"],
    )
    lap_amax = compute_amax_pct(L_density_volume, pct=lap_pct)
    log.info(
        f"density range  [{density_np.min():.4g}, {density_np.max():.4g}]   "
        f"L·density range  [{L_density.min():.4g}, {L_density.max():.4g}]   "
        f"contrast ±{lap_amax:.4g}  (σ={default_sigma:.2g}, p{lap_pct:g})"
    )

    # ---- Napari viewer ------------------------------------------------------
    viewer = napari.Viewer(title="Laplacian (simple)")
    viewer.dims.ndisplay = 3

    # Layer 0: original template (anatomical reference) -----------------------
    viewer.add_image(
        ctx["template_full"],
        name="template (reference)",
        colormap="gray",
        rendering="attenuated_mip",
        attenuation=0.5,
        opacity=float(args["template_opacity"]),
        blending="translucent",
        visible=True,
    )

    # Layer 1: L · density as a full-resolution volume ------------------------
    l_density_layer = viewer.add_image(
        L_density_volume,
        name="L · density",
        colormap=_napari_diverging_cmap(transparent_center=True, gamma=1.0),
        contrast_limits=(-lap_amax, lap_amax),
        rendering="translucent",
        opacity=0.95,
        blending="translucent",
        visible=True,
    )

    def show_L_density(sigma: float):
        """Recompute L · density with a fresh smoothing σ and refresh layer."""
        sigma = max(float(sigma), 0.0)
        d_np = density_at_nodes(
            ctx["sub_volume"], ctx["node_voxel_idx"], sigma=sigma,
        )
        Ld = apply_laplacian(ctx["laplacian_op"], d_np)
        Ld_vol = rasterize_to_full_res(
            Ld, ctx["node_voxel_idx"],
            ctx["sub_volume"].shape, ctx["template_full"].shape,
            ctx["sv_scale"], ctx["sv_translate"],
        )
        amax = compute_amax_pct(Ld_vol, pct=lap_pct)
        l_density_layer.data = Ld_vol
        l_density_layer.contrast_limits = (-amax, amax)
        print(
            f"[L·density σ={sigma:.2g}]   "
            f"density [{d_np.min():.4g}, {d_np.max():.4g}]   "
            f"L·density [{Ld.min():.4g}, {Ld.max():.4g}]   "
            f"sat ±{amax:.4g} (p{lap_pct:g})"
        )

    # ===== Eigenvector φ_k =====================================================
    default_pct = float(np.clip(args["eigvec_contrast_pct"], 50.0, 100.0))
    default_gamma = float(np.clip(args["eigvec_gamma"], 0.1, 2.0))
    initial_k = int(np.clip(args["initial_eigvec"], 0, K_max - 1))

    # Layer 2: φ_k as a full-resolution volume --------------------------------
    # Initialise with whatever k is shown first; .data, .colormap and
    # .contrast_limits all get rebuilt on every slider change.
    initial_phi = ctx["eigvec"][:, initial_k].cpu().numpy()
    initial_phi_vol = rasterize_to_full_res(
        initial_phi, ctx["node_voxel_idx"],
        ctx["sub_volume"].shape, ctx["template_full"].shape,
        ctx["sv_scale"], ctx["sv_translate"],
    )
    initial_eig_amax = compute_amax_pct(initial_phi, pct=default_pct)
    eigvec_volume_layer = viewer.add_image(
        initial_phi_vol,
        name="eigenvector φ_k",
        colormap=_napari_diverging_cmap(
            transparent_center=True, gamma=default_gamma,
        ),
        contrast_limits=(-initial_eig_amax, initial_eig_amax),
        rendering="translucent",
        opacity=0.95,
        blending="translucent",
        visible=True,
    )

    # Cached state between slider refreshes ------------------------------------
    eig_cache = {"k": None, "phi": None}

    def show_eigvec(k: int, pct: float, gamma: float):
        k = int(np.clip(k, 0, K_max - 1))

        if eig_cache["k"] != k:
            phi = ctx["eigvec"][:, k].cpu().numpy()
            eig_cache["k"] = k
            eig_cache["phi"] = phi
            # Re-rasterize the volume only when the mode index changes
            phi_vol = rasterize_to_full_res(
                phi, ctx["node_voxel_idx"],
                ctx["sub_volume"].shape, ctx["template_full"].shape,
                ctx["sv_scale"], ctx["sv_translate"],
            )
            eigvec_volume_layer.data = phi_vol
        else:
            phi = eig_cache["phi"]

        amax = compute_amax_pct(phi, pct=pct)

        # Rebake colormap (γ baked in) + update symmetric contrast
        eigvec_volume_layer.colormap = _napari_diverging_cmap(
            transparent_center=True, gamma=gamma,
        )
        eigvec_volume_layer.contrast_limits = (-amax, amax)

        # Stats — what you need to debug Fiedler-type modes
        lam = float(ctx["eigval"][k].item())
        abs_phi = np.abs(phi)
        p50 = float(np.percentile(abs_phi, 50))
        p99 = float(np.percentile(abs_phi, 99))
        print(
            f"[φ_{k}]  λ={lam:.6g}   "
            f"raw [{phi.min():.4g}, {phi.max():.4g}]   "
            f"|p50|={p50:.4g}   |p99|={p99:.4g}   "
            f"sat ±{amax:.4g} (p{pct:.0f})   γ={gamma:.2g}"
        )

    show_eigvec(initial_k, default_pct, default_gamma)

    # ----- Σ φ_k (first N modes) -----------------------------------------------
    default_sum_N = int(np.clip(args["initial_sum_N"], 1, K_max))

    initial_sum = eigvec_sum_at_nodes(ctx["eigvec"], default_sum_N)
    initial_sum_vol = rasterize_to_full_res(
        initial_sum, ctx["node_voxel_idx"],
        ctx["sub_volume"].shape, ctx["template_full"].shape,
        ctx["sv_scale"], ctx["sv_translate"],
    )
    initial_sum_amax = compute_amax_pct(initial_sum, pct=default_pct)
    eigvec_sum_layer = viewer.add_image(
        initial_sum_vol,
        name="Σ φ_k  (first N modes)",
        colormap=_napari_diverging_cmap(
            transparent_center=True, gamma=default_gamma,
        ),
        contrast_limits=(-initial_sum_amax, initial_sum_amax),
        rendering="translucent",
        opacity=0.95,
        blending="translucent",
        visible=True,
    )

    sum_cache = {"N": None, "sum": None}

    def show_sum(N: int, pct: float, gamma: float):
        N = int(np.clip(N, 1, K_max))

        if sum_cache["N"] != N:
            s = eigvec_sum_at_nodes(ctx["eigvec"], N)
            sum_cache["N"] = N
            sum_cache["sum"] = s
            s_vol = rasterize_to_full_res(
                s, ctx["node_voxel_idx"],
                ctx["sub_volume"].shape, ctx["template_full"].shape,
                ctx["sv_scale"], ctx["sv_translate"],
            )
            eigvec_sum_layer.data = s_vol
        else:
            s = sum_cache["sum"]

        amax = compute_amax_pct(s, pct=pct)
        eigvec_sum_layer.colormap = _napari_diverging_cmap(
            transparent_center=True, gamma=gamma,
        )
        eigvec_sum_layer.contrast_limits = (-amax, amax)

        abs_s = np.abs(s)
        p50 = float(np.percentile(abs_s, 50))
        p99 = float(np.percentile(abs_s, 99))
        print(
            f"[Σφ_k N={N}]   "
            f"raw [{s.min():.4g}, {s.max():.4g}]   "
            f"|p50|={p50:.4g}   |p99|={p99:.4g}   "
            f"sat ±{amax:.4g} (p{pct:.0f})   γ={gamma:.2g}"
        )

    show_sum(default_sum_N, default_pct, default_gamma)

    # ---- Spectrum widget (λ_k vs k) ------------------------------------------
    spectrum_widget, update_spectrum_k, update_spectrum_N = make_spectrum_widget(
        ctx["eigval"], initial_k=initial_k, initial_N=default_sum_N,
    )
    if spectrum_widget is not None:
        viewer.window.add_dock_widget(
            spectrum_widget, name="λ_k spectrum", area="left",
        )

    # ---- Dock widget: σ, k, N, γ, pct ---------------------------------------
    # State tracker — only re-rasterize the layers whose driving slider changed.
    slider_state = {
        "sigma": default_sigma,
        "k":     initial_k,
        "N":     default_sum_N,
        "gam":   default_gamma,
        "pct":   default_pct,
    }

    @magicgui(
        auto_call=True,
        density_sigma={"label": "L·density: density smoothing σ (voxels)",
                       "min": 0.0, "max": 10.0, "step": 0.1},
        eigvec_idx={"label": "eigenvector index k",
                    "min": 0, "max": K_max - 1, "step": 1},
        sum_N={"label": "Σ φ_k  (modes summed N)",
               "min": 1, "max": K_max, "step": 1},
        eigvec_gamma={"label": "γ  (γ<1 boosts small magnitudes)",
                      "min": 0.1, "max": 2.0, "step": 0.05},
        eigvec_pct={"label": "saturation percentile (φ_k, Σφ_k)",
                    "min": 50.0, "max": 100.0, "step": 0.5},
    )
    def controls(
        density_sigma: float = default_sigma,
        eigvec_idx: int = initial_k,
        sum_N: int = default_sum_N,
        eigvec_gamma: float = default_gamma,
        eigvec_pct: float = default_pct,
    ):
        chg_sig = density_sigma != slider_state["sigma"]
        chg_k   = eigvec_idx    != slider_state["k"]
        chg_N   = sum_N         != slider_state["N"]
        chg_gam = eigvec_gamma  != slider_state["gam"]
        chg_pct = eigvec_pct    != slider_state["pct"]
        slider_state.update(
            sigma=density_sigma, k=eigvec_idx, N=sum_N,
            gam=eigvec_gamma, pct=eigvec_pct,
        )
        if chg_sig:
            show_L_density(density_sigma)
        if chg_k or chg_gam or chg_pct:
            show_eigvec(eigvec_idx, eigvec_pct, eigvec_gamma)
        if chg_N or chg_gam or chg_pct:
            show_sum(sum_N, eigvec_pct, eigvec_gamma)
        if chg_k:
            update_spectrum_k(eigvec_idx)
        if chg_N:
            update_spectrum_N(sum_N)

    viewer.window.add_dock_widget(controls, name="eigenvector", area="right")

    print("\n" + "=" * 64)
    print("Simple Laplacian viewer ready.")
    print(f"  nodes        : {N:,}")
    print(f"  eigenmodes   : {K_max}")
    print(f"  template     : {ctx['template_full'].shape}")
    print(f"  stride       : {ctx['sv_scale'].astype(int).tolist()}")
    print(f"  layers       : 'template (reference)',")
    print(f"                 'L · density',")
    print(f"                 'eigenvector φ_k',")
    print(f"                 'Σ φ_k  (first N modes)'")
    print(f"  widgets      : 'λ_k spectrum'        (left)")
    print(f"                 'eigenvector' sliders (right)")
    print(f"  sliders      : density smoothing σ   (default {default_sigma:.2g} voxels)")
    print(f"                 eigenvector index k   (0 … {K_max - 1})")
    print(f"                 modes summed N        (1 … {K_max})")
    print(f"                 γ                     (default {default_gamma:.2g})")
    print(f"                 saturation pct        (default {default_pct:.1f})")
    print(f"  tip: σ=2-3 voxels usually clears the speckle and leaves the major edges.")
    print("=" * 64 + "\n")

    napari.run()


if __name__ == "__main__":
    main()