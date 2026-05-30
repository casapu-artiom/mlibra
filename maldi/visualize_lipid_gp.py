"""visualize_lipid_gp.py — visualize a simple spectral GP fit per lipid.

This is a *separate* tool from your trained ManifoldLGP (which is a deep
latent GP model). Here we fit an independent spectral GP for each lipid,
directly using the precomputed manifold eigendecomposition. It's a quick
"what would the simplest GP do with this lipid" sanity check, useful for:

  - seeing whether the kernel produces sensible per-lipid reconstructions
  - inspecting how the prediction depends on ν, ℓ, noise σ²
  - comparing predicted vs. measured at training locations

Layers in the viewer:

  0. template (reference)              — anatomical context (Volume)
  1. KNN graph nodes (faint)           — all stride=4 graph nodes
  2. training data points              — MaLDI measurement locations,
                                         colored by raw observed value
  3. GP posterior mean @ graph nodes   — prediction at every graph node
  4. GP posterior mean @ dense voxels  — Nyström interpolated whole-brain
                                         reconstruction (off by default)

Controls: lipid index, ν, ℓ, noise σ², num_modes, render_stride.

Sources of subtle bugs:
  - coord_mean/coord_std must match what was used at GP training time. We
    re-derive them from the *training* MaLDI rows (`xccf, yccf, zccf` mean
    and std). If your inducing points / training pipeline used different
    statistics, predictions will look spatially shifted. Override via
    `--coord-mean-file` / `--coord-std-file` if you want.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.cm as cm

# Reuse data loading + eigensolve infrastructure
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from visualize_laplacian_simple import (  # noqa: E402
    parse_args as _parse_base_args,
    setup as _setup_base,
    _napari_diverging_cmap,
)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> dict:
    extra = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    extra.add_argument("--maldi-file", required=True,
                       help="Parquet file with lipid measurements + coords.")
    extra.add_argument("--slices-dataset-file", required=True,
                       help="JSON with section IDs for train/test split.")
    extra.add_argument("--available-lipids-file", default=None,
                       help="npy file listing lipid names. If omitted, "
                            "lipid names are auto-detected from the parquet "
                            "(non-coordinate, non-metadata columns).")
    extra.add_argument("--initial-lipid-idx", type=int, default=0,
                       help="Starting lipid index in the available-lipids list.")
    extra.add_argument("--initial-lipid-name", default=None,
                       help="Override initial lipid by name (e.g. 'PA 36:1 PA 38:4').")
    extra.add_argument("--noise-sigma", type=float, default=0.3,
                       help="Observation noise σ for the GP. Larger = more "
                            "smoothing.")
    extra.add_argument("--lipid-log-transform", action="store_true",
                       help="Apply log(x + ε) transform to lipid values before "
                            "fitting (matches --log-transform in the main "
                            "experiment).")
    extra.add_argument("--render-stride", type=int, default=4,
                       help="Voxel stride for the dense-grid prediction layer. "
                            "1 = full-res (very slow), 4 = same as graph.")
    extra.add_argument("--max-render-points", type=int, default=300_000,
                       help="Subsample cap for the dense-grid prediction layer.")
    extra.add_argument("--training-subsample", type=int, default=-1,
                       help="If >0, randomly subsample training points to "
                            "this many for the GP fit. Speeds up exploration.")
    extra.add_argument("--bump-scale", type=float, default=20.0,
                       help="Bump function support radius (× graphbandwidth).")
    extra.add_argument("--bump-decay", type=float, default=0.01,
                       help="Bump function boundary softness.")
    extra.add_argument("--coord-mean-file", default=None,
                       help="Optional .pth/.npy of (3,) coordinate mean. If "
                            "omitted, computed from training MaLDI rows.")
    extra.add_argument("--coord-std-file", default=None,
                       help="Optional .pth/.npy of (3,) coordinate std.")
    extra.add_argument("--annotations-file", default=None,
                       help="Optional .npy of integer anatomical region IDs "
                            "(same shape as the reference template). "
                            "Rendered as a colored Labels layer.")
    extra.add_argument("--eucl-subsample", type=int, default=3000,
                       help="Training subsample size for the Euclidean Matern "
                            "GP. Cholesky is O(N³), so keep this modest.")
    extra.add_argument("--eucl-batch-size", type=int, default=20_000,
                       help="Batch size when predicting at the dense graph "
                            "nodes (controls peak GPU memory).")
    extra.add_argument("--lengthscale-eucl", type=float, default=None,
                       help="Initial lengthscale for the Euclidean Matern GP. "
                            "If omitted, defaults to --lengthscale. "
                            "Because the manifold and Euclidean kernels live "
                            "in different spaces (spectral vs spatial), their "
                            "optimal lengthscales typically differ — set "
                            "this independently for a fair comparison.")

    extra_args, remaining_argv = extra.parse_known_args()

    # Forward only the args the base parser expects
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining_argv
    try:
        base = _parse_base_args()
    finally:
        sys.argv = original_argv

    base.update(vars(extra_args))
    return base


# =============================================================================
# Bump function (library or fallback)
# =============================================================================
try:
    from manifold_gp.utils import bump_function as _lib_bump_function
    _USING_LIB_BUMP = True
except ImportError:
    _USING_LIB_BUMP = False


def bump_function(d, scale, decay):
    if _USING_LIB_BUMP:
        d_t = d if torch.is_tensor(d) else torch.as_tensor(d)
        scale_t = (scale if torch.is_tensor(scale)
                   else torch.as_tensor(float(scale), dtype=d_t.dtype, device=d_t.device))
        decay_t = (decay if torch.is_tensor(decay)
                   else torch.as_tensor(float(decay), dtype=d_t.dtype, device=d_t.device))
        return _lib_bump_function(d_t, scale_t, decay_t)
    d = torch.as_tensor(d) if not torch.is_tensor(d) else d
    out = torch.zeros_like(d)
    inside = d < float(scale)
    if inside.any():
        u = (d[inside] / float(scale)).clamp(0.0, 1.0 - 1e-6)
        out[inside] = torch.exp(-float(decay) / (1.0 - u * u))
        out[inside] = out[inside] / float(np.exp(-float(decay)))
    return out


# =============================================================================
# Data loading
# =============================================================================
def load_section_filter(slices_file: Path) -> tuple:
    """Mirror of config.extract_filters from the main experiment code.

    The JSON has keys "train", "test", "ignore". Each is a list of entries
    like ["Sample", "==", "Female1"] or ["Section", "==", 23]. The transform
    used in the main pipeline is:

        train_filter = [[tuple(i)] for i in filters["train"]]

    This produces a pyarrow DNF filter where each entry is its own
    singleton OR clause — i.e. "match Sample==Female1 OR Section==23 OR ...".

    Returns (train_filter, test_filter, n_train, n_test).
    """
    assert str(slices_file).endswith(".json"), \
        f"Expected a JSON file for slices_dataset_file, got {slices_file}"
    with open(slices_file) as f:
        filters = json.load(f)
    train_raw = filters.get("train", [])
    test_raw = filters.get("test", [])
    # Exact same transform as config.extract_filters
    train_filter = [[tuple(i)] for i in train_raw] if train_raw else None
    test_filter = [[tuple(i)] for i in test_raw] if test_raw else None
    return train_filter, test_filter, len(train_raw), len(test_raw)


def load_lipid_names(args: dict, parquet_cols: list[str]) -> list[str]:
    """Resolve the list of lipid column names to use."""
    if args.get("available_lipids_file") is not None:
        names = np.load(args["available_lipids_file"], allow_pickle=True)
        names = [str(n) for n in names]
    else:
        meta_cols = {"x", "y", "z", "xccf", "yccf", "zccf",
                     "x_index", "y_index", "z_index",
                     "Section", "Sample"}
        names = [c for c in parquet_cols if c not in meta_cols]
    return names


def _read_maldi_subset(
    maldi_file: str, lipid_names: list[str], filter_expr,
    log: logging.Logger, label: str,
):
    """Load coords + lipid values for a given pyarrow filter expression."""
    if filter_expr is None:
        log.warning(f"No {label} filter; skipping {label} load.")
        return None, None
    df_coords = pd.read_parquet(
        maldi_file, columns=["xccf", "yccf", "zccf"], filters=filter_expr,
    )
    df_lipids = pd.read_parquet(
        maldi_file, columns=lipid_names, filters=filter_expr,
    )
    coords_mm = torch.from_numpy(df_coords.values.astype(np.float32))
    values = torch.from_numpy(df_lipids.values.astype(np.float32))
    log.info(f"Loaded {label}: {coords_mm.shape[0]:,} points × "
             f"{values.shape[1]} lipids")
    return coords_mm, values


def _preprocess_lipid_values(values, log_transform, col_means=None, col_stds=None):
    """Clamp negatives, optional log, z-score. If col_means/col_stds are
    None, compute from the data; otherwise apply the provided statistics."""
    if (values < 0).any():
        values = values.clamp(min=0.0)
    if log_transform:
        values = torch.log(values + 1e-10)
    if col_means is None:
        col_means = values.mean(dim=0)
        col_stds = values.std(dim=0).clamp(min=1e-6)
    values_z = (values - col_means) / col_stds
    return values, values_z, col_means, col_stds


def load_maldi_train_and_test(args: dict, log: logging.Logger):
    """Load both train and test MaLDI data with shared normalization stats.

    Test data is z-scored using the *training* col_means/col_stds — this
    matches what the main experiment pipeline does and gives meaningful
    comparison between train and test residuals.

    Returns dict with keys:
        train_coords_mm, train_values_raw, train_values_z
        test_coords_mm,  test_values_raw,  test_values_z   (None if no test)
        lipid_names, col_means, col_stds
    """
    maldi_file = args["maldi_file"]
    slices_file = args["slices_dataset_file"]

    # Column names without loading the full file
    parquet_meta = pd.read_parquet(maldi_file, columns=None).columns.tolist()
    log.info(f"MaLDI parquet has {len(parquet_meta)} columns; "
             f"sample: {parquet_meta[:5]}...")

    lipid_names = load_lipid_names(args, parquet_meta)
    log.info(f"Resolved {len(lipid_names)} lipid columns")

    train_filter, test_filter, n_train_clauses, n_test_clauses = \
        load_section_filter(Path(slices_file))
    log.info(f"Filter clauses — train: {n_train_clauses}, test: {n_test_clauses}")

    # ---- Train ----
    train_coords_mm, train_raw = _read_maldi_subset(
        maldi_file, lipid_names, train_filter, log, "train",
    )
    train_raw, train_z, col_means, col_stds = _preprocess_lipid_values(
        train_raw, log_transform=bool(args.get("lipid_log_transform")),
    )

    # ---- Test (apply train normalization) ----
    test_coords_mm, test_raw = _read_maldi_subset(
        maldi_file, lipid_names, test_filter, log, "test",
    )
    if test_raw is not None:
        test_raw, test_z, _, _ = _preprocess_lipid_values(
            test_raw, log_transform=bool(args.get("lipid_log_transform")),
            col_means=col_means, col_stds=col_stds,
        )
    else:
        test_z = None

    log.info(f"Train: {train_coords_mm.shape[0]:,} pts | "
             f"Test: {test_coords_mm.shape[0] if test_coords_mm is not None else 0:,} pts")

    return {
        "train_coords_mm": train_coords_mm,
        "train_values_raw": train_raw,
        "train_values_z": train_z,
        "test_coords_mm": test_coords_mm,
        "test_values_raw": test_raw,
        "test_values_z": test_z,
        "lipid_names": lipid_names,
        "col_means": col_means,
        "col_stds": col_stds,
    }


def get_coord_normalization(args: dict, train_coords_mm: torch.Tensor):
    """Resolve coord_mean/coord_std from optional files or compute from
    training data."""
    cm_file = args.get("coord_mean_file")
    cs_file = args.get("coord_std_file")
    if cm_file is not None and cs_file is not None:
        def _load(p):
            p = Path(p)
            if p.suffix == ".pth":
                return torch.load(p, map_location="cpu").float()
            return torch.from_numpy(np.load(p)).float()
        return _load(cm_file), _load(cs_file)
    coord_mean = train_coords_mm.mean(dim=0)
    coord_std = train_coords_mm.std(dim=0).clamp(min=1e-6)
    return coord_mean, coord_std


# =============================================================================
# Spectral GP — fit + predict
# =============================================================================
def nystrom_eigvecs_at_points(
    coords_z: torch.Tensor,
    ctx: dict, eigvec_K: torch.Tensor,
    nearest_neighbors: int = 10,
    bump_scale: float = 20.0,
    bump_decay: float = 0.01,
    batch_size: int = 20_000,
) -> torch.Tensor:
    """Interpolate the first K eigenvectors to arbitrary points via Nyström.

    Returns a (Q, K) tensor of interpolated eigenvector values. Out-of-
    support queries (beyond `bump_radius`) get zeros.
    """
    from scipy.spatial import cKDTree
    Q = coords_z.shape[0]
    K = eigvec_K.shape[1]
    device = eigvec_K.device
    dtype = eigvec_K.dtype
    laplacian_op = ctx["laplacian_op"]

    if "_query_kdt" not in ctx:
        ctx["_query_kdt"] = cKDTree(ctx["reference_nodes"].cpu().numpy())
    kdt = ctx["_query_kdt"]

    bump_radius = float(laplacian_op.graphbandwidth.squeeze()) * float(bump_scale)
    out = torch.zeros((Q, K), device=device, dtype=dtype)
    coords_cpu = coords_z.cpu().numpy()

    for s in range(0, Q, batch_size):
        e = min(s + batch_size, Q)
        d_b, idx_b = kdt.query(
            coords_cpu[s:e], k=nearest_neighbors, workers=-1,
        )
        edge_value = torch.from_numpy(
            (d_b.astype(np.float32) ** 2),
        ).to(device=device, dtype=dtype)
        edge_index = torch.from_numpy(idx_b.astype(np.int64)).to(device)

        sqrt_d_nearest = edge_value[:, 0].sqrt()
        within = sqrt_d_nearest < bump_radius

        if within.any():
            projected = laplacian_op.out_of_sample(
                eigvec_K, edge_value[within], edge_index[within],
            )
            bump_vals = bump_function(
                sqrt_d_nearest[within], bump_radius, float(bump_decay),
            )
            global_pos = s + torch.nonzero(within, as_tuple=True)[0]
            out[global_pos] = projected * bump_vals.unsqueeze(-1)
    return out


def fit_spectral_gp(
    eigval: torch.Tensor, eigvec_K: torch.Tensor,
    phi_train: torch.Tensor, y: torch.Tensor,
    nu: int, lengthscale: float, noise_sigma: float,
) -> torch.Tensor:
    """Fit the spectral GP and return the K-length coefficient vector μ_c
    such that the posterior mean at any point x* is Σ_k φ_k(x*) (μ_c)_k.

    Math: y ~ N(Φ c, σ²I), c ~ N(0, W) with W = diag(w_k).
          μ_c = σ⁻² Σ_c Φᵀ y,  Σ_c = (W⁻¹ + σ⁻² Φᵀ Φ)⁻¹
    """
    K = eigvec_K.shape[1]
    device = eigvec_K.device
    dtype = eigvec_K.dtype

    safe_lam = eigval[:K].clamp(min=0.0)
    w = (2.0 * float(nu) / (float(lengthscale) ** 2) + safe_lam).pow(-float(nu))
    inv_w = 1.0 / w.clamp(min=1e-30)

    sigma_sq = max(float(noise_sigma), 1e-6) ** 2
    A = torch.diag(inv_w) + (1.0 / sigma_sq) * (phi_train.T @ phi_train)
    rhs = (1.0 / sigma_sq) * (phi_train.T @ y.to(device=device, dtype=dtype))
    mu_c = torch.linalg.solve(A, rhs)
    return mu_c


# =============================================================================
# Euclidean Matern GP (baseline for comparison)
# =============================================================================
def _besselk_integer(nu_int: int, z: torch.Tensor) -> torch.Tensor:
    """Compute K_n(z) for positive integer n using the recursion
        K_{n+1}(z) = K_{n-1}(z) + (2n/z) K_n(z)
    starting from K_0 and K_1 which torch.special provides.
    """
    if nu_int == 0:
        return torch.special.modified_bessel_k0(z)
    if nu_int == 1:
        return torch.special.modified_bessel_k1(z)
    Kprev = torch.special.modified_bessel_k0(z)
    Kcurr = torch.special.modified_bessel_k1(z)
    for n in range(1, nu_int):
        Knext = Kprev + (2.0 * n / z) * Kcurr
        Kprev, Kcurr = Kcurr, Knext
    return Kcurr


def matern_euclidean_pairwise(
    X1: torch.Tensor, X2: torch.Tensor, nu: int, lengthscale: float,
) -> torch.Tensor:
    """Compute the (|X1|, |X2|) Matern kernel matrix on the device of X1.

    Uses torch.special bessel functions; supports integer ν via recursion
    plus a closed-form fast path for ν=0.5 (exponential kernel).
    """
    d = torch.cdist(X1, X2)
    eps = 1e-7
    d_safe = d.clamp(min=eps * float(lengthscale))
    if abs(float(nu) - 0.5) < 1e-6:
        return torch.exp(-d_safe / float(lengthscale))
    nu_int = int(round(float(nu)))
    scaled = math.sqrt(2.0 * nu_int) * d_safe / float(lengthscale)
    Kn = _besselk_integer(nu_int, scaled)
    factor = (2.0 ** (1 - nu_int)) / math.gamma(nu_int)
    out = factor * scaled.pow(nu_int) * Kn
    # K(0) = 1; the formula above is singular at d=0, so substitute
    out = torch.where(d < eps * float(lengthscale), torch.ones_like(out), out)
    return out


def fit_euclidean_gp(
    X_train: torch.Tensor, y_train: torch.Tensor,
    nu: int, lengthscale: float, noise_sigma: float,
) -> tuple:
    """Fit a Euclidean Matern GP. Returns (alpha, L) where alpha is the dual
    coefficient vector and L is the Cholesky factor of (K + σ²I) — the
    latter is returned so the caller can cache it across lipid changes.

    Predictions: μ_*(x*) = K(x*, X_train) @ alpha
    """
    K = matern_euclidean_pairwise(X_train, X_train, nu, lengthscale)
    n = K.shape[0]
    sigma_sq = max(float(noise_sigma), 1e-6) ** 2
    K_reg = K + sigma_sq * torch.eye(n, device=K.device, dtype=K.dtype)
    # Add a tiny jitter for numerical stability
    K_reg.diagonal().add_(1e-6)
    L = torch.linalg.cholesky(K_reg)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)
    return alpha, L


def predict_euclidean_batched(
    X_query: torch.Tensor, X_train: torch.Tensor, alpha: torch.Tensor,
    nu: int, lengthscale: float, batch_size: int = 20_000,
) -> torch.Tensor:
    """Predict at X_query in batches to keep peak memory bounded."""
    N_q = X_query.shape[0]
    out = torch.zeros(N_q, dtype=alpha.dtype, device=alpha.device)
    for s in range(0, N_q, batch_size):
        e = min(s + batch_size, N_q)
        K_batch = matern_euclidean_pairwise(
            X_query[s:e], X_train, nu, lengthscale,
        )
        out[s:e] = K_batch @ alpha
    return out


# =============================================================================
# Coloring helpers (mirror visualize_laplacian.py)
# =============================================================================
def color_nodes_sequential(layer, values: np.ndarray, gamma: float, cmap_name: str):
    vmin, vmax = float(values.min()), float(values.max())
    if vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1) ** float(gamma)
    else:
        norm = np.zeros_like(values, dtype=np.float32)
    colors = cm.get_cmap(cmap_name)(norm)
    layer.face_color = colors
    layer.border_color = colors


def color_nodes_diverging(
    layer, values: np.ndarray, cmap_name: str = "RdBu_r",
    pct: float = 99.0, gamma: float = 0.5,
):
    abs_values = np.abs(values)
    amax = max(float(np.percentile(abs_values, pct)), 1e-12)
    sign = np.sign(values)
    rel = np.clip(np.abs(values) / amax, 0, 1) ** float(gamma)
    norm = 0.5 + 0.5 * sign * rel
    colors = cm.get_cmap(cmap_name)(np.clip(norm, 0, 1))
    layer.face_color = colors
    layer.border_color = colors


# =============================================================================
# Info panel: per-lipid metrics + value/error distributions
# =============================================================================
def make_info_panel():
    """Return (widget, update_lipid_meta, update_manifold, update_euclidean,
    update_hyperparams, update_summary, mark_summary_stale, recompute_btn).

    The recompute_btn is for the caller to wire to its own all-lipid compute
    routine (we can't compute here — that needs the GP fitters from main()).
    """
    try:
        from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTextEdit,
                                    QPushButton, QLabel)
        from qtpy.QtGui import QFont
        from matplotlib.figure import Figure
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except ImportError:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError as exc:
        return None, *(lambda *a, **k: None for _ in range(7)), None

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(4)

    # Metric table — monospace for column alignment
    metrics = QTextEdit()
    metrics.setReadOnly(True)
    mono = QFont("Monospace")
    mono.setStyleHint(QFont.TypeWriter)
    mono.setPointSize(9)
    metrics.setFont(mono)
    metrics.setMaximumHeight(220)
    layout.addWidget(metrics)

    def _new_fig(title):
        fig = Figure(figsize=(4.0, 1.7), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(120)
        return fig, ax, canvas

    fig_obs, ax_obs, canvas_obs = _new_fig("observed values (raw)")
    fig_terr, ax_terr, canvas_terr = _new_fig("train errors (z)")
    fig_xerr, ax_xerr, canvas_xerr = _new_fig("test errors (z)")
    layout.addWidget(canvas_obs)
    layout.addWidget(canvas_terr)
    layout.addWidget(canvas_xerr)

    # ---- Summary section: aggregate stats across ALL lipids -------------
    summary_label = QLabel("─" * 38 + "\nSummary across all lipids")
    summary_label.setFont(mono)
    layout.addWidget(summary_label)

    summary_text = QTextEdit()
    summary_text.setReadOnly(True)
    summary_text.setFont(mono)
    summary_text.setMaximumHeight(170)
    summary_text.setText("(click 'recompute summary' to populate)")
    layout.addWidget(summary_text)

    fig_scatter = Figure(figsize=(4.0, 2.6), constrained_layout=True)
    ax_scatter = fig_scatter.add_subplot(1, 1, 1)
    ax_scatter.set_title("test corr: manifold vs Eucl. (per lipid)", fontsize=8)
    ax_scatter.tick_params(axis="both", labelsize=7)
    canvas_scatter = FigureCanvas(fig_scatter)
    canvas_scatter.setMinimumHeight(180)
    layout.addWidget(canvas_scatter)

    recompute_btn = QPushButton("recompute summary (all lipids)")
    layout.addWidget(recompute_btn)

    state = {
        "lipid_idx": -1,
        "lipid_name": "—",
        "y_train_raw": None,
        "y_test_raw": None,
        "manifold": {},
        "euclidean": {},
        "hyper": {"nu": None, "ls_m": None, "ls_e": None,
                  "sigma": None, "K": None},
    }

    def _fmt(v):
        if v is None:
            return "    —    "
        if isinstance(v, float):
            if not np.isfinite(v):
                return "   nan   "
            if abs(v) < 1e-3 or abs(v) > 1e4:
                return f"{v:>9.3g}"
            return f"{v:>9.4f}"
        return f"{str(v):>9s}"

    def _render_metrics():
        lines = []
        lines.append(f"Lipid {state['lipid_idx']:3d}: {state['lipid_name']}")
        h = state["hyper"]
        if h["nu"] is not None:
            lines.append(
                f"  ν={h['nu']}, σ={h['sigma']:.3g}, K_modes={h['K']}"
            )
            lines.append(
                f"  ℓ_manif={h['ls_m']:.3g}, ℓ_eucl={h['ls_e']:.3g}"
            )
        lines.append("─" * 38)
        lines.append(f"{'metric':<14} {'Manifold':>9}  {'Eucl.':>9}")
        m = state["manifold"]
        e = state["euclidean"]
        for label, key in [
            ("train RMSE(z)", "train_rmse"),
            ("test  RMSE(z)", "test_rmse"),
            ("train MAE(z)",  "train_mae"),
            ("test  MAE(z)",  "test_mae"),
            ("train corr",    "train_corr"),
            ("test  corr",    "test_corr"),
        ]:
            lines.append(f"{label:<14} {_fmt(m.get(key))}  {_fmt(e.get(key))}")
        # Observed value summary
        y_tr = state["y_train_raw"]
        y_te = state["y_test_raw"]
        if y_tr is not None and y_tr.size > 0:
            lines.append("")
            lines.append(f"train raw: n={y_tr.size:,}  "
                         f"min={y_tr.min():.3g}, max={y_tr.max():.3g}, "
                         f"med={np.median(y_tr):.3g}")
        if y_te is not None and y_te.size > 0:
            lines.append(f"test  raw: n={y_te.size:,}  "
                         f"min={y_te.min():.3g}, max={y_te.max():.3g}, "
                         f"med={np.median(y_te):.3g}")
        metrics.setText("\n".join(lines))

    def _redraw_obs():
        ax_obs.clear()
        ax_obs.set_title("observed vs predicted values (raw)", fontsize=8)
        ax_obs.tick_params(axis="both", labelsize=7)
        y_tr = state["y_train_raw"]
        y_te = state["y_test_raw"]
        m_tr = state["manifold"].get("train_pred_raw")
        m_te = state["manifold"].get("test_pred_raw")
        e_tr = state["euclidean"].get("train_pred_raw")
        e_te = state["euclidean"].get("test_pred_raw")

        # Collect everything to choose global bin range
        all_arrays = [a for a in (y_tr, y_te, m_tr, m_te, e_tr, e_te)
                      if a is not None and a.size > 0]
        if not all_arrays:
            canvas_obs.draw_idle()
            return
        all_vals = np.concatenate(all_arrays)
        if all_vals.max() > all_vals.min():
            bins = np.linspace(all_vals.min(), all_vals.max(), 50)
        else:
            bins = 10

        # 1. Observed (true) values — filled gray backdrop
        obs_arrays = [a for a in (y_tr, y_te) if a is not None and a.size > 0]
        if obs_arrays:
            y_obs = np.concatenate(obs_arrays)
            ax_obs.hist(y_obs, bins=bins, alpha=0.45, color="0.4",
                        label=f"observed (n={y_obs.size})", zorder=1)

        # 2. Manifold predictions — blue step outline
        m_arrays = [a for a in (m_tr, m_te) if a is not None and a.size > 0]
        if m_arrays:
            m_all = np.concatenate(m_arrays)
            ax_obs.hist(m_all, bins=bins, histtype="step",
                        linewidth=1.5, color="C0",
                        label=f"manifold pred (n={m_all.size})", zorder=3)

        # 3. Euclidean predictions — orange step outline
        e_arrays = [a for a in (e_tr, e_te) if a is not None and a.size > 0]
        if e_arrays:
            e_all = np.concatenate(e_arrays)
            ax_obs.hist(e_all, bins=bins, histtype="step",
                        linewidth=1.5, color="C1",
                        label=f"Eucl. pred (n={e_all.size})", zorder=2)

        ax_obs.legend(fontsize=7, loc="upper right")
        canvas_obs.draw_idle()

    def _redraw_errors():
        for ax, canvas, key, title in [
            (ax_terr, canvas_terr, "train_err_z", "train errors (z)"),
            (ax_xerr, canvas_xerr, "test_err_z",  "test errors (z)"),
        ]:
            ax.clear()
            ax.set_title(title, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            m_err = state["manifold"].get(key)
            e_err = state["euclidean"].get(key)
            arrs = [a for a in (m_err, e_err) if a is not None and a.size > 0]
            if arrs:
                lo = min(a.min() for a in arrs)
                hi = max(a.max() for a in arrs)
                lim = max(abs(lo), abs(hi)) * 1.05
                bins = np.linspace(-lim, lim, 50) if lim > 0 else 10
                if m_err is not None and m_err.size > 0:
                    ax.hist(m_err, bins=bins, alpha=0.55, color="C0",
                            label="manifold")
                if e_err is not None and e_err.size > 0:
                    ax.hist(e_err, bins=bins, alpha=0.55, color="C1",
                            label="Eucl.")
                ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
                ax.legend(fontsize=7, loc="upper right")
            canvas.draw_idle()

    # ---- Public update callbacks --------------------------------------
    def update_lipid_meta(idx, name, y_train_raw, y_test_raw):
        state["lipid_idx"] = int(idx)
        state["lipid_name"] = str(name)
        state["y_train_raw"] = y_train_raw
        state["y_test_raw"] = y_test_raw
        # Clear stale predictions from the previous lipid — fresh per-lipid
        # predictions will arrive via update_manifold / update_euclidean.
        for k in ("train_pred_raw", "test_pred_raw",
                  "train_err_z", "test_err_z"):
            state["manifold"].pop(k, None)
            state["euclidean"].pop(k, None)
        _redraw_obs()
        _redraw_errors()
        _render_metrics()

    def _compute_metrics(err_z, y_obs_raw, y_pred_raw):
        rmse = float(np.sqrt(np.mean(err_z ** 2))) if err_z.size > 0 else float("nan")
        mae = float(np.mean(np.abs(err_z))) if err_z.size > 0 else float("nan")
        if y_obs_raw is not None and y_obs_raw.size > 1 and np.std(y_obs_raw) > 1e-12:
            corr = float(np.corrcoef(y_obs_raw, y_pred_raw)[0, 1])
        else:
            corr = float("nan")
        return rmse, mae, corr

    def update_manifold(train_err_z, train_obs_raw, train_pred_raw,
                        test_err_z, test_obs_raw, test_pred_raw):
        tr_rmse, tr_mae, tr_corr = _compute_metrics(train_err_z, train_obs_raw, train_pred_raw)
        state["manifold"]["train_err_z"] = train_err_z
        state["manifold"]["train_pred_raw"] = train_pred_raw
        state["manifold"]["train_rmse"] = tr_rmse
        state["manifold"]["train_mae"] = tr_mae
        state["manifold"]["train_corr"] = tr_corr
        if test_err_z is not None:
            te_rmse, te_mae, te_corr = _compute_metrics(test_err_z, test_obs_raw, test_pred_raw)
            state["manifold"]["test_err_z"] = test_err_z
            state["manifold"]["test_pred_raw"] = test_pred_raw
            state["manifold"]["test_rmse"] = te_rmse
            state["manifold"]["test_mae"] = te_mae
            state["manifold"]["test_corr"] = te_corr
        _redraw_errors()
        _redraw_obs()
        _render_metrics()

    def update_euclidean(train_err_z, train_obs_raw, train_pred_raw,
                         test_err_z, test_obs_raw, test_pred_raw):
        tr_rmse, tr_mae, tr_corr = _compute_metrics(train_err_z, train_obs_raw, train_pred_raw)
        state["euclidean"]["train_err_z"] = train_err_z
        state["euclidean"]["train_pred_raw"] = train_pred_raw
        state["euclidean"]["train_rmse"] = tr_rmse
        state["euclidean"]["train_mae"] = tr_mae
        state["euclidean"]["train_corr"] = tr_corr
        if test_err_z is not None:
            te_rmse, te_mae, te_corr = _compute_metrics(test_err_z, test_obs_raw, test_pred_raw)
            state["euclidean"]["test_err_z"] = test_err_z
            state["euclidean"]["test_pred_raw"] = test_pred_raw
            state["euclidean"]["test_rmse"] = te_rmse
            state["euclidean"]["test_mae"] = te_mae
            state["euclidean"]["test_corr"] = te_corr
        _redraw_errors()
        _redraw_obs()
        _render_metrics()

    def update_hyperparams(nu, ls_m, ls_e, sigma, num_modes):
        state["hyper"]["nu"] = nu
        state["hyper"]["ls_m"] = ls_m
        state["hyper"]["ls_e"] = ls_e
        state["hyper"]["sigma"] = sigma
        state["hyper"]["K"] = num_modes
        _render_metrics()

    def update_summary(summary):
        """Render the aggregate summary table + scatter plot.

        `summary` is a dict from compute_all_lipid_summary() with keys:
          rmse_train_m, rmse_train_e, rmse_test_m, rmse_test_e (np arrays),
          corr_train_m, corr_train_e, corr_test_m, corr_test_e (np arrays),
          n_lipids, hypers, m_wins, e_wins.
        """
        if summary is None:
            return
        s = summary
        lines = []
        n = s["n_lipids"]
        h = s.get("hypers", {})
        lines.append(
            f"At ν={h.get('nu', '?')}, σ={h.get('sigma', 0):.3g}, "
            f"K_modes={h.get('K', '?')}"
        )
        lines.append(
            f"   ℓ_m={h.get('ls_m', 0):.3g}, ℓ_e={h.get('ls_e', 0):.3g}"
        )
        lines.append("─" * 38)
        lines.append(f"{'metric':<16} {'Manifold':>9}  {'Eucl.':>9}")

        def _mean_nz(a):
            if a is None:
                return float("nan")
            v = a[np.isfinite(a)]
            return float(v.mean()) if v.size > 0 else float("nan")

        def _median_nz(a):
            if a is None:
                return float("nan")
            v = a[np.isfinite(a)]
            return float(np.median(v)) if v.size > 0 else float("nan")

        rows = [
            ("mean train RMSE", "rmse_train_m", "rmse_train_e", _mean_nz),
            ("mean test  RMSE", "rmse_test_m",  "rmse_test_e",  _mean_nz),
            ("mean train corr", "corr_train_m", "corr_train_e", _mean_nz),
            ("mean test  corr", "corr_test_m",  "corr_test_e",  _mean_nz),
            ("med  test  corr", "corr_test_m",  "corr_test_e",  _median_nz),
        ]
        for label, mk, ek, fn in rows:
            mv = fn(s.get(mk))
            ev = fn(s.get(ek))
            lines.append(f"{label:<16} {_fmt(mv)}  {_fmt(ev)}")
        lines.append("")
        # Win counts
        tr_mw = s.get("train_m_wins", 0)
        tr_ew = s.get("train_e_wins", 0)
        tr_v = s.get("train_valid", n)
        lines.append(f"train corr wins: M={tr_mw}/{tr_v}  E={tr_ew}/{tr_v}")
        if s.get("corr_test_m") is not None:
            te_mw = s.get("test_m_wins", 0)
            te_ew = s.get("test_e_wins", 0)
            te_v = s.get("test_valid", n)
            lines.append(f"test  corr wins: M={te_mw}/{te_v}  E={te_ew}/{te_v}")
        summary_text.setText("\n".join(lines))
        state["hyper"]["summary_hypers"] = (
            h.get("nu"), h.get("ls_m"), h.get("ls_e"),
            h.get("sigma"), h.get("K"),
        )
        state["hyper"]["summary_stale"] = False

        # ---- Scatter plot of test corr (manifold vs Euclidean) ----
        ax_scatter.clear()
        ax_scatter.set_title("test corr per lipid: M vs E", fontsize=8)
        ax_scatter.tick_params(axis="both", labelsize=7)
        ax_scatter.set_xlabel("Eucl. test corr", fontsize=7)
        ax_scatter.set_ylabel("manifold test corr", fontsize=7)
        cm = s.get("corr_test_m")
        ce = s.get("corr_test_e")
        if cm is None:
            cm = s.get("corr_train_m")
            ce = s.get("corr_train_e")
            ax_scatter.set_title("train corr per lipid: M vs E", fontsize=8)
            ax_scatter.set_xlabel("Eucl. train corr", fontsize=7)
            ax_scatter.set_ylabel("manifold train corr", fontsize=7)
        if cm is not None and ce is not None:
            mask = np.isfinite(cm) & np.isfinite(ce)
            ax_scatter.scatter(ce[mask], cm[mask], s=12, alpha=0.55,
                               color="C0", edgecolor="none")
            # Diagonal y=x line
            lo = float(min(cm[mask].min(), ce[mask].min(), -0.1))
            hi = float(max(cm[mask].max(), ce[mask].max(), 0.1))
            pad = 0.05 * (hi - lo)
            ax_scatter.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                            color="black", linewidth=0.8, alpha=0.5,
                            linestyle="--")
            ax_scatter.set_xlim(lo - pad, hi + pad)
            ax_scatter.set_ylim(lo - pad, hi + pad)
            ax_scatter.axhline(0, color="gray", linewidth=0.4, alpha=0.3)
            ax_scatter.axvline(0, color="gray", linewidth=0.4, alpha=0.3)
        canvas_scatter.draw_idle()

    def mark_summary_stale():
        """Indicate that summary doesn't match current sliders."""
        if not state["hyper"].get("summary_stale", False):
            cur = summary_text.toPlainText()
            if cur and not cur.startswith("(stale)"):
                summary_text.setText("(stale — click 'recompute summary')\n\n"
                                     + cur)
            state["hyper"]["summary_stale"] = True

    return (widget, update_lipid_meta, update_manifold, update_euclidean,
            update_hyperparams, update_summary, mark_summary_stale,
            recompute_btn)


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.get("verbose") else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("visualize_lipid_gp")

    # ---- Setup: eigendecomp + KNN ----------------------------------------
    ctx = _setup_base(args, log)
    K_modes = int(ctx["eigvec"].shape[1])
    N_nodes = int(ctx["eigvec"].shape[0])
    log.info(f"Manifold setup: {N_nodes:,} graph nodes, {K_modes} eigenmodes")

    # Expose coord transform attrs on ctx (used for dense-grid Nyström)
    ctx["stride"] = int(args["stride"])
    ctx["threshold"] = int(args["threshold"])

    # ---- Load MaLDI train + test data ------------------------------------
    data = load_maldi_train_and_test(args, log)
    has_test = data["test_coords_mm"] is not None and data["test_coords_mm"].shape[0] > 0

    # Use TRAIN coords for the global normalization (test inherits same stats)
    coord_mean, coord_std = get_coord_normalization(args, data["train_coords_mm"])
    log.info(f"Coord mean (mm): {coord_mean.tolist()}, "
             f"std: {coord_std.tolist()}")

    train_coords_z = ((data["train_coords_mm"] - coord_mean) / coord_std).to(
        ctx["device"], ctx["eigvec"].dtype,
    )
    if has_test:
        test_coords_z = ((data["test_coords_mm"] - coord_mean) / coord_std).to(
            ctx["device"], ctx["eigvec"].dtype,
        )
    else:
        test_coords_z = None

    # Build reference_nodes (z-scored graph node coords) — required by the
    # Nyström interpolator's KDTree. We adapt to whatever the base setup()
    # exposes:
    log.info(f"Available ctx keys: {sorted(ctx.keys())}")

    if "reference_nodes" in ctx:
        reference_nodes = ctx["reference_nodes"]
        log.info("Using ctx['reference_nodes'] directly")
    elif "voxel_offset" in ctx and "voxel_scale_mm" in ctx:
        # Full pipeline: sub_voxel * scale + offset = mm
        node_mm = (
            ctx["node_voxel_idx"].astype(np.float32)
            * np.asarray(ctx["voxel_scale_mm"], dtype=np.float32)
            + np.asarray(ctx["voxel_offset"], dtype=np.float32)
        )
        reference_nodes = (
            (torch.from_numpy(node_mm) - coord_mean) / coord_std
        ).to(ctx["device"], ctx["eigvec"].dtype)
        log.info("Built reference_nodes via voxel_offset + voxel_scale_mm")
    elif "sv_scale" in ctx and "sv_translate" in ctx:
        # Simpler pipeline: convert sub-voxel to full-template voxel, then
        # to mm using the 25 µm Allen CCF voxel size.
        # sv_scale and sv_translate may be per-axis arrays (shape (3,)).
        CCF_VOXEL_SIZE_MM = 0.025
        _sv_scale = np.asarray(ctx["sv_scale"], dtype=np.float32)
        _sv_translate = np.asarray(ctx["sv_translate"], dtype=np.float32)
        full_voxel = (
            ctx["node_voxel_idx"].astype(np.float32) * _sv_scale
            + _sv_translate
        )
        node_mm = full_voxel * CCF_VOXEL_SIZE_MM
        reference_nodes = (
            (torch.from_numpy(node_mm) - coord_mean) / coord_std
        ).to(ctx["device"], ctx["eigvec"].dtype)
        log.warning(
            f"Built reference_nodes assuming Allen CCF voxel size = "
            f"{CCF_VOXEL_SIZE_MM} mm. If your atlas has a different "
            f"resolution, training points and graph nodes will be in "
            f"different scales and the GP fit will be wrong."
        )
    else:
        raise RuntimeError(
            f"Can't reconstruct reference_nodes from ctx. Available keys: "
            f"{sorted(ctx.keys())}. Expected at least one of: "
            f"'reference_nodes', ('voxel_offset', 'voxel_scale_mm'), or "
            f"('sv_scale', 'sv_translate')."
        )

    ctx["reference_nodes"] = reference_nodes
    log.info(f"reference_nodes: shape {tuple(reference_nodes.shape)}, "
             f"mean={reference_nodes.mean(dim=0).cpu().tolist()}, "
             f"std={reference_nodes.std(dim=0).cpu().tolist()}")

    # Optional subsample for speed — applied SYMMETRICALLY to keep all
    # aligned arrays in lockstep across the lot.
    def _maybe_subsample(coords_z, raw, z, mm, label, n_target):
        if n_target <= 0 or coords_z.shape[0] <= n_target:
            return coords_z, raw, z, mm
        rng = np.random.default_rng(args.get("source_seed", 0))
        sel = rng.choice(coords_z.shape[0], n_target, replace=False)
        log.info(f"Subsampled {label} from {coords_z.shape[0]:,} → {n_target:,}")
        return coords_z[sel], raw[sel], z[sel], mm[sel]

    train_coords_z, train_raw, train_z, train_mm = _maybe_subsample(
        train_coords_z, data["train_values_raw"], data["train_values_z"],
        data["train_coords_mm"], "train", int(args["training_subsample"]),
    )
    if has_test:
        test_coords_z, test_raw, test_z, test_mm = _maybe_subsample(
            test_coords_z, data["test_values_raw"], data["test_values_z"],
            data["test_coords_mm"], "test", int(args["training_subsample"]),
        )

    # ---- Nyström interpolation: Φ at train and test points ---------------
    log.info("Interpolating eigvecs at train points via Nyström...")
    eigvec_K = ctx["eigvec"][:, :K_modes]
    phi_train = nystrom_eigvecs_at_points(
        train_coords_z, ctx, eigvec_K,
        nearest_neighbors=min(10, args["knn_k"]),
        bump_scale=float(args["bump_scale"]),
        bump_decay=float(args["bump_decay"]),
    )
    nonzero_train = (phi_train.abs().sum(dim=1) > 0)
    if nonzero_train.sum().item() < train_coords_z.shape[0] * 0.95:
        log.warning(f"Only {int(nonzero_train.sum())} / {train_coords_z.shape[0]} "
                    f"train points within bump support. Raise --bump-scale "
                    f"if this is unexpected.")

    phi_train = phi_train[nonzero_train]
    train_z_keep = train_z[nonzero_train.cpu()]
    train_raw_keep = train_raw[nonzero_train.cpu()]
    train_mm_keep = train_mm[nonzero_train.cpu()]
    train_coords_z_keep = train_coords_z[nonzero_train]
    log.info(f"Train set for GP fit: {phi_train.shape[0]:,} points")

    if has_test:
        log.info("Interpolating eigvecs at test points via Nyström...")
        phi_test = nystrom_eigvecs_at_points(
            test_coords_z, ctx, eigvec_K,
            nearest_neighbors=min(10, args["knn_k"]),
            bump_scale=float(args["bump_scale"]),
            bump_decay=float(args["bump_decay"]),
        )
        nonzero_test = (phi_test.abs().sum(dim=1) > 0)
        if nonzero_test.sum().item() < test_coords_z.shape[0] * 0.95:
            log.warning(f"Only {int(nonzero_test.sum())} / {test_coords_z.shape[0]} "
                        f"test points within bump support.")
        phi_test = phi_test[nonzero_test]
        test_z_keep = test_z[nonzero_test.cpu()]
        test_raw_keep = test_raw[nonzero_test.cpu()]
        test_mm_keep = test_mm[nonzero_test.cpu()]
        test_coords_z_keep = test_coords_z[nonzero_test]
        log.info(f"Test set: {phi_test.shape[0]:,} points")
    else:
        phi_test = None
        test_z_keep = test_raw_keep = test_mm_keep = None
        test_coords_z_keep = None

    # ---- Pick initial lipid ----------------------------------------------
    lipid_names = data["lipid_names"]
    n_lipids = len(lipid_names)

    if args.get("initial_lipid_name"):
        target = args["initial_lipid_name"].strip()
        match_idxs = [i for i, n in enumerate(lipid_names) if n == target]
        if len(match_idxs) == 0:
            match_idxs = [i for i, n in enumerate(lipid_names)
                          if target.lower() in n.lower()]
        if len(match_idxs) >= 1:
            initial_lipid_idx = match_idxs[0]
            log.info(f"Initial lipid by name '{target}' → "
                     f"idx {initial_lipid_idx}: {lipid_names[initial_lipid_idx]}")
        else:
            log.warning(f"No lipid matches '{target}'; using idx 0.")
            initial_lipid_idx = 0
    else:
        initial_lipid_idx = int(np.clip(args["initial_lipid_idx"], 0, n_lipids - 1))

    # ---- Dense-grid query coords (computed lazily on first use) ----------
    # We capture the voxel→mm transform once here so build_dense_query_grid
    # doesn't have to re-do the ctx-key dance every time.
    if "voxel_offset" in ctx and "voxel_scale_mm" in ctx:
        _vox_offset = np.asarray(ctx["voxel_offset"], dtype=np.float32)
        _vox_scale = np.asarray(ctx["voxel_scale_mm"], dtype=np.float32)
        def _subvoxel_to_mm(sub_voxel: np.ndarray) -> np.ndarray:
            return sub_voxel * _vox_scale + _vox_offset
    else:
        # Fallback: full_voxel * 0.025 (CCF voxel size)
        CCF_VOXEL_SIZE_MM = 0.025
        _sv_scale = np.asarray(ctx["sv_scale"], dtype=np.float32)
        _sv_translate = np.asarray(ctx["sv_translate"], dtype=np.float32)
        def _subvoxel_to_mm(sub_voxel: np.ndarray) -> np.ndarray:
            full_voxel = sub_voxel * _sv_scale + _sv_translate
            return full_voxel * CCF_VOXEL_SIZE_MM

    def build_dense_query_grid(render_stride: int):
        tmpl = ctx["template_full"]
        mask = tmpl > ctx["threshold"]
        sub_mask = mask[::render_stride, ::render_stride, ::render_stride]
        sub_idx = np.argwhere(sub_mask).astype(np.int32)
        voxel_idx = sub_idx * render_stride
        mm = _subvoxel_to_mm(voxel_idx.astype(np.float32) / float(ctx["stride"]))
        coords_z = (
            (torch.from_numpy(mm) - coord_mean) / coord_std
        ).to(ctx["eigvec"].device, ctx["eigvec"].dtype)
        return coords_z, voxel_idx

    # ---- Napari setup -----------------------------------------------------
    if args.get("no_launch"):
        log.info("--no-launch passed; precompute OK, exiting.")
        return

    import napari
    from magicgui import magicgui

    viewer = napari.Viewer(title="lipid GP visualizer")
    viewer.dims.ndisplay = 3

    # Full graph node positions in template voxel coords
    all_node_positions = (
        ctx["node_voxel_idx"].astype(np.float32) * ctx["sv_scale"]
        + ctx["sv_translate"]
    )
    # Helper: convert mm coords (any tensor) → full-template voxel positions
    def _mm_to_voxel_positions(coords_mm_np):
        if "voxel_offset" in ctx and "voxel_scale_mm" in ctx:
            _vox_offset_inv = np.asarray(ctx["voxel_offset"], dtype=np.float32)
            _vox_scale_inv = np.asarray(ctx["voxel_scale_mm"], dtype=np.float32)
            sub_voxel = (coords_mm_np - _vox_offset_inv) / _vox_scale_inv
            return (sub_voxel * float(ctx["stride"])).astype(np.float32)
        CCF_VOXEL_SIZE_MM = 0.025
        return (coords_mm_np / CCF_VOXEL_SIZE_MM).astype(np.float32)

    train_voxel_positions = _mm_to_voxel_positions(train_mm_keep.cpu().numpy())
    if has_test:
        test_voxel_positions = _mm_to_voxel_positions(test_mm_keep.cpu().numpy())

    # ---- Layer 0: template ------------------------------------------------
    tpl = ctx["template_full"]
    tpl_max = float(np.percentile(tpl[tpl > 0], 99)) if (tpl > 0).any() else 1.0
    viewer.add_image(
        tpl, name="0  template (reference)",
        colormap="gray", rendering="attenuated_mip",
        attenuation=0.05, contrast_limits=(0.0, tpl_max),
        opacity=0.5, blending="translucent",
    )

    # ---- Layer 1: annotations (anatomical labels) ------------------------
    # The Labels layer may fail on the same vispy "Volume needs a 3D array"
    # error we've seen elsewhere. When it does, napari leaves an orphan
    # layer in the model list — we clean up by name to avoid breaking
    # subsequent add_points calls.
    def _safe_add_labels(viewer, name, data, **kwargs):
        try:
            return viewer.add_labels(data, name=name, **kwargs)
        except Exception as exc:
            log.warning(f"Failed to add Labels layer '{name}': {exc}")
            for lyr in list(viewer.layers):
                if lyr.name == name:
                    try:
                        viewer.layers.remove(lyr)
                        log.warning(f"  Removed orphaned model layer '{name}'.")
                    except Exception:
                        pass
            return None

    if args.get("annotations_file"):
        try:
            ann = np.load(args["annotations_file"])
            log.info(f"Loaded annotations: shape {ann.shape}, "
                     f"{len(np.unique(ann)):,} unique labels")
            ann_layer = _safe_add_labels(
                viewer, "1  annotations (atlas)",
                ann.astype(np.int32),
                opacity=0.35,
                blending="translucent",
                visible=False,
            )
            if ann_layer is None:
                log.warning(
                    "Falling back to Points-based annotation rendering "
                    "(sub-sampled voxels colored by label)."
                )
                sub_stride = 6
                ann_sub = ann[::sub_stride, ::sub_stride, ::sub_stride]
                nz_mask = ann_sub > 0
                nz_positions = np.argwhere(nz_mask).astype(np.float32)
                if nz_positions.shape[0] > 0:
                    labels_at = ann_sub[nz_mask]
                    full_pos = nz_positions * sub_stride
                    # Subsample further if still too many points
                    max_ann_pts = 200_000
                    if full_pos.shape[0] > max_ann_pts:
                        rng_ann = np.random.default_rng(0)
                        sel = rng_ann.choice(full_pos.shape[0], max_ann_pts, replace=False)
                        full_pos = full_pos[sel]
                        labels_at = labels_at[sel]
                    # Cycle colors over distinct label IDs
                    unique_labels = np.unique(labels_at)
                    cmap = cm.get_cmap("tab20")
                    label_to_color = {
                        int(l): cmap(i % 20)
                        for i, l in enumerate(unique_labels)
                    }
                    colors = np.array(
                        [label_to_color[int(l)] for l in labels_at],
                        dtype=np.float32,
                    )
                    viewer.add_points(
                        full_pos,
                        name="1  annotations (atlas, sparse fallback)",
                        size=1.5,
                        face_color=colors,
                        border_color=colors,
                        opacity=0.6,
                        blending="translucent",
                        visible=False,
                    )
                    log.info(f"Added {full_pos.shape[0]:,} annotation points "
                             f"(fallback, stride={sub_stride})")
        except Exception as e:
            log.warning(f"Failed to load annotations: {e}")
    else:
        log.info("No --annotations-file given; skipping annotations layer.")

    # ---- Layer 2: KNN graph nodes (faint fabric) -------------------------
    viewer.add_points(
        all_node_positions, name="2  KNN graph nodes (fabric)",
        size=0.6, face_color="white", border_color="white",
        opacity=0.25, blending="additive", visible=False,
    )

    # ---- Layer 3: training data — raw lipid values -----------------------
    train_raw_layer = viewer.add_points(
        train_voxel_positions, name="3  train data (raw lipid)",
        size=2.5, face_color="white", border_color="white",
        symbol="o", opacity=0.95, blending="translucent", visible=True,
    )

    # ---- Layer 4: test data — raw lipid values ---------------------------
    if has_test:
        test_raw_layer = viewer.add_points(
            test_voxel_positions, name="4  test data (raw lipid)",
            size=2.5, face_color="white", border_color="white",
            symbol="square", opacity=0.95, blending="translucent", visible=True,
        )
    else:
        test_raw_layer = None

    # ---- Layer 5: GP pred @ train points ---------------------------------
    train_pred_layer = viewer.add_points(
        train_voxel_positions, name="5  GP pred @ train pts",
        size=2.5, face_color="white", border_color="white",
        symbol="o", opacity=0.95, blending="translucent", visible=False,
    )

    # ---- Layer 6: GP pred @ test points ----------------------------------
    if has_test:
        test_pred_layer = viewer.add_points(
            test_voxel_positions, name="6  GP pred @ test pts",
            size=2.5, face_color="white", border_color="white",
            symbol="square", opacity=0.95, blending="translucent", visible=False,
        )
    else:
        test_pred_layer = None

    # ---- Layer 7: GP errors @ train (signed, z-scored) -------------------
    train_err_layer = viewer.add_points(
        train_voxel_positions, name="7  GP error @ train (z)",
        size=2.5, face_color="white", border_color="white",
        symbol="o", opacity=0.95, blending="translucent", visible=True,
    )

    # ---- Layer 8: GP errors @ test (signed, z-scored) --------------------
    if has_test:
        test_err_layer = viewer.add_points(
            test_voxel_positions, name="8  GP error @ test (z)",
            size=2.5, face_color="white", border_color="white",
            symbol="square", opacity=0.95, blending="translucent", visible=True,
        )
    else:
        test_err_layer = None

    # ---- Layer 9: Euclidean GP pred @ train pts --------------------------
    eucl_train_pred_layer = viewer.add_points(
        train_voxel_positions, name="9  Eucl. GP pred @ train pts",
        size=2.5, face_color="white", border_color="white",
        symbol="o", opacity=0.95, blending="translucent", visible=False,
    )

    # ---- Layer 10: Euclidean GP pred @ test pts --------------------------
    if has_test:
        eucl_test_pred_layer = viewer.add_points(
            test_voxel_positions, name="10 Eucl. GP pred @ test pts",
            size=2.5, face_color="white", border_color="white",
            symbol="square", opacity=0.95, blending="translucent", visible=False,
        )
    else:
        eucl_test_pred_layer = None

    # ---- Layer 11: Euclidean GP errors @ train (signed, z-scored) --------
    eucl_train_err_layer = viewer.add_points(
        train_voxel_positions, name="11 Eucl. GP error @ train (z)",
        size=2.5, face_color="white", border_color="white",
        symbol="o", opacity=0.95, blending="translucent", visible=False,
    )

    # ---- Layer 12: Euclidean GP errors @ test (signed, z-scored) ---------
    if has_test:
        eucl_test_err_layer = viewer.add_points(
            test_voxel_positions, name="12 Eucl. GP error @ test (z)",
            size=2.5, face_color="white", border_color="white",
            symbol="square", opacity=0.95, blending="translucent", visible=False,
        )
    else:
        eucl_test_err_layer = None

    # ---- Layer 13: manifold GP pred @ all graph nodes --------------------
    pred_nodes_layer = viewer.add_points(
        all_node_positions, name="13 manifold GP pred @ graph nodes",
        size=0.9, face_color="white", border_color="white",
        opacity=0.85, blending="translucent", visible=False,
    )

    # ---- Layer 14: Euclidean GP pred @ all graph nodes -------------------
    eucl_graph_layer = viewer.add_points(
        all_node_positions, name="14 Eucl. GP pred @ graph nodes",
        size=0.9, face_color="white", border_color="white",
        opacity=0.85, blending="translucent", visible=False,
    )

    # ---- Layer 15: manifold GP pred @ dense voxels -----------------------
    dense_layer = viewer.add_points(
        np.zeros((1, 3), dtype=np.float32),
        name=f"15 manifold GP pred @ dense voxels (stride={args['render_stride']})",
        size=0.9, face_color="white", border_color="white",
        opacity=0.85, blending="translucent", visible=False,
    )

    # ---- Layer 16: Euclidean GP pred @ dense voxels ----------------------
    eucl_dense_layer = viewer.add_points(
        np.zeros((1, 3), dtype=np.float32),
        name=f"16 Eucl. GP pred @ dense voxels (stride={args['render_stride']})",
        size=0.9, face_color="white", border_color="white",
        opacity=0.85, blending="translucent", visible=False,
    )

    # ---- State for sliders ------------------------------------------------
    state = dict(
        lipid_idx=initial_lipid_idx,
        nu=int(args["nu"]),
        lengthscale=float(args["lengthscale"]),
        lengthscale_eucl=float(
            args["lengthscale_eucl"] if args.get("lengthscale_eucl") is not None
            else args["lengthscale"]
        ),
        noise_sigma=float(args["noise_sigma"]),
        num_modes=K_modes,
        render_stride=int(args["render_stride"]),
        gamma=0.7,
    )

    # ---- Predict + refresh ------------------------------------------------
    rng_render = np.random.default_rng(args.get("source_seed", 0))

    # Info panel created HERE (before refresh functions are defined) so its
    # callbacks can be captured by the refresh closures.
    (info_widget, info_update_lipid, info_update_manifold,
     info_update_euclidean, info_update_hyper,
     info_update_summary, info_mark_stale,
     info_recompute_btn) = make_info_panel()
    if info_widget is not None:
        viewer.window.add_dock_widget(info_widget, name="lipid GP stats",
                                      area="right")
        info_update_hyper(state["nu"], state["lengthscale"],
                          state["lengthscale_eucl"],
                          state["noise_sigma"], state["num_modes"])

    def predict_for_current_lipid():
        """Compute everything dependent on the current lipid + GP hypers.

        Returns dict with:
          mu_c                     — (K,) coefficient vector
          y_pred_train_z           — (N_train,) pred at train, z-score scale
          y_pred_test_z            — (N_test,) or None
          y_pred_full_graph_z      — (N_nodes,) pred at all graph nodes
          + corresponding _raw versions and signed errors (z-scored)
        """
        K = state["num_modes"]
        phi_train_K = phi_train[:, :K]
        y_train_z_tensor = train_z_keep[:, state["lipid_idx"]].to(
            ctx["device"], ctx["eigvec"].dtype,
        )
        mu_c = fit_spectral_gp(
            ctx["eigval"], phi_train_K, phi_train_K, y_train_z_tensor,
            nu=state["nu"], lengthscale=state["lengthscale"],
            noise_sigma=state["noise_sigma"],
        )
        # Predictions in z-scored space
        y_pred_train_z = (phi_train_K @ mu_c).cpu().numpy()
        y_pred_full_z = (ctx["eigvec"][:, :K] @ mu_c).cpu().numpy()
        if has_test:
            phi_test_K = phi_test[:, :K]
            y_pred_test_z = (phi_test_K @ mu_c).cpu().numpy()
        else:
            y_pred_test_z = None

        # De-standardize to raw scale
        mean_l = data["col_means"][state["lipid_idx"]].item()
        std_l = data["col_stds"][state["lipid_idx"]].item()

        y_train_obs_z = train_z_keep[:, state["lipid_idx"]].cpu().numpy()
        if has_test:
            y_test_obs_z = test_z_keep[:, state["lipid_idx"]].cpu().numpy()
        else:
            y_test_obs_z = None

        return {
            "mu_c": mu_c,
            "y_pred_train_z": y_pred_train_z,
            "y_pred_test_z": y_pred_test_z,
            "y_pred_full_z": y_pred_full_z,
            "y_pred_train_raw": y_pred_train_z * std_l + mean_l,
            "y_pred_test_raw": (y_pred_test_z * std_l + mean_l) if y_pred_test_z is not None else None,
            "y_pred_full_raw": y_pred_full_z * std_l + mean_l,
            "y_train_obs_z": y_train_obs_z,
            "y_test_obs_z": y_test_obs_z,
            "err_train_z": y_train_obs_z - y_pred_train_z,
            "err_test_z": (y_test_obs_z - y_pred_test_z) if y_pred_test_z is not None else None,
        }

    # ---- Euclidean GP setup: pre-build train subsample + cache state -----
    # Cholesky on the Euclidean kernel matrix is O(N³) and dominates the
    # cost. We subsample to a smaller fixed set, cache the Cholesky factor
    # against (nu, lengthscale, noise_sigma), and only recompute when those
    # change. We ALSO cache the K matrices for predictions at train/test
    # points — those depend only on (nu, lengthscale), not noise sigma.
    eucl_n_target = min(int(args["eucl_subsample"]),
                        train_coords_z.shape[0])
    rng_e = np.random.default_rng(args.get("source_seed", 0))
    eucl_sel = rng_e.choice(train_coords_z.shape[0], eucl_n_target, replace=False)
    X_train_eucl = train_coords_z[eucl_sel]
    y_train_eucl_z_all = train_z[eucl_sel.tolist() if isinstance(eucl_sel, np.ndarray) else eucl_sel]
    y_train_eucl_device = y_train_eucl_z_all.to(
        ctx["device"], ctx["eigvec"].dtype,
    )
    X_nodes_eucl = ctx["reference_nodes"]
    log.info(f"Euclidean GP subsample: {eucl_n_target:,} train points")

    _eucl_caches = {
        "L_key": None, "L": None,
        "K_key": None, "K_train_full": None, "K_test_full": None,
    }

    def _ensure_eucl_L(nu, ls, sig):
        key = (nu, ls, sig)
        if _eucl_caches["L_key"] == key:
            return
        log.info(f"Computing Euclidean Cholesky (ν={nu}, ℓ={ls:.3g}, "
                 f"σ={sig:.3g}, N={X_train_eucl.shape[0]})...")
        K = matern_euclidean_pairwise(X_train_eucl, X_train_eucl, nu, ls)
        n = K.shape[0]
        sigma_sq = max(sig, 1e-6) ** 2
        K_reg = K + sigma_sq * torch.eye(n, device=K.device, dtype=K.dtype)
        K_reg.diagonal().add_(1e-6)
        _eucl_caches["L"] = torch.linalg.cholesky(K_reg)
        _eucl_caches["L_key"] = key

    def _ensure_eucl_K_matrices(nu, ls):
        """Cache K(X_train_full, X_train_eucl) and K(X_test_full, X_train_eucl).
        These only depend on (ν, ℓ), not on σ — so they survive σ slider scrubs."""
        key = (nu, ls)
        if _eucl_caches["K_key"] == key:
            return
        log.info(f"Computing Euclidean train/test K matrices "
                 f"(ν={nu}, ℓ={ls:.3g})...")
        _eucl_caches["K_train_full"] = matern_euclidean_pairwise(
            train_coords_z_keep, X_train_eucl, nu, ls,
        )
        if has_test:
            _eucl_caches["K_test_full"] = matern_euclidean_pairwise(
                test_coords_z_keep, X_train_eucl, nu, ls,
            )
        _eucl_caches["K_key"] = key

    def refresh_euclidean_predictions():
        """Compute Euclidean GP predictions for all visible Euclidean layers.

        When the info panel exists we ALSO compute train+test predictions
        unconditionally so the metric panel stays current — these are cheap
        given the cached K matrices.
        """
        needs_train_layer = (eucl_train_pred_layer.visible
                             or eucl_train_err_layer.visible)
        needs_test_layer = (has_test
                            and (eucl_test_pred_layer.visible
                                 or eucl_test_err_layer.visible))
        needs_graph = eucl_graph_layer.visible
        needs_dense = eucl_dense_layer.visible
        info_active = info_widget is not None
        # If the info panel is active, always compute train (and test) for
        # metrics — cheap with cached K.
        do_train = needs_train_layer or info_active
        do_test = (has_test and (needs_test_layer or info_active))

        if not (do_train or do_test or needs_graph or needs_dense):
            return

        nu, ls, sig = state["nu"], state["lengthscale_eucl"], state["noise_sigma"]
        _ensure_eucl_L(nu, ls, sig)
        if do_train or do_test:
            _ensure_eucl_K_matrices(nu, ls)

        # Solve for dual coefficients α = (K + σ²I)⁻¹ y. Cheap given cached L.
        y_z = y_train_eucl_device[:, state["lipid_idx"]]
        alpha = torch.cholesky_solve(
            y_z.unsqueeze(-1), _eucl_caches["L"],
        ).squeeze(-1)

        mean_l = data["col_means"][state["lipid_idx"]].item()
        std_l = data["col_stds"][state["lipid_idx"]].item()
        gamma = state["gamma"]

        bits = [f"[Euclidean ν={nu} ℓ_e={ls:.3g} σ={sig:.3g}]"]

        # Will be captured for the info panel
        eucl_train_err_z = eucl_test_err_z = None
        eucl_train_pred_raw = eucl_test_pred_raw = None
        y_train_obs_raw = y_test_obs_raw = None

        if do_train:
            y_pred_train_z = (
                _eucl_caches["K_train_full"] @ alpha
            ).cpu().numpy()
            eucl_train_pred_raw = y_pred_train_z * std_l + mean_l
            y_obs_z = train_z_keep[:, state["lipid_idx"]].cpu().numpy()
            y_train_obs_raw = train_raw_keep[:, state["lipid_idx"]].cpu().numpy()
            eucl_train_err_z = y_obs_z - y_pred_train_z
            if eucl_train_pred_layer.visible:
                color_nodes_sequential(eucl_train_pred_layer, eucl_train_pred_raw,
                                       gamma, "magma")
            if eucl_train_err_layer.visible:
                color_nodes_diverging(eucl_train_err_layer, eucl_train_err_z,
                                      "RdBu_r", gamma=gamma)
            rmse = float(np.sqrt(np.mean(eucl_train_err_z ** 2)))
            bits.append(f"train RMSE(z)={rmse:.4g}")

        if do_test:
            y_pred_test_z = (
                _eucl_caches["K_test_full"] @ alpha
            ).cpu().numpy()
            eucl_test_pred_raw = y_pred_test_z * std_l + mean_l
            y_obs_z = test_z_keep[:, state["lipid_idx"]].cpu().numpy()
            y_test_obs_raw = test_raw_keep[:, state["lipid_idx"]].cpu().numpy()
            eucl_test_err_z = y_obs_z - y_pred_test_z
            if eucl_test_pred_layer.visible:
                color_nodes_sequential(eucl_test_pred_layer, eucl_test_pred_raw,
                                       gamma, "magma")
            if eucl_test_err_layer.visible:
                color_nodes_diverging(eucl_test_err_layer, eucl_test_err_z,
                                      "RdBu_r", gamma=gamma)
            rmse = float(np.sqrt(np.mean(eucl_test_err_z ** 2)))
            bits.append(f"test RMSE(z)={rmse:.4g}")

        if needs_graph:
            y_pred_z = predict_euclidean_batched(
                X_nodes_eucl, X_train_eucl, alpha,
                nu=nu, lengthscale=ls,
                batch_size=int(args["eucl_batch_size"]),
            ).cpu().numpy()
            y_pred_raw = y_pred_z * std_l + mean_l
            color_nodes_sequential(eucl_graph_layer, y_pred_raw,
                                   gamma, "magma")
            bits.append(f"graph range=[{y_pred_raw.min():.4g}, {y_pred_raw.max():.4g}]")

        if needs_dense:
            coords_z, voxel_idx = build_dense_query_grid(state["render_stride"])
            log.info(f"Eucl. dense grid: {coords_z.shape[0]:,} query points")
            y_pred_z = predict_euclidean_batched(
                coords_z, X_train_eucl, alpha,
                nu=nu, lengthscale=ls,
                batch_size=int(args["eucl_batch_size"]),
            ).cpu().numpy()
            y_pred_raw = y_pred_z * std_l + mean_l
            positions = voxel_idx.astype(np.float32)
            max_pts = int(args["max_render_points"])
            if positions.shape[0] > max_pts:
                sel = rng_render.choice(
                    positions.shape[0], size=max_pts, replace=False,
                )
                positions = positions[sel]
                y_pred_raw = y_pred_raw[sel]
            eucl_dense_layer.data = positions
            color_nodes_sequential(eucl_dense_layer, y_pred_raw,
                                   gamma, "magma")
            bits.append(f"dense {positions.shape[0]:,} pts")

        print("  ".join(bits))

        # Update info panel with Euclidean metrics
        if info_active:
            info_update_euclidean(
                eucl_train_err_z, y_train_obs_raw, eucl_train_pred_raw,
                eucl_test_err_z, y_test_obs_raw, eucl_test_pred_raw,
            )

    def compute_all_lipid_summary():
        """Fit BOTH GPs for every lipid at current hyperparams and return
        aggregated metrics. Batched: kernel matrices and Cholesky factors
        are shared across lipids; only the linear-system RHS varies.

        Cost (typical): manifold ~50ms, Euclidean ~200ms, metrics ~10ms.
        """
        import time
        t0 = time.time()
        nu = state["nu"]
        ls_m = state["lengthscale"]
        ls_e = state["lengthscale_eucl"]
        sig = state["noise_sigma"]
        K = state["num_modes"]
        device = ctx["device"]
        dtype = ctx["eigvec"].dtype

        # ---- Manifold: batched solve ----
        phi_train_K = phi_train[:, :K]
        Y_train_z_all = train_z_keep.to(device, dtype)  # (N_tr, n_lipids)
        safe_lam = ctx["eigval"][:K].clamp(min=0.0)
        w = (2.0 * nu / (ls_m ** 2) + safe_lam).pow(-nu)
        inv_w = 1.0 / w.clamp(min=1e-30)
        sigma_sq = max(sig, 1e-6) ** 2
        A = torch.diag(inv_w) + (1.0 / sigma_sq) * (phi_train_K.T @ phi_train_K)
        rhs = (1.0 / sigma_sq) * (phi_train_K.T @ Y_train_z_all)  # (K, n_lipids)
        mu_c_batch = torch.linalg.solve(A, rhs)
        y_pred_train_m_z = phi_train_K @ mu_c_batch
        if has_test:
            Y_test_z_all = test_z_keep.to(device, dtype)
            y_pred_test_m_z = phi_test[:, :K] @ mu_c_batch
        else:
            Y_test_z_all = None
            y_pred_test_m_z = None

        # ---- Euclidean: batched solve ----
        _ensure_eucl_L(nu, ls_e, sig)
        _ensure_eucl_K_matrices(nu, ls_e)
        alpha_batch = torch.cholesky_solve(
            y_train_eucl_device, _eucl_caches["L"],
        )  # (N_eucl, n_lipids)
        y_pred_train_e_z = _eucl_caches["K_train_full"] @ alpha_batch
        if has_test:
            y_pred_test_e_z = _eucl_caches["K_test_full"] @ alpha_batch
        else:
            y_pred_test_e_z = None

        # ---- Vectorized per-lipid metrics ----
        def _column_rmse(y_true, y_pred):
            return (y_true - y_pred).pow(2).mean(dim=0).sqrt().cpu().numpy()

        def _column_corr(y_true, y_pred):
            y_c = y_true - y_true.mean(dim=0, keepdim=True)
            p_c = y_pred - y_pred.mean(dim=0, keepdim=True)
            num = (y_c * p_c).sum(dim=0)
            den = (y_c.pow(2).sum(dim=0)
                   * p_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-30)
            return (num / den).cpu().numpy()

        rmse_train_m = _column_rmse(Y_train_z_all, y_pred_train_m_z)
        rmse_train_e = _column_rmse(Y_train_z_all, y_pred_train_e_z)
        corr_train_m = _column_corr(Y_train_z_all, y_pred_train_m_z)
        corr_train_e = _column_corr(Y_train_z_all, y_pred_train_e_z)

        summary = {
            "n_lipids": Y_train_z_all.shape[1],
            "hypers": {"nu": nu, "ls_m": ls_m, "ls_e": ls_e,
                       "sigma": sig, "K": K},
            "rmse_train_m": rmse_train_m, "rmse_train_e": rmse_train_e,
            "corr_train_m": corr_train_m, "corr_train_e": corr_train_e,
        }
        if has_test:
            summary["rmse_test_m"] = _column_rmse(Y_test_z_all, y_pred_test_m_z)
            summary["rmse_test_e"] = _column_rmse(Y_test_z_all, y_pred_test_e_z)
            summary["corr_test_m"] = _column_corr(Y_test_z_all, y_pred_test_m_z)
            summary["corr_test_e"] = _column_corr(Y_test_z_all, y_pred_test_e_z)

        # Win counts (by correlation, higher = better)
        def _wins(a, b):
            valid = np.isfinite(a) & np.isfinite(b)
            return (int(((a > b) & valid).sum()),
                    int(((b > a) & valid).sum()),
                    int(valid.sum()))

        tm, te, tv = _wins(corr_train_m, corr_train_e)
        summary["train_m_wins"] = tm
        summary["train_e_wins"] = te
        summary["train_valid"] = tv
        if has_test:
            xm, xe, xv = _wins(summary["corr_test_m"], summary["corr_test_e"])
            summary["test_m_wins"] = xm
            summary["test_e_wins"] = xe
            summary["test_valid"] = xv

        dt = time.time() - t0
        log.info(
            f"All-lipid summary computed in {dt*1000:.0f}ms "
            f"(n={summary['n_lipids']}, ν={nu}, ℓ_m={ls_m:.3g}, "
            f"ℓ_e={ls_e:.3g}, σ={sig:.3g})"
        )
        return summary

    def refresh_predictions():
        """Re-fit manifold GP and recolor all dependent layers."""
        out = predict_for_current_lipid()
        gamma = state["gamma"]

        # Raw observation layers (per-lipid raw scale)
        y_train_raw = train_raw_keep[:, state["lipid_idx"]].cpu().numpy()
        color_nodes_sequential(train_raw_layer, y_train_raw, gamma, "magma")
        if has_test:
            y_test_raw = test_raw_keep[:, state["lipid_idx"]].cpu().numpy()
            color_nodes_sequential(test_raw_layer, y_test_raw, gamma, "magma")

        # Prediction layers (same raw scale as raw layers, but using the
        # train-set's vmin/vmax as a common scale so train/test/pred colors
        # are directly comparable)
        color_nodes_sequential(train_pred_layer, out["y_pred_train_raw"],
                               gamma, "magma")
        if has_test:
            color_nodes_sequential(test_pred_layer, out["y_pred_test_raw"],
                                   gamma, "magma")

        # Error layers (signed, z-scored — diverging colormap)
        color_nodes_diverging(train_err_layer, out["err_train_z"],
                              "RdBu_r", gamma=gamma)
        if has_test:
            color_nodes_diverging(test_err_layer, out["err_test_z"],
                                  "RdBu_r", gamma=gamma)

        # Per-node prediction (raw scale)
        color_nodes_sequential(pred_nodes_layer, out["y_pred_full_raw"],
                               gamma, "magma")

        # ---- Log summary + info panel ----
        lipid_name = lipid_names[state["lipid_idx"]]
        train_rmse_z = float(np.sqrt(np.mean(out["err_train_z"] ** 2)))
        train_corr = float(np.corrcoef(y_train_raw,
                                       out["y_pred_train_raw"])[0, 1])
        bits = [
            f"[lipid {state['lipid_idx']:3d} '{lipid_name}']",
            f"train RMSE(z)={train_rmse_z:.4g}",
            f"train corr={train_corr:+.3f}",
        ]
        y_test_raw_local = None
        if has_test:
            test_rmse_z = float(np.sqrt(np.mean(out["err_test_z"] ** 2)))
            y_test_raw_local = test_raw_keep[:, state["lipid_idx"]].cpu().numpy()
            test_corr = float(np.corrcoef(y_test_raw_local,
                                          out["y_pred_test_raw"])[0, 1])
            bits.append(f"test RMSE(z)={test_rmse_z:.4g}")
            bits.append(f"test corr={test_corr:+.3f}")
        print("  ".join(bits))

        # Update info panel (lipid context + manifold metrics)
        if info_widget is not None:
            info_update_lipid(state["lipid_idx"], lipid_name,
                              y_train_raw, y_test_raw_local)
            info_update_manifold(
                out["err_train_z"], y_train_raw, out["y_pred_train_raw"],
                out["err_test_z"], y_test_raw_local, out["y_pred_test_raw"],
            )
            info_update_hyper(state["nu"], state["lengthscale"],
                              state["lengthscale_eucl"],
                              state["noise_sigma"], state["num_modes"])

        return out["mu_c"]

    last_mu_c = [None]

    def refresh_dense(mu_c=None):
        """Compute dense-grid predictions for the current lipid."""
        if not dense_layer.visible:
            return
        if mu_c is None:
            out = predict_for_current_lipid()
            mu_c = out["mu_c"]
        coords_z, voxel_idx = build_dense_query_grid(state["render_stride"])
        log.info(f"Dense grid: {coords_z.shape[0]:,} query points")
        phi_dense = nystrom_eigvecs_at_points(
            coords_z, ctx, ctx["eigvec"][:, :state["num_modes"]],
            nearest_neighbors=10,
            bump_scale=float(args["bump_scale"]),
            bump_decay=float(args["bump_decay"]),
        )
        y_dense_z = (phi_dense @ mu_c).cpu().numpy()
        mean_l = data["col_means"][state["lipid_idx"]].item()
        std_l = data["col_stds"][state["lipid_idx"]].item()
        y_dense_raw = y_dense_z * std_l + mean_l
        mask = phi_dense.abs().sum(dim=1).cpu().numpy() > 1e-12
        positions = voxel_idx[mask].astype(np.float32)
        y_keep_d = y_dense_raw[mask]
        max_pts = int(args["max_render_points"])
        if positions.shape[0] > max_pts:
            sel = rng_render.choice(positions.shape[0], size=max_pts, replace=False)
            positions = positions[sel]
            y_keep_d = y_keep_d[sel]
        if positions.shape[0] == 0:
            return
        dense_layer.data = positions
        color_nodes_sequential(dense_layer, y_keep_d, state["gamma"], "magma")
        print(f"[dense pred] rendered {positions.shape[0]:,} voxels, "
              f"raw range [{y_keep_d.min():.4g}, {y_keep_d.max():.4g}]")

    last_mu_c[0] = refresh_predictions()

    def _on_dense_visibility(event):
        if dense_layer.visible:
            refresh_dense(mu_c=last_mu_c[0])
    dense_layer.events.visible.connect(_on_dense_visibility)

    def _on_eucl_visibility(event):
        refresh_euclidean_predictions()

    for lyr in [
        eucl_train_pred_layer, eucl_test_pred_layer,
        eucl_train_err_layer, eucl_test_err_layer,
        eucl_graph_layer, eucl_dense_layer,
    ]:
        if lyr is not None:
            lyr.events.visible.connect(_on_eucl_visibility)

    # ---- Controls dock ----------------------------------------------------
    @magicgui(
        auto_call=True,
        lipid_idx={"label": "lipid index",
                   "min": 0, "max": n_lipids - 1, "step": 1},
        nu={"label": "ν (integer)", "min": 1, "max": 6, "step": 1},
        lengthscale={"label": "ℓ (manifold)", "min": 1e-3, "max": 100.0, "step": 1e-3},
        lengthscale_eucl={"label": "ℓ (Eucl.)", "min": 1e-3, "max": 100.0, "step": 1e-3},
        noise_sigma={"label": "σ noise", "min": 0.01, "max": 5.0, "step": 0.01},
        num_modes={"label": "num modes", "min": 10, "max": K_modes, "step": 10},
        render_stride={"label": "dense stride", "min": 1, "max": 8, "step": 1},
        gamma={"label": "color gamma", "min": 0.1, "max": 2.0, "step": 0.05},
    )
    def controls(
        lipid_idx: int = state["lipid_idx"],
        nu: int = state["nu"],
        lengthscale: float = state["lengthscale"],
        lengthscale_eucl: float = state["lengthscale_eucl"],
        noise_sigma: float = state["noise_sigma"],
        num_modes: int = state["num_modes"],
        render_stride: int = state["render_stride"],
        gamma: float = state["gamma"],
    ):
        chg_lipid = lipid_idx != state["lipid_idx"]
        chg_nu    = nu != state["nu"]
        chg_ls_m  = lengthscale != state["lengthscale"]            # manifold ℓ
        chg_ls_e  = lengthscale_eucl != state["lengthscale_eucl"]  # Eucl. ℓ
        chg_sig   = noise_sigma != state["noise_sigma"]
        chg_K     = num_modes != state["num_modes"]
        chg_rs    = render_stride != state["render_stride"]
        chg_gamma = gamma != state["gamma"]

        state.update(
            lipid_idx=lipid_idx, nu=nu, lengthscale=lengthscale,
            lengthscale_eucl=lengthscale_eucl,
            noise_sigma=noise_sigma, num_modes=num_modes,
            render_stride=render_stride, gamma=gamma,
        )

        # Manifold: only the manifold ℓ matters (chg_ls_m, not chg_ls_e).
        # Other shared params: lipid, nu, σ, num_modes, gamma.
        if (chg_lipid or chg_nu or chg_ls_m or chg_sig or chg_K or chg_gamma):
            last_mu_c[0] = refresh_predictions()

        if dense_layer.visible and (
            chg_lipid or chg_nu or chg_ls_m or chg_sig
            or chg_K or chg_rs or chg_gamma
        ):
            refresh_dense(mu_c=last_mu_c[0])

        # Euclidean: only the Eucl. ℓ matters (chg_ls_e, not chg_ls_m).
        if (chg_lipid or chg_nu or chg_ls_e or chg_sig or chg_gamma or (
            eucl_dense_layer.visible and chg_rs
        )):
            refresh_euclidean_predictions()

        # Aggregate summary depends on hypers; mark stale on relevant changes.
        # (Lipid scrubs don't affect the all-lipid summary.)
        if (chg_nu or chg_ls_m or chg_ls_e or chg_sig or chg_K) and info_mark_stale is not None:
            info_mark_stale()

    viewer.window.add_dock_widget(controls, name="lipid GP controls", area="right")

    # Trigger initial Euclidean fit so the info panel populates with both
    # Manifold and Euclidean columns from the start. The first call here is
    # the "slow" one (Cholesky); subsequent calls reuse the cached factor.
    refresh_euclidean_predictions()

    # Wire the "recompute summary" button to the batched all-lipid compute.
    # Auto-compute once at startup so the user sees the headline numbers
    # without having to click first.
    if info_recompute_btn is not None:
        def _do_recompute_summary():
            info_recompute_btn.setEnabled(False)
            info_recompute_btn.setText("computing... (please wait)")
            try:
                summary = compute_all_lipid_summary()
                info_update_summary(summary)
            finally:
                info_recompute_btn.setEnabled(True)
                info_recompute_btn.setText("recompute summary (all lipids)")
        info_recompute_btn.clicked.connect(_do_recompute_summary)
        # Initial compute (everything is already cached / about to be)
        _do_recompute_summary()

    # ---- Banner -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("Lipid GP visualizer ready.")
    print(f"  graph nodes  : {N_nodes:,}")
    print(f"  eigenmodes   : {K_modes}")
    print(f"  train pts    : {phi_train.shape[0]:,} (after bump filter)")
    if has_test:
        print(f"  test pts     : {phi_test.shape[0]:,} (after bump filter)")
    print(f"  Eucl. GP sub.: {eucl_n_target:,} (subsample for Cholesky)")
    print(f"  lipids       : {n_lipids}")
    print(f"  initial lipid: [{initial_lipid_idx}] '{lipid_names[initial_lipid_idx]}'")
    print("\nLayers (z-order, bottom → top):")
    print("  0  template (reference)            — anatomical context")
    print("  1  annotations (atlas)             — anatomical region labels (off)")
    print("  2  KNN graph nodes (fabric)        — all graph nodes (off)")
    print("  3  train data (raw lipid)          — MaLDI measurements at train pts (circles)")
    print("  4  test data (raw lipid)           — MaLDI measurements at test pts (squares)")
    print("  5  manifold GP pred @ train (off)  — predicted values at train pts")
    print("  6  manifold GP pred @ test (off)   — predicted values at test pts")
    print("  7  manifold GP error @ train (z)   — signed obs−pred in z-units")
    print("  8  manifold GP error @ test (z)    — signed obs−pred in z-units")
    print("  9  Eucl. GP pred @ train (off)     — baseline-kernel predictions at train pts")
    print("  10 Eucl. GP pred @ test (off)      — baseline-kernel predictions at test pts")
    print("  11 Eucl. GP error @ train (off)    — baseline-kernel obs−pred in z-units")
    print("  12 Eucl. GP error @ test (off)     — baseline-kernel obs−pred in z-units")
    print("  13 manifold GP @ graph nodes (off) — per-node manifold prediction")
    print("  14 Eucl. GP @ graph nodes (off)    — per-node Euclidean baseline")
    print("  15 manifold GP @ dense voxels (off)— whole-brain manifold reconstruction")
    print("  16 Eucl. GP @ dense voxels (off)   — whole-brain Euclidean reconstruction")
    print("\nControls: lipid index, ν, ℓ, σ noise, num modes, dense stride, gamma")
    print("\nTips:")
    print("  · Train = circles, test = squares — same colormap across all layers.")
    print("  · For each lipid: compare layers 7 vs 11 (train err) and 8 vs 12 (test err).")
    print("    Where Eucl. errors are larger than manifold errors, the manifold")
    print("    geometry is doing real work.")
    print("  · Cached K matrices for train/test mean lipid scrubs are fast.")
    print("    Hyper changes (ν, ℓ, σ) trigger recompute (slower).")
    print("=" * 72 + "\n")

    napari.run()


if __name__ == "__main__":
    main()