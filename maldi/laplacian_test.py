#!/usr/bin/env python
# encoding: utf-8
"""
Laplacian PSD diagnostic — single config.

One invocation = one (knn_method, normalization, threshold, knn_k,
graphbandwidth, cross_region_inflation) cell = one row in --out CSV.

The sweep itself lives in submit/run_eigenvector_sweep.sh, which generates
all the config combinations and submits one runai job per combination so
they can run in parallel. Each job writes to a config-specific CSV; a
notebook concatenates them at the end.

Flow per invocation:
  1. Build the KNN graph + Laplacian for this single config.
  2. Try to load cached eigenpairs; otherwise eigensolve and save.
     The cache key matches visualize_laplacian.py's, so anything this
     script computes is also reusable by the visualiser.
  3. Compute PSD-violation diagnostics from the eigenvalues.
  4. Emit one CSV row.

Diagnostics:
  ratio_min_over_max   primary PSD-violation signal
                       > -1e-10  : clean PSD (float noise only)
                       -1e-10 to -1e-6 : Lanczos jitter, harmless
                       < -1e-6   : real PSD violation
  n_below_matern_floor eigvals < -2ν/ℓ² ⇒ Matern spectral density goes
                       negative ⇒ manifold-Matern kernel loses PSD even
                       if the Laplacian itself is fine
  spectral_gap         λ_2 − λ_1; tiny gap ⇒ near-disconnected components
  condition_number     λ_max / λ_min_positive; large ⇒ near-singular

Extended diagnostics (columns prefixed diag_*), answering why the implicit-
manifold GP can underperform a Euclidean Matern:
  Q1/Q3 diag_geo_*, diag_dspec_*  geodesic vs spectral vs euclidean distance
                       (opt-in: --geodesic-anchors N). geo_euc_spearman ~1 and
                       ratio_p95 ~1 ⇒ manifold buys nothing here.
  Q2  diag_ortho_l2_offmax vs diag_ortho_deg_offmax  reveals whether the
                       eigvecs are degree-orthonormal (kernel assumes L2) ⇒
                       density-modulated prior variance (diag_varproxy_ratio).
                       diag_eig_resid_* convergence; diag_weyl_* intrinsic dim;
                       diag_n_components vs diag_n_zero_modes connectivity.
       diag_oos_*      (1 - bw²λ)² OOS-denominator zero-crossings; participation
                       ratio = effective #modes the kernel actually uses.
  Q4  diag_maldi_*     (opt-in: --maldi-file + a fold). Configure the fold the
                       same way the per-lipid trainer does: --slices-dataset-file
                       + --available-lipids-file (+ --lipids-file) via MaldiConfig,
                       or manually with --fold-filter / --fold-column. Reports snap
                       + bump coverage, spectral content of the lipid signal vs
                       prior weight, and manifold-vs-euclidean structure match.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

import scipy.sparse as _sp
import scipy.sparse.csgraph as _csg
from scipy.stats import spearmanr as _spearmanr

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges, labels_for_nodes_from_sub_atlas
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
)
from utils import coord_norm_from_reference, crop_or_stride_volume, reference_ccf_from_subvolume

try:
    from manifold_gp.utils import bump_function as _lib_bump_function
    _USING_LIB_BUMP = True
except ImportError:
    _USING_LIB_BUMP = False


def bump_function(d: torch.Tensor, scale, decay) -> torch.Tensor:
    """Same bump used by visualize_laplacian.py / RiemannGP.modulation."""
    if _USING_LIB_BUMP:
        d_t = d if torch.is_tensor(d) else torch.as_tensor(d)
        scale_t = scale if torch.is_tensor(scale) else torch.as_tensor(float(scale), dtype=d_t.dtype, device=d_t.device)
        decay_t = decay if torch.is_tensor(decay) else torch.as_tensor(float(decay), dtype=d_t.dtype, device=d_t.device)
        return _lib_bump_function(d_t, scale_t, decay_t)
    scale = float(scale); decay = float(decay)
    out = torch.zeros_like(d)
    inside = d < scale
    if inside.any():
        u = (d[inside] / scale).clamp(0.0, 1.0 - 1e-6)
        out[inside] = torch.exp(-decay / (1.0 - u * u)) / float(np.exp(-decay))
    return out


# -------------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------------
def analyze_eigvals(
    eigval: np.ndarray,
    nu: int,
    lengthscale: float,
    tol_zero: float = 1e-10,
    tol_neg:  float = 1e-6,
) -> dict:
    """Compute PSD-violation indicators from an eigenvalue vector.

    Tolerances are relative to |λ_max|. tol_zero=1e-10 separates "exactly
    zero up to Lanczos noise" from real positive eigenvalues. tol_neg=1e-6
    separates Lanczos drift (~1e-7 of λ_max) from genuine PSD violations.

    The Matern floor 2ν/ℓ² is the threshold below which the spectral density
    (2ν/ℓ² + λ)^(-ν) becomes negative — eigenvalues there break manifold-
    Matern PSD even if the Laplacian itself is fine.
    """
    ev = np.asarray(eigval, dtype=np.float64).ravel()
    if ev.size == 0:
        return {}
    lam_min = float(ev.min())
    lam_max = float(ev.max())
    scale = max(abs(lam_max), 1e-30)

    pos = ev[ev > 0]
    lam_min_pos = float(pos.min()) if pos.size else float("nan")

    ev_sorted = np.sort(ev)
    spectral_gap = float(ev_sorted[1] - ev_sorted[0]) if ev.size >= 2 else float("nan")

    matern_floor = -2.0 * float(nu) / (float(lengthscale) ** 2)
    n_below_matern = int(np.sum(ev < matern_floor))

    return {
        "n_total":                int(ev.size),
        "lambda_min":             lam_min,
        "lambda_max":             lam_max,
        "ratio_min_over_max":     lam_min / scale,
        "n_zero_exact":           int(np.sum(ev == 0.0)),
        "n_zero_eps":             int(np.sum(np.abs(ev) < tol_zero * scale)),
        "n_negative":             int(np.sum(ev < 0)),
        "n_negative_significant": int(np.sum(ev < -tol_neg * scale)),
        "n_below_matern_floor":   n_below_matern,
        "matern_floor":           matern_floor,
        "spectral_gap":           spectral_gap,
        "condition_number":       (lam_max / lam_min_pos) if pos.size else float("inf"),
        "lambda_min_positive":    lam_min_pos,
    }


# -------------------------------------------------------------------------
# Kernel-PSD evaluation: builds three Gram matrices on a sample of test
# points and runs PSD diagnostics on each. The point of this is to test
# whether the deployed manifold-Matern kernel is PSD on both training
# (in-sample) and non-training (out-of-sample) points — the Laplacian
# being PSD doesn't guarantee the kernel is, because the kernel uses a
# truncated Nyström extension of the eigenfunctions plus a bump-modulated
# Euclidean fallback.
# -------------------------------------------------------------------------
def sample_test_points(
    reference_nodes: torch.Tensor,
    sub_volume: np.ndarray,
    voxel_offset, voxel_scale_mm: float,
    coord_mean: torch.Tensor, coord_std: torch.Tensor,
    threshold: int,
    n_on: int, n_off: int,
    rng: np.random.Generator,
):
    """Return (test_points (N,3), n_on_actual).

    On-manifold ("in-sample"): random subset of reference_nodes, i.e.
    voxels that are above threshold and are training graph nodes.

    Off-manifold ("out-of-sample"): random voxels in sub_volume that are
    *below* threshold (template>0 but template<=threshold) — these are
    real off-manifold positions in the brain coordinate frame and exercise
    the bump's transition region. Falls back to jittered training nodes
    if there aren't enough sub-threshold voxels.
    """
    device = reference_nodes.device
    N = reference_nodes.shape[0]
    n_on  = min(n_on, N)
    on_idx = rng.choice(N, size=n_on, replace=False)
    pts_on = reference_nodes[on_idx].clone()

    off_mask = (sub_volume > 0) & (sub_volume <= threshold)
    off_zyx = np.argwhere(off_mask)
    if off_zyx.shape[0] > 0:
        take = min(n_off, off_zyx.shape[0])
        pick = rng.choice(off_zyx.shape[0], size=take, replace=False)
        off_idx_full = off_zyx[pick].astype(np.float32)
        oz, oy, ox = voxel_offset
        # Convert the chosen sub-volume voxels back to mm using the same
        # convention as reference_ccf_from_subvolume so coord_mean/std apply.
        if float(voxel_scale_mm) == 0.025:
            off_mm = (off_idx_full + np.array([oz, oy, ox], dtype=np.float32)) * 0.025
        else:
            off_mm = off_idx_full * voxel_scale_mm
        off_mm_t = torch.from_numpy(off_mm).to(device)
        pts_off = (off_mm_t - coord_mean.to(device)) / coord_std.to(device)
    else:
        pts_off = torch.empty(0, 3, dtype=pts_on.dtype, device=device)

    if pts_off.shape[0] < n_off:
        # Fallback: jitter training nodes to fill the off-manifold sample.
        deficit = n_off - pts_off.shape[0]
        seed_idx = rng.choice(N, size=deficit, replace=True)
        jitter = torch.from_numpy(
            rng.normal(0, 0.05, size=(deficit, 3)).astype(np.float32)
        ).to(device)
        pts_off = torch.cat([pts_off, reference_nodes[seed_idx] + jitter], dim=0)

    return torch.cat([pts_on, pts_off], dim=0), pts_on.shape[0]


def sample_test_points_uniform_distance(
    reference_nodes: torch.Tensor,
    knn,
    n_on: int, n_off: int,
    dist_max: float, n_bins: int,
    rng: np.random.Generator,
    oversample: int = 60,
):
    """Seed from the manifold and spread off-manifold points UNIFORMLY across
    distance-from-manifold.

    Returns (test_points (N,3), n_on_actual, dist_per_point (N,) in z-units).

    The first `n_on_actual` rows are exact manifold nodes (distance 0). The rest
    are points `node + t * unit_direction` whose *actual* nearest-node distance
    (measured by reusing the graph's `knn` 1-NN search, exactly the same snap
    the kernel uses) is binned into `n_bins` equal-width bins on (0, dist_max]
    and subsampled to ~equal counts per bin, giving roughly uniform coverage of
    the distance axis instead of the naturally clustered distribution. Random
    directions + measured distance avoids needing the manifold normal: offsets
    running along the manifold land in low bins, offsets that exit the tissue
    fill the high bins. dist_max / dist bins are in standardized z-units.
    """
    device = reference_nodes.device
    nodes = reference_nodes.detach().cpu().numpy().astype(np.float64)
    N = nodes.shape[0]

    n_on = min(n_on, N)
    on_idx = rng.choice(N, size=n_on, replace=False)
    pts_on = nodes[on_idx]
    d_on = np.zeros(n_on, dtype=np.float64)

    M = max(int(n_off) * int(oversample), int(n_bins) * 200)
    seed = rng.choice(N, size=M, replace=True)
    dirs = rng.normal(size=(M, 3))
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    t = rng.uniform(0.0, float(dist_max) * 1.5, size=M)
    cand = nodes[seed] + dirs * t[:, None]
    cand_t = torch.tensor(cand, dtype=torch.float32, device=device).contiguous()
    val, _ = knn.search(cand_t, 1)
    d_cand = val.squeeze(-1).clamp(min=0).sqrt().detach().cpu().numpy().astype(np.float64)

    edges = np.linspace(0.0, float(dist_max), int(n_bins) + 1)
    per_bin = max(1, int(n_off) // int(n_bins))
    sel_pts, sel_d, short = [], [], 0
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        in_bin = np.where((d_cand > lo) & (d_cand <= hi))[0]
        if in_bin.size == 0:
            short += 1; continue
        take = min(per_bin, in_bin.size)
        if in_bin.size < per_bin:
            short += 1
        pick = rng.choice(in_bin, size=take, replace=False)
        sel_pts.append(cand[pick]); sel_d.append(d_cand[pick])
    if short:
        logging.info(f"uniform-distance sampler: {short}/{n_bins} distance bins "
                     f"under-filled (manifold too dense / dist_max too large there).")
    pts_off = np.concatenate(sel_pts, 0) if sel_pts else np.empty((0, 3))
    d_off = np.concatenate(sel_d, 0) if sel_d else np.empty((0,))

    pts = np.concatenate([pts_on, pts_off], 0).astype(np.float32)
    dist = np.concatenate([d_on, d_off], 0).astype(np.float64)
    return torch.from_numpy(pts).to(device), int(pts_on.shape[0]), dist


def matern_euclidean_pairwise(
    coords: torch.Tensor, nu: float, lengthscale: float,
) -> np.ndarray:
    """Pairwise Euclidean Matern kernel on `coords` (N×3). Same formula as
    visualize_laplacian.py:matern_euclidean_at_source, vectorised to N×N."""
    from scipy.special import kv, gamma as gamma_fn
    pts = coords.detach().cpu().numpy()
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    nu_f = float(nu); ls = float(lengthscale)
    K = np.ones_like(d, dtype=np.float64)
    nz = d > 0
    if nz.any():
        z = np.sqrt(2.0 * nu_f) * d[nz] / ls
        K[nz] = ((2.0 ** (1.0 - nu_f)) / gamma_fn(nu_f)) * (z ** nu_f) * kv(nu_f, z)
    return K


def evaluate_kernel_psd(
    matern_kernel: RiemannMaternKernel,
    test_points: torch.Tensor,
    graphbandwidth: float,
    bump_scale: float, bump_decay: float,
    n_on_manifold: int,
    analyze_kwargs: dict,
    per_point_csv: "Path | None" = None,
    plot_dir: "Path | None" = None,
    plot_tag: str = "",
    coord_std: "float | None" = None,
) -> dict:
    """PSD-diagnose the deployed manifold Gram matrix:

      K_m = features(x) · features(y)            (deployed kernel)

    Since features() already applies the bump to out-of-sample rows, K_m IS the
    kernel the model uses (forward() returns features·featuresᵀ). It is a Gram
    matrix, hence PSD in exact arithmetic; a meaningfully negative eigenvalue
    therefore flags numerically blown-up / ill-conditioned features (e.g. from
    the out-of-sample (1-bw²λ)² denominator), not an invalid formula.

    Reductions on the in-sample block and the out-of-sample block are reported
    separately — trouble on the in-sample block (exact eigenvectors at training
    nodes) is a much stronger signal than on the off-manifold block (Nyström
    extension, naturally less well-conditioned on novel queries).

    The bump weight b is still reported per point (and dumped vs distance) to
    characterize the test set, even though it's already folded into K_m.
    """
    with torch.no_grad():
        feat = matern_kernel.features(test_points)        # (N, num_modes)
        K_m = (feat @ feat.t()).detach().cpu().numpy().astype(np.float64)
        K_m = 0.5 * (K_m + K_m.T)
        edge_value, _ = matern_kernel.knn.search(test_points, 1)
        d_nearest = edge_value.sqrt().squeeze(-1)
        b = bump_function(
            d_nearest,
            scale=bump_scale * float(graphbandwidth),
            decay=float(bump_decay),
        ).detach().cpu().numpy().astype(np.float64)

    n_on  = int(n_on_manifold)
    n_off = int(test_points.shape[0] - n_on)
    out: dict[str, Any] = {
        "n_test_on":   n_on,
        "n_test_off":  n_off,
        "bump_min":    float(b.min()),
        "bump_mean":   float(b.mean()),
        "bump_max":    float(b.max()),
        "bump_on_mean":  float(b[:n_on].mean()) if n_on > 0 else float("nan"),
        "bump_off_mean": float(b[n_on:].mean()) if n_off > 0 else float("nan"),
        "K_m_max":     float(np.abs(K_m).max()),
    }
    # Full-sample diagnostics, then submatrix diagnostics for the in-sample
    # block (K[:n_on, :n_on]) and the out-of-sample block (K[n_on:, n_on:]).
    blocks = [
        ("",   slice(None),  slice(None)),
        ("on_",  slice(0, n_on), slice(0, n_on)),
        ("off_", slice(n_on, None), slice(n_on, None)),
    ]
    for tag, mat in [("K_m", K_m)]:
        for blk_prefix, ri, ci in blocks:
            sub = mat[ri, ci]
            if sub.size == 0:
                continue
            try:
                ev = np.linalg.eigvalsh(sub)
            except np.linalg.LinAlgError as e:
                out[f"{tag}_{blk_prefix}status"] = f"EIG_FAIL:{e}"
                continue
            diag = analyze_eigvals(ev, **analyze_kwargs)
            for k, v in diag.items():
                out[f"{tag}_{blk_prefix}{k}"] = v

    # ---- Optional per-point dump: kernel quantities vs distance-from-manifold.
    # This is what a uniform-distance test sample is for -- it lets you read the
    # bump and the prior variance as smooth functions of distance to the graph.
    if per_point_csv is not None or plot_dir is not None:
        d_np = d_nearest.detach().cpu().numpy().astype(np.float64)
        Km_diag = np.clip(np.diag(K_m), 0.0, None)
        on_flag = np.zeros(d_np.shape[0], dtype=int); on_flag[:n_on] = 1
        order = np.argsort(d_np)
        df = {
            "dist_z": d_np[order],
            "on_manifold": on_flag[order],
            "bump": b[order],
            "Km_diag": Km_diag[order],          # deployed manifold prior variance
            "manifold_prior_std": np.sqrt(Km_diag[order]),
        }
        if coord_std is not None:
            df["dist_mm"] = d_np[order] * float(coord_std)
        import pandas as _pd
        _df = _pd.DataFrame(df)
        if per_point_csv is not None:
            Path(per_point_csv).parent.mkdir(parents=True, exist_ok=True)
            _df.to_csv(per_point_csv, index=False)
        if plot_dir is not None:
            _try_plot_kernel_vs_distance(plot_dir, plot_tag, _df, coord_std)
        out["per_point_n"] = int(d_np.shape[0])
        out["per_point_dist_max"] = float(d_np.max())

    return out


def _try_plot_kernel_vs_distance(plot_dir, tag, df, coord_std):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    x = df["dist_z"].to_numpy()
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(x, df["bump"].to_numpy(), color="tab:blue", lw=1.5, label="bump weight")
    ax1.set_xlabel("distance to nearest manifold node (z-units"
                   + (f"; 1z={coord_std:.3f}mm)" if coord_std else ")"))
    ax1.set_ylabel("bump weight", color="tab:blue")
    ax1.set_ylim(-0.02, 1.05); ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(x, df["manifold_prior_std"].to_numpy(), color="tab:red", lw=1.2,
             alpha=0.85, label="manifold prior std sqrt(Km)")
    ax2.set_ylabel("prior std", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="center right")
    ax1.set_title(f"Kernel vs distance-from-manifold  [{tag}]")
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / f"kernel_vs_distance_{tag}.png", dpi=120)
    plt.close(fig)


# =========================================================================
# EXTENDED DIAGNOSTICS
#
# These answer the four questions about *why* the implicit-manifold GP can
# underperform a Euclidean Matern, and emit flat scalar metrics so they slot
# straight into the one-row-per-config CSV:
#
#   Q1  do the eigvecs/eigvals approximate geodesic distance?   (geodesic_*)
#   Q2  is the eigen/Laplacian computation healthy / bug-free?  (eigen_health_*)
#   Q3  is geodesic distance even meaningful, or ~= euclidean?  (geo_euc_*)
#   Q4  does the kernel match the real MALDI lipid structure?   (maldi_*)
#
# Cheap checks (eigen-health, bandwidth/OOS) run by default. The expensive
# geodesic (Dijkstra) and MALDI-data checks are opt-in: geodesic via
# --geodesic-anchors > 0, MALDI via --maldi-file. Pass --diag-plot-dir to
# also dump PNGs (off by default so the sweep stays headless).
# =========================================================================
def _spectral_density_np(eigval: np.ndarray, nu: float, lengthscale: float) -> np.ndarray:
    """Matches RiemannMaternKernel.spectral_density()."""
    safe = np.clip(eigval.astype(np.float64), 0.0, None)
    return (2.0 * float(nu) / float(lengthscale) ** 2 + safe) ** (-float(nu))


def _kernel_features_np(eigval: np.ndarray, eigvec: np.ndarray,
                        nu: float, lengthscale: float,
                        node_idx: np.ndarray) -> np.ndarray:
    """In-sample feature map for selected nodes, matching
    RiemannKernel.features(): f_k(i) = sqrt(S_k/sum(S) * N) * phi_k(i)."""
    S = _spectral_density_np(eigval, nu, lengthscale)
    S = S / max(S.sum(), 1e-12)
    scale = np.sqrt(S * eigvec.shape[0])
    return scale[None, :] * eigvec[node_idx]


def eigen_health_diagnostics(
    laplacian_op: GraphLaplacianOperator,
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    edge_index: torch.Tensor,
    norm: str,
    n_res_modes: int = 64,
) -> dict:
    """Q2 — health of the eigen-decomposition and the graph.

    * connected components vs number of ~zero eigenvalues
    * orthonormality in BOTH inner products: phi^T phi (L2) and phi^T D phi
      (degree). For randomwalk, LaplacianEigensolver._postprocess returns
      DEGREE-orthonormal vectors (it applies D^-1/2 and does NOT re-L2-
      normalize), while the kernel's feature scaling assumes a uniform (L2)
      measure -> a degree-modulated, non-stationary prior variance. These two
      numbers make that visible.
    * eigen-residual ||L phi - lambda phi||/|lambda| via the REAL operator
      matmul (faithful to deployment, both norms).
    * Weyl-law slope of lambda_k vs k -> intrinsic dimension estimate.
    """
    N, M = int(eigvec.shape[0]), int(eigvec.shape[1])
    out: dict[str, Any] = {}

    # --- connected components (CPU, from edge_index) ---
    ei = edge_index.detach().cpu().numpy()
    data = np.ones(ei.shape[1], dtype=np.float64)
    A = _sp.coo_matrix((data, (ei[0], ei[1])), shape=(N, N))
    A = A + A.T
    n_comp, _ = _csg.connected_components(A, directed=False)
    ev_np = eigval.detach().cpu().numpy().astype(np.float64)
    lam_scale = max(abs(ev_np).max(), 1e-30)
    n_zero = int(np.sum(np.abs(ev_np) < 1e-8 * lam_scale))
    out["diag_n_components"] = int(n_comp)
    out["diag_n_zero_modes"] = n_zero

    # --- orthonormality (L2 vs degree) on a column subset ---
    k = min(M, 200)
    cols = torch.linspace(0, M - 1, k).round().long().to(eigvec.device)
    Phi = eigvec.index_select(1, cols)
    eye = torch.eye(k, device=eigvec.device, dtype=Phi.dtype)
    G_l2 = Phi.t() @ Phi
    deg = laplacian_op.degree_mat.to(Phi.dtype)
    G_D = Phi.t() @ (deg.view(-1, 1) * Phi)
    out["diag_ortho_l2_offmax"] = float((G_l2 - eye).abs().max().item())
    out["diag_ortho_deg_offmax"] = float((G_D - eye).abs().max().item())
    # marginal-variance non-uniformity proxy: sum_k phi_k(i)^2 across nodes
    var_proxy = (eigvec ** 2).sum(dim=1)
    out["diag_varproxy_ratio"] = float(
        (var_proxy.max() / var_proxy.clamp(min=1e-30).min()).item()
    )

    # --- eigen-residuals via the real operator matmul ---
    nres = min(n_res_modes, M)
    sel = torch.linspace(0, M - 1, nres).round().long().to(eigvec.device)
    V = eigvec.index_select(1, sel)
    lam = eigval.index_select(0, sel)
    with torch.no_grad():
        LV = laplacian_op._matmul(V)
    resid = torch.linalg.norm(LV - lam.view(1, -1) * V, dim=0)
    rel = (resid / lam.abs().clamp(min=1e-12)).detach().cpu().numpy()
    out["diag_eig_resid_median"] = float(np.median(rel))
    out["diag_eig_resid_p90"] = float(np.percentile(rel, 90))
    out["diag_eig_resid_max"] = float(rel.max())

    # --- Weyl law ---
    pos = ev_np[ev_np > lam_scale * 1e-8]
    if pos.size > 10:
        kk = np.arange(1, pos.size + 1)
        lo, hi = int(0.1 * pos.size), int(0.8 * pos.size)
        slope = float(np.polyfit(np.log(kk[lo:hi]), np.log(pos[lo:hi]), 1)[0])
        out["diag_weyl_slope"] = slope
        out["diag_weyl_dim"] = float(2.0 / slope) if slope > 0 else float("nan")
    return out


def bandwidth_oos_diagnostics(
    eigval: torch.Tensor, graphbandwidth: float, nu: float, lengthscale: float,
) -> dict:
    """Cheap bandwidth/OOS sanity.

    The out-of-sample spectral density divides by (1 - bw^2 lambda)^2 (clamped
    at 1e-6). If bw^2*lambda crosses 1 for high modes, that correction blows
    up / sign-flips on OFF-graph test points. The eigvals are FROZEN at the
    solve-time bandwidth, so if training learns a different graphbandwidth the
    misalignment worsens. Also report the spectral-density participation ratio
    (effective number of modes the kernel actually uses)."""
    ev = eigval.detach().cpu().numpy().astype(np.float64)
    z = (float(graphbandwidth) ** 2) * ev
    n_bad = int(np.sum(z >= 1.0))
    out = {
        "diag_oos_bw2lam_max": float(z.max()),
        "diag_oos_denom_crossings": n_bad,
        "diag_oos_first_crossing_mode": int(np.argmax(z >= 1.0)) if n_bad else -1,
    }
    S = _spectral_density_np(ev, nu, lengthscale)
    S = S / max(S.sum(), 1e-30)
    out["diag_smat_participation_ratio"] = float((S.sum() ** 2) / (S ** 2).sum())
    return out


def geodesic_distance_diagnostics(
    edge_index: torch.Tensor, edge_value: torch.Tensor,
    reference_nodes: torch.Tensor,
    eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: float, lengthscale: float,
    n_anchors: int, seed: int = 0,
    plot_dir: Path | None = None, plot_tag: str = "",
) -> dict:
    """Q1 + Q3 — geodesic vs spectral vs euclidean distance.

    Geodesic = Dijkstra on the graph with euclidean edge lengths
    (sqrt(edge_value)). Spectral = the kernel-induced distance from the frozen
    eigenpairs + Matern spectral density. Reports Spearman correlations and the
    geodesic/euclidean ratio. If geodesic ~= euclidean the manifold buys
    nothing here; if d_spec tracks geodesic poorly at full num_modes you're
    mode-starved or the basis is wrong (see eigen-health)."""
    N = int(reference_nodes.shape[0])
    rng = np.random.default_rng(seed)
    anchors = rng.choice(N, size=min(n_anchors, N), replace=False)

    ei = edge_index.detach().cpu().numpy()
    ev = edge_value.detach().cpu().numpy().astype(np.float64)
    lengths = np.sqrt(np.maximum(ev, 0.0))
    Wlen = _sp.coo_matrix((lengths, (ei[0], ei[1])), shape=(N, N)).tocsr()
    Dgeo_full = _csg.dijkstra(Wlen, directed=False, indices=anchors)  # (A, N)
    reach = float(np.isfinite(Dgeo_full).mean())
    Dgeo = Dgeo_full[:, anchors]

    coords = reference_nodes.detach().cpu().numpy().astype(np.float64)[anchors]
    diff = coords[:, None, :] - coords[None, :, :]
    Deuc = np.sqrt((diff ** 2).sum(-1))

    iu = np.triu_indices(len(anchors), k=1)
    geo_v, euc_v = Dgeo[iu], Deuc[iu]
    ok = np.isfinite(geo_v)
    ratio = geo_v[ok] / np.maximum(euc_v[ok], 1e-9)

    ev_np = eigval.detach().cpu().numpy()
    evec_np = eigvec.detach().cpu().numpy()

    def dspec(modes):
        F = _kernel_features_np(ev_np[:modes], evec_np[:, :modes],
                                nu, lengthscale, anchors)
        Kg = F @ F.T
        dK = np.diag(Kg)
        return np.sqrt(np.clip(dK[:, None] + dK[None, :] - 2 * Kg, 0, None))[iu]

    M = evec_np.shape[1]
    m_low = max(1, M // 10)
    ds_full = dspec(M)
    ds_low = dspec(m_low)
    out = {
        "diag_geo_reachable_frac": reach,
        "diag_geo_euc_spearman": float(_spearmanr(geo_v[ok], euc_v[ok]).statistic),
        "diag_geo_euc_ratio_median": float(np.median(ratio)),
        "diag_geo_euc_ratio_p95": float(np.percentile(ratio, 95)),
        "diag_dspec_geo_spearman_full": float(_spearmanr(ds_full[ok], geo_v[ok]).statistic),
        "diag_dspec_geo_spearman_low": float(_spearmanr(ds_low[ok], geo_v[ok]).statistic),
        "diag_dspec_euc_spearman_full": float(_spearmanr(ds_full[ok], euc_v[ok]).statistic),
        "diag_geo_n_anchors": int(len(anchors)),
        "diag_geo_modes_low": int(m_low),
    }
    if plot_dir is not None:
        _try_plot_geodesic(plot_dir, plot_tag, euc_v[ok], geo_v[ok],
                           ds_full[ok], out)
    return out


def _signal_variogram_metrics(dist: np.ndarray, dissim_abs: np.ndarray,
                              n_bins: int = 24) -> dict:
    """Empirical (semi)variogram of a signal against ONE distance metric, over
    flat pair arrays. Pairs are split into n_bins equal-count (quantile) bins of
    `dist`; per bin we form the semivariance gamma = 0.5*mean(dissim^2). Returns:
      r2          fraction of squared-dissimilarity variance explained by binned
                  distance (higher => distance organizes the signal better)
      nugget_sill nugget/sill = gamma in the nearest bin / plateau (mean of the
                  top-quartile bins). Low => the metric explains the near field;
                  high => a signal jump at distance~0 the metric is blind to (a
                  fold: distance-near but signal-far pairs pile up at small h)
      spearman    monotonicity of |dissim| vs distance"""
    d = np.asarray(dist, float).ravel()
    a = np.asarray(dissim_abs, float).ravel()
    y = a ** 2
    if d.size < 3 * n_bins:
        n_bins = max(3, d.size // 3)
    edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return dict(r2=float("nan"), nugget_sill=float("nan"), spearman=float("nan"))
    edges[-1] = np.inf
    pred = np.full_like(y, y.mean())
    gammas = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        if sel.any():
            g_k = float(y[sel].mean())
            pred[sel] = g_k
            gammas.append(0.5 * g_k)                         # ascending-distance order
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    gammas = np.asarray(gammas, float)
    nugget = float(gammas[0])
    k = max(1, len(gammas) // 4)
    sill = float(np.mean(np.sort(gammas)[-k:]))              # plateau ~ top quartile
    return dict(
        r2=float(1.0 - ss_res / max(ss_tot, 1e-30)),
        nugget_sill=float(nugget / max(sill, 1e-30)),
        spearman=float(_spearmanr(d, a).statistic),
    )


def _signal_variogram_compare(dists: dict, dissim_abs: np.ndarray,
                              mask: np.ndarray, n_bins: int = 24,
                              local_quantile: float = 0.15) -> dict:
    """Compare candidate distance fields (name -> (A,H) array) by how cleanly the
    signal's variogram decays along each, over the masked pairs. For each metric
    we report the variogram over ALL pairs (global) AND over only that metric's
    nearest `local_quantile` fraction of pairs (local). The local read matters
    for oscillatory/antiphase signals: globally the semivariance is non-monotone
    (so R2 is tiny for every metric), but within the near-field the signal IS
    monotone in dissimilarity, so the fold pollution -- distance-near but
    signal-far cross-region pairs -- shows up sharply as a high local nugget.
    Returns name -> {r2, nugget_sill, spearman, *_local, local_cutoff}."""
    y_all = dissim_abs[mask]
    nan3 = dict(r2=float("nan"), nugget_sill=float("nan"), spearman=float("nan"))
    out = {}
    for name, dm in dists.items():
        x_all = dm[mask]
        g = _signal_variogram_metrics(x_all, y_all, n_bins)
        cutoff = float(np.quantile(x_all, local_quantile))   # this metric's near-field
        sel = x_all <= cutoff
        loc = (_signal_variogram_metrics(x_all[sel], y_all[sel], n_bins)
               if sel.sum() >= 3 else nan3)
        out[name] = dict(
            r2=g["r2"], nugget_sill=g["nugget_sill"], spearman=g["spearman"],
            r2_local=loc["r2"], nugget_sill_local=loc["nugget_sill"],
            spearman_local=loc["spearman"], local_cutoff=cutoff,
        )
    return out


def _kernel_vs_data_core(
    Xs: np.ndarray, Y: np.ndarray,
    knn, reference_nodes: torch.Tensor,
    eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: float, lengthscale: float, graphbandwidth: float, bump_scale: float,
    seed: int = 0, n_pairs: int = 40000,
    plot_dir: Path | None = None, plot_tag: str = "",
    node_labels=None, edge_index=None,
    edge_value=None, vario_local_quantile: float = 0.15,
    vario_anchors: int = 200, vario_max_targets: int = 4000,
) -> dict:
    """Shared analysis: given standardized coords Xs (N,3) and z-scored lipid
    values Y (N,K) — already in the GRAPH's coordinate space — report:
      * snap distances + fraction within the bump support (bump_scale*bw)
      * spectral content of the lipid signal on the eigenbasis vs prior weight
      * Spearman(kernel correlation, -|y_i - y_j|) for manifold vs euclidean.
    Used by both the parquet front-end and the run-dir (fold) front-end."""
    device = reference_nodes.device
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device).contiguous()
    val, idx = knn.search(Xs_t, 1)
    snap_d = val.squeeze(-1).clamp(min=0).sqrt().detach().cpu().numpy()
    snap_i = idx.squeeze(-1).detach().cpu().numpy().astype(np.int64)
    support = float(bump_scale) * float(graphbandwidth)

    N = int(reference_nodes.shape[0])
    out: dict[str, Any] = {
        "diag_maldi_n_points": int(Xs.shape[0]),
        "diag_maldi_snap_median": float(np.median(snap_d)),
        "diag_maldi_snap_p95": float(np.percentile(snap_d, 95)),
        "diag_maldi_bump_support": support,
        "diag_maldi_frac_in_support": float(np.mean(snap_d < support)),
    }

    sig = Y.mean(axis=1)
    node_sum = np.zeros(N); node_cnt = np.zeros(N)
    np.add.at(node_sum, snap_i, sig)
    np.add.at(node_cnt, snap_i, 1.0)
    hit = node_cnt > 0
    s_node = np.zeros(N)
    s_node[hit] = node_sum[hit] / node_cnt[hit]
    out["diag_maldi_node_coverage"] = float(hit.mean())

    if node_labels is not None and edge_index is not None:
        nl = (node_labels.detach().cpu().numpy()
              if torch.is_tensor(node_labels) else np.asarray(node_labels))
        ei = (edge_index.detach().cpu().numpy()
              if torch.is_tensor(edge_index) else np.asarray(edge_index))
        # keep only the upper triangle to avoid double-counting
        upper = ei[0] < ei[1]
        ei = ei[:, upper]; nl_i, nl_j = nl[ei[0]], nl[ei[1]]
        cross_e = nl_i != nl_j
        valid_e = hit[ei[0]] & hit[ei[1]]
        diff_e  = np.abs(s_node[ei[0][valid_e]] - s_node[ei[1][valid_e]])
        cv = cross_e[valid_e]
        if cv.sum() > 0 and (~cv).sum() > 0:
            cj = float(diff_e[cv].mean())
            ij = float(diff_e[~cv].mean())
            out["diag_maldi_cross_signal_jump"] = cj
            out["diag_maldi_intra_signal_jump"] = ij
            out["diag_maldi_cross_intra_ratio"] = cj / (ij + 1e-10)

    evec_np = eigvec.detach().cpu().numpy()
    ev_np = eigval.detach().cpu().numpy()
    s = s_node.copy(); s[hit] -= s[hit].mean()
    c = evec_np.T @ s
    energy = c ** 2; energy /= max(energy.sum(), 1e-30)
    cum = np.cumsum(energy)
    S = _spectral_density_np(ev_np, nu, lengthscale); S /= S.sum()
    cumS = np.cumsum(S)
    M = len(c)
    mode_at = lambda f, cu: int(np.searchsorted(cu, f)) + 1
    out["diag_maldi_e50_mode"] = mode_at(.5, cum)
    out["diag_maldi_e90_mode"] = mode_at(.9, cum)
    out["diag_maldi_e90_mode_frac"] = float(mode_at(.9, cum) / M)
    out["diag_maldi_prior_e90_mode"] = mode_at(.9, cumS)
    out["diag_maldi_tail_energy_beyond_modes"] = float(max(0.0, 1.0 - cum[-1]))

    rng = np.random.default_rng(seed)
    hit_nodes = np.flatnonzero(hit)
    if hit_nodes.size >= 2:
        if hit_nodes.size > 4000:
            hit_nodes = rng.choice(hit_nodes, 4000, replace=False)
        a = rng.choice(hit_nodes, n_pairs); b = rng.choice(hit_nodes, n_pairs)
        keep = a != b; a, b = a[keep], b[keep]
        F = _kernel_features_np(ev_np, evec_np, nu, lengthscale,
                                np.concatenate([a, b]))
        Fa, Fb = F[:len(a)], F[len(a):]
        Kab = (Fa * Fb).sum(1)
        Kman = Kab / np.sqrt(np.clip((Fa * Fa).sum(1) * (Fb * Fb).sum(1), 1e-30, None))
        coords = reference_nodes.detach().cpu().numpy().astype(np.float64)
        dd = np.sqrt(((coords[a] - coords[b]) ** 2).sum(1))
        from scipy.special import kv, gamma as gfn
        z = np.sqrt(2 * nu) * dd / lengthscale
        Keuc = np.ones_like(dd); nz = dd > 0
        Keuc[nz] = (2 ** (1 - nu) / gfn(nu)) * (z[nz] ** nu) * kv(nu, z[nz])
        dissim = np.abs(s_node[a] - s_node[b])
        out["diag_maldi_match_spearman_manifold"] = float(_spearmanr(Kman, -dissim).statistic)
        out["diag_maldi_match_spearman_euclidean"] = float(_spearmanr(Keuc, -dissim).statistic)
        if plot_dir is not None:
            _try_plot_maldi(plot_dir, plot_tag, cum, cumS, dd, dissim)

    # signal variogram (covariance vs distance): does the lipid signal decay
    # along the manifold (graph-geodesic) metric or the euclidean one? Model-free
    # (no kernel/eigenbasis), so immune to the truncated-feature and far-pair
    # dilution that can flip the match_spearman read. Distances are taken only
    # between HIT nodes (where signal exists); the geodesic uses inflated edge
    # lengths (sqrt(edge_value)) when given, else euclidean lengths from coords.
    hit_idx = np.flatnonzero(hit)
    if edge_index is not None and hit_idx.size >= 50:
        ei_v = (edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index)
                else np.asarray(edge_index))
        coords_all = reference_nodes.detach().cpu().numpy().astype(np.float64)
        if edge_value is not None:
            ev_v = (edge_value.detach().cpu().numpy() if torch.is_tensor(edge_value)
                    else np.asarray(edge_value)).astype(np.float64)
            lengths = np.sqrt(np.maximum(ev_v, 0.0))
        else:
            lengths = np.linalg.norm(
                coords_all[ei_v[0]] - coords_all[ei_v[1]], axis=1)
        rngv = np.random.default_rng(seed + 1)
        n_anc = min(int(vario_anchors), hit_idx.size)
        anchors_v = rngv.choice(hit_idx, size=n_anc, replace=False)
        tgt = (rngv.choice(hit_idx, size=int(vario_max_targets), replace=False)
               if hit_idx.size > int(vario_max_targets) else hit_idx)
        Wlen = _sp.coo_matrix((lengths, (ei_v[0], ei_v[1])), shape=(N, N)).tocsr()
        Dgeo = _csg.dijkstra(Wlen, directed=False, indices=anchors_v)[:, tgt]  # (A,T)
        diff = coords_all[anchors_v][:, None, :] - coords_all[tgt][None, :, :]
        Deuc = np.sqrt((diff ** 2).sum(-1))                                   # (A,T)
        dissim_v = np.abs(s_node[anchors_v][:, None] - s_node[tgt][None, :])  # (A,T)
        finite = np.isfinite(Dgeo) & (Dgeo > 0)
        if finite.sum() >= 3 * 24:
            vg = _signal_variogram_compare(
                {"manifold_geodesic": Dgeo, "euclidean": Deuc}, dissim_v, finite,
                local_quantile=vario_local_quantile)
            for mname, key in (("manifold_geodesic", "man"), ("euclidean", "euc")):
                v = vg[mname]
                out[f"diag_maldi_vario_{key}_r2"] = v["r2"]
                out[f"diag_maldi_vario_{key}_r2_local"] = v["r2_local"]
                out[f"diag_maldi_vario_{key}_nugget_sill"] = v["nugget_sill"]
                out[f"diag_maldi_vario_{key}_nugget_sill_local"] = v["nugget_sill_local"]
            out["diag_maldi_vario_n_anchors"] = int(n_anc)
            out["diag_maldi_vario_reachable_frac"] = float(finite.mean())
    return out


def _parse_fold_filter(fold_filter_json: str | None,
                       fold_column: str | None,
                       fold_test_values: list | None):
    """Turn CLI fold args into a pyarrow `filters` expression — the SAME shape
    as MaldiConfig.test_filter (which is what the experiment passes to
    read_parquet). Two ways to specify a fold:

      --fold-filter '[["Section","in",["S1","S2"]]]'   (general; copy your
          MaldiConfig.test_filter verbatim, or hand-write it). JSON: a list of
          [col, op, val] predicates (AND), or a list of such lists (OR of ANDs).
      --fold-column Section --fold-test-values S1 S2    (convenience → "in").

    Returns a pyarrow-style filters object, or None (whole parquet)."""
    if fold_filter_json:
        spec = json.loads(fold_filter_json)

        def to_tuples(x):
            # innermost predicate [col, op, val] -> tuple
            if (isinstance(x, list) and len(x) == 3 and isinstance(x[0], str)
                    and isinstance(x[1], str)):
                return tuple(x)
            if isinstance(x, list):
                return [to_tuples(e) for e in x]
            return x
        return to_tuples(spec)
    if fold_column and fold_test_values:
        return [(fold_column, "in", list(fold_test_values))]
    return None


def maldi_data_diagnostics(
    maldi_file: str, lipids: list, log_transform: bool,
    knn, reference_nodes: torch.Tensor,
    eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: float, lengthscale: float,
    graphbandwidth: float, bump_scale: float,
    fold_filter=None,
    max_rows: int = 400000, n_pairs: int = 40000, seed: int = 0,
    plot_dir: Path | None = None, plot_tag: str = "",
    coord_mean=None, coord_std=None,
    node_labels=None, edge_index=None,
    edge_value=None, vario_local_quantile: float = 0.15,
) -> dict:
    """Q4 — does the kernel match the MALDI lipid structure on a given FOLD?

    The fold is configured ON THE COMMAND LINE via `fold_filter` (a pyarrow
    filter expression == MaldiConfig.test_filter), so no trained run dir is
    needed. Reads coords + the requested lipids for the fold's rows the way
    experiment.py does, snaps each point to the nearest graph node, and reports
    snap/bump coverage, the spectral content of the lipid signal vs the prior
    weight, and the manifold-vs-euclidean structure match.

    Coordinate normalization: reuses the reference-template normalization from
    utils.coord_norm_from_reference (passed in as coord_mean/coord_std) — the
    SAME per-axis mean + scalar isotropic std the deployed model and
    reference_nodes use, so the snapping/distances are in the model's space.
    Falls back to global whole-brain parquet stats only if those aren't given."""
    import pandas as pd
    coord_cols = ["xccf", "yccf", "zccf"]
    lip_cols = [str(l) for l in lipids]

    # --- coord normalization ---
    # Reuse the reference-template normalization (utils.coord_norm_from_reference,
    # threaded in as coord_mean/coord_std) -- the SAME mean/std the deployed model
    # and reference_nodes use. Fall back to global parquet stats only if absent.
    if coord_mean is not None and coord_std is not None:
        cm = (coord_mean.detach().cpu().numpy() if torch.is_tensor(coord_mean)
              else np.asarray(coord_mean)).astype(np.float64).reshape(-1)
        cs = (coord_std.detach().cpu().numpy() if torch.is_tensor(coord_std)
              else np.asarray(coord_std)).astype(np.float64)
        cs = np.maximum(cs, 1e-6)
        logging.info(f"[maldi] coord norm from reference template (util): "
                     f"mean={cm}, std={cs}")
    else:
        g = pd.read_parquet(maldi_file, columns=coord_cols)  # all rows, 3 cols
        if max_rows and len(g) > max_rows:
            g = g.sample(max_rows, random_state=0)
        gx = g[coord_cols].to_numpy(np.float64)
        cm, cs = gx.mean(0), gx.std(0)
        cs[cs < 1e-6] = 1e-6
        logging.info(f"[maldi] coord stats from GLOBAL parquet: mean={cm}, std={cs}")

    # --- fold rows: coords + requested lipids ---
    df = pd.read_parquet(maldi_file, columns=coord_cols + lip_cols,
                         filters=fold_filter)
    if df.shape[0] == 0:
        raise RuntimeError(
            f"Fold filter {fold_filter!r} selected 0 rows from {maldi_file}. "
            "Check the column name / values."
        )
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)
    X = df[coord_cols].to_numpy(np.float64).copy()
    Y = df[lip_cols].to_numpy(np.float64).copy()
    Y[Y < 0] = 0.0
    if log_transform:
        Y = np.log(Y + 1e-10)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)

    Xs = (X - cm) / cs
    out = _kernel_vs_data_core(
        Xs, Y, knn, reference_nodes, eigval, eigvec,
        nu, lengthscale, graphbandwidth, bump_scale,
        seed=seed, n_pairs=n_pairs, plot_dir=plot_dir, plot_tag=plot_tag,
        node_labels=node_labels, edge_index=edge_index,
        edge_value=edge_value, vario_local_quantile=vario_local_quantile,
    )
    out["diag_maldi_n_lipids"] = int(Y.shape[1])
    return out

def _safe_filename(name: str) -> str:
    """Lipid name -> slug, matching lgp_experiment_per_lipid.safe_filename:
    'PA 36:1 PA 38:4' -> 'PA_36-1_PA_38-4'."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace(":", "-"))
    return s.strip("_")


def load_fold_from_run_dir(run_dir: Path, lipids: list | None = None):
    """Load a particular experiment FOLD straight from a trained run dir
    (the one with config.json / lipid_names.json / graph_meta.npz /
    predictions/<slug>/). Returns the exact held-out test points and the
    z-scored ground truth the GP was scored against — no parquet filter
    reconstruction, and the exact coord normalization from graph_meta.

    Returns dict: coords_mm (N,3), Y (N,K z-scored true), pred (N,K z pred or
    None), names (K), coord_mean (3), coord_std (3), log_transform, config."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "lipid_names.json") as f:
        names_all = json.load(f)

    gm_path = run_dir / "graph_meta.npz"
    if not gm_path.exists():
        raise FileNotFoundError(
            f"{gm_path} not found. graph_meta.npz (with coord_mean/coord_std) "
            "is written only for manifold runs; for a euclidean run pass "
            "--coord-mean/--coord-std and use --maldi-file instead."
        )
    gm = np.load(gm_path)
    coord_mean = gm["coord_mean"].astype(np.float64)
    coord_std = gm["coord_std"].astype(np.float64)

    pred_root = run_dir / "predictions"
    # which lipids: those requested (by name) that are on disk, else all on disk
    want = set(lipids) if lipids else None
    sel_names, coords_ref = [], None
    cols_true, cols_pred = [], []
    for name in names_all:
        if want is not None and name not in want:
            continue
        slug = _safe_filename(name)
        d = pred_root / slug
        cpath, tpath = d / "test_coords_mm.npy", d / "test_true_z.npy"
        if not (cpath.exists() and tpath.exists()):
            continue
        coords = np.load(cpath).astype(np.float64)
        true_z = np.load(tpath).astype(np.float64).ravel()
        if coords_ref is None:
            coords_ref = coords
        if coords.shape[0] != coords_ref.shape[0] or true_z.shape[0] != coords_ref.shape[0]:
            logging.warning(f"[fold] {slug}: point count mismatch; skipping.")
            continue
        sel_names.append(name)
        cols_true.append(true_z)
        ppath = d / "test_pred_z.npy"
        cols_pred.append(np.load(ppath).astype(np.float64).ravel()
                         if ppath.exists() else None)
    if coords_ref is None or not sel_names:
        raise RuntimeError(
            f"No usable predictions/<slug>/ with test_coords_mm.npy + "
            f"test_true_z.npy found under {pred_root} for the requested lipids."
        )
    Y = np.stack(cols_true, axis=1)
    pred = (np.stack(cols_pred, axis=1)
            if all(p is not None for p in cols_pred) else None)
    return {
        "coords_mm": coords_ref, "Y": Y, "pred": pred, "names": sel_names,
        "coord_mean": coord_mean, "coord_std": coord_std,
        "log_transform": bool(config.get("log_transform", False)),
        "config": config,
    }


def fold_data_diagnostics(
    run_dir: Path, lipids: list | None,
    knn, reference_nodes: torch.Tensor,
    eigval: torch.Tensor, eigvec: torch.Tensor,
    nu: float, lengthscale: float, graphbandwidth: float, bump_scale: float,
    n_pairs: int = 40000, seed: int = 0,
    plot_dir: Path | None = None, plot_tag: str = "",
    node_labels=None, edge_index=None,
    edge_value=None, vario_local_quantile: float = 0.15,
) -> dict:
    """Q4 (run-dir / fold front-end). Loads the exact held-out test fold from a
    trained run dir, standardizes coords with that run's coord_mean/std (from
    graph_meta), and runs the kernel-vs-data analysis. Also reports the run's
    achieved per-lipid test Spearman so kernel-structure quality can be related
    to actual fold performance, and flags kernel-hyperparameter mismatches
    between config.json and this script's flags."""
    fold = load_fold_from_run_dir(run_dir, lipids)
    Xs = (fold["coords_mm"] - fold["coord_mean"]) / fold["coord_std"]
    out = _kernel_vs_data_core(
        Xs, fold["Y"], knn, reference_nodes, eigval, eigvec,
        nu, lengthscale, graphbandwidth, bump_scale,
        seed=seed, n_pairs=n_pairs, plot_dir=plot_dir, plot_tag=plot_tag,
        node_labels=node_labels, edge_index=edge_index,
        edge_value=edge_value, vario_local_quantile=vario_local_quantile,
    )
    out["diag_fold_name"] = Path(run_dir).name
    out["diag_fold_n_lipids"] = int(fold["Y"].shape[1])
    out["diag_fold_n_test"] = int(fold["Y"].shape[0])
    # achieved test performance the run actually got (mean per-lipid Spearman)
    if fold["pred"] is not None:
        rs = [float(_spearmanr(fold["pred"][:, j], fold["Y"][:, j]).statistic)
              for j in range(fold["Y"].shape[1])]
        out["diag_fold_test_spearman_mean"] = float(np.nanmean(rs))
    # cross-check kernel hyperparameters against the run's config
    cfg = fold["config"]
    for cli_val, cfg_key, label in (
        (graphbandwidth, "graphbandwidth", "graphbandwidth"),
        (nu, "nu", "nu"),
    ):
        try:
            cv = float(cfg.get(cfg_key))
        except (TypeError, ValueError):
            continue
        if abs(cv - float(cli_val)) > 1e-9:
            logging.warning(
                f"[fold] {label} mismatch: run config={cv} but this script "
                f"used {cli_val}. The diagnostic kernel won't match the fold's "
                f"kernel — pass matching flags for a faithful comparison."
            )
    return out


def _resolve_lipids(spec, lipid_names):
    """Names OR integer indices -> list of names. Mirrors
    lgp_experiment_per_lipid.resolve_lipids (de-dup, preserve order)."""
    if spec is None:
        return list(lipid_names)
    out = []
    for tok in spec:
        try:
            i = int(tok)
            if 0 <= i < len(lipid_names):
                out.append(i); continue
        except (ValueError, TypeError):
            pass
        if tok in lipid_names:
            out.append(lipid_names.index(tok))
        else:
            logging.warning(f"[fold] lipid spec {tok!r} not found; skipped.")
    seen = set()
    return [lipid_names[i] for i in out if not (i in seen or seen.add(i))]


def _or_filters(f1, f2):
    """OR two pyarrow filter expressions (list-of-tuples or list-of-lists)."""
    def groups(f):
        if f is None:
            return None
        return f if (f and isinstance(f[0], list)) else [f]
    g1, g2 = groups(f1), groups(f2)
    if g1 is None or g2 is None:
        return None
    return g1 + g2


def fold_and_lipids_from_config(args: dict, log: logging.Logger):
    """Build the EXACT fold filter + lipid set the per-lipid training uses, by
    constructing the same MaldiConfig from --slices-dataset-file /
    --available-lipids-file (and friends), then reading config.section_filter /
    config.test_filter / config.selected_lipids_names. This is the faithful
    analogue of how lgp_experiment_per_lipid.py defines a fold.

    Returns (fold_filter, lipid_names). `--fold-side` picks train/test/all."""
    try:
        from config import MaldiConfig
    except Exception as e:
        raise RuntimeError(
            "Could not import MaldiConfig (`from config import MaldiConfig`). "
            "Run from the same environment as lgp_experiment_per_lipid.py, or "
            "use the manual --fold-filter / --fold-column instead. "
            f"(import error: {e})"
        )

    # MaldiConfig.from_args reads a dict shaped like the per-lipid experiment's
    # parsed args. Start from this script's args and fill in the per-lipid keys
    # (with harmless defaults) that MaldiConfig may require.
    import tempfile
    cfg_args = dict(args)
    defaults = {
        "mode": args.get("mode", "per_lipid"),
        "exp_name": args.get("exp_name", "laplacian_diag"),
        "output_dir": args.get("output_dir") or tempfile.mkdtemp(prefix="diagcfg_"),
        "dataset_path": args.get("dataset_path"),
        "maldi_file": args.get("maldi_file"),
        "available_lipids_file": args.get("available_lipids_file"),
        "slices_dataset_file": args.get("slices_dataset_file"),
        "template_name": args.get("template_name"),
        "reference_file": args.get("reference_file"),
        "annotations_file": args.get("annotations_file"),
        "num_inducing": args.get("num_inducing", 1000),
        "latent_dim": args.get("latent_dim", 1),
        "n_pixels": args.get("n_pixels", 10),
        "seed": args.get("seed", 42),
        "log_transform": args.get("log_transform", False),
        "region_bbox": args.get("region_bbox"),
        "device": args.get("device", "cpu"),
    }
    for k, v in defaults.items():
        cfg_args.setdefault(k, v)

    config = MaldiConfig.from_args(cfg_args)

    side = args.get("fold_side", "test")
    if side == "train":
        filt = config.section_filter
    elif side == "all":
        filt = _or_filters(config.section_filter, config.test_filter)
    else:
        filt = config.test_filter
    log.info(f"[fold] MaldiConfig fold-side={side} filter: {filt}")

    names_all = list(config.selected_lipids_names)
    spec = list(args.get("lipids") or [])
    if args.get("lipids_file"):
        with open(args["lipids_file"]) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    spec.append(line)
    spec = spec or None
    lipid_names = _resolve_lipids(spec, names_all)
    log.info(f"[fold] resolved {len(lipid_names)} lipids "
             f"(of {len(names_all)} available).")
    return filt, lipid_names


# ---- optional plotting (guarded; only when --diag-plot-dir given) ----------
def _try_plot_geodesic(plot_dir, tag, euc, geo, dspec, metrics):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    ax[0].scatter(euc, geo, s=4, alpha=.3)
    lim = [0, np.nanmax(geo)]; ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set(xlabel="euclidean", ylabel="geodesic",
              title=f"geo vs euc (rho={metrics['diag_geo_euc_spearman']:.3f})")
    ax[1].scatter(geo, dspec, s=4, alpha=.3)
    ax[1].set(xlabel="geodesic", ylabel="d_spec (full modes)",
              title=f"spec vs geo (rho={metrics['diag_dspec_geo_spearman_full']:.3f})")
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / f"geodesic_{tag}.png", dpi=120)
    plt.close(fig)


def _try_plot_maldi(plot_dir, tag, cum, cumS, dd, dissim):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    ax[0].plot(np.arange(1, len(cum) + 1), cum, label="data signal energy")
    ax[0].plot(np.arange(1, len(cumS) + 1), cumS, label="Matern prior weight")
    ax[0].set(xlabel="mode k", ylabel="cumulative fraction",
              title="data vs prior spectral content"); ax[0].legend()
    bins = np.linspace(0, np.percentile(dd, 99), 25)
    bi = np.digitize(dd, bins)
    mv = [dissim[bi == k].mean() if np.any(bi == k) else np.nan
          for k in range(1, len(bins))]
    ax[1].plot(0.5 * (bins[1:] + bins[:-1]), mv, "o-")
    ax[1].set(xlabel="euclidean distance", ylabel="mean |lipid_i - lipid_j|",
              title="empirical variogram")
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / f"maldi_{tag}.png", dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------------
# Graph construction — replicates visualize_laplacian.py:setup() but without
# the kernel and napari pieces. Returns the laplacian_op for the eigensolver.
# -------------------------------------------------------------------------
def build_graph_and_laplacian(
    args: dict, knn_method: str, norm: str,
    knn_k: int, graphbandwidth: float, cross_region_inflation: float,
    threshold: int,
    device: torch.device, graphs_cache: KnnGraphCache,
):
    """Build the laplacian_op for one sweep cell.

    knn_k, graphbandwidth, cross_region_inflation, threshold are passed
    explicitly because they vary across the sweep — args[...] holds the
    *defaults* used for the non-swept fields (template, stride, etc.).

    cross_region_inflation only affects faiss_atlas_weighted; it's ignored
    for the other methods and excluded from their cache keys so we don't
    create spurious duplicate work.

    Returns (laplacian_op, graph_key_for_eigvec_cache, n_nodes, n_edges).
    """
    template_full = np.load(args["reference_file"])
    annotations_full = np.load(args["annotations_file"]) if args["annotations_file"] else None

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_full, annotations_full,
        stride=args["stride"], region_bbox=args["region_bbox"],
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    reference_nodes_mm = torch.tensor(reference_ccf, dtype=torch.float32)
    coord_mean, coord_std = coord_norm_from_reference(template_full)
    reference_nodes = ((reference_nodes_mm - coord_mean) / coord_std).to(device)

    graph_key_parts = {
        "template": args["template_name"],
        "stride":   1 if args["region_bbox"] is not None else args["stride"],
        "thresh":   threshold,
        "method":   knn_method,
        "k":        knn_k,
        "nlist":    args["n_list"],
        "bbox":     tuple(args["region_bbox"]) if args["region_bbox"] is not None else None,
    }
    
    if knn_method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"]  = 3
    graph_key = make_graph_key(graph_key_parts)

    node_labels = None

    if knn_method == "faiss":
        knn, edge_index, edge_value = graphs_cache.train_or_load(
            key=graph_key, method="faiss", coords=reference_nodes,
            k=knn_k, nlist=args["n_list"], extra=graph_key_parts,
            device=device, force_recompute=False,
        )
    elif knn_method == "anatomical_atlas":
        if sub_atlas is None:
            raise RuntimeError(
                "anatomical_atlas requires --annotations-file; not provided."
            )
        knn, edge_index, edge_value = graphs_cache.train_or_load(
            key=graph_key, method="anatomical_atlas", volume=sub_volume,
            threshold=threshold, atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=knn_k, nlist=args["n_list"],
            extra=graph_key_parts, device=device, force_recompute=False,
        )
    elif knn_method == "faiss_atlas_weighted":
        if sub_atlas is None:
            raise RuntimeError(
                "faiss_atlas_weighted requires --annotations-file; not provided."
            )
        base_parts = dict(graph_key_parts); base_parts["method"] = "faiss"
        base_key = make_graph_key(base_parts)
        knn, edge_index, edge_value = graphs_cache.train_or_load(
            key=base_key, method="faiss", coords=reference_nodes,
            k=knn_k, nlist=args["n_list"], extra=base_parts,
            device=device, force_recompute=False,
        )
        node_labels = labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, threshold)
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=cross_region_inflation, treat_zero_as_cross=True,
        )
        graph_key_parts["weighting"] = f"atlas_x{cross_region_inflation:g}"
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"unknown knn_method: {knn_method!r}")

    bw_tensor = torch.tensor(float(graphbandwidth), device=device)
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0], bw_tensor, norm,
    )
    n_nodes = int(knn.x.shape[0])
    n_edges = int(edge_index.shape[1])
    return {
        "laplacian_op":    laplacian_op,
        "graph_key":       graph_key,
        "n_nodes":         n_nodes,
        "n_edges":         n_edges,
        "knn":             knn,
        "edge_index":      edge_index,
        "edge_value":      edge_value,
        "node_labels":     node_labels,
        "reference_nodes": reference_nodes,
        "sub_volume":      sub_volume,
        "voxel_offset":    voxel_offset,
        "voxel_scale_mm":  voxel_scale_mm,
        "coord_mean":      coord_mean,
        "coord_std":       coord_std,
    }

# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Inputs
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None,
                   help="Required for anatomical_atlas and faiss_atlas_weighted.")
    p.add_argument("--eigenvector-dir", required=True, type=Path,
                   help="Same dir visualize_laplacian.py uses. eigvecs/ and knn/ subdirs.")
    # Graph params (single values — the sweep across configs lives in the
    # submit shell, which spawns one job per (method, norm, thresh, k, bw,
    # inflation) combination)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--num-modes", type=int, default=1300)
    p.add_argument("--region-bbox", type=int, nargs=6, default=None)
    p.add_argument("--knn-method", required=True,
                   choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted"])
    p.add_argument("--normalization", required=True,
                   choices=["symmetric", "randomwalk"])
    p.add_argument("--knn-k", type=int, required=True)
    p.add_argument("--graphbandwidth", type=float, required=True)
    p.add_argument("--threshold", type=int, required=True)
    p.add_argument("--cross-region-inflation", type=float, default=10.0,
                   help=("Inflation factor for cross-region edges. Ignored "
                         "unless knn_method == faiss_atlas_weighted."))
    # Solver
    p.add_argument("--solver-backend", default="cupy", choices=["cupy", "scipy"])
    p.add_argument("--solver-tol", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    # Diagnostic params
    p.add_argument("--nu", type=int, default=2,
                   help="Matern smoothness for the matern-floor diagnostic and the kernel build.")
    p.add_argument("--lengthscale", type=float, default=1.0,
                   help="Matern lengthscale for the matern-floor diagnostic and the kernel build.")
    p.add_argument("--tol-zero", type=float, default=1e-10)
    p.add_argument("--tol-neg",  type=float, default=1e-6)
    # Kernel-PSD evaluation
    p.add_argument("--bump-scale", type=float, default=3.0,
                   help="Bump radius as a multiple of graphbandwidth.")
    p.add_argument("--bump-decay", type=float, default=0.05,
                   help="Bump sharpness.")
    p.add_argument("--n-test-on",  type=int, default=200,
                   help="In-sample test points (random subset of training nodes).")
    p.add_argument("--n-test-off", type=int, default=200,
                   help=("Out-of-sample test points (sub-threshold voxels in the "
                         "brain). Falls back to jittered training nodes if there "
                         "aren't enough sub-threshold voxels."))
    p.add_argument("--test-seed",  type=int, default=42,
                   help="RNG seed for test-point sampling.")
    p.add_argument("--test-sampling", choices=["subthreshold", "uniform_distance"],
                   default="uniform_distance",
                   help=("Off-manifold test sampling. 'subthreshold' (default) uses "
                         "real sub-threshold voxels. 'uniform_distance' seeds from "
                         "manifold nodes and spreads points UNIFORMLY across "
                         "distance-from-manifold (probe bump/kernel vs distance)."))
    p.add_argument("--test-dist-max", type=float, default=1.0,
                   help="(uniform_distance) max distance-from-manifold to span, z-units.")
    p.add_argument("--test-dist-bins", type=int, default=30,
                   help="(uniform_distance) number of equal-width distance bins.")
    p.add_argument("--per-point-kernel", action="store_true",
                   help=("Dump per-point bump + prior variance vs distance-from-manifold "
                         "to --diag-plot-dir (CSV + plot). Auto-on for uniform_distance."))
    p.add_argument("--skip-kernel-psd", action="store_true",
                   help="Skip the kernel Gram-matrix PSD evaluation; emit only Laplacian diagnostics.")
    # Extended diagnostics (Q1-Q4). Cheap eigen-health + bandwidth/OOS run by
    # default; turn them off with --skip-extended-diagnostics. Geodesic and
    # MALDI checks are opt-in (see below).
    p.add_argument("--skip-extended-diagnostics", action="store_true",
                   help="Skip the cheap eigen-health + bandwidth/OOS diagnostics.")
    p.add_argument("--eig-resid-modes", type=int, default=64,
                   help="How many modes to spot-check for ||L phi - lambda phi||.")
    p.add_argument("--geodesic-anchors", type=int, default=300,
                   help=("Q1/Q3: run Dijkstra geodesic vs spectral vs euclidean "
                         "from this many anchor nodes. 0 = skip (it is the slow "
                         "part on large graphs). 150-300 is plenty."))
    p.add_argument("--geodesic-seed", type=int, default=0)
    p.add_argument("--vario-local-quantile", type=float, default=0.15,
                   help=("Q4 signal variogram: the local (near-field) variogram "
                         "uses each metric's nearest fraction of pairs. Sharper "
                         "manifold-vs-euclidean read for oscillatory signals."))
    p.add_argument("--run-dir", type=Path, default=None,
                   help=("Q4 (optional): a trained per-lipid run dir, if you "
                         "happen to have one — uses its exact held-out fold + "
                         "coord normalization. NOT required; prefer --maldi-file "
                         "+ --fold-* to configure a fold directly on the CLI."))
    p.add_argument("--maldi-file", default=None,
                   help="Q4: MALDI parquet (enables the data-driven kernel check).")
    p.add_argument("--lipids", nargs="*", default=None,
                   help="Selected lipid column names (as in experiment.py).")
    p.add_argument("--log-transform", action="store_true",
                   help="Apply log(x+1e-10) to lipids, matching experiment.py.")
    # ---- fold via the experiment's own split machinery (preferred) ----
    # Mirrors lgp_experiment_per_lipid.py: the slices file + available-lipids
    # file go through MaldiConfig, which yields the exact section/test filters
    # and lipid names the training used.
    p.add_argument("--kernel", required=True)
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    p.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=2000, help="Batch size for training")
    p.add_argument("--slices-dataset-file", default=None,
                   help="Fold split JSON (e.g. .../splits/fold_3.json). With "
                        "--available-lipids-file, builds the SAME fold filter "
                        "the per-lipid training uses, via MaldiConfig.")
    p.add_argument("--available-lipids-file", default=None,
                   help="Available-lipids .npy (as in run_lgp_per_lipid.sh).")
    p.add_argument("--dataset-path", default=None,
                   help="Dataset path passed to MaldiConfig.")
    p.add_argument("--lipids-file", default=None,
                   help="Text file with one lipid name (or index) per line; "
                        "blanks/'#' ignored. Resolved against the available "
                        "lipids, exactly like the per-lipid trainer.")
    p.add_argument("--fold-side", choices=["test", "train", "all"], default="test",
                   help="Which side of the split to run Q4 on (default: test).")
    p.add_argument("--mode", default="per_lipid", help="Passed to MaldiConfig.")
    p.add_argument("--exp-name", default="laplacian_diag",
                   help="Passed to MaldiConfig (no training output is written).")
    p.add_argument("--output-dir", default=None,
                   help="Passed to MaldiConfig (defaults to a temp dir).")
    p.add_argument("--num-inducing", type=int, default=1000,
                   help="Passed to MaldiConfig for compatibility.")
    p.add_argument("--latent-dim", type=int, default=1,
                   help="Passed to MaldiConfig for compatibility.")
    p.add_argument("--maldi-max-rows", type=int, default=400000)
    p.add_argument("--diag-plot-dir", type=Path, default=None,
                   help="If set, dump diagnostic PNGs here (off by default).")
    # Output
    p.add_argument("--out", type=Path, required=True,
                   help=("Output CSV path. One row will be written. The submit "
                         "shell builds a config-unique filename per job so "
                         "parallel jobs don't collide; the notebook concatenates "
                         "all CSVs in the output directory."))
    p.add_argument("--skip-if-row-exists", action="store_true",
                   help=("If --out already exists and contains a row with the "
                         "same cache_key, exit immediately without recomputing. "
                         "Lets you re-submit the whole sweep safely."))
    return vars(p.parse_args())


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eigvec_dir = Path(args["eigenvector_dir"]) / "eigvecs"
    knn_dir    = Path(args["eigenvector_dir"]) / "knn"
    eigvec_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args["device"])
    graphs_cache = KnnGraphCache(cache_dir=knn_dir, verbose=True)

    ncv_min = max(1500, 3 * args["num_modes"] + 20)
    knn_method = args["knn_method"]
    norm       = args["normalization"]
    threshold  = args["threshold"]
    knn_k      = args["knn_k"]
    bw         = args["graphbandwidth"]
    infl       = args["cross_region_inflation"] if knn_method == "faiss_atlas_weighted" else float("nan")

    t0 = time.time()
    row: dict[str, Any] = {
        "knn_method":             knn_method,
        "normalization":          norm,
        "threshold":              threshold,
        "graphbandwidth":         bw,
        "knn_k":                  knn_k,
        "cross_region_inflation": (infl if knn_method == "faiss_atlas_weighted" else ""),
        "stride":                 args["stride"],
        "num_modes":              args["num_modes"],
        "nu":                     args["nu"],
        "lengthscale":            args["lengthscale"],
    }
    logging.info(
        f"config: {knn_method} {norm} thr={threshold} k={knn_k} bw={bw:g}"
        + (f" infl={infl:g}" if knn_method == "faiss_atlas_weighted" else "")
    )

    # Add bump/test fields to the row schema so they appear in the CSV
    # even when --skip-kernel-psd is set (as blanks).
    row["bump_scale"] = args["bump_scale"]
    row["bump_decay"] = args["bump_decay"]

    try:
        built = build_graph_and_laplacian(
            args, knn_method, norm,
            knn_k=knn_k, graphbandwidth=bw, cross_region_inflation=infl,
            threshold=threshold,
            device=device, graphs_cache=graphs_cache,
        )
        laplacian_op = built["laplacian_op"]
        graph_key    = built["graph_key"]

        eigvec_key_parts = {
            "graph": graph_key, "norm": norm, "bw": bw, "modes": args["num_modes"],
        }
        ekey = make_eig_key(eigvec_key_parts)
        row["cache_key"] = ekey

        if args["skip_if_row_exists"] and args["out"].exists():
            with open(args["out"]) as f:
                for r in csv.DictReader(f):
                    if r.get("cache_key", "") == ekey and r.get("status", "") == "OK":
                        logging.info(f"row already exists in {args['out']} — exiting")
                        return

        solver = LaplacianEigensolver(
            num_modes=args["num_modes"], backend=args["solver_backend"],
            tol=args["solver_tol"], ncv_min=ncv_min, verbose=True,
        )
        eigval, eigvec = solver.compute_or_load(
            laplacian_op, cache_dir=eigvec_dir, key=ekey,
            graphbandwidth=bw, laplacian_normalization=norm,
            extra=eigvec_key_parts, force_recompute=False, device=device,
        )
        analyze_kwargs = dict(
            nu=args["nu"], lengthscale=args["lengthscale"],
            tol_zero=args["tol_zero"], tol_neg=args["tol_neg"],
        )
        ev_np = eigval.detach().cpu().numpy()
        row.update(analyze_eigvals(ev_np, **analyze_kwargs))
        row["fp_n_nodes"] = built["n_nodes"]
        row["fp_n_edges"] = built["n_edges"]

        # Kernel-Gram PSD evaluation: build the deployed Matern kernel and
        # evaluate it at sampled in-sample (training nodes) and out-of-sample
        # test points (sub-threshold voxels, or uniform-across-distance when
        # --test-sampling uniform_distance). Reports the deployed Gram matrix K_m
        # at the full block plus the in-sample-only (on_) and out-of-sample-only
        # (off_) sub-blocks; optionally dumps bump/prior-variance vs distance.
        if not args["skip_kernel_psd"]:
            rng = np.random.default_rng(args["test_seed"])
            if args["test_sampling"] == "uniform_distance":
                test_pts, n_on, _test_dist = sample_test_points_uniform_distance(
                    reference_nodes=built["reference_nodes"],
                    knn=built["knn"],
                    n_on=args["n_test_on"], n_off=args["n_test_off"],
                    dist_max=args["test_dist_max"], n_bins=args["test_dist_bins"],
                    rng=rng,
                )
            else:
                test_pts, n_on = sample_test_points(
                    reference_nodes=built["reference_nodes"],
                    sub_volume=built["sub_volume"],
                    voxel_offset=built["voxel_offset"],
                    voxel_scale_mm=built["voxel_scale_mm"],
                    coord_mean=built["coord_mean"], coord_std=built["coord_std"],
                    threshold=threshold,
                    n_on=args["n_test_on"], n_off=args["n_test_off"],
                    rng=rng,
                )
            matern_kernel = RiemannMaternKernel(
                nu=args["nu"],
                lengthscale=args["lengthscale"],
                knn=built["knn"],
                edge_index=built["edge_index"],
                edge_value=built["edge_value"],
                eigval=eigval, eigvec=eigvec,
                nearest_neighbors=knn_k,
                laplacian_normalization=norm,
                num_modes=args["num_modes"],
                bump_scale=args["bump_scale"],
                bump_decay=args["bump_decay"],
                graphbandwidth_init=bw,
            ).to(device)
            _dump_pp = args["per_point_kernel"] or args["test_sampling"] == "uniform_distance"
            _pp_dir = args["diag_plot_dir"] if (_dump_pp and args["diag_plot_dir"]) else None
            _pp_csv = (Path(_pp_dir) / f"kernel_vs_distance_{ekey}.csv") if _pp_dir else None
            _cs = built["coord_std"]
            _cs = float(_cs.mean()) if torch.is_tensor(_cs) else float(np.mean(_cs))
            kdiag = evaluate_kernel_psd(
                matern_kernel, test_pts,
                graphbandwidth=bw,
                bump_scale=args["bump_scale"], bump_decay=args["bump_decay"],
                n_on_manifold=n_on,
                analyze_kwargs=analyze_kwargs,
                per_point_csv=_pp_csv,
                plot_dir=_pp_dir,
                plot_tag=ekey,
                coord_std=_cs,
            )
            row.update(kdiag)
            del matern_kernel
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ---- Extended diagnostics (Q1-Q4) --------------------------------
        plot_tag = ekey
        if not args["skip_extended_diagnostics"]:
            row.update(eigen_health_diagnostics(
                laplacian_op, eigval, eigvec, built["edge_index"], norm,
                n_res_modes=args["eig_resid_modes"],
            ))
            row.update(bandwidth_oos_diagnostics(
                eigval, graphbandwidth=bw,
                nu=args["nu"], lengthscale=args["lengthscale"],
            ))
        if args["geodesic_anchors"] > 0:
            row.update(geodesic_distance_diagnostics(
                built["edge_index"], built["edge_value"],
                built["reference_nodes"], eigval, eigvec,
                nu=args["nu"], lengthscale=args["lengthscale"],
                n_anchors=args["geodesic_anchors"], seed=args["geodesic_seed"],
                plot_dir=args["diag_plot_dir"], plot_tag=plot_tag,
            ))
        if args["run_dir"] is not None:
            row.update(fold_data_diagnostics(
                args["run_dir"], args["lipids"],
                knn=built["knn"], reference_nodes=built["reference_nodes"],
                eigval=eigval, eigvec=eigvec,
                nu=args["nu"], lengthscale=args["lengthscale"],
                graphbandwidth=bw, bump_scale=args["bump_scale"],
                plot_dir=args["diag_plot_dir"], plot_tag=plot_tag,
                node_labels=built["node_labels"], edge_index=built["edge_index"],
                edge_value=built["edge_value"],
                vario_local_quantile=args["vario_local_quantile"],
            ))
        elif args["maldi_file"] and (args["slices_dataset_file"]
                                      or args["lipids"] or args["lipids_file"]):
            log = logging.getLogger()
            if not args["slices_dataset_file"]:
                raise RuntimeError(
                        "--slices-dataset-file needs --available-lipids-file "
                        "(MaldiConfig derives lipid names from it)."
                    )
            if not args["available_lipids_file"]:
                raise RuntimeError(
                    "--slices-dataset-file needs --available-lipids-file "
                    "(MaldiConfig derives lipid names from it)."
                )
            fold_filter, lipid_names = fold_and_lipids_from_config(args, log)
            row.update(maldi_data_diagnostics(
                args["maldi_file"], lipid_names, args["log_transform"],
                knn=built["knn"], reference_nodes=built["reference_nodes"],
                eigval=eigval, eigvec=eigvec,
                nu=args["nu"], lengthscale=args["lengthscale"],
                graphbandwidth=bw, bump_scale=args["bump_scale"],
                fold_filter=fold_filter,
                max_rows=args["maldi_max_rows"],
                plot_dir=args["diag_plot_dir"], plot_tag=plot_tag,
                coord_mean=built["coord_mean"], coord_std=built["coord_std"],
                node_labels=built["node_labels"], edge_index=built["edge_index"],
                edge_value=built["edge_value"],
                vario_local_quantile=args["vario_local_quantile"],
            ))

        row["status"]   = "OK"
        row["wall_sec"] = round(time.time() - t0, 2)
        logging.info(
            f"OK   λ_min={row['lambda_min']:+.3e} ratio={row['ratio_min_over_max']:+.2e} "
            + (
                f"K_m_full={row.get('K_m_ratio_min_over_max', float('nan')):+.2e} "
                f"K_m_on={row.get('K_m_on_ratio_min_over_max', float('nan')):+.2e} "
                f"K_m_off={row.get('K_m_off_ratio_min_over_max', float('nan')):+.2e} "
                if not args["skip_kernel_psd"] else ""
            )
            + f"({row['wall_sec']:.1f}s)"
        )
        # Compact extended-diagnostic summary in the log (CSV has everything).
        if not args["skip_extended_diagnostics"]:
            logging.info(
                "diag orthoL2=%.2e orthoD=%.2e varproxy=%.1fx resid_med=%.1e "
                "ncomp=%s weyl_dim=%.2f part_ratio=%.0f"
                % (row.get("diag_ortho_l2_offmax", float("nan")),
                   row.get("diag_ortho_deg_offmax", float("nan")),
                   row.get("diag_varproxy_ratio", float("nan")),
                   row.get("diag_eig_resid_median", float("nan")),
                   row.get("diag_n_components", "?"),
                   row.get("diag_weyl_dim", float("nan")),
                   row.get("diag_smat_participation_ratio", float("nan")))
            )
        if args["geodesic_anchors"] > 0:
            logging.info(
                "diag geo: geo~euc rho=%.3f ratio_p95=%.2f | dspec~geo rho=%.3f "
                "(low modes %.3f)"
                % (row.get("diag_geo_euc_spearman", float("nan")),
                   row.get("diag_geo_euc_ratio_p95", float("nan")),
                   row.get("diag_dspec_geo_spearman_full", float("nan")),
                   row.get("diag_dspec_geo_spearman_low", float("nan")))
            )
        if args["run_dir"] is not None or (
            args["maldi_file"] and (args["slices_dataset_file"]
                                    or args["lipids"] or args["lipids_file"])):
            logging.info(
                "diag maldi: in_support=%.3f snap_med=%.3f | e90_mode_frac=%.2f | "
                "match man=%.3f euc=%.3f"
                % (row.get("diag_maldi_frac_in_support", float("nan")),
                   row.get("diag_maldi_snap_median", float("nan")),
                   row.get("diag_maldi_e90_mode_frac", float("nan")),
                   row.get("diag_maldi_match_spearman_manifold", float("nan")),
                   row.get("diag_maldi_match_spearman_euclidean", float("nan")))
            )
        if args["run_dir"] is not None and "diag_fold_test_spearman_mean" in row:
            logging.info(
                "diag fold: %s  n_test=%s n_lipids=%s  achieved_test_spearman=%.3f"
                % (row.get("diag_fold_name", "?"),
                   row.get("diag_fold_n_test", "?"),
                   row.get("diag_fold_n_lipids", "?"),
                   row.get("diag_fold_test_spearman_mean", float("nan")))
            )
    except Exception as e:
        row["status"] = "ERROR"
        row["error"]  = f"{type(e).__name__}: {e}"
        row["wall_sec"] = round(time.time() - t0, 2)
        logging.error(f"ERROR {type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
        # Still write the error row — the notebook can filter status!=OK.

    write_csv(args["out"], [row])
    logging.info(f"wrote 1 row to {args['out']}")


def write_csv(path: Path, rows: list[dict]):
    id_cols  = ["knn_method", "normalization", "graphbandwidth", "knn_k",
                "cross_region_inflation", "stride", "threshold",
                "num_modes", "nu", "lengthscale",
                "bump_scale", "bump_decay"]
    status_col = ["status", "wall_sec"]
    lap_cols = [
        "n_total", "lambda_min", "lambda_max", "ratio_min_over_max",
        "n_zero_exact", "n_zero_eps", "n_negative", "n_negative_significant",
        "n_below_matern_floor", "matern_floor",
        "spectral_gap", "condition_number", "lambda_min_positive",
    ]
    # Kernel-PSD bookkeeping
    kpsd_meta = ["n_test_on", "n_test_off",
                 "bump_min", "bump_mean", "bump_max",
                 "bump_on_mean", "bump_off_mean",
                 "K_m_max"]
    # Per-Gram-matrix diagnostics. Only K_m is reported (the deployed kernel),
    # at the full block plus the in-sample (on_) and out-of-sample (off_) blocks.
    block_prefixes = ["", "on_", "off_"]
    kernel_diag_fields = [
        "n_total", "lambda_min", "lambda_max", "ratio_min_over_max",
        "n_zero_exact", "n_zero_eps", "n_negative", "n_negative_significant",
        "n_below_matern_floor", "matern_floor",
        "spectral_gap", "condition_number", "lambda_min_positive",
    ]
    k_cols: list[str] = []
    for mat in ("K_m",):
        for blk in block_prefixes:
            for f in kernel_diag_fields:
                k_cols.append(f"{mat}_{blk}{f}")
            k_cols.append(f"{mat}_{blk}status")
    k_cols += ["per_point_n", "per_point_dist_max"]
    fp_cols   = ["fp_n_nodes", "fp_n_edges"]
    # Extended diagnostics (Q1-Q4). Blank when the corresponding check was
    # skipped (cheap eigen-health/bandwidth run by default; geodesic and maldi
    # are opt-in).
    diag_cols = [
        # Q2 eigen / graph health
        "diag_n_components", "diag_n_zero_modes",
        "diag_ortho_l2_offmax", "diag_ortho_deg_offmax", "diag_varproxy_ratio",
        "diag_eig_resid_median", "diag_eig_resid_p90", "diag_eig_resid_max",
        "diag_weyl_slope", "diag_weyl_dim",
        # bandwidth / OOS sanity
        "diag_oos_bw2lam_max", "diag_oos_denom_crossings",
        "diag_oos_first_crossing_mode", "diag_smat_participation_ratio",
        # Q1 / Q3 geodesic vs spectral vs euclidean
        "diag_geo_reachable_frac", "diag_geo_euc_spearman",
        "diag_geo_euc_ratio_median", "diag_geo_euc_ratio_p95",
        "diag_dspec_geo_spearman_full", "diag_dspec_geo_spearman_low",
        "diag_dspec_euc_spearman_full", "diag_geo_n_anchors", "diag_geo_modes_low",
        # Q4 MALDI data-driven
        "diag_maldi_n_points", "diag_maldi_snap_median", "diag_maldi_snap_p95",
        "diag_maldi_bump_support", "diag_maldi_frac_in_support",
        "diag_maldi_node_coverage", "diag_maldi_e50_mode", "diag_maldi_e90_mode",
        "diag_maldi_e90_mode_frac", "diag_maldi_prior_e90_mode",
        "diag_maldi_tail_energy_beyond_modes",
        "diag_maldi_match_spearman_manifold", "diag_maldi_match_spearman_euclidean",
        "diag_maldi_cross_signal_jump", "diag_maldi_intra_signal_jump",
        "diag_maldi_cross_intra_ratio",
        # Q4 signal variogram (covariance vs distance), global + local
        "diag_maldi_vario_man_r2", "diag_maldi_vario_man_r2_local",
        "diag_maldi_vario_euc_r2", "diag_maldi_vario_euc_r2_local",
        "diag_maldi_vario_man_nugget_sill", "diag_maldi_vario_man_nugget_sill_local",
        "diag_maldi_vario_euc_nugget_sill", "diag_maldi_vario_euc_nugget_sill_local",
        "diag_maldi_vario_n_anchors", "diag_maldi_vario_reachable_frac",
        "diag_maldi_n_lipids",
        # Q4 fold-specific (only when --run-dir is used)
        "diag_fold_name", "diag_fold_n_lipids", "diag_fold_n_test",
        "diag_fold_test_spearman_mean",
    ]
    misc_cols = ["cache_key", "error"]
    columns = id_cols + status_col + lap_cols + kpsd_meta + k_cols + fp_cols + diag_cols + misc_cols

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()