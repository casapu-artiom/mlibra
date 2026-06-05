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
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    nu: float, lengthscale: float,
    n_on_manifold: int,
    analyze_kwargs: dict,
) -> dict:
    """Build the three Gram matrices and PSD-diagnose each:

      K_m     = features(x) · features(y)            (bare manifold)
      K_bm    = b(x) · K_m(x,y) · b(y)               (bump-modulated)
      K_full  = K_bm + (1-b(x))(1-b(y)) · K_eucl     (deployed kernel)

    Reductions on the in-sample block and the out-of-sample block are
    reported separately too — non-PSD restricted to the in-sample block
    is a much stronger signal (the truncated Nyström features aren't even
    PSD at training nodes) than non-PSD only on the off-manifold block
    (Nyström extension can lose PSD on novel queries even when fine on
    training nodes).
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

    K_eu = matern_euclidean_pairwise(test_points, nu, lengthscale)
    K_bm = b[:, None] * K_m * b[None, :]
    one_m_b = 1.0 - b
    K_full = K_bm + (one_m_b[:, None] * one_m_b[None, :]) * K_eu

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
        "K_eu_max":    float(np.abs(K_eu).max()),
        "K_m_max":     float(np.abs(K_m).max()),
    }
    # Full-sample diagnostics, then submatrix diagnostics for the in-sample
    # block (K[:n_on, :n_on]) and the out-of-sample block (K[n_on:, n_on:]).
    blocks = [
        ("",   slice(None),  slice(None)),
        ("on_",  slice(0, n_on), slice(0, n_on)),
        ("off_", slice(n_on, None), slice(n_on, None)),
    ]
    for tag, mat in [("K_m", K_m), ("K_bm", K_bm), ("K_full", K_full)]:
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
    return out


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
    coord_mean = reference_nodes_mm.mean(dim=0)
    coord_std = reference_nodes_mm.std(dim=0).clamp(min=1e-6)
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
    p.add_argument("--skip-kernel-psd", action="store_true",
                   help="Skip the kernel Gram-matrix PSD evaluation; emit only Laplacian diagnostics.")
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
        # (sub-threshold voxels) test points. Three Gram matrices reported:
        # K_m (bare), K_bm (bump-modulated), K_full (with Euclidean fallback).
        # Each Gram matrix is diagnosed at the full block as well as the
        # in-sample-only (on_) and out-of-sample-only (off_) sub-blocks.
        if not args["skip_kernel_psd"]:
            rng = np.random.default_rng(args["test_seed"])
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
            kdiag = evaluate_kernel_psd(
                matern_kernel, test_pts,
                graphbandwidth=bw,
                bump_scale=args["bump_scale"], bump_decay=args["bump_decay"],
                nu=args["nu"], lengthscale=args["lengthscale"],
                n_on_manifold=n_on,
                analyze_kwargs=analyze_kwargs,
            )
            row.update(kdiag)
            del matern_kernel
            if device.type == "cuda":
                torch.cuda.empty_cache()

        row["status"]   = "OK"
        row["wall_sec"] = round(time.time() - t0, 2)
        logging.info(
            f"OK   λ_min={row['lambda_min']:+.3e} ratio={row['ratio_min_over_max']:+.2e} "
            + (
                f"K_m_full={row.get('K_m_ratio_min_over_max', float('nan')):+.2e} "
                f"K_m_on={row.get('K_m_on_ratio_min_over_max', float('nan')):+.2e} "
                f"K_m_off={row.get('K_m_off_ratio_min_over_max', float('nan')):+.2e} "
                f"K_full_full={row.get('K_full_ratio_min_over_max', float('nan')):+.2e} "
                if not args["skip_kernel_psd"] else ""
            )
            + f"({row['wall_sec']:.1f}s)"
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
                 "K_eu_max", "K_m_max"]
    # Per-Gram-matrix diagnostics. Each matrix (K_m, K_bm, K_full) is reported
    # at the full block plus the in-sample (on_) and out-of-sample (off_) blocks.
    block_prefixes = ["", "on_", "off_"]
    kernel_diag_fields = [
        "n_total", "lambda_min", "lambda_max", "ratio_min_over_max",
        "n_zero_exact", "n_zero_eps", "n_negative", "n_negative_significant",
        "n_below_matern_floor", "matern_floor",
        "spectral_gap", "condition_number", "lambda_min_positive",
    ]
    k_cols: list[str] = []
    for mat in ("K_m", "K_bm", "K_full"):
        for blk in block_prefixes:
            for f in kernel_diag_fields:
                k_cols.append(f"{mat}_{blk}{f}")
            k_cols.append(f"{mat}_{blk}status")
    fp_cols   = ["fp_n_nodes", "fp_n_edges"]
    misc_cols = ["cache_key", "error"]
    columns = id_cols + status_col + lap_cols + kpsd_meta + k_cols + fp_cols + misc_cols

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()