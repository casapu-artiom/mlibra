#!/usr/bin/env python
# encoding: utf-8
"""
Laplacian PSD diagnostic sweep.

For each (knn_method, normalization) config:
  1. Try to load cached eigenpairs from <eigenvector-dir>/eigvecs/.
  2. If missing, build the KNN graph + Laplacian and run the eigensolver
     (LaplacianEigensolver.compute_or_load, which writes the cache for next time).
  3. Compute PSD-violation diagnostics from the eigenvalues alone.
  4. Append a row to a CSV.

This script is the same setup as visualize_laplacian.py minus the napari UI —
it shares the same cache keys, so anything it computes is reusable by the
visualiser and vice versa.

The diagnostics computed per config are documented in `analyze_eigvals` below.
Key indicators:

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
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges, labels_for_nodes_from_sub_atlas
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
)
from utils import crop_or_stride_volume, reference_ccf_from_subvolume


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
    return laplacian_op, graph_key, n_nodes, n_edges


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Inputs (match visualize_laplacian.py)
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None,
                   help="Required for anatomical_atlas and faiss_atlas_weighted.")
    p.add_argument("--eigenvector-dir", required=True, type=Path,
                   help="Same dir visualize_laplacian.py uses. eigvecs/ and knn/ subdirs.")
    # Graph params (defaults shared across the sweep)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--num-modes", type=int, default=1300)
    p.add_argument("--region-bbox", type=int, nargs=6, default=None)
    # Sweep dimensions
    p.add_argument("--knn-methods", nargs="+",
                   default=["faiss", "anatomical_atlas", "faiss_atlas_weighted"])
    p.add_argument("--normalizations", nargs="+",
                   default=["symmetric", "randomwalk"])
    p.add_argument("--knn-ks", type=int, nargs="+", default=[120],
                   help="List of k values for the KNN graph.")
    p.add_argument("--graphbandwidths", type=float, nargs="+", default=[0.1],
                   help="List of graph bandwidths.")
    p.add_argument("--thresholds", type=int, nargs="+", default=[5],
                   help=("List of tissue-mask thresholds. Each value defines "
                         "a different graph (different node set), so this "
                         "axis multiplies the work in full."))
    p.add_argument("--cross-region-inflations", type=float, nargs="+", default=[10.0],
                   help=("List of inflation factors. Only iterated when "
                         "knn_method == faiss_atlas_weighted — for the other "
                         "methods inflation is irrelevant and would create "
                         "duplicate work."))
    # Solver
    p.add_argument("--solver-backend", default="cupy", choices=["cupy", "scipy"])
    p.add_argument("--solver-tol", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    # Diagnostic params
    p.add_argument("--nu", type=int, default=2,
                   help="Matern smoothness for the matern-floor diagnostic.")
    p.add_argument("--lengthscale", type=float, default=1.0,
                   help="Matern lengthscale for the matern-floor diagnostic.")
    p.add_argument("--tol-zero", type=float, default=1e-10)
    p.add_argument("--tol-neg",  type=float, default=1e-6)
    # Output / control
    p.add_argument("--out", type=Path, default=Path("laplacian_psd_sweep.csv"))
    p.add_argument("--append", action="store_true",
                   help="Append to existing CSV instead of overwriting; skips configs already present.")
    p.add_argument("--skip-on-error", action="store_true",
                   help="If a config fails to build/eigensolve, log and continue (status=ERROR).")
    return vars(p.parse_args())


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eigvec_dir = Path(args["eigenvector_dir"]) / "eigvecs"
    knn_dir    = Path(args["eigenvector_dir"]) / "knn"
    eigvec_dir.mkdir(parents=True, exist_ok=True)

    # If --append, read existing rows so we can skip already-done configs.
    done_keys: set[str] = set()
    existing_rows: list[dict] = []
    if args["append"] and args["out"].exists():
        with open(args["out"]) as f:
            for r in csv.DictReader(f):
                existing_rows.append(r)
                done_keys.add(r.get("cache_key", ""))

    device = torch.device(args["device"])
    graphs_cache = KnnGraphCache(cache_dir=knn_dir, verbose=True)

    rows: list[dict[str, Any]] = list(existing_rows)
    ncv_min = max(1500, 3 * args["num_modes"] + 20)

    # Build the full sweep grid. The inflation axis is only meaningful for
    # `faiss_atlas_weighted` — for the other methods we collapse it to a
    # single sentinel so the identical config doesn't run N times.
    configs: list[tuple[str, str, int, int, float, float]] = []
    for knn_method, norm, threshold, knn_k, bw in product(
        args["knn_methods"], args["normalizations"],
        args["thresholds"], args["knn_ks"], args["graphbandwidths"],
    ):
        if knn_method == "faiss_atlas_weighted":
            for infl in args["cross_region_inflations"]:
                configs.append((knn_method, norm, threshold, knn_k, bw, float(infl)))
        else:
            configs.append((knn_method, norm, threshold, knn_k, bw, float("nan")))

    logging.info(
        f"sweep grid: {len(configs)} configs "
        f"({len(args['knn_methods'])} methods × {len(args['normalizations'])} norms × "
        f"{len(args['thresholds'])} thresholds × {len(args['knn_ks'])} ks × "
        f"{len(args['graphbandwidths'])} bws "
        f"× inflations only for faiss_atlas_weighted)"
    )

    for i, (knn_method, norm, threshold, knn_k, bw, infl) in enumerate(configs):
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
            f"[{i+1}/{len(configs)}] {knn_method:22s} {norm:11s} "
            f"thr={threshold} k={knn_k} bw={bw:g}"
            + (f" infl={infl:g}" if knn_method == "faiss_atlas_weighted" else "")
        )
        try:
            laplacian_op, graph_key, n_nodes, n_edges = build_graph_and_laplacian(
                args, knn_method, norm,
                knn_k=knn_k, graphbandwidth=bw, cross_region_inflation=infl,
                threshold=threshold,
                device=device, graphs_cache=graphs_cache,
            )
            eigvec_key_parts = {
                "graph": graph_key, "norm": norm, "bw": bw, "modes": args["num_modes"],
            }
            ekey = make_eig_key(eigvec_key_parts)
            row["cache_key"] = ekey
            if ekey in done_keys:
                logging.info(f"     SKIP — already in CSV ({ekey})")
                continue

            solver = LaplacianEigensolver(
                num_modes=args["num_modes"], backend=args["solver_backend"],
                tol=args["solver_tol"], ncv_min=ncv_min, verbose=True,
            )
            eigval, _eigvec = solver.compute_or_load(
                laplacian_op, cache_dir=eigvec_dir, key=ekey,
                graphbandwidth=bw, laplacian_normalization=norm,
                extra=eigvec_key_parts, force_recompute=False, device=device,
            )
            ev_np = eigval.detach().cpu().numpy()
            diag = analyze_eigvals(
                ev_np, nu=args["nu"], lengthscale=args["lengthscale"],
                tol_zero=args["tol_zero"], tol_neg=args["tol_neg"],
            )
            row.update(diag)
            row["fp_n_nodes"] = n_nodes
            row["fp_n_edges"] = n_edges
            row["status"]     = "OK"
            row["wall_sec"]   = round(time.time() - t0, 2)
            logging.info(
                f"     OK   λ_min={row['lambda_min']:+.3e} λ_max={row['lambda_max']:.3e} "
                f"ratio={row['ratio_min_over_max']:+.2e} "
                f"n_neg={row['n_negative']:5d} n_neg_sig={row['n_negative_significant']:5d} "
                f"({row['wall_sec']:.1f}s)"
            )
            # Free GPU memory before the next config — eigvecs can be huge.
            del eigval, _eigvec, laplacian_op
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            row["status"] = "ERROR"
            row["error"]  = f"{type(e).__name__}: {e}"
            row["wall_sec"] = round(time.time() - t0, 2)
            logging.error(f"     ERROR {type(e).__name__}: {e}")
            if not args["skip_on_error"]:
                logging.error(traceback.format_exc())
                raise
            logging.error(traceback.format_exc())
        rows.append(row)

        # Write incremental — protects long sweeps against interrupts.
        write_csv(args["out"], rows)

    write_csv(args["out"], rows)
    logging.info(f"done. {len(rows)} rows in {args['out']}")


def write_csv(path: Path, rows: list[dict]):
    id_cols  = ["knn_method", "normalization", "graphbandwidth", "knn_k",
                "cross_region_inflation", "stride", "threshold",
                "num_modes", "nu", "lengthscale"]
    status_col = ["status", "wall_sec"]
    diag_cols = [
        "n_total", "lambda_min", "lambda_max", "ratio_min_over_max",
        "n_zero_exact", "n_zero_eps", "n_negative", "n_negative_significant",
        "n_below_matern_floor", "matern_floor",
        "spectral_gap", "condition_number", "lambda_min_positive",
    ]
    fp_cols   = ["fp_n_nodes", "fp_n_edges"]
    misc_cols = ["cache_key", "error"]
    columns = id_cols + status_col + diag_cols + fp_cols + misc_cols

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()