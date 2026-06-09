#!/usr/bin/env python
# encoding: utf-8
"""
toy_box_cylinder_laplacian_test.py
====================================

Validate the laplacian_test diagnostic suite on the box+cylinder toy geometry.

Why this toy (analogous to toy_laplacian_test.py for the Swiss roll):
  - True surface labels (LABEL_CYL vs LABEL_BOX) are known, so every diagnostic
    has a KNOWN expected answer we can check.
  - The "fold" is quantified: cross-surface signal jump ~2 (antiphase), minimum
    cross-surface distance ~0.5 units.

Strategy: reproduce build_graph_and_laplacian's `built` dict from the toy point
cloud, then call laplacian_test's diagnostic functions unchanged.

  --mode surface   (default): data on the surfaces; manifold-favourable.
  --mode volume :  data fills the box; Euclidean-favourable.

TOY-TRUTH CHECKS (validated against known structure):
  1. Cross-surface fold:  Euclidean distance between cylinder and box wall is
     SMALL but the graph geodesic (with inflation) should be large.
  2. Graph is disconnected into 2 components without inflation (surfaces too far
     for KNN to bridge); with inflation applied, cross-surface edges are penalised
     so the spectral separation is even clearer.
  3. Signal is antiphase across the gap: diag_maldi_match_spearman_manifold
     should exceed euclidean when signal='geodesic' + mode='surface'.
  4. OOS denominator safe: bw^2 * lambda_max < 0.8.

Run from the directory containing laplacian_test.py and utils.py (i.e. maldi/),
or set TOY_PROJECT_ROOT to the project root.  From toy_example/:

    python toy_box_cylinder_laplacian_test.py --signal geodesic --mode surface \\
        --inflation 50 --num-modes 200

    python toy_box_cylinder_laplacian_test.py --signal geodesic --mode volume \\
        --inflation 1 --num-modes 200
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors toy_laplacian_test.py exactly)
# ---------------------------------------------------------------------------
def _bootstrap_paths():
    here = Path(__file__).resolve().parent
    root = next((p for p in [here, *here.parents]
                 if (p / "manifold_gp").is_dir() and (p / "maldi").is_dir()),
                here.parent)
    root = Path(os.environ.get("TOY_PROJECT_ROOT", root)).resolve()
    for d in (root, here, root / "maldi"):
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    return root

_PROJECT_ROOT = _bootstrap_paths()

from toy_box_cylinder import make_box_cylinder, LABEL_CYL, LABEL_BOX, fold_report
from manifold_gp.utils.nearest_neighbors import NearestNeighbors
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

import laplacian_test as LT

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# Build the `built` dict (analogous to build_graph_and_laplacian's output)
# ---------------------------------------------------------------------------

def build_toy_built(args, device):
    """Generate the box+cylinder toy and build the graph Laplacian.

    Returns:
        built      dict  mirroring build_graph_and_laplacian's return value
        data       dict  from make_box_cylinder
        X_raw      (N,3) raw 3D coordinates (before normalization)
        bw         float graph bandwidth
    """
    data = make_box_cylinder(
        n=args.n,
        box_half=args.box_half,
        cyl_radius=args.cyl_radius,
        cyl_half_height=args.cyl_half_height,
        mode=args.mode,
        signal_kind=args.signal,
        n_azimuthal_cycles=args.n_azimuthal_cycles,
        seed=args.seed,
    )
    X_raw  = data["X"].astype(np.float64)
    labels = data["labels"]

    # per-axis standardization (matches maldi internal normalization)
    cm = X_raw.mean(0)
    cs = X_raw.std(0); cs[cs < 1e-6] = 1e-6
    Xs = ((X_raw - cm) / cs).astype(np.float32)

    coords = torch.as_tensor(Xs, device=device).contiguous()
    knn    = NearestNeighbors(coords)
    edge_index, edge_value = knn.graph(args.knn_k)

    bw = (args.graphbandwidth if args.graphbandwidth > 0
          else float(np.sqrt(np.median(edge_value.detach().cpu().numpy()))))

    if args.inflation and args.inflation != 1.0:
        edge_index, edge_value, _ = inflate_cross_region_edges(
            edge_index, edge_value, labels,
            inflation=args.inflation, treat_zero_as_cross=False)

    lap = GraphLaplacianOperator(
        edge_value, edge_index, coords.shape[0],
        torch.tensor(bw, device=device), args.norm, True)

    built = dict(
        laplacian_op=lap,
        graph_key="box_cylinder",
        n_nodes=int(coords.shape[0]),
        n_edges=int(edge_index.shape[1]),
        knn=knn,
        edge_index=edge_index,
        edge_value=edge_value,
        reference_nodes=coords,
        sub_volume=np.zeros((2, 2, 2), np.int32),   # empty -> jitter fallback
        voxel_offset=(0, 0, 0),
        voxel_scale_mm=1.0,
        coord_mean=torch.tensor(cm, dtype=torch.float32),
        coord_std=torch.tensor(cs, dtype=torch.float32),
    )
    return built, data, X_raw, bw


def write_toy_parquet(path, X_raw, signal):
    import pandas as pd
    df = pd.DataFrame({
        "xccf": X_raw[:, 0], "yccf": X_raw[:, 1], "zccf": X_raw[:, 2],
        "sig":     signal.astype(np.float64),
        "sig_alt": np.cos(np.arcsin(np.clip(signal, -1, 1))).astype(np.float64),
    })
    df.to_parquet(path)
    return ["sig", "sig_alt"]


def graph_geodesic_from(edge_index, edge_value, n, anchors):
    ei = edge_index.detach().cpu().numpy()
    ev = np.sqrt(np.clip(edge_value.detach().cpu().numpy(), 0, None))
    W  = sp.csr_matrix((ev, (ei[0], ei[1])), shape=(n, n))
    W  = W.maximum(W.T)
    return csg.dijkstra(W, directed=False, indices=anchors)   # (A, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Geometry
    ap.add_argument("--n",               type=int,   default=4_000)
    ap.add_argument("--box-half",        type=float, default=4.0)
    ap.add_argument("--cyl-radius",      type=float, default=3.5,
                    help="cylinder radius; radial gap = box_half - cyl_radius")
    ap.add_argument("--cyl-half-height", type=float, default=3.0)
    ap.add_argument("--mode",            choices=["surface", "volume"],
                    default="surface")
    ap.add_argument("--signal",          choices=["geodesic", "ambient"],
                    default="geodesic",
                    help="geodesic: antiphase surface signal (manifold-favourable); "
                         "ambient: smooth 3D (Euclidean-favourable)")
    ap.add_argument("--n-azimuthal-cycles", type=int, default=2)
    # Graph / Laplacian
    ap.add_argument("--knn-k",           type=int,   default=15)
    ap.add_argument("--num-modes",       type=int,   default=200)
    ap.add_argument("--nu",              type=int,   default=2)
    ap.add_argument("--lengthscale",     type=float, default=1.0)
    ap.add_argument("--inflation",       type=float, default=50.0)
    ap.add_argument("--norm",            choices=["symmetric", "randomwalk"],
                    default="randomwalk")
    ap.add_argument("--graphbandwidth",  type=float, default=0.0)
    ap.add_argument("--bump-scale",      type=float, default=0.1)
    ap.add_argument("--bump-decay",      type=float, default=0.01)
    # Diagnostics
    ap.add_argument("--n-test-on",       type=int,   default=200)
    ap.add_argument("--n-test-off",      type=int,   default=200)
    ap.add_argument("--eig-resid-modes", type=int,   default=64)
    ap.add_argument("--geodesic-anchors", type=int,  default=200)
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",             default=None,
                    help="optional CSV path for one-row summary")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    built, data, X_raw, bw = build_toy_built(args, device)
    labels = data["labels"]

    # ---- fold quality report -------------------------------------------------
    print(f"\nFold report (n_src=100, k={args.knn_k}):")
    fr = fold_report(data, n_src=100, k=args.knn_k)
    print(f"  intra-surface signal jump (median):  {fr['median_intra_surface_signal_jump']:.3f}")
    if "median_cross_surface_dist" in fr:
        print(f"  cross-surface distance:  "
              f"min={fr['min_cross_surface_dist']:.3f}  "
              f"median={fr['median_cross_surface_dist']:.3f}")
        print(f"  cross-surface signal jump:  "
              f"median={fr['median_cross_surface_signal_jump']:.3f}  "
              f"p90={fr['p90_cross_surface_signal_jump']:.3f}")

    # ---- compute eigenbasis --------------------------------------------------
    analyze_kw = dict(nu=args.nu, lengthscale=args.lengthscale,
                      tol_zero=1e-10, tol_neg=1e-6)

    solver = LaplacianEigensolver(
        num_modes=args.num_modes,
        backend="cupy" if device.type == "cuda" else "scipy",
        ncv_min=max(1500, 3 * args.num_modes + 20), verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        eigval, eigvec = solver.compute_or_load(
            built["laplacian_op"], cache_dir=tmp, key="boxcyl",
            graphbandwidth=bw, laplacian_normalization=args.norm,
            force_recompute=True, device=device)

    row = {
        "knn_method": f"box_cyl(infl={args.inflation:g},mode={args.mode})",
        "normalization": args.norm, "graphbandwidth": bw,
        "knn_k": args.knn_k, "num_modes": args.num_modes,
        "nu": args.nu, "lengthscale": args.lengthscale,
        "signal": args.signal, "mode": args.mode,
    }

    # ---- run the laplacian_test suite ----------------------------------------
    ev_np = eigval.detach().cpu().numpy()
    row.update(LT.analyze_eigvals(ev_np, **analyze_kw))
    row["fp_n_nodes"], row["fp_n_edges"] = built["n_nodes"], built["n_edges"]

    rng = np.random.default_rng(42)
    test_pts, n_on = LT.sample_test_points(
        reference_nodes=built["reference_nodes"],
        sub_volume=built["sub_volume"],
        voxel_offset=built["voxel_offset"],
        voxel_scale_mm=built["voxel_scale_mm"],
        coord_mean=built["coord_mean"],
        coord_std=built["coord_std"],
        threshold=0, n_on=args.n_test_on, n_off=args.n_test_off, rng=rng)

    kernel = RiemannMaternKernel(
        nu=args.nu, lengthscale=args.lengthscale,
        knn=built["knn"], edge_index=built["edge_index"],
        edge_value=built["edge_value"], eigval=eigval, eigvec=eigvec,
        nearest_neighbors=args.knn_k, laplacian_normalization=args.norm,
        num_modes=args.num_modes, bump_scale=args.bump_scale,
        bump_decay=args.bump_decay, graphbandwidth_init=bw).to(device)

    # on-graph self-distance check
    with torch.no_grad():
        self_d, _ = built["knn"].search(built["reference_nodes"], 1)
        frac_on = float((self_d[:, 0] < 1e-8).double().mean())
    print(f"\n  [diag] node self-dist^2 max={float(self_d[:,0].max()):.2e}  "
          f"frac<1e-8(on-graph)={frac_on:.1%}"
          + ("  <-- nodes MISROUTED to OOS path!" if frac_on < 0.999 else ""))

    row.update(LT.evaluate_kernel_psd(
        kernel, test_pts, graphbandwidth=bw,
        bump_scale=args.bump_scale, bump_decay=args.bump_decay,
        n_on_manifold=n_on, analyze_kwargs=analyze_kw))
    row.update(LT.eigen_health_diagnostics(
        built["laplacian_op"], eigval, eigvec, built["edge_index"], args.norm,
        n_res_modes=args.eig_resid_modes))
    row.update(LT.bandwidth_oos_diagnostics(
        eigval, graphbandwidth=bw, nu=args.nu, lengthscale=args.lengthscale))
    row.update(LT.geodesic_distance_diagnostics(
        built["edge_index"], built["edge_value"], built["reference_nodes"],
        eigval, eigvec, nu=args.nu, lengthscale=args.lengthscale,
        n_anchors=args.geodesic_anchors, seed=0, plot_dir=None,
        plot_tag="box_cyl"))

    with tempfile.TemporaryDirectory() as tmp:
        pq = os.path.join(tmp, "boxcyl.parquet")
        lip_names = write_toy_parquet(pq, X_raw, data["signal"])
        row.update(LT.maldi_data_diagnostics(
            pq, lip_names, False,
            knn=built["knn"], reference_nodes=built["reference_nodes"],
            eigval=eigval, eigvec=eigvec, nu=args.nu, lengthscale=args.lengthscale,
            graphbandwidth=bw, bump_scale=args.bump_scale,
            fold_filter=None, max_rows=args.n, plot_dir=None, plot_tag="box_cyl"))

    # ---- TOY-TRUTH CHECKS ----------------------------------------------------

    # Graph geodesic vs Euclidean and vs true surface distance
    # Use a sample of anchor points, preferring cylinder points
    rng2 = np.random.default_rng(0)
    n_anchors = min(64, built["n_nodes"])
    if args.mode == "surface":
        # sample anchors from the cylinder (the "inner layer")
        cyl_idx = np.where(data["labels"] == LABEL_CYL)[0]
        A = rng2.choice(cyl_idx, size=min(n_anchors, len(cyl_idx)), replace=False)
    else:
        A = rng2.choice(built["n_nodes"], size=n_anchors, replace=False)

    g    = graph_geodesic_from(built["edge_index"], built["edge_value"],
                               built["n_nodes"], A)
    intr = data["intrinsic"].astype(np.float64)
    euc  = np.linalg.norm(X_raw[A][:, None, :] - X_raw[None, :, :], axis=2)
    surf_dist = np.linalg.norm(intr[A][:, None, :] - intr[None, :, :], axis=2)

    m = np.isfinite(g) & (g > 0)
    geo_vs_surf = float(spearmanr(g[m], surf_dist[m]).correlation)
    euc_vs_surf = float(spearmanr(euc[m], surf_dist[m]).correlation)

    def get(k): return row.get(k, float("nan"))

    print(f"\n=== laplacian_test on box+cylinder toy "
          f"(signal={args.signal}, mode={args.mode}, "
          f"inflation={args.inflation:g}, modes={args.num_modes}, "
          f"norm={args.norm}) ===")

    print("  RAW DIAGNOSTICS (selected):")
    for k in ["lambda_min", "lambda_max", "ratio_min_over_max", "spectral_gap",
              "n_below_matern_floor",
              "diag_n_components", "diag_n_zero_modes", "diag_ortho_l2_offmax",
              "diag_varproxy_ratio", "diag_eig_resid_median", "diag_weyl_dim",
              "diag_oos_bw2lam_max", "diag_oos_denom_crossings",
              "diag_oos_first_crossing_mode", "diag_smat_participation_ratio",
              "diag_geo_reachable_frac", "diag_geo_euc_spearman",
              "diag_geo_euc_ratio_p95", "diag_dspec_geo_spearman_full",
              "diag_dspec_geo_spearman_low", "diag_dspec_euc_spearman_full",
              "diag_maldi_n_points", "diag_maldi_frac_in_support",
              "diag_maldi_e90_mode", "diag_maldi_prior_e90_mode",
              "diag_maldi_match_spearman_manifold",
              "diag_maldi_match_spearman_euclidean"]:
        v = get(k)
        print(f"    {k:<40s} {v:>12.4g}" if isinstance(v, (int, float))
              else f"    {k:<40s} {v}")

    print("\n  TOY-TRUTH CHECKS:")

    def check(name, cond, detail):
        print(f"    [{'PASS' if cond else 'CHECK'}] {name}: {detail}")

    # 1. Graph geodesic better than Euclidean at tracking surface structure
    check(
        "graph geodesic encodes surface structure better than Euclidean",
        geo_vs_surf > euc_vs_surf + 0.05,
        f"graph_geo~surf={geo_vs_surf:.3f}  euc~surf={euc_vs_surf:.3f}  "
        f"(gap should be larger with inflation; if small, inflation too weak "
        f"or graph disconnected -- Dijkstra returns inf)")

    # 2. Cross-surface Euclidean distance is small but graph geodesic is large
    if args.mode == "surface" and "median_cross_surface_dist" in fr:
        cyl_idxs = np.where(data["labels"] == LABEL_CYL)[0]
        box_idxs = np.where(data["labels"] == LABEL_BOX)[0]
        # sample a few cross-surface pairs from the cylinder
        sample_c = rng2.choice(cyl_idxs, size=min(20, len(cyl_idxs)), replace=False)
        sample_b = rng2.choice(box_idxs, size=min(20, len(box_idxs)), replace=False)
        cross_euc = np.linalg.norm(
            X_raw[sample_c][:, None, :] - X_raw[None, sample_b, :], axis=2).min()
        cross_g   = graph_geodesic_from(
            built["edge_index"], built["edge_value"],
            built["n_nodes"], sample_c[:10])
        cross_g_min = np.nanmin(cross_g[:, sample_b])
        ratio = cross_g_min / (cross_euc + 1e-9)
        check(
            "cross-surface: geodesic >> Euclidean (the fold)",
            ratio > 2.0,
            f"min cross-surface Euclidean={cross_euc:.3f}  "
            f"min cross-surface graph_geo={cross_g_min:.3f}  "
            f"ratio={ratio:.1f}  (large ratio = strong fold; >2 is good)")

    # 3. Signal match: manifold should beat Euclidean for geodesic signal in surface mode
    man = get("diag_maldi_match_spearman_manifold")
    euc_m = get("diag_maldi_match_spearman_euclidean")
    if args.signal == "geodesic" and args.mode == "surface":
        check(
            "manifold match beats Euclidean (geodesic signal, surface mode)",
            man > euc_m,
            f"man={man:.3f}  euc={euc_m:.3f}  (manifold should win; "
            f"if not, try larger --inflation or more --num-modes)")
    elif args.signal == "ambient" or args.mode == "volume":
        check(
            "Euclidean match beats manifold (ambient/volume mode)",
            euc_m >= man,
            f"euc={euc_m:.3f}  man={man:.3f}  (Euclidean should win here)")

    # 4. Graph connectivity
    check(
        "graph connected (single component)",
        get("diag_n_components") == 1,
        f"diag_n_components={get('diag_n_components')}  "
        f"(if > 1: gap is too large for kNN to bridge -- "
        f"use larger --knn-k or smaller --cyl-radius)")

    # 5. OOS stability
    check(
        "OOS denominator safe (bw^2*lambda_max < 0.8)",
        get("diag_oos_bw2lam_max") < 0.8,
        f"diag_oos_bw2lam_max={get('diag_oos_bw2lam_max'):.3f}  "
        f"crossings={get('diag_oos_denom_crossings')}")

    print(f"\n  Graph geodesic vs intrinsic distance:")
    print(f"    graph_geodesic ~ surface_distance  Spearman={geo_vs_surf:.3f}")
    print(f"    euclidean      ~ surface_distance  Spearman={euc_vs_surf:.3f}")
    print(f"    (higher graph_geo = better graph encodes surface geometry)")
    print(f"\n  NOTE: if diag_n_components > 1, the KNN graph cannot bridge the "
          f"{args.box_half - args.cyl_radius:.2f}-unit radial gap.  Dijkstra "
          f"returns inf, fold checks use only connected pairs.  With inflation, "
          f"cross-surface AFFINITY is near-zero even if KNN edges are present.")
    print(f"\n  Sweep --inflation 1 -> 50 -> 500 on the surface+geodesic case to "
          f"watch: manifold_match improves, geo_vs_surf improves, and the "
          f"surface mode should clearly outperform the volume mode for manifold GP.")

    if args.out:
        row["toy_geo_vs_surf_spearman"] = geo_vs_surf
        row["toy_euc_vs_surf_spearman"] = euc_vs_surf
        LT.write_csv(Path(args.out), [row])
        print(f"\n  wrote row -> {args.out}")


if __name__ == "__main__":
    main()
