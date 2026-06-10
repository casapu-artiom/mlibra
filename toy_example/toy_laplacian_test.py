#!/usr/bin/env python
# encoding: utf-8
"""
toy_laplacian_validate.py
=========================
Validate the ENTIRE laplacian_test diagnostic suite on the folded-manifold toy.

Why the toy: its intrinsic coords ARE the true geodesic, the fold is known, and
the signal is known -- so every diagnostic has a value we can CHECK, not just run.

Strategy: reproduce build_graph_and_laplacian's `built` dict from the toy point
cloud, then call laplacian_test's diagnostic functions UNCHANGED (so the suite is
validated as-is). Two front-end dependencies are handled without forking:
  * sample_test_points  -> pass an empty sub_volume; it uses its jitter fallback
                           for off-manifold points (no atlas volume needed).
  * maldi_data_diagnostics -> write a temp parquet (xccf,yccf,zccf + signal as
                           'lipid' columns) with RAW coords; the function recomputes
                           the same per-axis normalization, so its node-snap matches
                           our graph nodes exactly. (We therefore build the graph on
                           PER-AXIS standardized coords for frame consistency.)

Then a TOY-TRUTH CHECK layer compares the headline diagnostics against the known
ground truth (graph-geodesic vs true geodesic, geo-vs-euclidean contrast, Q4
manifold-vs-euclidean match, OOS bw^2*lambda).

Run on the GPU box, FROM the directory that holds laplacian_test.py, utils.py and
toy_manifold.py (so `import laplacian_test` resolves its `from utils import ...`):
    # manifold-favourable (tight fold, geodesic signal): manifold checks should pass
    python toy_laplacian_test.py --signal geodesic --gap 0.3 --thickness 0 --inflation 50
    # euclidean-favourable (ambient signal): the Q4 check expects euclidean to win
    python toy_laplacian_test.py --signal ambient_grid --gap 1.5 --thickness 0 --inflation 50
    # plain vs atlas contrast
    python toy_laplacian_test.py --inflation 1  --num-modes 300 --gap 0.3
"""
from __future__ import annotations
import argparse, inspect, os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

# --- make the three sibling folders importable regardless of CWD -----------
# Layout:  project/
#            maldi/        laplacian_test.py, utils.py
#            manifold_gp/  (the package)
#            toy_example/  this file, toy_manifold.py
# laplacian_test.py itself does `from utils import ...`, so maldi/ must be on the
# path too -- and ahead of toy_example/ in case both have a utils.py.
def _bootstrap_paths():
    here = Path(__file__).resolve().parent                      # .../toy_example
    root = next((p for p in [here, *here.parents]
                 if (p / "manifold_gp").is_dir() and (p / "maldi").is_dir()),
                here.parent)                                    # project/
    # env override if the auto-detect is wrong: TOY_PROJECT_ROOT=/path/to/project
    root = Path(os.environ.get("TOY_PROJECT_ROOT", root)).resolve()
    # insert order => priority: maldi (utils, laplacian_test) > toy_example
    # (toy_manifold) > root (manifold_gp package).
    for d in (root, here, root / "maldi"):
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    return root
_PROJECT_ROOT = _bootstrap_paths()

from toy_manifold import make_folded_manifold
from manifold_gp.utils.nearest_neighbors import NearestNeighbors
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

import laplacian_test as LT          # the suite under validation

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
def build_toy_built(args, device):
    """Reproduce build_graph_and_laplacian's return dict from the toy."""
    # signature-tolerant call: pass signal_kind/n_turns/r0/height/wavelength_gaps
    # only if this make_folded_manifold supports them.
    gen_kw = dict(n=args.n, n_turns=args.n_turns, gap=args.gap, r0=args.r0,
                  height=args.height, cycles_per_turn=args.cycles_per_turn,
                  thickness=args.thickness, seed=args.seed,
                  signal_kind=args.signal, wavelength_gaps=args.wavelength_gaps)
    params = inspect.signature(make_folded_manifold).parameters
    if "signal_kind" not in params and args.signal != "geodesic":
        raise SystemExit("This make_folded_manifold has no signal_kind; only the "
                         "geodesic signal is available.")
    d = make_folded_manifold(**{k: v for k, v in gen_kw.items() if k in params})
    X = d["X"].astype(np.float64)                       # raw "mm-like" coords
    cm, cs = X.mean(0), X.std(0)
    cs[cs < 1e-6] = 1e-6
    Xs = ((X - cm) / cs).astype(np.float32)             # per-axis (matches maldi internal)
    labels = np.floor(d["t"] / (2 * np.pi)).astype(np.int64)

    coords = torch.as_tensor(Xs, device=device).contiguous()
    knn = NearestNeighbors(coords)
    edge_index, edge_value = knn.graph(args.knn_k)
    bw = args.graphbandwidth if args.graphbandwidth > 0 \
        else float(np.sqrt(np.median(edge_value.detach().cpu().numpy())))
    if args.inflation and args.inflation != 1.0:
        edge_index, edge_value, _ = inflate_cross_region_edges(
            edge_index, edge_value, labels,
            inflation=args.inflation, treat_zero_as_cross=False)

    lap = GraphLaplacianOperator(
        edge_value, edge_index, coords.shape[0],
        torch.tensor(bw, device=device), args.norm, True)

    built = dict(
        laplacian_op=lap, graph_key="toy",
        n_nodes=int(coords.shape[0]), n_edges=int(edge_index.shape[1]),
        knn=knn, edge_index=edge_index, edge_value=edge_value,
        reference_nodes=coords,
        sub_volume=np.zeros((2, 2, 2), np.int32),       # empty -> jitter fallback
        voxel_offset=(0, 0, 0), voxel_scale_mm=1.0,
        coord_mean=torch.tensor(cm, dtype=torch.float32),
        coord_std=torch.tensor(cs, dtype=torch.float32),
    )
    return built, d, X, bw


def write_toy_parquet(path, X_raw, signal):
    import pandas as pd
    df = pd.DataFrame({
        "xccf": X_raw[:, 0], "yccf": X_raw[:, 1], "zccf": X_raw[:, 2],
        "sig":     signal.astype(np.float64),
        "sig_alt": np.cos(np.arcsin(np.clip(signal, -1, 1))).astype(np.float64),  # 2nd channel
    })
    df.to_parquet(path)
    return ["sig", "sig_alt"]


def graph_geodesic_from(edge_index, edge_value, n, anchors):
    ei = edge_index.detach().cpu().numpy()
    ev = np.sqrt(np.clip(edge_value.detach().cpu().numpy(), 0, None))
    W = sp.csr_matrix((ev, (ei[0], ei[1])), shape=(n, n))
    W = W.maximum(W.T)
    return csg.dijkstra(W, directed=False, indices=anchors)   # (A, n)


def reference_distance_correlation(edge_index, edge_value, X_raw, n_ref, rng):
    """Inducing-point view of the metric: pick `n_ref` reference (inducing)
    nodes, compute GRAPH-GEODESIC and EUCLIDEAN distance from every node to each
    reference, and report how correlated those two distance fields are. This is
    exactly the cross-distance an inducing-point GP sees in K(x, z): if geodesic
    and euclidean agree (high Spearman), a manifold kernel gains little over a
    euclidean one at these inducing points; if they decorrelate (low Spearman),
    the folds matter and the manifold kernel is doing real work.

    Reports both the pooled correlation over all point->reference pairs and the
    per-reference correlation (each inducing point's own distance field, then
    summarized) -- the per-reference number avoids mixing different references'
    distance scales and is the more faithful inducing-point read."""
    n = X_raw.shape[0]
    ref = rng.choice(n, size=min(int(n_ref), n), replace=False)
    g = graph_geodesic_from(edge_index, edge_value, n, ref)                  # (R, N)
    euc = np.linalg.norm(X_raw[ref][:, None, :] - X_raw[None, :, :], axis=2)  # (R, N)
    finite = np.isfinite(g) & (g > 0)
    gm, em = g[finite], euc[finite]
    per = []
    for r in range(len(ref)):
        msk = finite[r]
        if msk.sum() >= 10:
            per.append(float(spearmanr(g[r, msk], euc[r, msk]).correlation))
    per = np.asarray(per, float)
    return dict(
        n_ref=int(len(ref)),
        reachable_frac=float(finite.mean()),
        spearman_global=float(spearmanr(gm, em).correlation),
        pearson_global=float(np.corrcoef(gm, em)[0, 1]),
        spearman_per_ref_median=float(np.median(per)) if per.size else float("nan"),
        spearman_per_ref_p10=float(np.percentile(per, 10)) if per.size else float("nan"),
        spearman_per_ref_p90=float(np.percentile(per, 90)) if per.size else float("nan"),
    )


def variogram_metrics(dist, dissim_abs, n_bins=24):
    """Empirical (semi)variogram of a signal against ONE distance metric, over
    flat pair arrays. Pairs are split into n_bins equal-count (quantile) bins of
    `dist`; per bin we form the semivariance gamma = 0.5*mean(dissim^2). Returns:
      r2          fraction of squared-dissimilarity variance explained by binned
                  distance (higher => distance organizes the signal better)
      nugget_sill nugget/sill = gamma in the nearest bin / plateau (mean of the
                  top-quartile bins). Low => the metric explains the near field;
                  high => a signal jump at distance~0 the metric is blind to (the
                  fold: euclidean-near but signal-far pairs pile up at small h)
      spearman    monotonicity of |dissim| vs distance"""
    d = np.asarray(dist, float).ravel()
    a = np.asarray(dissim_abs, float).ravel()
    y = a ** 2                                              # squared diffs
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
            gammas.append(0.5 * g_k)                        # ascending-distance order
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    gammas = np.asarray(gammas, float)
    nugget = float(gammas[0])
    k = max(1, len(gammas) // 4)
    sill = float(np.mean(np.sort(gammas)[-k:]))             # plateau ~ top quartile
    return dict(
        r2=float(1.0 - ss_res / max(ss_tot, 1e-30)),
        nugget_sill=float(nugget / max(sill, 1e-30)),
        spearman=float(spearmanr(d, a).correlation),
    )


def variogram_compare(dists, dissim_abs, mask, n_bins=24, local_quantile=0.15):
    """Compare candidate distance fields (name -> (A,N) array) by how cleanly the
    signal's variogram decays along each, over the masked pairs. For each metric
    we report the variogram over ALL pairs (global) AND over only that metric's
    nearest `local_quantile` fraction of pairs (local). The local read matters
    for oscillatory/antiphase signals: globally the semivariance is non-monotone
    (so R2 is tiny for every metric), but within the near-field the signal IS
    monotone in dissimilarity, so the fold pollution -- euclidean-near but
    signal-far cross-region pairs -- shows up sharply as a high euclidean local
    nugget. Returns name -> {r2, nugget_sill, spearman, *_local, local_cutoff};
    the metric with the higher (esp. local) r2 and lower nugget_sill is the one
    the signal lives on -> the kernel should use it."""
    y_all = dissim_abs[mask]
    nan3 = dict(r2=float("nan"), nugget_sill=float("nan"), spearman=float("nan"))
    out = {}
    for name, dm in dists.items():
        x_all = dm[mask]
        g = variogram_metrics(x_all, y_all, n_bins)
        cutoff = float(np.quantile(x_all, local_quantile))   # this metric's near-field
        sel = x_all <= cutoff
        loc = variogram_metrics(x_all[sel], y_all[sel], n_bins) if sel.sum() >= 3 else nan3
        out[name] = dict(
            r2=g["r2"], nugget_sill=g["nugget_sill"], spearman=g["spearman"],
            r2_local=loc["r2"], nugget_sill_local=loc["nugget_sill"],
            spearman_local=loc["spearman"], local_cutoff=cutoff,
        )
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-turns", type=float, default=3.5)
    ap.add_argument("--gap", type=float, default=10.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=20.0)
    ap.add_argument("--cycles-per-turn", type=float, default=1.5)
    ap.add_argument("--thickness", type=float, default=1.5)
    ap.add_argument("--signal",
                    choices=["geodesic", "ambient_x", "ambient_grid", "radial", "euclidean"],
                    default="geodesic",
                    help="geodesic -> fold-respecting signal (manifold-favourable); "
                         "ambient_*/radial -> 3D-smooth (euclidean-favourable). Flips "
                         "the expected winner of the Q4 match check.")
    ap.add_argument("--wavelength-gaps", type=float, default=3.0,
                    help="ambient-signal wavelength in units of --gap")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--num-modes", type=int, default=300)
    ap.add_argument("--nu", type=int, default=2)
    ap.add_argument("--lengthscale", type=float, default=1.0)
    ap.add_argument("--inflation", type=float, default=50.0)
    ap.add_argument("--norm", choices=["symmetric", "randomwalk"], default="randomwalk")
    ap.add_argument("--graphbandwidth", type=float, default=0.0)
    ap.add_argument("--bump-scale", type=float, default=0.1)
    ap.add_argument("--bump-decay", type=float, default=0.01)
    ap.add_argument("--n-test-on", type=int, default=1000)
    ap.add_argument("--n-test-off", type=int, default=1000)
    ap.add_argument("--eig-resid-modes", type=int, default=64)
    ap.add_argument("--geodesic-anchors", type=int, default=300)
    ap.add_argument("--n-reference", type=int, default=1000,
                    help="inducing-point set size for the geodesic-vs-euclidean "
                         "reference distance correlation")
    ap.add_argument("--vario-local-quantile", type=float, default=0.15,
                    help="local variogram uses each metric's nearest fraction of "
                         "pairs (sharper read for oscillatory signals)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="optional CSV path (one row)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    built, d, X_raw, bw = build_toy_built(args, device)

    # ---- inducing-point distance correlation (geodesic vs euclidean) -------
    # Needs only the graph, not the eigenbasis. This is the metric an
    # inducing-point GP actually sees in K(x, z) for inducing points z.
    rdc = reference_distance_correlation(
        built["edge_index"], built["edge_value"], X_raw,
        n_ref=args.n_reference, rng=np.random.default_rng(args.seed))
    print(f"\nReference (inducing-point) distance correlation  "
          f"(n_ref={rdc['n_ref']}, reachable={rdc['reachable_frac']:.1%}):")
    print(f"  geodesic vs euclidean, pooled over all point->reference pairs:  "
          f"Spearman={rdc['spearman_global']:.3f}  Pearson={rdc['pearson_global']:.3f}")
    print(f"  per-reference Spearman:  median={rdc['spearman_per_ref_median']:.3f}  "
          f"[p10={rdc['spearman_per_ref_p10']:.3f}, p90={rdc['spearman_per_ref_p90']:.3f}]")
    print(f"  (low => geodesic decorrelates from euclidean => the manifold kernel "
          f"is doing real work at these inducing points; high => little to gain)")

    analyze_kwargs = dict(nu=args.nu, lengthscale=args.lengthscale,
                          tol_zero=1e-10, tol_neg=1e-6)

    solver = LaplacianEigensolver(
        num_modes=args.num_modes,
        backend="cupy" if device.type == "cuda" else "scipy",
        ncv_min=max(1500, 3 * args.num_modes + 20), verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        eigval, eigvec = solver.compute_or_load(
            built["laplacian_op"], cache_dir=tmp, key="toy",
            graphbandwidth=bw, laplacian_normalization=args.norm,
            force_recompute=True, device=device)

    row = {"knn_method": f"toy(infl={args.inflation:g})", "normalization": args.norm,
           "graphbandwidth": bw, "knn_k": args.knn_k, "num_modes": args.num_modes,
           "nu": args.nu, "lengthscale": args.lengthscale, "signal": args.signal}

    # ---- exactly the laplacian_test call sequence, on toy `built` ----------
    ev_np = eigval.detach().cpu().numpy()
    row.update(LT.analyze_eigvals(ev_np, **analyze_kwargs))
    row["fp_n_nodes"], row["fp_n_edges"] = built["n_nodes"], built["n_edges"]

    rng = np.random.default_rng(42)
    test_pts, n_on = LT.sample_test_points(
        reference_nodes=built["reference_nodes"], sub_volume=built["sub_volume"],
        voxel_offset=built["voxel_offset"], voxel_scale_mm=built["voxel_scale_mm"],
        coord_mean=built["coord_mean"], coord_std=built["coord_std"],
        threshold=0, n_on=args.n_test_on, n_off=args.n_test_off, rng=rng)
    kernel = RiemannMaternKernel(
        nu=args.nu, lengthscale=args.lengthscale, knn=built["knn"],
        edge_index=built["edge_index"], edge_value=built["edge_value"],
        eigval=eigval, eigvec=eigvec, nearest_neighbors=args.knn_k,
        laplacian_normalization=args.norm, num_modes=args.num_modes,
        bump_scale=args.bump_scale, bump_decay=args.bump_decay,
        graphbandwidth_init=bw).to(device)

    # on-graph misrouting check: nodes should be detected on-graph (faiss self-dist^2
    # < 1e-8); float32 faiss self-distances ~1e-6 can misroute them to the OOS path,
    # whose (1 - bw^2*lambda)^2 denominator blows up as bw^2*lambda_max -> 1.
    with torch.no_grad():
        self_d, _ = built["knn"].search(built["reference_nodes"], 1)
        frac_on = float((self_d[:, 0] < 1e-8).double().mean())
    print(f"  [diag] node self-dist^2 max={float(self_d[:, 0].max()):.2e}  "
          f"frac<1e-8(on-graph)={frac_on:.1%}"
          + ("  <-- nodes MISROUTED to OOS path (loosen the 1e-8 tol in "
             "riemann_kernel.features)" if frac_on < 0.999 else ""))

    row.update(LT.evaluate_kernel_psd(
        kernel, test_pts, graphbandwidth=bw,
        bump_scale=args.bump_scale, bump_decay=args.bump_decay,
        n_on_manifold=n_on, analyze_kwargs=analyze_kwargs))

    row.update(LT.eigen_health_diagnostics(
        built["laplacian_op"], eigval, eigvec, built["edge_index"], args.norm,
        n_res_modes=args.eig_resid_modes))
    row.update(LT.bandwidth_oos_diagnostics(
        eigval, graphbandwidth=bw, nu=args.nu, lengthscale=args.lengthscale))
    row.update(LT.geodesic_distance_diagnostics(
        built["edge_index"], built["edge_value"], built["reference_nodes"],
        eigval, eigvec, nu=args.nu, lengthscale=args.lengthscale,
        n_anchors=args.geodesic_anchors, seed=0, plot_dir=None, plot_tag="toy"))

    with tempfile.TemporaryDirectory() as tmp:
        pq = os.path.join(tmp, "toy.parquet")
        lip_names = write_toy_parquet(pq, X_raw, d["signal"])
        row.update(LT.maldi_data_diagnostics(
            pq, lip_names, False,
            knn=built["knn"], reference_nodes=built["reference_nodes"],
            eigval=eigval, eigvec=eigvec, nu=args.nu, lengthscale=args.lengthscale,
            graphbandwidth=bw, bump_scale=args.bump_scale,
            fold_filter=None, max_rows=args.n, plot_dir=None, plot_tag="toy"))

    # ---- TOY-TRUTH CHECKS --------------------------------------------------
    # independent graph-geodesic vs the toy's TRUE geodesic (intrinsic coords)
    A = np.random.default_rng(0).choice(built["n_nodes"],
                                        size=min(64, built["n_nodes"]), replace=False)
    g = graph_geodesic_from(built["edge_index"], built["edge_value"], built["n_nodes"], A)
    intr = d["intrinsic"].astype(np.float64)
    true_geo = np.linalg.norm(intr[A][:, None, :] - intr[None, :, :], axis=2)
    euc = np.linalg.norm(X_raw[A][:, None, :] - X_raw[None, :, :], axis=2)
    m = np.isfinite(g) & (g > 0)
    geo_vs_true = float(spearmanr(g[m], true_geo[m]).correlation)
    euc_vs_true = float(spearmanr(euc[m], true_geo[m]).correlation)

    # signal variogram (covariance vs distance): which metric does the SIGNAL
    # actually decay along? reuses the same anchor pairs / mask as above.
    sig_node = d["signal"].astype(np.float64)
    dissim   = np.abs(sig_node[A][:, None] - sig_node[None, :])
    vg = variogram_compare(
        {"manifold_geodesic": g, "euclidean": euc, "true_geodesic": true_geo},
        dissim, m, local_quantile=args.vario_local_quantile)

    def get(k): return row.get(k, float("nan"))
    print(f"\n=== laplacian_test on toy  (signal={args.signal}, inflation={args.inflation:g}, "
          f"modes={args.num_modes}, gap={args.gap}, thickness={args.thickness}, "
          f"t/gap={args.thickness/args.gap:.2f}, norm={args.norm}) ===")
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
        print(f"    {k:<40s} {v:>12.4g}" if isinstance(v, (int, float)) else f"    {k:<40s} {v}")

    print(f"\n  SIGNAL VARIOGRAM (covariance vs distance; g=global over all pairs, "
          f"l=local nearest {args.vario_local_quantile:.0%}; higher R2 / lower "
          f"nugget:sill = the metric the signal decays along):")
    for name in ("manifold_geodesic", "euclidean", "true_geodesic"):
        vm = vg[name]
        print(f"    {name:<18s} R2 g/l={vm['r2']:>6.3f}/{vm['r2_local']:>6.3f}"
              f"  nug:sill g/l={vm['nugget_sill']:>5.3f}/{vm['nugget_sill_local']:>5.3f}"
              f"  rho g/l={vm['spearman']:>6.3f}/{vm['spearman_local']:>6.3f}")
    print("    (local is the sharper read for oscillatory/antiphase signals; a "
          "large EUCLIDEAN local nugget:sill is the fold at distance~0)")

    print("\n  TOY-TRUTH CHECKS (known ground truth):")
    def check(name, cond, detail):
        print(f"    [{'PASS' if cond else 'CHECK'}] {name}: {detail}")

    # signal variogram: which metric the signal decays along (model-free, no
    # kernel/eigenbasis). The LOCAL read is the discriminator -- globally an
    # oscillatory signal is non-monotone so every metric's R2 is tiny.
    vman, veuc = vg["manifold_geodesic"], vg["euclidean"]
    if args.signal in ("ambient_x", "ambient_grid", "radial", "euclidean"):
        check("signal variogram cleaner along EUCLIDEAN than manifold (local, ambient)",
              veuc["r2_local"] >= vman["r2_local"],
              f"local R2 euc={veuc['r2_local']:.3f} man={vman['r2_local']:.3f}  "
              f"(global euc={veuc['r2']:.3f} man={vman['r2']:.3f}; smooth 3D signal "
              f"-> euclidean organizes it)")
    else:
        check("signal variogram cleaner along MANIFOLD geodesic (local, geodesic signal)",
              vman["r2_local"] > veuc["r2_local"],
              f"local R2 man={vman['r2_local']:.3f} euc={veuc['r2_local']:.3f}  |  "
              f"local nug:sill man={vman['nugget_sill_local']:.3f} "
              f"euc={veuc['nugget_sill_local']:.3f}  (global R2 man={vman['r2']:.3f} "
              f"euc={veuc['r2']:.3f}; euclidean local nugget should be large -- the "
              f"fold injects a signal jump at euclidean distance ~0)")
    check("graph-geodesic encodes geometry euclidean can't",
          geo_vs_true > euc_vs_true + 0.15,
          f"graph_geo~true={geo_vs_true:.3f}  vs  euclidean~true={euc_vs_true:.3f}  "
          f"(gap should grow with inflation; ~0.8 is the kNN-graph ceiling. If this "
          f"is small, inflation is too weak -- sweep 50->500->1000)")
    check("fold makes euclidean != geodesic", get("diag_geo_euc_spearman") < 0.9,
          f"diag_geo_euc_spearman={get('diag_geo_euc_spearman'):.3f} "
          f"(lower => stronger fold)")
    check("graph connected (single component)", get("diag_n_components") == 1,
          f"diag_n_components={get('diag_n_components')}")
    man = get("diag_maldi_match_spearman_manifold")
    euc = get("diag_maldi_match_spearman_euclidean")
    if args.signal in ("ambient_x", "ambient_grid", "radial", "euclidean"):
        check("euclidean match beats manifold (ambient signal)", euc > man,
              f"euc={euc:.3f} man={man:.3f}  (ambient signal is single-valued/smooth in "
              f"3D -> euclidean-favourable; the atlas prior severs edges it wants)")
    else:
        check("manifold match beats euclidean (geodesic signal)", man > euc,
              f"man={man:.3f} euc={euc:.3f}  (geodesic signal is fold-respecting -> "
              f"manifold-favourable)")
    check("OOS denominator safe (bw^2*lambda_max < 0.8)",
          get("diag_oos_bw2lam_max") < 0.8,
          f"diag_oos_bw2lam_max={get('diag_oos_bw2lam_max'):.3f}, "
          f"crossings={get('diag_oos_denom_crossings')}")
    print(f"\n  NOTE: the geodesic diagnostic runs Dijkstra on edge LENGTHS "
          f"(x sqrt(inflation)),\n  but the Laplacian eigenbasis uses the AFFINITY "
          f"exp(-edge_value/4bw^2) (x inflation on\n  SQUARED distance => near-cut). So "
          f"the eigenbasis is more fold-aware than the\n  Dijkstra-geodesic number implies"
          f" -- dspec/match can pass even when geo_euc does not.\n"
          f"  expectation: inflation>1 should raise graph_geo~TRUE above euclidean and\n"
          f"  improve the Q4 manifold match; inflation=1 collapses both toward euclidean.\n"
          f"  Run inflation in (1, 50, 500, 1000) and watch the trend.")

    if args.out:
        row["toy_graph_geo_vs_true_spearman"] = geo_vs_true
        row["toy_euc_vs_true_spearman"] = euc_vs_true
        row["toy_vario_man_r2"] = vg["manifold_geodesic"]["r2"]
        row["toy_vario_euc_r2"] = vg["euclidean"]["r2"]
        row["toy_vario_man_r2_local"] = vg["manifold_geodesic"]["r2_local"]
        row["toy_vario_euc_r2_local"] = vg["euclidean"]["r2_local"]
        row["toy_vario_man_nugget_sill"] = vg["manifold_geodesic"]["nugget_sill"]
        row["toy_vario_euc_nugget_sill"] = vg["euclidean"]["nugget_sill"]
        row["toy_vario_man_nugget_sill_local"] = vg["manifold_geodesic"]["nugget_sill_local"]
        row["toy_vario_euc_nugget_sill_local"] = vg["euclidean"]["nugget_sill_local"]
        row["toy_refdist_geo_euc_spearman"] = rdc["spearman_global"]
        LT.write_csv(Path(args.out), [row])
        print(f"\n  wrote row -> {args.out}")


if __name__ == "__main__":
    main()