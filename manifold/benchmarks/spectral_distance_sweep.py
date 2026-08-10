#!/usr/bin/env python
# encoding: utf-8
"""
spectral_distance_sweep.py
==========================

Iterate over a grid of graph parameters and, for each, measure how well the
MALDI lipid **data covariance** decays along several distance metrics on the
graph nodes:

    euclidean      straight-line distance in the model's coordinate space
    geodesic       Dijkstra shortest path on the (optionally atlas-inflated) graph
    spectral       kernel-induced distance from the frozen Laplacian eigenpairs at
                     the FULL mode budget, K = Matern(L). Two forms (--spectral-norm):
                       correlation (default) d_spec=sqrt(2(1-cos)),
                                             cos = K_ij/sqrt(K_ii*K_jj). Strips the
                                             per-node kernel diagonal (node variance),
                                             so it measures separation, not node norm.
                                             Matches the corrected notebook d_spec.
                       rkhs                  the raw d_spec=sqrt(K_ii+K_jj-2K_ij); the
                                             diagonal contaminates it when node
                                             variance is heterogeneous.
    spectral_low<N>  the same spectral distance truncated to N modes. Pass a LIST
                     via --low-modes (e.g. 64 256 1000) to compare several
                     truncations side-by-side — tells you how many modes are
                     needed to capture the covariance structure. With no list, a
                     single `spectral_low` at --low-mode-frac of the budget is used.

The question this answers: "which graph parameters give a (spectral) distance
along which the lipid covariance is most cleanly organized?" — i.e. the best
spectral correlation parameters. It does this PER LIPID (and aggregates), so you
can see which lipids the manifold geometry actually helps.

FEATURE MODE (--feature-mode)
-----------------------------
  snap     (default) MALDI points are snapped + averaged onto graph nodes, and
           spectral features are the exact on-graph eigenvectors. bump_scale /
           bump_decay are inert here (endpoints are graph nodes).
  nystrom  endpoints are the REAL MALDI measurements at their TRUE coords; their
           spectral features come from the deployed Nystrom OOS map (bump-weighted
           eigenvectors) — the actual prediction-time path. Here bump_scale and
           bump_decay MATTER and are swept as axes (--bump-scales / --bump-decays).
           cov uses each point's own value; points beyond bump_scale*graphbandwidth
           get zero features (reported as off_support_frac) -> treated as
           decorrelated (d_spec->sqrt(2) under the correlation norm).

STRIDE / MODE BUDGET
--------------------
Resolution (stride) and the eigen-mode budget are coupled: a finer stride means
more nodes and supports more modes. So instead of a free num_modes axis, you
sweep (stride, max_modes) PAIRS via --stride-modes, e.g.

    --stride-modes 8:1300 4:6000 2:20000

means {stride 8 with 1300 modes, stride 4 with 6000 modes, stride 2 with 20000}.

WHAT IS COMPUTED, per (config, lipid, distance-metric)
------------------------------------------------------
Pairs of graph nodes (anchors x targets, restricted to nodes that carry signal)
are binned by distance. Per bin we form the empirical data covariance
C(h)=mean_{pairs in bin} s_i*s_j (s = the lipid's mean-centered node signal). The
binned decay Spearman is reported under BOTH axis normalizations (cf. the notebook
Q4b vs Q4c) because they can disagree sharply for a saturating metric like the
spectral distance — equal-WIDTH bins flatter a metric that piles all its pairs near
one end, while equal-COUNT bins are robust to that. We then report, per distance
metric:

    cov_dist_spearman          Spearman(distance, pairwise covariance), RAW per-pair
                               (1-sample, noisy). Negative = covariance decays.
    decay_strength             = -cov_dist_spearman  (raw rank metric)
    decay_strength_quantile    -Spearman over EQUAL-COUNT (quantile) bins  [Q4c;
                               robust, the honest ranking metric]
    decay_strength_uniform     -Spearman over EQUAL-WIDTH [min,max] bins (bins with
                               < --uniform-min-count pairs dropped)  [Q4b; flatters
                               saturating metrics]
    cov_curve_corr             Pearson corr of the binned (h_bar, C_bar) curve
    vario_r2                   R^2 of the binned semivariogram

OUTPUTS (resumable — see RESUME)
--------------------------------
  <out-dir>/rows/<config-slug>.csv   per-config rows (lipid x metric), written
                                     atomically as soon as the config finishes

This script ONLY computes per-config rows — it does NOT aggregate or rank. That
is a separate step so the sweep can fan out across many one-config jobs (each job
runs ONE config, writing its own rows/<slug>.csv to a shared dir; see
submit/run_spectral_distance_sweep.sh). Once the jobs finish, build the report
with:

  python manifold/benchmarks/spectral_sweep_report.py --out-dir <out-dir> \
      --objective spectral --rank-stat decay_strength_quantile

which globs rows/*.csv and writes per_lipid.csv / per_config.csv / ranked.csv.

RESUME
------
Each config writes rows/<slug>.csv on completion (tmp + atomic rename). Re-running
the SAME command skips any config whose rows file already exists, so a preempted
run picks up where it left off. --force recomputes. The KNN graph + eigenpairs are
also cached on disk (same keys as laplacian_test.py / the deployed kernel), so even
a recomputed config reuses the expensive eigensolve.

DATA SELECTION (no fold)
------------------------
There is NO train/test fold: the correlogram is computed over ALL measured MALDI
points, since covariance-vs-distance is a property of the graph geometry + lipid
fields, not of a held-out split. Whole samples can be dropped with
--exclude-samples (default: GBA1, Pregnant1, Pregnant2, Pregnant4) via a predicate
on --sample-column. Lipid columns come from --available-lipids-file, optionally
narrowed by --lipids / --lipids-file.

Run in the laplacian_test.py environment (torch, faiss, manifold_gp).

RUN
---
  python spectral_distance_sweep.py \
      --template-name reference \
      --reference-file  /path/reference_image.npy \
      --annotations-file /path/level_15annot.npy \
      --eigenvector-dir /path/eigenvectors \
      --maldi-file /path/maindata_minimal.parquet \
      --available-lipids-file /path/maindata_minimal_available_lipids.npy \
      --lipids-file maldi/data/lipid_subset.txt \
      --exclude-samples GBA1 Pregnant1 Pregnant2 Pregnant4 \
      --knn-methods faiss_atlas_weighted --normalizations randomwalk \
      --thresholds 5 --knn-ks 60 120 --bandwidths 0.05 0.1 0.2 \
      --inflations 10 50 --stride-modes 8:1300 4:6000 \
      --low-modes 64 256 1000 --out-dir ./spectral_sweep

  # off-node MALDI points through the bump, sweeping bump scale/decay:
  python spectral_distance_sweep.py ... \
      --feature-mode nystrom --bump-scales 1 2 3 --bump-decays 0.01 0.1

  # then aggregate + rank (separate step; see spectral_sweep_report.py):
  python spectral_sweep_report.py --out-dir ./spectral_sweep --objective spectral
"""
from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
import scipy.sparse as _sp
import scipy.sparse.csgraph as _csg

# Moved from maldi/ to benchmarks/; the maldi siblings (laplacian_test, utils)
# live in ../maldi, so put that on sys.path.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maldi"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manifold"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manifold" / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse the exact graph/eigs + lipid-name machinery from the single-config diagnostic.
from laplacian_test import (
    build_graph_and_laplacian,
    _resolve_lipids,
    _kernel_features_np,
)
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import KnnGraphCache

log = logging.getLogger("spectral_distance_sweep")
BASE_METRICS = ["euclidean", "geodesic", "spectral"]


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return (f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s")


def parse_stride_modes(tokens: list[str]) -> list[tuple[int, int]]:
    """'8:1300' -> (stride=8, max_modes=1300)."""
    pairs = []
    for t in tokens:
        if ":" not in t:
            raise argparse.ArgumentTypeError(
                f"--stride-modes entry {t!r} must be STRIDE:MODES, e.g. 4:6000")
        s, m = t.split(":")
        pairs.append((int(s), int(m)))
    return pairs


# ===========================================================================
# MALDI signal load — coords + per-lipid values over ALL measured points
# (NO train/test fold). Optionally excludes whole samples (e.g. disease /
# pregnancy brains). The covariance-vs-distance diagnostic is a property of the
# graph geometry + lipid fields, so it wants the whole dataset, not a held-out
# slice. Coord normalization is the reference-template one (config-independent),
# so only the node-snapping changes per graph.
# ===========================================================================
def _resolve_lipid_names(args: dict) -> list[str]:
    """Lipid columns to read: all names in --available-lipids-file, optionally
    narrowed by --lipids / --lipids-file (names or integer indices). No fold /
    MaldiConfig involved."""
    if not args.get("available_lipids_file"):
        raise RuntimeError("--available-lipids-file is required to know the lipid columns.")
    names_all = list(np.load(args["available_lipids_file"], allow_pickle=True))
    spec = list(args.get("lipids") or [])
    if args.get("lipids_file"):
        with open(args["lipids_file"]) as f:
            spec += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return _resolve_lipids(spec or None, names_all)


def load_maldi_signal(args: dict):
    from utils import coord_norm_from_reference
    lipid_names = _resolve_lipid_names(args)
    coord_cols = ["xccf", "yccf", "zccf"]
    lip_cols = [str(l) for l in lipid_names]

    # exclude whole samples (e.g. GBA1 / Pregnant*) — predicate pushdown on the
    # sample column; the column itself need not be in `columns`.
    excl = list(args.get("exclude_samples") or [])
    filt = [(args["sample_column"], "not in", excl)] if excl else None
    log.info(f"reading {args['maldi_file']} (all points"
             + (f", excluding {args['sample_column']} in {excl}" if excl else "")
             + ") ...")
    df = pd.read_parquet(args["maldi_file"], columns=coord_cols + lip_cols,
                         filters=filt)
    if df.shape[0] == 0:
        raise RuntimeError(f"0 rows after sample exclusion {excl!r}")
    if args["maldi_max_rows"] and len(df) > args["maldi_max_rows"]:
        df = df.sample(args["maldi_max_rows"], random_state=0)

    X = df[coord_cols].to_numpy(np.float64)
    Y = df[lip_cols].to_numpy(np.float64).copy()
    Y[Y < 0] = 0.0
    if args["log_transform"]:
        Y = np.log(Y + 1e-10)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)

    template_full = np.load(args["reference_file"])
    coord_mean, coord_std = coord_norm_from_reference(template_full)
    cm = np.asarray(coord_mean, np.float64).reshape(-1)
    cs = np.maximum(np.asarray(coord_std, np.float64), 1e-6)
    Xs = (X - cm) / cs
    log.info(f"{len(df):,} points x {len(lip_cols)} lipids loaded.")
    return Xs.astype(np.float32), Y.astype(np.float64), lip_cols


# ===========================================================================
# Per-bin data-covariance-vs-distance metrics for ONE lipid against ONE
# distance field. dist/si/sj are flat arrays over the SAME pair set.
# ===========================================================================
def _binned_decay_spearman(dist: np.ndarray, cov: np.ndarray,
                           edges: np.ndarray, min_count: int) -> float:
    """Spearman(mean-dist, mean-cov) over the given bin edges; bins with fewer than
    min_count pairs are dropped. Negative => covariance decays cleanly with distance.
    Mirrors the notebook's binned Q4b/Q4c rho (Spearman is rank-invariant, so using
    h_bar vs bin-centers / a normalized [0,1] axis makes no difference)."""
    nb = len(edges) - 1
    if nb < 3:
        return np.nan
    idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    safe = np.maximum(cnt, 1)
    h_bar = np.bincount(idx, weights=dist, minlength=nb) / safe
    c_bar = np.bincount(idx, weights=cov,  minlength=nb) / safe
    ok = cnt >= min_count
    if ok.sum() < 3:
        return np.nan
    return float(spearmanr(h_bar[ok], c_bar[ok]).statistic)


def cov_distance_metrics(dist: np.ndarray, si: np.ndarray, sj: np.ndarray,
                         n_bins: int, uniform_min_count: int = 30) -> dict:
    nan = dict(cov_dist_spearman=np.nan, decay_strength=np.nan,
               cov_dist_spearman_quantile=np.nan, decay_strength_quantile=np.nan,
               cov_dist_spearman_uniform=np.nan, decay_strength_uniform=np.nan,
               cov_curve_corr=np.nan, vario_r2=np.nan)
    cov = si * sj                 # pairwise data covariance contribution
    y = (si - sj) ** 2            # squared dissimilarity (semivariogram target)
    if dist.size < 3 * n_bins:
        n_bins = max(3, dist.size // 3)
    edges = np.unique(np.quantile(dist, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return nan
    edges[-1] = np.inf
    idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    cnt = np.bincount(idx, minlength=nb).astype(float)
    safe = np.maximum(cnt, 1)
    h_bar = np.bincount(idx, weights=dist, minlength=nb) / safe
    c_bar = np.bincount(idx, weights=cov,  minlength=nb) / safe
    ok = cnt > 0

    g_bar = np.bincount(idx, weights=y, minlength=nb) / safe
    pred = g_bar[idx]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    sp = float(spearmanr(dist, cov).statistic)                       # raw per-pair (noisy)
    pc = float(pearsonr(h_bar[ok], c_bar[ok])[0]) if ok.sum() >= 3 else np.nan

    # binned decay Spearman under BOTH axis normalizations (cf. notebook Q4b vs Q4c):
    #   quantile = equal-count bins (robust to a metric's pile-up; the honest ranking)
    #   uniform  = equal-WIDTH bins on [min,max] (flatters a saturating metric)
    sp_q = _binned_decay_spearman(dist, cov, edges, min_count=1)      # quantile reuses `edges`
    lo, hi = float(np.min(dist)), float(np.max(dist))
    if hi > lo:
        u_edges = np.linspace(lo, hi, nb + 1); u_edges[-1] = np.inf
        sp_u = _binned_decay_spearman(dist, cov, u_edges, min_count=uniform_min_count)
    else:
        sp_u = np.nan
    neg = lambda v: (-v if np.isfinite(v) else np.nan)
    return dict(cov_dist_spearman=sp, decay_strength=-sp,
                cov_dist_spearman_quantile=sp_q, decay_strength_quantile=neg(sp_q),
                cov_dist_spearman_uniform=sp_u, decay_strength_uniform=neg(sp_u),
                cov_curve_corr=pc, vario_r2=r2)


# ===========================================================================
# One graph config -> per-lipid rows for all distance metrics.
# ===========================================================================
def evaluate_config(c: dict, args: dict, Xs: np.ndarray, Y: np.ndarray,
                    lipid_names: list[str], metrics: list[str],
                    graphs_cache, eigvec_dir: Path, device) -> list[dict]:
    knn_method, norm = c["knn_method"], c["normalization"]
    bw = c["graphbandwidth"]
    num_modes = c["num_modes"]
    # stride varies per config (paired with the mode budget), so override here.
    cfg_args = {**args, "stride": c["stride"]}

    log.info(f"  building graph + laplacian (stride={c['stride']}) ...")
    built = build_graph_and_laplacian(
        cfg_args, knn_method, norm,
        knn_k=c["knn_k"], graphbandwidth=bw,
        cross_region_inflation=c["cross_region_inflation"], threshold=c["threshold"],
        device=device, graphs_cache=graphs_cache,
    )
    ekey = make_eig_key({"graph": built["graph_key"], "norm": norm,
                         "bw": bw, "modes": num_modes})
    ncv_min = resolve_ncv_min(num_modes, args["ncv_min"])
    solver = LaplacianEigensolver(num_modes=num_modes,
                                  backend=args["solver_backend"],
                                  tol=args["solver_tol"], ncv_min=ncv_min,
                                  verbose=True)
    log.info(f"  eigensolve / load ({num_modes} modes, key={ekey[:12]}...) ...")
    eigval, eigvec = solver.compute_or_load(
        built["laplacian_op"], cache_dir=eigvec_dir, key=ekey,
        graphbandwidth=bw, laplacian_normalization=norm,
        extra={"graph": built["graph_key"], "norm": norm, "bw": bw,
               "modes": num_modes},
        force_recompute=False, device=device,
    )

    reference_nodes = built["reference_nodes"]
    N = int(reference_nodes.shape[0])
    knn = built["knn"]
    K = Y.shape[1]
    fmode = args["feature_mode"]

    coords = reference_nodes.detach().cpu().numpy().astype(np.float64)
    ei = built["edge_index"].detach().cpu().numpy()
    ev = built["edge_value"].detach().cpu().numpy().astype(np.float64)
    lengths = np.sqrt(np.maximum(ev, 0.0))
    ev_np = eigval.detach().cpu().numpy()
    evec_np = eigvec.detach().cpu().numpy()
    M = eigvec.shape[1]

    # resolve the mode count for every spectral metric requested. spectral=full
    # budget M; spectral_low<N>=N (clamped to M); bare spectral_low=frac*M.
    spec_modes: dict[str, int] = {}
    for name in metrics:
        if name == "spectral":
            spec_modes[name] = M
        elif name == "spectral_low":
            spec_modes[name] = max(1, int(round(M * args["low_mode_frac"])))
        elif name.startswith("spectral_low"):
            req = int(name[len("spectral_low"):])
            cnt = min(req, M)
            if req > M:
                log.warning(f"  {name}: requested {req} modes > budget {M}; "
                            f"clamped to {M} (== full spectral).")
            spec_modes[name] = max(1, cnt)

    rng = np.random.default_rng(args["seed"])

    # -------------------------------------------------------------------
    # Endpoint selection differs by feature mode:
    #   snap    : anchors/targets are GRAPH NODES (MALDI snapped + averaged);
    #             spectral features = exact on-graph eigenvectors (bump-free).
    #   nystrom : anchors/targets are REAL MALDI POINTS at true coords;
    #             spectral features = deployed Nystrom OOS map (bump-weighted),
    #             so bump_scale/bump_decay matter. cov uses each point's own value.
    # Both produce: anchor coords Xa (na,3), target coords Xt (nt,3), per-lipid
    # signal matrices Sa (na,K)/St (nt,K), and the nearest graph node of each
    # endpoint (a_nodes/t_nodes) for the geodesic + low-freq fallback.
    # -------------------------------------------------------------------
    if fmode == "snap":
        Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device).contiguous()
        _, idx = knn.search(Xs_t, 1)
        snap_i = idx.squeeze(-1).detach().cpu().numpy().astype(np.int64)
        node_cnt = np.zeros(N); np.add.at(node_cnt, snap_i, 1.0)
        hit = node_cnt > 0
        node_sum = np.zeros((N, K)); np.add.at(node_sum, snap_i, Y)
        s_node = np.zeros((N, K))
        s_node[hit] = node_sum[hit] / node_cnt[hit, None]
        s_node[hit] -= s_node[hit].mean(0, keepdims=True)
        hit_idx = np.flatnonzero(hit)
        log.info(f"  snapped: {int(hit.sum()):,}/{N:,} nodes carry signal "
                 f"(coverage {hit.mean():.1%})")
        if hit_idx.size < 50:
            log.warning(f"  only {hit_idx.size} hit nodes; skipping config.")
            return []
        n_anc = min(args["vario_anchors"], hit_idx.size)
        a_nodes = rng.choice(hit_idx, size=n_anc, replace=False)
        t_nodes = (rng.choice(hit_idx, size=args["vario_max_targets"], replace=False)
                   if hit_idx.size > args["vario_max_targets"] else hit_idx)
        Xa, Xt = coords[a_nodes], coords[t_nodes]
        Sa, St = s_node[a_nodes], s_node[t_nodes]
        off_frac = 0.0
    else:  # nystrom
        P = Xs.shape[0]
        n_anc = min(args["vario_anchors"], P)
        a_idx = rng.choice(P, size=n_anc, replace=False)
        t_idx = (rng.choice(P, size=args["vario_max_targets"], replace=False)
                 if P > args["vario_max_targets"] else np.arange(P))
        Xa = Xs[a_idx].astype(np.float64); Xt = Xs[t_idx].astype(np.float64)
        Sa, St = Y[a_idx], Y[t_idx]                       # per-point z-scored values
        # nearest graph node of each endpoint (geodesic + low-freq fallback)
        a_nodes = knn.search(torch.tensor(Xs[a_idx], dtype=torch.float32, device=device),
                             1)[1].squeeze(-1).detach().cpu().numpy().astype(np.int64)
        t_nodes = knn.search(torch.tensor(Xs[t_idx], dtype=torch.float32, device=device),
                             1)[1].squeeze(-1).detach().cpu().numpy().astype(np.int64)

    # spectral feature map per endpoint set, at a given mode budget. Memoized by
    # (endpoint, modes) so the full-M features shared by the spectral metric and
    # the nystrom coverage diagnostic are only computed once.
    _feat_cache: dict = {}

    def _features(tag, coords_np, node_idx, modes):
        key = (tag, modes)
        if key in _feat_cache:
            return _feat_cache[key]
        if fmode == "snap":
            F = _kernel_features_np(ev_np[:modes], evec_np[:, :modes],
                                    c["nu"], c["lengthscale"], node_idx)
        else:
            kern = RiemannMaternKernel(
                nu=c["nu"], knn=knn,
                edge_index=built["edge_index"], edge_value=built["edge_value"],
                eigval=eigval, eigvec=eigvec,
                nearest_neighbors=c["knn_k"], num_modes=modes,
                bump_scale=c["bump_scale"], bump_decay=c["bump_decay"],
                laplacian_normalization=norm, graphbandwidth_init=bw,
            ).to(device).eval()
            kern.lengthscale = torch.tensor(float(c["lengthscale"]), device=device)
            with torch.no_grad():
                xt = torch.as_tensor(np.asarray(coords_np, np.float32),
                                     device=device).contiguous()
                F = kern.features(xt).cpu().numpy()
            del kern
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _feat_cache[key] = F
        return F

    def spectral_dist(modes: int) -> np.ndarray:
        Fa = _features("a", Xa, a_nodes, modes); Ft = _features("t", Xt, t_nodes, modes)
        Kat = Fa @ Ft.T
        dKa = (Fa * Fa).sum(1); dKt = (Ft * Ft).sum(1)
        if args["spectral_norm"] == "correlation":
            # strip the per-node kernel diagonal: d = sqrt(2(1-cos)). Off-bump
            # nystrom points have zero features -> cos~0 -> d~sqrt(2) (decorrelated).
            cos = Kat / np.sqrt(np.clip(dKa[:, None] * dKt[None, :], 1e-24, None))
            return np.sqrt(np.clip(2.0 * (1.0 - cos), 0, None))
        return np.sqrt(np.clip(dKa[:, None] + dKt[None, :] - 2 * Kat, 0, None))

    # bump coverage diagnostic (nystrom only): fraction of endpoints the kernel
    # cannot represent (zero OOS features beyond bump_scale*graphbandwidth).
    if fmode == "nystrom":
        Ffull_a = _features("a", Xa, a_nodes, M); Ffull_t = _features("t", Xt, t_nodes, M)
        off_a = ((Ffull_a ** 2).sum(1) <= 1e-12); off_t = ((Ffull_t ** 2).sum(1) <= 1e-12)
        off_frac = float((off_a.sum() + off_t.sum()) / (len(off_a) + len(off_t)))
        log.info(f"  nystrom bump (bs={c['bump_scale']}, bd={c['bump_decay']}): "
                 f"{off_frac:.1%} of endpoints off-support (zero features)")

    n_tgt = len(t_nodes)
    log.info(f"  distances: {n_anc} anchors x {n_tgt} targets [{fmode}] "
             f"({'/'.join(metrics)}; modes={spec_modes}) ...")
    Deuc = np.sqrt(((Xa[:, None] - Xt[None, :]) ** 2).sum(-1))
    Wlen = _sp.coo_matrix((lengths, (ei[0], ei[1])), shape=(N, N)).tocsr()
    Dgeo = _csg.dijkstra(Wlen, directed=False, indices=a_nodes)[:, t_nodes]

    # common finite pair set (geodesic reachability gates all metrics)
    finite = np.isfinite(Dgeo) & (Dgeo > 0)
    if finite.sum() < 3 * args["n_bins"]:
        log.warning(f"  too few reachable pairs ({int(finite.sum())}); skipping.")
        return []
    fr = finite.ravel()
    AIp = np.repeat(np.arange(n_anc), n_tgt)[fr]       # positions into anchor arrays
    BIp = np.tile(np.arange(n_tgt), n_anc)[fr]         # positions into target arrays
    dist_flat = {"euclidean": Deuc.ravel()[fr], "geodesic": Dgeo.ravel()[fr]}
    spec_cache: dict[int, np.ndarray] = {}             # reuse if counts coincide
    for name, modes in spec_modes.items():
        if modes not in spec_cache:
            spec_cache[modes] = spectral_dist(modes).ravel()[fr]
        dist_flat[name] = spec_cache[modes]

    log.info(f"  scoring {len(lipid_names)} lipids over "
             f"{int(finite.sum()):,} reachable pairs ...")
    base = {k: c[k] for k in ("knn_method", "normalization", "threshold",
                              "knn_k", "graphbandwidth", "cross_region_inflation",
                              "nu", "lengthscale", "stride", "num_modes",
                              "bump_scale", "bump_decay")}
    base["config"] = c["_slug"]
    base["feature_mode"] = fmode
    base["spectral_norm"] = args["spectral_norm"]
    base["n_nodes"] = N
    base["off_support_frac"] = float(off_frac)
    base["reachable_frac"] = float(finite.mean())
    rows = []
    for k, name in enumerate(lipid_names):
        si = Sa[AIp, k]; sj = St[BIp, k]
        for metric in metrics:
            m = cov_distance_metrics(dist_flat[metric], si, sj, args["n_bins"],
                                     uniform_min_count=args["uniform_min_count"])
            rows.append({**base, "lipid": name, "metric": metric,
                         "metric_modes": spec_modes.get(metric, np.nan), **m})

    del eigval, eigvec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# ===========================================================================
# Grid + driver
# ===========================================================================
def slug(x):
    return str(x).replace(".", "p").replace("-", "n")


def enumerate_configs(a, stride_modes) -> list[dict]:
    tag = {"faiss": "fa", "anatomical_atlas": "aa", "faiss_atlas_weighted": "fw"}
    ntag = {"symmetric": "sym", "randomwalk": "rw"}
    # bump only affects the nystrom feature path; in snap mode it is inert, so we
    # fix it to a single placeholder to avoid enumerating duplicate configs.
    nystrom = (a.feature_mode == "nystrom")
    bump_scales = a.bump_scales if nystrom else [a.bump_scales[0]]
    bump_decays = a.bump_decays if nystrom else [a.bump_decays[0]]
    out = []
    for (method, norm, thr, k, bw, nu, ls, (stride, modes), bs, bd) in itertools.product(
            a.knn_methods, a.normalizations, a.thresholds, a.knn_ks,
            a.bandwidths, a.nus, a.lengthscales, stride_modes,
            bump_scales, bump_decays):
        infls = a.inflations if method == "faiss_atlas_weighted" else [a.inflations[0]]
        for infl in infls:
            s = (f"{tag.get(method, method)}-{ntag.get(norm, norm)}-t{thr}"
                 f"-k{k}-bw{slug(bw)}")
            if method == "faiss_atlas_weighted":
                s += f"-i{slug(infl)}"
            s += f"-nu{nu}-ls{slug(ls)}-s{stride}-nm{modes}"
            if nystrom:                               # keep snap slugs unchanged
                s += f"-bs{slug(bs)}-bd{slug(bd)}"
            out.append(dict(knn_method=method, normalization=norm, threshold=thr,
                            knn_k=k, graphbandwidth=bw, cross_region_inflation=infl,
                            nu=nu, lengthscale=ls, stride=stride, num_modes=modes,
                            bump_scale=bs, bump_decay=bd, _slug=s))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # fixed inputs
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None)
    p.add_argument("--eigenvector-dir", required=True)
    p.add_argument("--maldi-file", required=True)
    p.add_argument("--available-lipids-file", default=None,
                   help="npy of lipid names = the parquet's lipid columns")
    p.add_argument("--lipids-file", default=None,
                   help="optional subset of lipids (names or indices), one per line")
    p.add_argument("--log-transform", action="store_true")
    # NO train/test fold: use ALL measured points, optionally dropping whole samples.
    p.add_argument("--exclude-samples", nargs="*",
                   default=["GBA1", "Pregnant1", "Pregnant2", "Pregnant4"],
                   help="sample IDs to drop entirely (e.g. disease/pregnancy brains); "
                        "pass with no values to keep everything")
    p.add_argument("--sample-column", default="Sample",
                   help="parquet column holding the sample ID for --exclude-samples")
    p.add_argument("--maldi-max-rows", type=int, default=400000)
    # sweep axes
    p.add_argument("--knn-methods", nargs="+",
                   choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted"],
                   default=["faiss_atlas_weighted"])
    p.add_argument("--normalizations", nargs="+",
                   choices=["randomwalk", "symmetric"], default=["randomwalk"])
    p.add_argument("--thresholds", type=int, nargs="+", default=[5])
    p.add_argument("--knn-ks", type=int, nargs="+", default=[120])
    p.add_argument("--bandwidths", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--inflations", type=float, nargs="+", default=[10.0])
    p.add_argument("--nus", type=int, nargs="+", default=[2])
    p.add_argument("--lengthscales", type=float, nargs="+", default=[1.0])
    p.add_argument("--feature-mode", choices=["snap", "nystrom"], default="snap",
                   help="how off-node MALDI points get spectral features. 'snap' "
                        "(default): snap+average onto graph nodes, exact on-graph "
                        "eigenvectors (bump-free). 'nystrom': evaluate at the points' "
                        "TRUE coords via the deployed Nystrom OOS map (bump-weighted "
                        "eigenvectors), so bump_scale/bump_decay matter.")
    p.add_argument("--bump-scales", type=float, nargs="+", default=[1.0],
                   help="bump support = bump_scale*graphbandwidth (nystrom only; "
                        "swept as an axis there, inert in snap mode)")
    p.add_argument("--bump-decays", type=float, nargs="+", default=[0.01],
                   help="bump falloff sharpness (nystrom only)")
    p.add_argument("--spectral-norm", choices=["correlation", "rkhs"],
                   default="correlation",
                   help="spectral distance form: 'correlation' (default) = "
                        "sqrt(2(1-cos)), strips the per-node kernel diagonal (matches "
                        "the corrected notebook d_spec); 'rkhs' = raw "
                        "sqrt(K_ii+K_jj-2K_ij), contaminated by node variance.")
    p.add_argument("--stride-modes", nargs="+", default=None,
                   help="STRIDE:MAX_MODES pairs, e.g. 8:1300 4:6000. Couples "
                        "resolution to the mode budget. Falls back to "
                        "--stride x --num-modes if omitted.")
    # low-mode spectral variant(s). Pass a LIST of absolute mode counts to get
    # one spectral_low<N> metric each (e.g. --low-modes 64 256 1000), so several
    # truncations are compared side-by-side in a single sweep. Each is clamped to
    # the config's full budget M. If no list is given (and not --no-low-modes), a
    # single frac-based `spectral_low` metric (--low-mode-frac of M) is used.
    p.add_argument("--low-modes", type=int, nargs="*", default=[],
                   help="absolute mode count(s) for the spectral_low<N> metric(s); "
                        "e.g. --low-modes 64 256 1000. Empty => fall back to "
                        "--low-mode-frac.")
    p.add_argument("--low-mode-frac", type=float, default=0.1,
                   help="fraction of the full mode budget for the single "
                        "`spectral_low` metric, used only when --low-modes is empty")
    p.add_argument("--no-low-modes", action="store_true",
                   help="disable the spectral_low metric(s) entirely")
    # non-swept knobs / fallbacks
    p.add_argument("--stride", type=int, default=4,
                   help="fallback stride if --stride-modes is omitted")
    p.add_argument("--num-modes", type=int, default=1300,
                   help="fallback mode budget if --stride-modes is omitted")
    p.add_argument("--ncv-min", type=int, default=-1)
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--region-bbox", type=int, nargs=6, default=None)
    p.add_argument("--solver-backend", default="cupy", choices=["cupy", "scipy"])
    p.add_argument("--solver-tol", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    # variogram sampling
    p.add_argument("--vario-anchors", type=int, default=200)
    p.add_argument("--vario-max-targets", type=int, default=4000)
    p.add_argument("--n-bins", type=int, default=24)
    p.add_argument("--uniform-min-count", type=int, default=30,
                   help="min pairs per equal-WIDTH bin for the uniform decay Spearman "
                        "(sparse bins dropped; mirrors the notebook Q4b filter)")
    # output (ranking/aggregation lives in spectral_sweep_report.py)
    p.add_argument("--out-dir", default="./spectral_sweep")
    p.add_argument("--force", action="store_true",
                   help="recompute configs even if their rows/<slug>.csv exists")
    return p.parse_args()


def main():
    a = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = out_dir / "rows"; rows_dir.mkdir(parents=True, exist_ok=True)

    args = vars(a)
    args["region_bbox"] = tuple(a.region_bbox) if a.region_bbox else None
    args["lipids"] = None

    stride_modes = (parse_stride_modes(a.stride_modes) if a.stride_modes
                    else [(a.stride, a.num_modes)])
    # Distance metrics for this run. Low-mode spectral variants: one
    # spectral_low<N> per requested absolute count, else a single frac-based
    # spectral_low. metric_tag goes in the rows filename so a re-run with a
    # DIFFERENT low-modes set doesn't resume off stale rows (graph/eigs caches
    # are still shared — only the cheap per-pair scoring is redone).
    metrics = list(BASE_METRICS)
    if a.no_low_modes:
        metric_tag = "nolow"
    elif a.low_modes:
        metrics += [f"spectral_low{n}" for n in a.low_modes]
        metric_tag = "lm" + "-".join(str(n) for n in a.low_modes)
    else:
        metrics.append("spectral_low")
        metric_tag = f"lmf{slug(a.low_mode_frac)}"
    # feature mode is part of the rows-file identity (snap vs nystrom give
    # different numbers for the same config slug; the slug only carries bump in
    # nystrom mode), so a re-run in the other mode never resumes off stale rows.
    metric_tag = f"{a.feature_mode}-{metric_tag}"

    def row_path_for(c):
        return rows_dir / f"{c['_slug']}__{metric_tag}.csv"

    device = torch.device(a.device)
    eigvec_dir = Path(a.eigenvector_dir) / "eigvecs"; eigvec_dir.mkdir(parents=True, exist_ok=True)
    graphs_cache = KnnGraphCache(cache_dir=Path(a.eigenvector_dir) / "knn", verbose=True)

    Xs, Y, lipid_names = load_maldi_signal(args)

    configs = enumerate_configs(a, stride_modes)
    n = len(configs)
    done0 = sum(row_path_for(c).exists() and not a.force for c in configs)
    log.info(f"Sweep: {n} config(s) [stride-modes={stride_modes}] x "
             f"{len(lipid_names)} lipids x {len(metrics)} metrics ({metrics}). "
             f"{done0} already done, {n - done0} to run.")

    t_start = time.time()
    durations = []
    for i, c in enumerate(configs, 1):
        row_path = row_path_for(c)
        if row_path.exists() and not a.force:
            log.info(f"[{i}/{n}] SKIP (done): {c['_slug']}")
            continue
        eta = ""
        if durations:
            avg = sum(durations) / len(durations)
            eta = f"  ~ETA {_fmt_dur(avg * (n - i + 1))}"
        log.info(f"[{i}/{n}] RUN {c['_slug']}{eta}")
        t0 = time.time()
        try:
            rows = evaluate_config(c, args, Xs, Y, lipid_names, metrics,
                                   graphs_cache, eigvec_dir, device)
        except Exception as e:
            oom = "out of memory" in str(e).lower() or "OutOfMemory" in type(e).__name__
            hint = ("  (OOM: lower this stride's modes in --stride-modes, or cap "
                    "--ncv-min, e.g. ncv just above num_modes)" if oom else "")
            log.warning(f"[{i}/{n}] FAILED {c['_slug']}: {e}{hint}", exc_info=not oom)
            # reclaim GPU memory so one OOM doesn't cascade into every later config
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache(); torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    import cupy
                    cupy.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            continue
        if not rows:
            log.warning(f"[{i}/{n}] no rows for {c['_slug']} (skipped internally)")
            continue
        # tmp + atomic rename: a kill mid-write can't leave a truncated rows file
        # that resume would mistake for done.
        tmp = row_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(row_path)
        dt = time.time() - t0
        durations.append(dt)
        log.info(f"[{i}/{n}] DONE {c['_slug']} in {_fmt_dur(dt)} "
                 f"({len(rows)} rows) -> {row_path.name}")

    # No aggregation here: this run only produces per-config rows. Aggregation +
    # ranking is a separate step (spectral_sweep_report.py) so the sweep can fan
    # out across many one-config jobs sharing this rows/ dir.
    n_done = sum(row_path_for(c).exists() for c in configs)
    log.info(f"Total wall: {_fmt_dur(time.time() - t_start)} "
             f"over {len(configs)} config(s); {n_done} rows file(s) present.")
    log.info(f"Rows in {rows_dir}/*.csv. Build the report with:\n"
             f"    python manifold/benchmarks/spectral_sweep_report.py --out-dir {out_dir}")


if __name__ == "__main__":
    main()
