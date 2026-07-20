#!/usr/bin/env python
# encoding: utf-8
"""
Benchmark the Nyström out-of-sample extension of graph-Laplacian eigenvectors.

Idea
----
The kernel extends eigenvectors to unseen points with a Nyström formula
(`GraphLaplacianOperator.out_of_sample`). How good is that extension? We can
measure it directly by exploiting the fact that the strided atlas grids are
nested: the stride-`COARSE` voxels are a SUBSET of the stride-`FINE` voxels
(e.g. stride 8 ⊂ stride 4, since 0,8,16,… ⊂ 0,4,8,12,16,…). So:

  1. Solve the eigenproblem on the COARSE graph (stride 8)  -> the source we
     extend FROM.
  2. Solve the eigenproblem on the FINE graph (stride 4)    -> the ground truth.
  3. The FINE nodes that are NOT coarse nodes are the "in-between" points the
     coarse graph never saw. Nyström-extend the coarse eigenvectors to those
     points and compare against the true fine eigenvectors there.

Aligning two eigenbases
-----------------------
Eigenvectors from two different graphs are only comparable up to (a) per-mode
sign, (b) rotation inside near-degenerate clusters, (c) the discrete L2 scale
(coarse entries are ~sqrt(N_fine/N_coarse) larger), and (d) mode reordering.
We absorb all of that with a single linear map fit ONLY on the shared
(coarse) nodes via least squares:

      T = argmin_T || Ec · T  -  Ef|shared ||_F      (solved with lstsq, so it
                                                      doesn't assume Ec is
                                                      orthonormal -- randomwalk
                                                      eigvecs are D-orthonormal)

`T` (modes × modes) rewrites the fine basis in terms of the coarse basis using
information available at the shared nodes only. We then push the Nyström
extension through the SAME `T` at the held-out in-between nodes and score it
against the true fine eigenvectors. The residual therefore reflects two honest
things at once: the Nyström extension error, and any part of the fine spectrum
the coarse graph simply cannot resolve (aliasing) -- which is exactly the
practical question "is a stride-8 solve good enough to predict stride-4?".

Reported metrics (all on the held-out test points)
  * relative L2 error per fine mode  ||Ê_k - E_k|| / ||E_k||
  * |correlation| per fine mode
  * cumulative subspace reconstruction error using the first m modes
  * a self-consistency check: Nyström at the SHARED nodes must reproduce the
    coarse eigenvectors (≈ machine eps) -- a guard that the coordinate spaces
    and indexing line up.
  * the extrapolation regime: NN distance of test points to the coarse graph,
    in units of the graph bandwidth.

Scoring at real MALDI points (--maldi-file, --maldi-mode)
-----------------------------------------------------------
By default the held-out test points are the nested grid's "in-between" nodes
(synthetic). `--maldi-file` brings the ACTUAL measured MALDI voxel coordinates
into the picture -- the real query distribution the kernel Nyström-extends to
at inference -- in one of two modes:

  * --maldi-mode direct (default once --maldi-file is set): score DIRECTLY at
    the MALDI coordinates. They are (almost) never exact nodes of EITHER
    graph, so there is no literal ground truth there. We treat the FINE
    graph's own Nyström extension at those points as a reference (it aliases
    far less than the coarse graph) and score the COARSE graph's extension
    (mapped through the same alignment `T`) against it. Honest about
    coarse-vs-fine degradation at the real points, but NOT an absolute-truth
    comparison the way the grid-based mode is.

  * --maldi-mode distance-match: keep literal ground truth by staying on the
    FINE grid's in-between nodes, but SUBSAMPLE them so their NN-distance-to-
    the-coarse-graph distribution matches the real MALDI points' distribution
    (quantile-binned resampling). This answers "how good is the extension at
    the extrapolation distances MALDI points actually sit at", with an exact
    fine eigenvector as ground truth -- the grid mode's honesty plus the
    MALDI mode's realistic extrapolation regime.

Usage
-----
    python nystrom_benchmark.py \
        --reference-volume reference_image.npy \
        --annotations-volume level_15annot.npy \
        --output-path /home/casap/mlibra/output/eigenvectors \
        --coarse-stride 8 --fine-stride 4 --modes 1000 \
        --bandwidth 0.1 --normalization randomwalk --threshold 5

`--normalization randomwalk --bandwidth 0.1` matches the lgp / per-lipid
production kernel (plain faiss graph; atlas edge-weighting is NOT applied here).
COARSE and FINE must share `--threshold`; run once per threshold to sweep.
Graphs + eigenpairs are (re)computed and cached under output-path on first use.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from manifold_gp.utils.nearest_neighbors import KnnGraphCache, make_key as knn_make_key
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as eigen_make_key,
)
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator

# Moved from maldi/ to benchmarks/; the maldi sibling `utils` lives in ../maldi.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maldi"))
from utils import crop_or_stride_volume, reference_ccf_from_subvolume


# ---------------------------------------------------------------------------
# Build (or load) a graph + its eigenvectors for one stride, mirroring
# compute_eigenvectors.py exactly so the cache keys match.
# ---------------------------------------------------------------------------
class StrideSolve:
    """Everything about one stride's graph + eigenpairs, in one place."""

    def __init__(self, mm_coords, coord_mean, coord_std, knn_model, op,
                 eigval, eigvec):
        self.mm_coords = mm_coords        # (N,3) raw mm, the shared key space
        self.coord_mean = coord_mean      # (3,) per-axis, this graph's space
        self.coord_std = coord_std        # (3,) per-axis
        self.knn = knn_model              # FAISS index over standardized coords
        self.op = op                      # GraphLaplacianOperator
        self.eigval = eigval              # (M,)
        self.eigvec = eigvec              # (N, M), L2-normalized over N

    @property
    def n_nodes(self) -> int:
        return self.mm_coords.shape[0]

    def standardize(self, mm_xyz: np.ndarray) -> torch.Tensor:
        """Put raw mm coords into THIS graph's standardized space (the space
        its FAISS index and bandwidth live in)."""
        z = (mm_xyz - self.coord_mean) / self.coord_std
        return torch.from_numpy(z).float().to(self.eigvec.device).contiguous()


def build_stride(reference_image, annotation_volume, *, stride, template,
                 threshold, k, nlist, bandwidth, modes, ncv_min, normalization,
                 device, cache_path, force_recompute) -> StrideSolve:
    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        reference_image, annotation_volume, stride)
    mm_coords = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold)
    n_nodes = mm_coords.shape[0]
    logging.info(f"[stride {stride}] tissue nodes (thresh={threshold}): {n_nodes:,}")

    # Per-axis standardization, identical to compute_eigenvectors.py.
    coord_mean = mm_coords.mean(axis=0)
    coord_std = mm_coords.std(axis=0)
    reference_coords = torch.from_numpy(
        (mm_coords - coord_mean) / coord_std
    ).float().to(device).contiguous()

    knn_config = {
        "template": template, "stride": stride, "method": "faiss",
        "k": k, "thresh": threshold, "bbox": None,
    }
    knn_key = knn_make_key(knn_config)
    knn_cache = KnnGraphCache(cache_dir=cache_path / "knn", verbose=True)
    knn_model, edge_index, edge_value = knn_cache.train_or_load(
        key=knn_key, method="faiss", k=k, nlist=nlist,
        coords=reference_coords, volume=sub_volume, threshold=threshold,
        atlas_volume=sub_atlas, connectivity=1,
        force_recompute=force_recompute, device=device,
    )

    bw_tensor = torch.tensor(bandwidth, dtype=torch.float32, device=device)
    op = GraphLaplacianOperator(
        x=edge_value, idx=edge_index,
        operator_dimension=knn_model.x.shape[0],
        graphbandwidth=bw_tensor, normalization=normalization, self_loops=True,
    )

    eigen_key = eigen_make_key({"graph": knn_key, "norm": normalization,
                                "bw": bandwidth, "modes": modes})
    solver = LaplacianEigensolver(
        num_modes=modes, backend="cupy" if device.type == "cuda" else "scipy",
        ncv_min=resolve_ncv_min(modes, ncv_min), verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op=op, cache_dir=cache_path / "eigvecs", key=eigen_key,
        graphbandwidth=bandwidth, laplacian_normalization=normalization,
        force_recompute=force_recompute, device=device,
    )
    return StrideSolve(mm_coords, coord_mean, coord_std, knn_model, op,
                       eigval, eigvec)


# ---------------------------------------------------------------------------
# Match the nested grids: which fine nodes coincide with a coarse node?
# ---------------------------------------------------------------------------
def match_nested(coarse_mm: np.ndarray, fine_mm: np.ndarray, decimals=4):
    """Return (shared_fine_idx, shared_coarse_idx, oos_fine_idx).

    Both coordinate sets are exact multiples of (stride*0.025) mm in the same
    full-res frame, so a rounded-tuple hash gives exact, unambiguous matches.
    """
    key = lambda a: np.round(a, decimals)
    coarse_lookup: Dict[tuple, int] = {
        tuple(r): i for i, r in enumerate(key(coarse_mm))
    }
    shared_fine, shared_coarse, oos_fine = [], [], []
    for i, r in enumerate(key(fine_mm)):
        ci = coarse_lookup.get(tuple(r))
        if ci is None:
            oos_fine.append(i)
        else:
            shared_fine.append(i)
            shared_coarse.append(ci)
    return (np.asarray(shared_fine, dtype=np.int64),
            np.asarray(shared_coarse, dtype=np.int64),
            np.asarray(oos_fine, dtype=np.int64))


# ---------------------------------------------------------------------------
# Load real MALDI voxel coordinates, in the SAME raw-mm "ccf" frame as
# reference_ccf_from_subvolume (z, y, x order) -- see augment_nodes_with_maldi
# in maldi/utils.py, which concatenates the two coordinate sets directly and
# is the existing evidence that the parquet's coord_cols already line up with
# that axis order. Deduplicated the same way maldi_voxels_standardized is
# (many MALDI rows share a voxel).
# ---------------------------------------------------------------------------
def load_maldi_query_points(maldi_file, filt, coord_cols, decimals=6) -> np.ndarray:
    import pandas as pd
    df = pd.read_parquet(maldi_file, columns=list(coord_cols), filters=filt)
    coords = df.values.astype(np.float32)
    uniq = np.unique(np.round(coords, decimals), axis=0)
    return uniq.astype(np.float32)


# ---------------------------------------------------------------------------
# Nyström-extend coarse eigenvectors to a set of raw-mm query points.
# ---------------------------------------------------------------------------
def nystrom_extend(coarse: StrideSolve, query_mm: np.ndarray, k: int,
                   batch_size: int = 20000
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (psi, nn_dist): psi is (Q, M) extended eigvecs in the coarse
    basis; nn_dist is (Q,) Euclidean distance to the nearest coarse node in the
    coarse standardized space. Both are numpy arrays on the host.

    Batched over query points: out_of_sample materializes (B, k, M), which at
    Q~5e5, k=15, M=1000 is ~26 GiB if done in one shot -- so we chunk and move
    each batch's result to the CPU."""
    psi_parts, dist_parts = [], []
    for start in range(0, query_mm.shape[0], batch_size):
        q = coarse.standardize(query_mm[start:start + batch_size])
        ev_k, ei_k = coarse.knn.search(q, k)        # squared dists, indices
        psi = coarse.op.out_of_sample(coarse.eigvec, ev_k, ei_k)
        psi_parts.append(psi.detach().cpu().numpy())
        dist_parts.append(ev_k[:, 0].clamp(min=0).sqrt().detach().cpu().numpy())
        if coarse.eigvec.device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(psi_parts, axis=0), np.concatenate(dist_parts, axis=0)


# ---------------------------------------------------------------------------
# Cheap half of nystrom_extend -- the KNN search only, no eigenvector
# extension -- for sizing/matching a query set by its extrapolation distance
# before paying for the expensive (B, k, M) out_of_sample call.
# ---------------------------------------------------------------------------
def nn_distance_to_coarse(coarse: StrideSolve, query_mm: np.ndarray, k: int,
                          batch_size: int = 100000) -> np.ndarray:
    parts = []
    for start in range(0, query_mm.shape[0], batch_size):
        q = coarse.standardize(query_mm[start:start + batch_size])
        ev_k, _ = coarse.knn.search(q, k)
        parts.append(ev_k[:, 0].clamp(min=0).sqrt().detach().cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Select points from `candidate_dist` (e.g. grid in-between nodes) whose
# NN-distance-to-coarse distribution matches `target_dist` (e.g. real MALDI
# points), via quantile-binned resampling without replacement.
# ---------------------------------------------------------------------------
def match_distance_distribution(candidate_dist: np.ndarray, target_dist: np.ndarray,
                                n_samples: int, seed: int = 0, n_bins: int = 20
                                ) -> np.ndarray:
    """Returns a sorted index array into `candidate_dist`. May be SHORTER than
    `n_samples` if some distance bins are underpopulated among the candidates
    (e.g. the grid simply has no points as far from the coarse graph as the
    farthest MALDI points) -- the caller should check and report the shortfall
    rather than have it silently padded with a different distribution."""
    rng = np.random.default_rng(seed)
    edges = np.quantile(target_dist, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    target_bin = np.digitize(target_dist, edges[1:-1])
    target_frac = np.bincount(target_bin, minlength=n_bins) / target_dist.size
    cand_bin = np.digitize(candidate_dist, edges[1:-1])

    selected = []
    for b in range(n_bins):
        want = int(round(target_frac[b] * n_samples))
        if want == 0:
            continue
        pool = np.where(cand_bin == b)[0]
        if pool.size == 0:
            continue
        take = min(want, pool.size)
        selected.append(rng.choice(pool, take, replace=False))
    sel = np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)
    return np.sort(sel)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_mode_rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(pred - true, axis=0)
    den = np.linalg.norm(true, axis=0) + 1e-30
    return num / den


def per_mode_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = pred - pred.mean(0, keepdims=True)
    t = true - true.mean(0, keepdims=True)
    num = (p * t).sum(0)
    den = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0) + 1e-30
    return num / den


def cumulative_subspace_error(pred: np.ndarray, true: np.ndarray,
                              steps: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Relative Frobenius error using the first m modes, for a grid of m."""
    M = true.shape[1]
    ms = np.unique(np.linspace(1, M, min(steps, M)).astype(int))
    errs = []
    for m in ms:
        num = np.linalg.norm(pred[:, :m] - true[:, :m])
        den = np.linalg.norm(true[:, :m]) + 1e-30
        errs.append(num / den)
    return ms, np.asarray(errs)


# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-volume", required=True)
    p.add_argument("--annotations-volume", required=True)
    p.add_argument("--output-path", required=True,
                   help="Cache dir (reuses knn/ and eigvecs/ like "
                        "compute_eigenvectors.py).")
    p.add_argument("--coarse-stride", type=int, default=8,
                   help="Stride we extend FROM (must be a multiple of the fine "
                        "stride so its grid is nested in the fine one).")
    p.add_argument("--fine-stride", type=int, default=4,
                   help="Stride that provides the ground truth.")
    p.add_argument("--template", default="allen_25um")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="Tissue threshold. COARSE and FINE use the SAME value "
                        "so their grids stay nested; run the script once per "
                        "threshold to sweep (e.g. 5 and 40).")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--modes", type=int, default=1000)
    p.add_argument("--bandwidth", type=float, default=0.1)
    p.add_argument("--normalization", choices=["randomwalk", "symmetric"],
                   default="randomwalk",
                   help="Laplacian normalization; randomwalk matches the lgp / "
                        "per-lipid production kernel. The Nyström extension uses "
                        "the matching out_of_sample branch.")
    p.add_argument("--ncv-min", type=int, default=-1)
    p.add_argument("--nystrom-k", type=int, default=-1,
                   help="Neighbors used by the Nyström extension. <=0 uses --k.")
    p.add_argument("--max-oos", type=int, default=-1,
                   help="Subsample the held-out test points (grid in-between "
                        "nodes, or MALDI points if --maldi-file is set) to this "
                        "many for the comparison (<=0 uses all). Caps memory of "
                        "the dense alignment math; the extension itself is cheap.")
    p.add_argument("--maldi-file", default=None,
                   help="Parquet of real MALDI measurements. If set, the "
                        "held-out test points are the ACTUAL measured voxel "
                        "coordinates instead of the synthetic grid in-between "
                        "nodes -- see the module docstring for what 'ground "
                        "truth' means in this mode.")
    p.add_argument("--maldi-filter", default=None,
                   help="Python literal for a pandas parquet row filter (the "
                        "same format as MaldiConfig.section_filter/test_filter, "
                        "e.g. \"[('section', 'in', [1,2,3])]\"). None = all rows.")
    p.add_argument("--maldi-coord-cols", default="xccf,yccf,zccf",
                   help="Comma-separated parquet columns, in the SAME axis "
                        "order as reference_ccf_from_subvolume's (z,y,x) mm "
                        "coords -- matches the default augment_nodes_with_maldi "
                        "uses in maldi/utils.py.")
    p.add_argument("--maldi-mode", choices=["direct", "distance-match"],
                   default="direct",
                   help="Only used with --maldi-file. 'direct' (default): score "
                        "AT the MALDI coordinates (fine-graph Nyström stands in "
                        "for ground truth, see module docstring). "
                        "'distance-match': score at grid in-between nodes (exact "
                        "ground truth) SUBSAMPLED to match the MALDI points' "
                        "NN-distance-to-coarse-graph distribution.")
    p.add_argument("--distance-match-bins", type=int, default=20,
                   help="distance-match only: number of quantile bins used to "
                        "match the two NN-distance distributions.")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None,
                   help="Path prefix for JSON + PNG report.")
    args = p.parse_args()

    if args.coarse_stride % args.fine_stride != 0:
        raise SystemExit(
            f"--coarse-stride ({args.coarse_stride}) must be a multiple of "
            f"--fine-stride ({args.fine_stride}) so the coarse grid is nested "
            f"in the fine grid; otherwise no nodes are exactly shared.")

    device = torch.device(args.device)
    cache_path = Path(args.output_path)
    nystrom_k = args.k if args.nystrom_k <= 0 else args.nystrom_k

    reference_image = np.load(args.reference_volume)
    annotation_volume = np.load(args.annotations_volume)

    common = dict(template=args.template, threshold=args.threshold, k=args.k,
                  nlist=args.n_list, bandwidth=args.bandwidth, modes=args.modes,
                  ncv_min=args.ncv_min, normalization=args.normalization,
                  device=device, cache_path=cache_path,
                  force_recompute=args.force_recompute)

    print(f"\n[nystrom] === COARSE solve (stride {args.coarse_stride}) ===")
    t0 = time.time()
    coarse = build_stride(reference_image, annotation_volume,
                          stride=args.coarse_stride, **common)
    print(f"[nystrom] === FINE solve (stride {args.fine_stride}, ground truth) ===")
    fine = build_stride(reference_image, annotation_volume,
                        stride=args.fine_stride, **common)
    build_seconds = time.time() - t0

    # ---- nested-grid matching ----
    shared_fine, shared_coarse, oos_fine = match_nested(
        coarse.mm_coords, fine.mm_coords)
    n_shared, n_oos = shared_fine.size, oos_fine.size
    print(f"\n[nystrom] coarse nodes={coarse.n_nodes:,}  fine nodes={fine.n_nodes:,}")
    print(f"[nystrom] shared (coarse∩fine)={n_shared:,}  "
          f"in-between (held-out)={n_oos:,}")
    matched_frac = n_shared / max(1, coarse.n_nodes)
    if matched_frac < 0.98:
        print(f"[nystrom] WARNING: only {matched_frac:.1%} of coarse nodes "
              f"matched a fine node -- the grids may not be perfectly nested "
              f"(threshold mismatch?); results still valid on the matched set.")
    if n_oos == 0:
        raise SystemExit("[nystrom] no in-between nodes to test -- pick a "
                         "coarse stride strictly larger than the fine stride.")

    # ---- basis alignment T fit on the SHARED nodes only ----
    # Ec: coarse eigvecs at shared nodes (orthonormal columns over coarse nodes
    # only when shared==all coarse, which holds for nested grids). Ef: fine
    # eigvecs at the same physical nodes.
    Ec_shared = coarse.eigvec[shared_coarse].detach().cpu().numpy()   # (S, M)
    Ef_shared = fine.eigvec[shared_fine].detach().cpu().numpy()       # (S, M)
    # Least squares Ec·T ≈ Ef on shared nodes (robust to Ec not being exactly
    # orthonormal on a strict subset of coarse nodes).
    T, *_ = np.linalg.lstsq(Ec_shared, Ef_shared, rcond=None)         # (M, M)

    # ---- coordinate-alignment guard + in-sample floor ----
    # Nyström at the shared nodes: the nearest coarse node must be the node
    # itself (distance ~0). If the two graphs' standardized spaces were
    # misaligned this distance would blow up -- so it guards the coord math.
    # The reconstruction error there is NOT zero (out_of_sample is a local
    # average, not an index lookup): it's the best-case Nyström smoothing
    # FLOOR that the held-out error is measured against.
    psi_shared, nn_dist_shared = nystrom_extend(
        coarse, fine.mm_coords[shared_fine], nystrom_k)
    pred_shared = psi_shared @ T
    floor_rel = float(np.linalg.norm(pred_shared - Ef_shared)
                      / (np.linalg.norm(Ef_shared) + 1e-30))
    print(f"[nystrom] coord-alignment guard: max NN distance at shared nodes = "
          f"{nn_dist_shared.max():.2e}  (must be ~0)")
    if nn_dist_shared.max() > 1e-4:
        print("[nystrom] WARNING: shared nodes are NOT landing on coarse nodes "
              "-- the coarse/fine standardized spaces may be misaligned; "
              "Nyström results below are unreliable.")
    print(f"[nystrom] in-sample Nyström floor (smoothing at on-graph nodes) "
          f"rel-Frob err = {floor_rel:.4f}")

    # ---- the actual test: Nyström-extend coarse modes to the held-out test
    #      points, map through T, compare to a fine-graph reference there ----
    maldi_mm = None
    if args.maldi_file is not None:
        coord_cols = tuple(c.strip() for c in args.maldi_coord_cols.split(","))
        filt = ast.literal_eval(args.maldi_filter) if args.maldi_filter else None
        maldi_mm = load_maldi_query_points(args.maldi_file, filt, coord_cols)
        print(f"\n[nystrom] loaded {maldi_mm.shape[0]:,} unique MALDI voxels "
              f"from {args.maldi_file}")
        if maldi_mm.shape[0] == 0:
            raise SystemExit("[nystrom] --maldi-file matched 0 rows -- check "
                             "--maldi-filter / --maldi-coord-cols.")

    if args.maldi_file is not None and args.maldi_mode == "direct":
        query_source = "maldi"
        query_mm = maldi_mm
        n_oos = query_mm.shape[0]
        if args.max_oos > 0 and n_oos > args.max_oos:
            rng = np.random.default_rng(0)
            sel = rng.choice(n_oos, args.max_oos, replace=False)
            query_mm = query_mm[np.sort(sel)]
            n_oos = query_mm.shape[0]
            print(f"[nystrom] subsampled MALDI points -> {n_oos:,} for scoring")

        # No literal fine eigenvector exists at an off-grid MALDI coordinate,
        # so the FINE graph's own Nyström extension there stands in as the
        # reference (see the module docstring: this measures coarse-vs-fine
        # degradation at the real points, not absolute truth).
        psi_true, nn_dist_fine = nystrom_extend(fine, query_mm, nystrom_k)
        true = psi_true                                               # (Q, M)
        print(f"[nystrom] fine-graph reference NN distance: mean="
              f"{nn_dist_fine.mean():.3g}  median={np.median(nn_dist_fine):.3g}")

        psi_oos, nn_dist = nystrom_extend(coarse, query_mm, nystrom_k)
        pred = psi_oos @ T                                            # (Q, M)

    elif args.maldi_file is not None and args.maldi_mode == "distance-match":
        query_source = "grid-distance-matched"
        target_dist = nn_distance_to_coarse(coarse, maldi_mm, nystrom_k)
        cand_dist = nn_distance_to_coarse(coarse, fine.mm_coords[oos_fine], nystrom_k)
        n_target = args.max_oos if args.max_oos > 0 else maldi_mm.shape[0]
        sel = match_distance_distribution(cand_dist, target_dist, n_target,
                                          seed=0, n_bins=args.distance_match_bins)
        if sel.size == 0:
            raise SystemExit("[nystrom] distance-match found 0 grid points in "
                             "any MALDI distance bin -- coarse/fine standardized "
                             "spaces may be misaligned, or the grids are too "
                             "sparse near the MALDI distance regime.")
        if sel.size < n_target:
            print(f"[nystrom] WARNING: distance-match only found {sel.size:,}/"
                  f"{n_target:,} grid points covering the MALDI NN-distance "
                  f"distribution (some distance bins are underpopulated on the "
                  f"grid) -- the matched sample is short, not padded from a "
                  f"different distribution.")
        oos_fine = oos_fine[sel]
        n_oos = oos_fine.size
        matched_dist = cand_dist[sel]
        print(f"[nystrom] distance-matched {n_oos:,} grid points  "
              f"(target NN-dist mean={target_dist.mean():.3g} median="
              f"{np.median(target_dist):.3g}  matched mean="
              f"{matched_dist.mean():.3g} median={np.median(matched_dist):.3g})")

        psi_oos, nn_dist = nystrom_extend(coarse, fine.mm_coords[oos_fine], nystrom_k)
        pred = psi_oos @ T                                                # (Q, M)
        true = fine.eigvec[oos_fine].detach().cpu().numpy()              # (Q, M)

    else:
        query_source = "grid"
        if args.max_oos > 0 and n_oos > args.max_oos:
            rng = np.random.default_rng(0)
            sel = rng.choice(n_oos, args.max_oos, replace=False)
            oos_fine = oos_fine[np.sort(sel)]
            n_oos = oos_fine.size
            print(f"[nystrom] subsampled in-between nodes -> {n_oos:,} for scoring")

        psi_oos, nn_dist = nystrom_extend(coarse, fine.mm_coords[oos_fine], nystrom_k)
        pred = psi_oos @ T                                                # (Q, M)
        true = fine.eigvec[oos_fine].detach().cpu().numpy()              # (Q, M)

    rel_l2 = per_mode_rel_l2(pred, true)
    corr = np.abs(per_mode_corr(pred, true))
    ms, cum_err = cumulative_subspace_error(pred, true)
    nn_dist_np = nn_dist
    bw_units = nn_dist_np / max(args.bandwidth, 1e-30)

    overall_rel = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-30))

    # ---- report ----
    def band(arr, lo, hi):
        m = (np.arange(len(arr)) >= lo) & (np.arange(len(arr)) < hi)
        return float(arr[m].mean()) if m.any() else float("nan")

    print(f"\n[nystrom] test-point ({query_source}) NN distance to coarse graph: "
          f"mean={nn_dist_np.mean():.3g}  median={np.median(nn_dist_np):.3g}  "
          f"(in bandwidth units: mean={bw_units.mean():.2f})")
    ref_desc = ("fine-graph Nyström reference" if query_source == "maldi"
                else "true fine eigenbasis")
    print(f"\n[nystrom] reconstruction of the {ref_desc} at held-out "
          f"({query_source}) points")
    print(f"{'mode band':<14}{'mean |corr|':>14}{'mean rel-L2':>14}")
    print("-" * 42)
    M = args.modes
    for lo, hi, lbl in [(0, 10, "0-9"), (10, 50, "10-49"),
                        (50, 100, "50-99"), (100, M, f"100-{M-1}")]:
        if lo >= M:
            continue
        hi = min(hi, M)
        print(f"{lbl:<14}{band(corr, lo, hi):>14.4f}{band(rel_l2, lo, hi):>14.4f}")
    print("-" * 42)
    print(f"{'ALL':<14}{corr.mean():>14.4f}{rel_l2.mean():>14.4f}")
    print(f"\n[nystrom] overall subspace rel-Frob error (all {M} modes) = "
          f"{overall_rel:.4f}")

    # find the mode where correlation first drops below thresholds
    for thr in (0.99, 0.95, 0.9):
        below = np.where(corr < thr)[0]
        first = int(below[0]) if below.size else M
        print(f"[nystrom] |corr| stays >= {thr:.2f} up to mode {first}")

    if args.out is not None:
        _write_report(args.out, args, coarse, fine, n_shared, n_oos, floor_rel,
                      overall_rel, rel_l2, corr, ms, cum_err, bw_units,
                      build_seconds, query_source)


def _write_report(out, args, coarse, fine, n_shared, n_oos, floor_rel,
                  overall_rel, rel_l2, corr, ms, cum_err, bw_units,
                  build_seconds, query_source):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "query_source": query_source, "maldi_file": args.maldi_file,
        "coarse_stride": args.coarse_stride, "fine_stride": args.fine_stride,
        "modes": args.modes, "bandwidth": args.bandwidth, "k": args.k,
        "nystrom_k": (args.k if args.nystrom_k <= 0 else args.nystrom_k),
        "threshold": args.threshold,
        "coarse_nodes": int(coarse.n_nodes), "fine_nodes": int(fine.n_nodes),
        "shared_nodes": int(n_shared), "in_between_nodes": int(n_oos),
        "in_sample_floor_rel_err": floor_rel,
        "overall_subspace_rel_err": overall_rel,
        "mean_abs_corr": float(corr.mean()),
        "mean_rel_l2": float(rel_l2.mean()),
        "nn_dist_bw_units_mean": float(bw_units.mean()),
        "build_seconds": build_seconds,
        "per_mode_abs_corr": corr.tolist(),
        "per_mode_rel_l2": rel_l2.tolist(),
        "cumulative_modes": ms.tolist(),
        "cumulative_rel_err": cum_err.tolist(),
    }
    json_path = out.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[nystrom] wrote {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[nystrom] matplotlib unavailable, skipping plot: {e}")
        return

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(corr, lw=1.2)
    ax[0].set(title="|correlation| per fine mode", xlabel="fine mode",
              ylabel="|corr|", ylim=(0, 1.02))
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(np.maximum(rel_l2, 1e-6), lw=1.2, color="C1")
    ax[1].set(title="relative L2 error per fine mode", xlabel="fine mode",
              ylabel="rel L2")
    ax[1].grid(alpha=0.3)
    ax[2].plot(ms, cum_err, "-o", ms=3, color="C2")
    ax[2].set(title="cumulative subspace rel-Frob error", xlabel="# modes used",
              ylabel="rel Frob err")
    ax[2].grid(alpha=0.3)
    fig.suptitle(
        f"Nyström: stride {args.coarse_stride} → stride {args.fine_stride}  "
        f"({n_oos:,} held-out {query_source} points, {args.modes} modes, "
        f"bw={args.bandwidth}, floor={floor_rel:.2f})")
    fig.tight_layout()
    png_path = out.with_suffix(".png")
    fig.savefig(png_path, dpi=130)
    print(f"[nystrom] wrote {png_path}")


if __name__ == "__main__":
    main()
