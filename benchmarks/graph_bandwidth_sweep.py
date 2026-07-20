#!/usr/bin/env python
# encoding: utf-8
"""
graph_bandwidth_sweep.py
========================

Sweep graph types / parameters and visualize, for each, the edge-length
distribution and how the heat-kernel affinity w_ij behaves across a grid of
graph bandwidths -- so you can pick a bw that is neither saturated (all
w_ij ~ 1, geometry washed out -> ~combinatorial Laplacian) nor dead (all
w_ij ~ 0, graph fragments -> degenerate spectrum).

ALIGNMENT
---------
Graphs are built with the SAME helpers and SAME cache keys as
visualize_laplacian.setup() / the experiment (KnnGraphCache, crop_or_stride,
coord_norm_from_reference), so the edge_value this analyzes is identical to what
the deployed kernel consumes. For faiss_atlas_weighted the full deployed
refinement is applied in-memory over the cached base faiss graph — root handling
(dissolve/cross/ignore) -> inflate_cross_region_edges -> denoise_labels_majority_vote
-> prune_cross_region_edges — matching lgp_experiment_per_lipid.py. The eigensolve
is skipped (edge weights don't need it), so this runs CPU-only and needs no eigvec
cache.

The affinity is computed EXACTLY as GraphLaplacianOperator.adjacency_unnorm_mat:

        w_ij = exp(-edge_value_ij / (4 * bw^2))            # edge_value = squared
                                                            # distance in z-space

What you get, per (graph_type, params):
  * edge-distance percentiles (sqrt(edge_value), standardized z-units), both as a
    combined overlay across configs and as a per-config small-multiples panel
    (edge_length_distributions.png / edge_length_distributions_per_graph.png)
  * for faiss_atlas_weighted: the same edge-length distribution BEFORE vs AFTER
    cross_region_inflation/denoise/prune, per config
    (edge_length_distributions_before_after.png; edge_d_median_raw in the CSV)
  * for faiss_atlas_weighted: one PNG per config
    (affinity_within_vs_cross_<label>.png) with the original (pre-inflation)
    distance split within/cross on top, and the resulting affinity
    w=exp(-d^2/4bw^2) split the same way at bw in {0.01,0.02,0.05,0.1,0.5,1.0}
    below, median/ratio annotated per panel
  * for each bw on the grid: percentiles of w_ij (5/25/50/75/95), the fraction
    saturated (w>0.9) and dead (w<0.01), and Sum_ij w_ij
  * a suggested bw from the median edge length, and an intrinsic-dimension
    estimate from the slope of log(Sum w) vs log(bw)
  * connected-component count vs bw (an edge survives iff w_ij >= eps): how the
    EFFECTIVE graph fragments as bw shrinks -- the fragmentation counterpart of
    frac_dead, and the multiplicity of the ~zero Laplacian eigenvalue. Reports
    the smallest bw that keeps the graph in one piece.

Plots (PNG) + tables (CSV) are written to --out-dir.

Run:
  python graph_bandwidth_sweep.py \
      --template-name allen_25um \
      --reference-file reference_image.npy \
      --annotations-file level_15annot.npy \
      --eigenvector-dir /home/casap/mlibra/output \
      --knn-methods faiss anatomical_atlas faiss_atlas_weighted \
      --knn-ks 15 \
      --strides 4 \
      --inflations 100 \
      --threshold 5 \
      --bandwidths 0.005 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5 1.0 \
      --device cpu \
      --out-dir ./bw_sweep

Must be importable in the same env as visualize_laplacian (torch, faiss,
manifold_gp). matplotlib optional but recommended.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Moved from maldi/ to benchmarks/; the maldi sibling `utils` (imported inside a
# function below) lives in ../maldi, so put that on sys.path at module load.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maldi"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

PCTS = [5, 25, 50, 75, 95]


# ===========================================================================
# Graph construction -- mirrors visualize_laplacian.setup() up to (and incl.)
# the KNN graph + optional atlas inflation. Heavy imports are lazy so the
# stats/plot helpers below can be imported & tested without the package.
# ===========================================================================
def build_graph(cfg: dict, common: dict, log: logging.Logger):
    import torch
    from manifold_gp.utils.nearest_neighbors import KnnGraphCache, make_key as make_graph_key
    from manifold_gp.utils.anatomical_knn import (
        inflate_cross_region_edges, labels_for_nodes_from_sub_atlas,
        dissolve_root_labels, denoise_labels_majority_vote,
        prune_cross_region_edges,
    )
    from utils import crop_or_stride_volume, reference_ccf_from_subvolume, coord_norm_from_reference

    method = cfg["method"]; k = cfg["k"]; stride = cfg["stride"]
    inflation = cfg.get("inflation"); threshold = cfg["threshold"]
    device = common["device"]
    n_list = common["n_list"]; bbox = common["region_bbox"]

    template_full = common["_template_full"]
    annotations_full = common["_annotations_full"]

    if bbox is not None:
        raise ValueError(
            "--region-bbox is no longer supported: crop_or_stride_volume() "
            "dropped the region_bbox argument. Use --strides to subsample."
        )
    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_full, annotations_full, stride,
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    reference_nodes_mm = torch.tensor(reference_ccf, dtype=torch.float32)
    coord_mean, coord_std = coord_norm_from_reference(template_full)
    reference_nodes = ((reference_nodes_mm - coord_mean) / coord_std).to(device)

    graphs = KnnGraphCache(cache_dir=Path(common["eigenvector_dir"]) / "knn", verbose=True)
    graph_key_parts = {
        "template": common["template_name"],
        "stride": (stride if bbox is None else 1),
        "thresh": threshold, "method": method, "k": k, "nlist": n_list,
        "bbox": (tuple(bbox) if bbox is not None else None),
    }
    if method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
    graph_key = make_graph_key(graph_key_parts)

    edge_value_raw = None   # set below for faiss_atlas_weighted (pre-inflation);
                            # for faiss / anatomical_atlas raw == final (no inflation)
    cross_mask = None       # set below for faiss_atlas_weighted; None = no atlas split
    cross_mask_raw = None   # cross split aligned to edge_value_raw (pre-prune edge
                            # count); NOT the same array as cross_mask whenever
                            # prune_cross_region_edges changes the edge count below
    if method == "faiss":
        knn, ei, ev = graphs.train_or_load(
            key=graph_key, method="faiss", coords=reference_nodes,
            k=k, nlist=n_list, extra=graph_key_parts, device=device,
            force_recompute=common["force_recompute_graph"],
        )
    elif method == "anatomical_atlas":
        knn, ei, ev = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas", volume=sub_volume,
            threshold=threshold, atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=k, nlist=n_list, extra=graph_key_parts,
            device=device, force_recompute=common["force_recompute_graph"],
        )
    elif method == "faiss_atlas_weighted":
        base = dict(graph_key_parts); base["method"] = "faiss"
        knn, ei, ev = graphs.train_or_load(
            key=make_graph_key(base), method="faiss", coords=reference_nodes,
            k=k, nlist=n_list, extra=base, device=device,
            force_recompute=common["force_recompute_graph"],
        )
        edge_value_raw = ev.detach().cpu().numpy().astype(np.float64).copy()
        node_labels = labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, threshold)
        # Mirror the deployed pipeline (lgp_experiment_per_lipid.py, faiss_atlas_
        # weighted): root handling -> soft cross-region inflation -> optional label
        # denoise -> optional hard prune. Same helpers, same order, so the
        # edge_value analyzed here matches what the kernel would consume.
        #   root_handling='dissolve' (default, = production): fold the label-0
        #     'root' catch-all into the nearest region, then edges are NOT inflated
        #     via zero-as-cross; 'cross' = legacy (root inflated as cross);
        #     'ignore' = keep root as its own label, no zero-as-cross inflation.
        root_mode = cfg.get("root_handling", "dissolve")
        if root_mode == "dissolve":
            node_labels = dissolve_root_labels(
                node_labels, reference_nodes.detach().cpu().numpy())
        # Cross split on the topology as it stands right before inflation --
        # same edge count/order as edge_value_raw above (inflate reweights in
        # place, doesn't touch edge_index), so this stays aligned even though
        # prune_cross_region_edges will later shrink ei/ev for the FINAL split.
        labels_raw = np.asarray(node_labels)
        s_idx0 = ei[0].detach().cpu().numpy(); d_idx0 = ei[1].detach().cpu().numpy()
        cross_mask_raw = labels_raw[s_idx0] != labels_raw[d_idx0]
        if root_mode == "cross":
            cross_mask_raw = cross_mask_raw | (labels_raw[s_idx0] == 0) | (labels_raw[d_idx0] == 0)
        ei, ev, _ = inflate_cross_region_edges(
            ei, ev, node_labels, inflation=float(inflation),
            treat_zero_as_cross=(root_mode == "cross"),
        )
        n_denoise = int(cfg.get("denoise", 0) or 0)
        prune = float(cfg.get("prune", 0.0) or 0.0)
        if n_denoise > 0:
            node_labels = denoise_labels_majority_vote(node_labels, ei, n_denoise)
        if prune > 0.0:
            ei, ev = prune_cross_region_edges(
                ei, ev, node_labels, prune, zero_is_region=False)
        # cross/within split on the FINAL (post denoise/prune) topology + labels,
        # for the affinity-space before/after plot below.
        labels_arr = np.asarray(node_labels)
        s_idx = ei[0].detach().cpu().numpy(); d_idx = ei[1].detach().cpu().numpy()
        cross_mask = labels_arr[s_idx] != labels_arr[d_idx]
        if root_mode == "cross":
            cross_mask = cross_mask | (labels_arr[s_idx] == 0) | (labels_arr[d_idx] == 0)
    else:
        raise ValueError(f"unknown method {method!r}")

    if cross_mask is None and method == "faiss" and sub_atlas is not None:
        # Diagnostic-only atlas split for the plain (uninflated) faiss graph:
        # shows what within/cross would look like on the UNMODIFIED graph --
        # the null case where nothing has deliberately separated the two
        # groups. Node ordering here matches faiss's reference_nodes coords,
        # same as the faiss_atlas_weighted base graph above; not extended to
        # anatomical_atlas since its node set/ordering isn't verified to match.
        diag_labels = labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, threshold)
        s_idx = ei[0].detach().cpu().numpy(); d_idx = ei[1].detach().cpu().numpy()
        cross_mask = (diag_labels[s_idx] != diag_labels[d_idx]) \
            | (diag_labels[s_idx] == 0) | (diag_labels[d_idx] == 0)

    edge_value = ev.detach().cpu().numpy().astype(np.float64)   # squared dist, z-space
    edge_index = ei.detach().cpu().numpy()                      # (2, E) node pairs
    if edge_value_raw is None:
        edge_value_raw = edge_value
    if cross_mask_raw is None:
        cross_mask_raw = cross_mask   # no prune ever ran -> raw/final topology identical
    return dict(
        label=cfg["label"], edge_value=edge_value, edge_value_raw=edge_value_raw,
        edge_index=edge_index, cross_mask=cross_mask, cross_mask_raw=cross_mask_raw,
        coord_std=float(coord_std), n_nodes=int(knn.x.shape[0]),
        n_edges=int(edge_value.shape[0]),
    )


# ===========================================================================
# Pure stats (no torch / package -- testable standalone)
# ===========================================================================
def affinity(edge_value_sq: np.ndarray, bw: float) -> np.ndarray:
    """w_ij = exp(-d^2 / (4 bw^2)) -- identical to adjacency_unnorm_mat."""
    return np.exp(-edge_value_sq / (4.0 * bw * bw))


def edge_distance_stats(edge_value_sq: np.ndarray) -> dict:
    d = np.sqrt(edge_value_sq)
    qs = np.percentile(d, PCTS)
    out = {f"edge_d_p{p}": float(v) for p, v in zip(PCTS, qs)}
    out["edge_d_min"] = float(d.min()); out["edge_d_max"] = float(d.max())
    out["edge_d_mean"] = float(d.mean())
    out["edge_d_median"] = float(np.median(d))
    return out


def w_stats_over_bw(edge_value_sq: np.ndarray, bw_grid) -> pd.DataFrame:
    rows = []
    for bw in bw_grid:
        w = affinity(edge_value_sq, bw)
        qs = np.percentile(w, PCTS)
        rows.append(dict(
            bw=float(bw),
            **{f"w_p{p}": float(v) for p, v in zip(PCTS, qs)},
            w_mean=float(w.mean()),
            frac_saturated=float((w > 0.9).mean()),
            frac_dead=float((w < 0.01).mean()),
            sum_w=float(w.sum()),
        ))
    return pd.DataFrame(rows)


def components_over_bw(edge_index, n_nodes, edge_value_sq, bw_grid,
                       eps: float = 0.01) -> pd.DataFrame:
    """Connected components of the graph once heat-killed edges are dropped
    (an edge survives iff w_ij >= eps). Topology is fixed by kNN, but a small bw
    numerically zeroes the long edges first, so as bw shrinks the effective graph
    fragments. Each extra component is one more ~zero Laplacian eigenvalue (a
    spurious mode) -- so this is the fragmentation counterpart of frac_dead.
    Reports, per bw: the component count, the largest-component node fraction,
    and the fraction of fully isolated nodes."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    src, dst = edge_index[0], edge_index[1]
    rows = []
    for bw in bw_grid:
        keep = affinity(edge_value_sq, bw) >= eps
        s, d = src[keep], dst[keep]
        adj = sp.coo_matrix((np.ones(s.shape[0], np.int8), (s, d)),
                            shape=(n_nodes, n_nodes))
        n_comp, labels = csg.connected_components(adj + adj.T, directed=False)
        counts = np.bincount(labels, minlength=n_comp)
        rows.append(dict(
            bw=float(bw),
            n_components=int(n_comp),
            largest_cc_frac=float(counts.max() / n_nodes),
            frac_isolated=float((counts == 1).sum() / n_nodes),
        ))
    return pd.DataFrame(rows)


def estimate_dim(bw_grid, sum_w):
    """Intrinsic-dim estimate: Sum_ij exp(-d^2/4bw^2) ~ bw^D in the scaling
    regime, so D = d log(Sum w) / d log(bw). Report local slopes + the max
    (steepest, most reliable) one."""
    lb = np.log(np.asarray(bw_grid, float))
    ls = np.log(np.asarray(sum_w, float) + 1e-300)
    slopes = np.gradient(ls, lb)
    return slopes, float(np.nanmax(slopes))


def suggested_bw(edge_stats: dict) -> float:
    """Median edge has weight ~ e^-1 when 4 bw^2 = median(d^2) ~ median(d)^2,
    i.e. bw = median(d)/2."""
    return edge_stats["edge_d_median"] / 2.0


# ===========================================================================
# Plots
# ===========================================================================
def plot_w_percentiles(results, sweeps, bw_grid, out_dir):
    if not HAVE_MPL:
        return
    n = len(results)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax, res, sw in zip(axes.ravel(), results, sweeps):
        ax.fill_between(sw["bw"], sw["w_p25"], sw["w_p75"], alpha=0.25,
                        label="w 25-75%")
        ax.plot(sw["bw"], sw["w_p50"], lw=2, label="w median")
        ax.plot(sw["bw"], sw["w_p5"], lw=1, ls="--", alpha=0.7)
        ax.plot(sw["bw"], sw["w_p95"], lw=1, ls="--", alpha=0.7)
        ax.axhline(0.9, color="red", ls=":", lw=1, label="saturated (0.9)")
        ax.axhline(0.01, color="gray", ls=":", lw=1, label="dead (0.01)")
        bw_sug = suggested_bw(edge_distance_stats(res["edge_value"]))
        ax.axvline(bw_sug, color="green", ls="-.", lw=1.2,
                   label=f"suggested bw={bw_sug:.3g}")
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("graph bandwidth bw (z-units)"); ax.set_ylabel("w_ij")
        ax.set_title(res["label"], fontsize=9)
        ax.legend(fontsize=6, loc="center left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Heat-kernel affinity w_ij percentiles vs graph bandwidth")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "w_percentiles_vs_bw.png", dpi=130)
    plt.close(fig)


def plot_window(results, sweeps, out_dir):
    if not HAVE_MPL:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for res, sw in zip(results, sweeps):
        a1.plot(sw["bw"], sw["frac_saturated"], marker="o", ms=3, label=res["label"])
        a2.plot(sw["bw"], sw["frac_dead"], marker="o", ms=3, label=res["label"])
    for a, t in [(a1, "fraction saturated (w>0.9)"), (a2, "fraction dead (w<0.01)")]:
        a.set_xscale("log"); a.set_xlabel("bw (z-units)"); a.set_ylabel(t)
        a.set_ylim(-0.02, 1.02); a.legend(fontsize=7)
    a1.axhline(0.5, color="k", ls=":", lw=0.8)
    fig.suptitle("Usable bandwidth window: avoid saturated (left=too big) and dead (right=too small)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "usable_window_vs_bw.png", dpi=130)
    plt.close(fig)


def plot_edge_distributions(results, out_dir):
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    alld = np.concatenate([np.sqrt(r["edge_value"]) for r in results])
    hi = float(np.percentile(alld, 99.5))
    bins = np.linspace(0, hi, 80)
    for r in results:
        d = np.sqrt(r["edge_value"])
        ax.hist(d, bins=bins, histtype="step", lw=1.6, density=True,
                label=f"{r['label']} (E={r['n_edges']:,})")
    ax.set_xlabel("edge length sqrt(edge_value) (standardized z-units)")
    ax.set_ylabel("density")
    ax.set_title("Edge-length distribution by graph")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "edge_length_distributions.png", dpi=130)
    plt.close(fig)


def plot_edge_distributions_per_config(results, out_dir):
    """Small-multiples version of plot_edge_distributions: one panel per graph
    config with its own bin range, so a config with a narrow/near-delta edge
    length distribution isn't flattened by others' wider spread on a shared
    axis (lattice graphs commonly have only a handful of discrete edge
    lengths -- see edge_distance_stats percentiles)."""
    if not HAVE_MPL:
        return
    n = len(results)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax, res in zip(axes.ravel(), results):
        d = np.sqrt(res["edge_value"])
        es = edge_distance_stats(res["edge_value"])
        hi = max(float(np.percentile(d, 99.5)), 1e-9)
        bins = np.linspace(0, hi, 60)
        ax.hist(d, bins=bins, color="tab:blue", alpha=0.8)
        for p in (5, 25, 50, 75, 95):
            ax.axvline(es[f"edge_d_p{p}"], color="k", ls=":", lw=0.8, alpha=0.5)
        bw_sug = suggested_bw(es)
        ax.axvline(bw_sug, color="green", ls="-.", lw=1.3,
                   label=f"suggested bw={bw_sug:.3g}")
        ax.set_xlabel("edge length sqrt(edge_value) (z-units)")
        ax.set_ylabel("count")
        ax.set_title(f"{res['label']}  (E={res['n_edges']:,})", fontsize=8)
        ax.legend(fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Edge-length distribution per graph config (dotted = p5/25/50/75/95)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "edge_length_distributions_per_graph.png", dpi=130)
    plt.close(fig)


def plot_edge_distributions_before_after(results, out_dir):
    """Per-config small multiples comparing the raw geometric edge-length
    distribution (pre cross_region_inflation) against the final one actually
    fed to the kernel (post inflation/denoise/prune). For faiss / anatomical_atlas
    configs the two are identical (no inflation applied) -- included anyway so
    the grid lines up 1:1 with the other per-config plot. Distributions are
    density-normalized since pruning can change edge counts between the two."""
    if not HAVE_MPL:
        return
    n = len(results)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax, res in zip(axes.ravel(), results):
        d_raw = np.sqrt(res["edge_value_raw"])
        d_post = np.sqrt(res["edge_value"])
        same = (d_raw.shape == d_post.shape) and np.allclose(d_raw, d_post)
        hi = max(float(np.percentile(d_post, 99.5)),
                 float(np.percentile(d_raw, 99.5)), 1e-9)
        bins = np.linspace(0, hi, 60)
        ax.hist(d_raw, bins=bins, density=True, histtype="step", lw=1.6,
                color="tab:gray", label=f"before (E={d_raw.size:,})")
        if not same:
            ax.hist(d_post, bins=bins, density=True, histtype="step", lw=1.6,
                    color="tab:red", label=f"after (E={d_post.size:,})")
        ax.set_xlabel("edge length (z-units)")
        ax.set_ylabel("density")
        title = res["label"] + ("  (no inflation)" if same else "")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Edge-length distribution before vs after cross-region inflation/denoise/prune")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "edge_length_distributions_before_after.png", dpi=130)
    plt.close(fig)


def plot_within_cross_by_bw(results, out_dir,
                            bw_values=(0.01, 0.02, 0.05, 0.1, 0.5, 1.0)):
    """Simple, one-config-per-figure view (kept deliberately plain after the
    combined grid version proved hard to read): top panel is the ORIGINAL
    edge distance (pre cross_region_inflation), split within- vs cross-region
    -- since they're geometrically the same lattice, this should look like
    one overlapping distribution. Below it, a fixed small grid of bandwidths
    shows the resulting AFFINITY w=exp(-d^2/4bw^2) (using the FINAL edge_value
    -- inflated for faiss_atlas_weighted, unmodified for plain faiss) split
    the same way, with median/ratio annotated in each title so the numbers
    are readable even when one side collapses to a spike matplotlib can
    barely render. Runs for any config with a cross_mask: faiss_atlas_weighted
    (the real inflated split) and plain faiss (the null case -- since nothing
    there deliberately separates within/cross, expect the two distributions,
    and the ratio, to stay close to each other at every bw). One PNG per
    config."""
    if not HAVE_MPL:
        return
    targets = [r for r in results if r.get("cross_mask") is not None]
    for res in targets:
        cross = res["cross_mask"]                  # aligned to edge_value (final, post-prune)
        cross_raw = res.get("cross_mask_raw", cross)  # aligned to edge_value_raw (pre-prune)
        d_raw = np.sqrt(res["edge_value_raw"])

        mosaic = [["dist", "dist", "dist"],
                 ["bw0", "bw1", "bw2"],
                 ["bw3", "bw4", "bw5"]]
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(12, 10))

        ax = axd["dist"]
        hi = max(float(np.percentile(d_raw, 99.5)), 1e-9)
        bins_d = np.linspace(0, hi, 60)
        ax.hist(d_raw[~cross_raw], bins=bins_d, density=True, alpha=0.6,
                color="tab:blue", label="within")
        ax.hist(d_raw[cross_raw], bins=bins_d, density=True, alpha=0.6,
                color="tab:red", label="cross")
        ax.set_title("Original edge distance (pre-inflation) -- within vs cross",
                     fontsize=12)
        ax.set_xlabel("distance (z-units)"); ax.set_ylabel("density")
        ax.legend(fontsize=9)

        bw_keys = ["bw0", "bw1", "bw2", "bw3", "bw4", "bw5"]
        for key, bw in zip(bw_keys, bw_values):
            ax = axd[key]
            w = affinity(res["edge_value"], bw)
            w_in, w_cr = w[~cross], w[cross]
            bins_w = np.linspace(0, 1, 40)
            ax.hist(w_in, bins=bins_w, density=True, alpha=0.6, color="tab:blue")
            ax.hist(w_cr, bins=bins_w, density=True, alpha=0.6, color="tab:red")
            m_in, m_cr = float(np.median(w_in)), float(np.median(w_cr))
            ratio = (m_in / m_cr) if m_cr > 0 else float("inf")
            ax.set_title(f"bw={bw:g}\nmedian within={m_in:.2g}  cross={m_cr:.2g}  "
                        f"ratio={ratio:.3g}x", fontsize=10)
            ax.set_xlabel("w_ij"); ax.set_ylabel("density")

        fig.suptitle(f"{res['label']}: distance -> affinity, within vs cross-region",
                    fontsize=13)
        fig.tight_layout()
        safe_label = res["label"].replace("/", "_")
        fig.savefig(Path(out_dir) / f"affinity_within_vs_cross_{safe_label}.png",
                   dpi=130)
        plt.close(fig)


def plot_dimension_curve(results, sweeps, out_dir):
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for res, sw in zip(results, sweeps):
        _, dim = estimate_dim(sw["bw"].to_numpy(), sw["sum_w"].to_numpy())
        ax.plot(sw["bw"], sw["sum_w"], marker="o", ms=3,
                label=f"{res['label']}  (dim~{dim:.1f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bw (z-units)"); ax.set_ylabel("Sum_ij w_ij(bw)")
    ax.set_title("Diffusion-maps curve: slope of log(Sum w) vs log(bw) ~ intrinsic dim")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "dimension_curve.png", dpi=130)
    plt.close(fig)


def plot_components_vs_bw(results, sweeps, out_dir):
    if not HAVE_MPL:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for res, sw in zip(results, sweeps):
        a1.plot(sw["bw"], sw["n_components"], marker="o", ms=3, label=res["label"])
        a2.plot(sw["bw"], sw["largest_cc_frac"], marker="o", ms=3, label=res["label"])
    a1.axhline(1, color="k", ls=":", lw=0.8, label="connected (1 comp)")
    a1.set_yscale("log"); a1.set_ylabel("# connected components (w >= eps)")
    a2.set_ylabel("largest-component node fraction"); a2.set_ylim(-0.02, 1.02)
    for a in (a1, a2):
        a.set_xscale("log"); a.set_xlabel("bw (z-units)"); a.legend(fontsize=7)
    fig.suptitle("Graph fragmentation vs bandwidth: below the knee the effective graph shatters")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "components_vs_bw.png", dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None)
    p.add_argument("--eigenvector-dir", required=True, help="root holding the knn/ cache")
    p.add_argument("--thresholds", type=int, nargs="+", default=[5],
                   help="tissue threshold(s) on the reference (reference > t); "
                        "swept like the other params. Feeds the graph cache key.")
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--region-bbox", type=int, nargs=6, default=None,
                   metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"))
    p.add_argument("--force-recompute-graph", action="store_true")

    p.add_argument("--knn-methods", nargs="+",
                   choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted"],
                   default=["faiss", "anatomical_atlas"])
    p.add_argument("--knn-ks", type=int, nargs="+", default=[15])
    p.add_argument("--strides", type=int, nargs="+", default=[4])
    p.add_argument("--inflations", type=float, nargs="+", default=[100.0],
                   help="cross-region inflation values (used only for faiss_atlas_weighted)")
    p.add_argument("--root-handling", nargs="+",
                   choices=["dissolve", "cross", "ignore"], default=["dissolve"],
                   help="atlas label-0 'root' handling (faiss_atlas_weighted only), "
                        "swept. 'dissolve' (default, = production) folds root into "
                        "the nearest region; 'cross' = legacy zero-as-cross; "
                        "'ignore' keeps root without zero-as-cross inflation.")
    p.add_argument("--denoise-labels", type=int, nargs="+", default=[0],
                   help="majority-vote label-smoothing passes before the prune "
                        "(faiss_atlas_weighted only). PAIRED by index with "
                        "--prune-cross-region (denoise_labels[i] goes with "
                        "prune_cross_region[i], not a cartesian product) -- the "
                        "two must have equal length. 0 = off.")
    p.add_argument("--prune-cross-region", type=float, nargs="+", default=[0.0],
                   help="fraction of cross-region edges to HARD-remove, "
                        "connectivity-preserving (faiss_atlas_weighted only). "
                        "PAIRED by index with --denoise-labels (same length "
                        "required). 0 = off. E.g. --denoise-labels 0 3 3 "
                        "--prune-cross-region 0.0 0.95 0.97 sweeps 3 variants: "
                        "off, and two (denoise=3, prune) combos.")

    p.add_argument("--bandwidths", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1,
                            0.15, 0.2, 0.3, 0.5, 1.0])
    p.add_argument("--component-threshold", type=float, default=0.01,
                   help="an edge counts as alive when w_ij >= this (matches the "
                        "'dead' line); used to count connected components vs bw. "
                        "Default 0.01.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="./bw_sweep")
    return p.parse_args()


def enumerate_configs(args):
    if len(args.denoise_labels) != len(args.prune_cross_region):
        raise ValueError(
            "--denoise-labels and --prune-cross-region must have the same "
            "length (paired by index, not a cartesian product): got "
            f"{args.denoise_labels} vs {args.prune_cross_region}"
        )
    dn_pr_pairs = list(zip(args.denoise_labels, args.prune_cross_region))

    cfgs = []
    for method, k, stride, thresh in itertools.product(
            args.knn_methods, args.knn_ks, args.strides, args.thresholds):
        if method == "faiss_atlas_weighted":
            for infl, root, (dn, pr) in itertools.product(
                    args.inflations, args.root_handling, dn_pr_pairs):
                lbl = f"atlasw_k{k}_s{stride}_t{thresh}_infl{infl:g}_{root}"
                if dn > 0:
                    lbl += f"_dn{dn}"
                if pr > 0:
                    lbl += f"_pr{pr:g}"
                cfgs.append(dict(method=method, k=k, stride=stride, threshold=thresh,
                                 inflation=infl, root_handling=root,
                                 denoise=dn, prune=pr, label=lbl))
        else:
            short = {"faiss": "faiss", "anatomical_atlas": "atlas"}[method]
            cfgs.append(dict(method=method, k=k, stride=stride, threshold=thresh,
                             label=f"{short}_k{k}_s{stride}_t{thresh}"))
    return cfgs


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("graph_bandwidth_sweep")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as _np
    common = dict(
        device=args.device, n_list=args.n_list,
        region_bbox=tuple(args.region_bbox) if args.region_bbox else None,
        eigenvector_dir=args.eigenvector_dir, template_name=args.template_name,
        force_recompute_graph=args.force_recompute_graph,
        _template_full=_np.load(args.reference_file),
        _annotations_full=_np.load(args.annotations_file) if args.annotations_file else None,
    )

    bw_grid = sorted(args.bandwidths)
    configs = enumerate_configs(args)
    print(f"Sweeping {len(configs)} graph(s) x {len(bw_grid)} bandwidths.\n")

    results, sweeps, edge_rows, sweep_rows = [], [], [], []
    for cfg in configs:
        if cfg["method"] != "faiss_atlas_weighted" and common["_annotations_full"] is None \
                and cfg["method"] == "anatomical_atlas":
            log.warning(f"skip {cfg['label']}: anatomical_atlas needs --annotations-file")
            continue
        try:
            res = build_graph(cfg, common, log)
        except Exception as e:
            log.warning(f"skip {cfg['label']}: {e}")
            continue
        es = edge_distance_stats(res["edge_value"])
        sw = w_stats_over_bw(res["edge_value"], bw_grid)
        comp = components_over_bw(res["edge_index"], res["n_nodes"],
                                  res["edge_value"], bw_grid,
                                  eps=args.component_threshold)
        sw = sw.merge(comp, on="bw")
        _, dim = estimate_dim(sw["bw"].to_numpy(), sw["sum_w"].to_numpy())
        bw_sug = suggested_bw(es)
        # smallest bw at which the effective graph is still a single component
        _conn = comp.loc[comp["n_components"] == 1, "bw"]
        bw_min_conn = float(_conn.min()) if len(_conn) else float("nan")

        es_raw_median = float(np.median(np.sqrt(res["edge_value_raw"])))

        results.append(res); sweeps.append(sw)
        edge_rows.append(dict(label=res["label"], n_nodes=res["n_nodes"],
                              n_edges=res["n_edges"], coord_std_mm=res["coord_std"],
                              suggested_bw=bw_sug, bw_min_connected=bw_min_conn,
                              dim_estimate=dim, edge_d_median_raw=es_raw_median,
                              **es))
        sw2 = sw.copy(); sw2.insert(0, "label", res["label"])
        sweep_rows.append(sw2)

        print(f"=== {res['label']}  (N={res['n_nodes']:,}, E={res['n_edges']:,}, "
              f"1z={res['coord_std']:.3f}mm) ===")
        print(f"  edge length (z): median={es['edge_d_median']:.4g}  "
              f"p5={es['edge_d_p5']:.4g}  p95={es['edge_d_p95']:.4g}  max={es['edge_d_max']:.4g}")
        print(f"  suggested bw ~ median/2 = {bw_sug:.4g} z ({bw_sug*res['coord_std']:.3f} mm)"
              f"   intrinsic-dim estimate ~ {dim:.1f}")
        if np.isnan(bw_min_conn):
            _top = comp.iloc[-1]
            print(f"  NEVER a single component in the swept bw range: "
                  f"{int(_top['n_components'])} comps even at bw={_top['bw']:g} "
                  f"(the kNN graph itself is disconnected; eps={args.component_threshold:g})")
        else:
            print(f"  stays connected (1 component) down to bw ~ {bw_min_conn:.4g} z"
                  f"   (eps={args.component_threshold:g})")
        # show where saturation/dead cross the data
        for _, row in sw.iterrows():
            flag = ""
            if row["frac_saturated"] > 0.9: flag = "  <-- SATURATED (geometry washed out)"
            elif row["frac_dead"] > 0.5: flag = "  <-- mostly DEAD (fragmenting)"
            print(f"    bw={row['bw']:<6g}  w 5/50/95 = "
                  f"{row['w_p5']:.3f}/{row['w_p50']:.3f}/{row['w_p95']:.3f}  "
                  f"sat={row['frac_saturated']:.2f} dead={row['frac_dead']:.2f}  "
                  f"comps={int(row['n_components'])}{flag}")
        print()

    if not results:
        print("No graphs built. Check args / annotations file.")
        return

    pd.DataFrame(edge_rows).to_csv(out_dir / "edge_and_suggested_bw.csv", index=False)
    pd.concat(sweep_rows, ignore_index=True).to_csv(out_dir / "w_percentiles_by_bw.csv", index=False)

    plot_w_percentiles(results, sweeps, bw_grid, out_dir)
    plot_window(results, sweeps, out_dir)
    plot_edge_distributions(results, out_dir)
    plot_edge_distributions_per_config(results, out_dir)
    plot_edge_distributions_before_after(results, out_dir)
    plot_within_cross_by_bw(results, out_dir)
    plot_dimension_curve(results, sweeps, out_dir)
    plot_components_vs_bw(results, sweeps, out_dir)

    print(f"[saved] {out_dir}/edge_and_suggested_bw.csv, w_percentiles_by_bw.csv"
          + (", *.png" if HAVE_MPL else " (matplotlib unavailable: no plots)"))


if __name__ == "__main__":
    main()