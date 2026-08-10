#!/usr/bin/env python
# encoding: utf-8
"""
bump_support_report.py
======================

How many MALDI train/test points fall inside the manifold kernel's bump support?
If most points sit OUTSIDE the bump radius, their manifold features are zeroed
and the Riemann-Matern GP silently degrades to its Euclidean fallback / prior
mean -- so it can only do worse than a plain Matern.

ZERO-MISALIGNMENT DESIGN
------------------------
This script does NOT reimplement the graph, the coordinate normalization, the
KNN engine, or the bump. It imports and calls `visualize_laplacian.setup()`,
which builds the *exact* deployed objects from the *exact* same cache keys the
viewer uses:

  * ctx["knn"]            : the FAISS NearestNeighbors index (same engine the
                            kernel queries -- not a cKDTree)
  * ctx["matern_kernel"]  : the real RiemannMaternKernel, from which we read the
                            live graphbandwidth, bump_scale, bump_decay
  * ctx["coord_mean/std"] : the shared standardized space
  * ctx["reference_nodes"]: the graph node set

and it computes the nearest-node distance and bump weight EXACTLY as
RiemannKernel.features() / laplacian_test.evaluate_kernel_psd() do:

    edge_value, _ = ctx["knn"].search(test_points, 1)        # squared L2
    d            = edge_value.sqrt()                          # standardized z
    within       = d < bump_scale * graphbandwidth           # support
    bump         = bump_function(d, bump_scale*bw, bump_decay)

Pass it the SAME flags you pass visualize_laplacian (template/reference/stride/
threshold/knn-method/knn-k/graphbandwidth/laplacian-norm/num-modes/bump-*/...),
plus the MALDI parquet and your train/test parquet filters. Because the cache
keys are identical, the graph and eigvecs are loaded (not recomputed) for any
parameter set you've already run, and bump_scale/decay sweeps never touch them.

Run:
  python bump_support_report.py \
      --template-name allen_25um \
      --reference-file reference_image.npy \
      --annotations-file level_15annot.npy \
      --eigenvector-dir /home/casap/mlibra/output \
      --knn-method anatomical_atlas --knn-k 15 --n-list 1 \
      --stride 4 --threshold 5 \
      --graphbandwidth 0.1 --laplacian-norm symmetric --num-modes 1300 \
      --bump-scale 20 --bump-decay 0.01 \
      --device cuda \
      --maldi maindata_minimal.parquet \
      --train-filter '[("Section","in",[...])]' \
      --test-filter  '[("Section","in",[...])]' \
      --bump-scale-sweep 1 2 5 10 20 \
      --bump-decay-sweep  0.01 0.05 0.1 \
      --out-dir ./bump_report

Must be importable in the SAME environment as visualize_laplacian (torch, faiss,
manifold_gp, etc.). Distances are computed once per group; --bump-scale-sweep and
--bump-decay-sweep then re-bin/re-weight them cheaply (bump_scale sets the support
radius alpha=bump_scale*bw; bump_decay only reshapes the in-support weight).

To sweep bump_scale x bump_decay across MANY graph configs (stride/threshold/knn),
use manifold/benchmarks/bump_support_sweep.sh, which runs one setup() per graph config and
concatenates every summary.csv into OUT_DIR/summary_all.csv.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# This script moved from maldi/ to benchmarks/; its maldi sibling modules
# (imported by bare name below) live in ../maldi, so put that on sys.path.
# manifold_gp is pip-installed and needs no shim.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maldi"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manifold"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manifold" / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse the viewer's real machinery -- this is the whole point.
import visualize_laplacian as vl
from visualize_laplacian import setup, bump_function

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ---------------------------------------------------------------------------
# Args: the subset visualize_laplacian.setup() consumes (verbatim names /
# defaults), plus MALDI + report flags. Pass the SAME values you give the viewer.
# ---------------------------------------------------------------------------
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ---- consumed by setup() (mirror visualize_laplacian.parse_args) ----
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--region-bbox", type=int, nargs=6, default=None,
                   metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"))
    p.add_argument("--knn-method", choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted"],
                   default="anatomical_atlas")
    p.add_argument("--cross-region-inflation", type=float, default=100.0)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--n-probe", dest="n_probe", default="8",
                   help="FAISS IVF nprobe (int or 'sqrt', default 8). MUST be > 1 "
                        "when nlist > 1 -- nprobe=1 with an IVF index builds a "
                        "FRAGMENTED graph (~nlist components).")
    p.add_argument("--laplacian-norm", choices=["symmetric", "randomwalk"], default="symmetric")
    p.add_argument("--graphbandwidth", type=float, required=True)
    p.add_argument("--eigenvector-dir", required=True)
    p.add_argument("--num-modes", type=int, default=200)
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute-eigvecs", action="store_true")
    p.add_argument("--nu", type=int, default=2)
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--bump-scale", type=float, default=0.1)
    p.add_argument("--bump-decay", type=float, default=0.05)
    p.add_argument("--device", default="cuda")

    # ---- MALDI + report ----
    p.add_argument("--maldi", required=True, help="MALDI parquet (maindata_minimal.parquet)")
    p.add_argument("--train-filter", default=None,
                   help="python-literal pyarrow filter (paste config.section_filter)")
    p.add_argument("--test-filter", default=None,
                   help="python-literal pyarrow filter (paste config.test_filter)")
    p.add_argument("--bump-scale-sweep", type=float, nargs="+", default=None,
                   help="extra bump_scale values to re-bin the SAME distances against.")
    p.add_argument("--bump-decay-sweep", type=float, nargs="+", default=None,
                   help="extra bump_decay values to re-weight the SAME distances "
                        "against. decay changes the in-support weight profile, not "
                        "the support radius (which is bump_scale*bw).")
    p.add_argument("--max-points", type=int, default=None,
                   help="optional per-group cap (random subsample, seeded).")
    p.add_argument("--search-batch-size", type=int, default=50_000,
                   help="batch size for knn.search to bound device memory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="./bump_report")
    return vars(p.parse_args())


def parse_filter(s):
    if s is None or not str(s).strip():
        return None
    return ast.literal_eval(s)


# ---------------------------------------------------------------------------
# MALDI coords, standardized in the SAME space setup() produced
# ---------------------------------------------------------------------------
def load_maldi_std(parquet_path, filt, coord_mean, coord_std, device,
                   max_points=None, seed=0):
    cols = ["xccf", "yccf", "zccf"]
    df = pd.read_parquet(parquet_path, columns=cols, filters=filt)
    if df.shape[0] == 0:
        return None
    pts = torch.tensor(df[cols].to_numpy(), dtype=torch.float32)   # positional axes
    if max_points is not None and pts.shape[0] > max_points:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(pts.shape[0], generator=g)[:max_points]
        pts = pts[sel]
    pts = (pts - coord_mean.cpu()) / coord_std.cpu()
    return pts.to(device).contiguous()


# ---------------------------------------------------------------------------
# Nearest-node distance via the kernel's OWN knn (FAISS), exactly as features()
# ---------------------------------------------------------------------------
def nearest_node_distance(pts_std, knn, batch):
    out = []
    N = pts_std.shape[0]
    with torch.no_grad():
        for i in range(0, N, batch):
            ev, _ = knn.search(pts_std[i:i + batch], 1)   # squared L2 (E,1) tensor
            ev = ev if torch.is_tensor(ev) else torch.as_tensor(np.asarray(ev))
            out.append(ev[:, 0].float().sqrt().detach().cpu())
    return torch.cat(out).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_distance_summary(name, d, bw):
    N = d.shape[0]
    qs = [0, 1, 5, 25, 50, 75, 90, 95, 99, 100]
    pc = np.percentile(d, qs)
    print(f"\n========== {name}  (N = {N:,}) ==========")
    print(f"  nearest-node distance d (standardized z-units):")
    print(f"      min {pc[0]:.4g}  p1 {pc[1]:.4g}  p5 {pc[2]:.4g}  p25 {pc[3]:.4g}  "
          f"median {pc[4]:.4g}")
    print(f"      p75 {pc[5]:.4g}  p90 {pc[6]:.4g}  p95 {pc[7]:.4g}  p99 {pc[8]:.4g}  "
          f"max {pc[9]:.4g}  mean {d.mean():.4g}")
    print(f"      within one bandwidth (d < bw={bw:g}): "
          f"{int((d < bw).sum()):,}  ({100*float((d<bw).mean()):.1f}%)")


def report_at_scale(name, d, bw, bump_scale, decay):
    N = d.shape[0]
    alpha = bump_scale * bw
    within = int((d < alpha).sum())
    beyond = N - within
    within_half = int((d < 0.5 * alpha).sum())
    b = bump_function(torch.from_numpy(d), alpha, decay).cpu().numpy()
    b_in = b[d < alpha]
    bump_ge_half = int((b > 0.5).sum())
    pct = lambda n: 100.0 * n / N
    bmean = float(b_in.mean()) if b_in.size else float("nan")
    # one line per (scale, decay) so the nested sweep stays readable; the support
    # membership (within/beyond) is decay-independent, the bump-weight cols aren't.
    print(f"    scale={bump_scale:<5g} decay={decay:<7g} alpha={alpha:<8g}z  "
          f"within={pct(within):5.1f}%  within a/2={pct(within_half):5.1f}%  "
          f"beyond(zeroed)={pct(beyond):5.1f}%  bump_mean_in={bmean:.3f}  "
          f"bump>0.5={pct(bump_ge_half):5.1f}%")
    return dict(group=name, n=N, bw=bw, bump_scale=bump_scale, alpha=alpha,
                bump_decay=decay, within_support=within, beyond_support=beyond,
                within_half_alpha=within_half,
                frac_within_support=within / N, frac_beyond_support=beyond / N,
                bump_gt_half=bump_ge_half,
                bump_mean_in_support=float(b_in.mean()) if b_in.size else float("nan"),
                d_median=float(np.percentile(d, 50)), d_p95=float(np.percentile(d, 95)),
                d_p99=float(np.percentile(d, 99)), d_max=float(d.max()), d_mean=float(d.mean()))


def maybe_hist(dists, bw, alphas, coord_std, out_dir):
    if not HAVE_MPL or out_dir is None or not dists:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    alld = np.concatenate([d for _, d in dists])
    hi = max(float(np.percentile(alld, 99.5)), max(alphas) * 1.1)
    bins = np.linspace(0, hi, 80)
    for name, d in dists:
        ax.hist(d, bins=bins, histtype="step", linewidth=1.6, label=f"{name} (n={d.size:,})")
    ax.axvline(bw, color="green", ls=":", label=f"bw = {bw:g}")
    for a in sorted(set(alphas)):
        ax.axvline(a, color="red", ls="--", alpha=0.7)
    ax.set_xlabel(f"nearest graph-node distance (z-units;  1 z = {float(coord_std):.4f} mm)")
    ax.set_ylabel("count")
    ax.set_title("MALDI distance to nearest manifold node  (red = bump support radii)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(out_dir) / "distance_hist.png"
    fig.savefig(p, dpi=130)
    print(f"\n[plot] {p}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("bump_support_report")

    # Build the EXACT deployed graph / space / kernel (cache-keyed identically).
    ctx = setup(args, log)
    kernel = ctx["matern_kernel"]
    bw = float(kernel.graphbandwidth.squeeze().item())          # live value
    base_scale = float(getattr(kernel, "bump_scale", args["bump_scale"]))
    decay = float(getattr(kernel, "bump_decay", args["bump_decay"]))
    coord_mean, coord_std = ctx["coord_mean"], ctx["coord_std"]
    device = ctx["device"]

    scales = [base_scale] + (list(args["bump_scale_sweep"]) if args["bump_scale_sweep"] else [])
    seen = set(); scales = [s for s in scales if not (s in seen or seen.add(s))]
    decays = [decay] + (list(args["bump_decay_sweep"]) if args["bump_decay_sweep"] else [])
    seen = set(); decays = [x for x in decays if not (x in seen or seen.add(x))]
    alphas = [s * bw for s in scales]

    print(f"\nbw (graphbandwidth) = {bw:g} z   coord_std = {float(coord_std):.5f} mm/z")
    print("support radii: " + "   ".join(
        f"bump_scale={s:g} -> alpha={s*bw:g} z = {s*bw*float(coord_std):.3f} mm"
        for s in scales))
    print("bump decays: " + "  ".join(f"{x:g}" for x in decays))

    # Graph-config columns, so a multi-config sweep (bump_support_sweep.sh) can
    # concatenate every run's summary.csv into one self-describing table.
    cfg_meta = dict(knn_method=args["knn_method"], knn_k=args["knn_k"],
                    stride=args["stride"], threshold=args["threshold"],
                    cross_region_inflation=args["cross_region_inflation"])

    out_dir = Path(args["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    tf, ef = parse_filter(args["train_filter"]), parse_filter(args["test_filter"])
    if tf is None and ef is None:
        groups = [("all", None)]
    else:
        groups = [("train", tf), ("test", ef)]

    rows, dists = [], []
    for name, filt in groups:
        pts = load_maldi_std(args["maldi"], filt, coord_mean, coord_std, device,
                             args["max_points"], args["seed"])
        if pts is None:
            print(f"\n[{name}] no points (empty filter result)."); continue
        d = nearest_node_distance(pts, ctx["knn"], args["search_batch_size"])
        np.save(out_dir / f"dist_{name}.npy", d.astype(np.float32))
        dists.append((name, d))
        print_distance_summary(name, d, bw)
        on_graph = int((d < 1e-4).sum())
        print(f"      on-graph (d<1e-4, snapped/exact eigvec): {on_graph:,}  "
              f"({100*on_graph/d.shape[0]:.1f}%)")
        for s in scales:
            for dec in decays:
                rows.append({**cfg_meta, **report_at_scale(name, d, bw, s, dec)})

    maybe_hist(dists, bw, alphas, coord_std, out_dir)

    if rows:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(rows, f, indent=2)
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        print(f"\n[saved] {out_dir}/summary.csv, summary.json, dist_*.npy"
              + (", distance_hist.png" if HAVE_MPL else ""))
        worst = min(r["frac_within_support"] for r in rows)
        print(f"\nTakeaway: smallest within-support fraction across all "
              f"(group, bump_scale) = {100*worst:.1f}%.")
        if worst < 0.5:
            print("  In at least one setting >50% of points are OUTSIDE the bump "
                  "support -> manifold features zeroed there. Raise bump_scale/bw "
                  "or check coordinate alignment.")


if __name__ == "__main__":
    main()