#!/usr/bin/env python
# encoding: utf-8
"""
spectral_sweep_report.py
========================

Aggregate + rank the per-config rows produced by spectral_distance_sweep.py.

The sweep itself only writes <out-dir>/rows/<config-slug>.csv (one file per
config, each row a (lipid x distance-metric) measurement). When the sweep is fanned
out across many one-config jobs (submit/run_spectral_distance_sweep.sh), every job
drops its own rows file into a SHARED rows/ dir; this script collects them all once
the jobs finish and produces the report:

  <out-dir>/per_lipid.csv     every rows/*.csv concatenated
  <out-dir>/per_config.csv    one row per (config, metric): mean/median over lipids
  <out-dir>/ranked.csv        per_config pivoted to one row per config + sorted by
                              the objective (higher = better)

It is intentionally light (pandas/numpy only — no torch/faiss), so it runs
anywhere the rows are readable, independent of the heavy sweep environment.

RANKING
-------
--rank-stat picks which per-(config,metric) statistic to compare (default
decay_strength_quantile = equal-count binned decay strength, the honest metric
robust to spectral-distance pile-up). --objective picks the metric (or derived
gain) that defines the sort key:

  euclidean | geodesic | spectral | spectral_low<N>   a single distance metric
  spectral_gain                                       spectral - euclidean
  lowmode_gain                                        spectral_low - spectral
                                                        (needs exactly one low-mode
                                                         metric in the results)

RUN
---
  python spectral_sweep_report.py --out-dir ./spectral_sweep \
      --objective spectral --rank-stat decay_strength_quantile --top 20
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


# config-identifying columns carried verbatim from the rows into per_config / ranked
CFG_COLS = ["config", "knn_method", "normalization", "threshold", "knn_k",
            "graphbandwidth", "cross_region_inflation", "nu", "lengthscale",
            "stride", "num_modes", "feature_mode", "bump_scale", "bump_decay",
            "spectral_norm", "n_nodes", "off_support_frac", "reachable_frac"]


def load_rows(out_dir: Path) -> pd.DataFrame:
    rows_dir = out_dir / "rows"
    paths = sorted(glob.glob(str(rows_dir / "*.csv")))
    if not paths:
        raise SystemExit(f"no per-config rows found in {rows_dir}/*.csv "
                         f"(run the sweep first).")
    frames, bad = [], []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:  # truncated / mid-write file — skip, don't abort
            bad.append((Path(p).name, str(e)))
    if bad:
        for name, err in bad:
            print(f"  [warn] unreadable {name}: {err}")
    if not frames:
        raise SystemExit(f"all {len(paths)} rows file(s) in {rows_dir} were unreadable.")
    per_lipid = pd.concat(frames, ignore_index=True)
    print(f"[loaded] {len(frames)} rows file(s) -> {len(per_lipid):,} rows, "
          f"{per_lipid['config'].nunique()} config(s), "
          f"{per_lipid['lipid'].nunique()} lipid(s), "
          f"metrics={sorted(per_lipid['metric'].unique())}")
    return per_lipid


def aggregate_and_rank(per_lipid: pd.DataFrame, out_dir: Path,
                       rank_stat: str, objective: str, top: int):
    grp = (per_lipid.groupby(["config", "metric"])
           .agg(metric_modes=("metric_modes", "first"),
                decay_strength_mean=("decay_strength", "mean"),
                decay_strength_median=("decay_strength", "median"),
                decay_strength_quantile_mean=("decay_strength_quantile", "mean"),
                decay_strength_uniform_mean=("decay_strength_uniform", "mean"),
                vario_r2_mean=("vario_r2", "mean"),
                cov_curve_corr_mean=("cov_curve_corr", "mean"),
                n_lipids=("lipid", "nunique"))
           .reset_index())
    cfg_cols = [c for c in CFG_COLS if c in per_lipid.columns]
    cfg_meta = per_lipid[cfg_cols].drop_duplicates("config")
    per_config = grp.merge(cfg_meta, on="config", how="left")
    per_config.to_csv(out_dir / "per_config.csv", index=False)

    stat = f"{rank_stat}_mean"
    pivot = per_config.pivot_table(index="config", columns="metric", values=stat)
    pivot.columns = [f"{stat}__{m}" for m in pivot.columns]
    ranked = cfg_meta.merge(pivot.reset_index(), on="config", how="left")

    def col(metric):
        name = f"{stat}__{metric}"
        if name not in ranked.columns:
            raise SystemExit(f"objective metric {metric!r} not in results "
                             f"(have: {[c for c in ranked.columns if c.startswith(stat)]})")
        return ranked[name]

    if objective == "spectral_gain":
        ranked["objective"] = col("spectral") - col("euclidean")
    elif objective == "lowmode_gain":
        # spectral_low<N> minus full spectral. With multiple low-mode metrics this
        # is ambiguous, so require exactly one (else point at a specific name).
        lows = [c[len(f"{stat}__"):] for c in ranked.columns
                if c.startswith(f"{stat}__spectral_low")]
        if len(lows) != 1:
            raise SystemExit(
                f"lowmode_gain needs exactly one spectral_low metric, found {lows}. "
                f"Pass --objective <name> directly (e.g. --objective {lows[0] if lows else 'spectral_low64'}).")
        ranked["objective"] = col(lows[0]) - col("spectral")
    else:
        ranked["objective"] = col(objective)
    ranked = ranked.sort_values("objective", ascending=False)
    ranked.to_csv(out_dir / "ranked.csv", index=False)

    print("\n" + "=" * 80)
    print(f"Objective: {objective}  (rank-stat: {stat}, higher=better)")
    best = ranked.iloc[0]
    print(f"BEST = {best['objective']:.4f}   config: {best['config']}")
    print("-" * 80)
    for c in ["knn_method", "normalization", "threshold", "knn_k", "graphbandwidth",
              "cross_region_inflation", "nu", "lengthscale", "stride", "num_modes",
              "feature_mode", "bump_scale", "bump_decay", "spectral_norm"]:
        if c in best:
            print(f"  {c:<24} {best[c]}")
    print("=" * 80)
    show = (["config", "stride", "num_modes"]
            + [c for c in ranked.columns if c.startswith(stat)] + ["objective"])
    show = [c for c in show if c in ranked.columns]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(f"\nTop {min(top, len(ranked))}:\n")
        print(ranked[show].head(top).to_string(index=False))
    print(f"\n[saved] {out_dir}/per_lipid.csv, per_config.csv, ranked.csv")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="./spectral_sweep",
                   help="sweep output dir holding rows/*.csv (report files are "
                        "written here too)")
    p.add_argument("--objective", default="spectral",
                   help="metric to rank configs by (euclidean|geodesic|spectral|"
                        "spectral_low<N>), or 'spectral_gain' (spectral-euclidean) "
                        "or 'lowmode_gain' (spectral_low-spectral)")
    p.add_argument("--rank-stat", default="decay_strength_quantile",
                   choices=["decay_strength_quantile", "decay_strength_uniform",
                            "decay_strength", "vario_r2", "cov_curve_corr"],
                   help="which per-(config,metric) stat to rank by. Default is the "
                        "equal-count (quantile) binned decay strength — the honest "
                        "metric robust to spectral-distance pile-up.")
    p.add_argument("--top", type=int, default=15)
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    if not out_dir.exists():
        raise SystemExit(f"--out-dir {out_dir} does not exist.")
    per_lipid = load_rows(out_dir)
    per_lipid.to_csv(out_dir / "per_lipid.csv", index=False)
    aggregate_and_rank(per_lipid, out_dir, a.rank_stat, a.objective, a.top)


if __name__ == "__main__":
    main()
