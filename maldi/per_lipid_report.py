#!/usr/bin/env python
"""
Aggregate report over per-lipid GP runs produced by ``run_lgp_per_lipid.sh``
(submitted to the cloud via ``submit/run_submit_baselines.sh``).

Each run lands in ``<root>/<exp_name>/`` with:

    config.json    # the full args dict (records the fold via slices_dataset_file)
    metrics.csv    # one row per lipid: test_rmse_z, test_corr, test_r2, ...
    summary.json   # the run's own aggregate
    FAILED.txt     # present only if the run diverged

This script walks ``<root>``, finds every directory that has a ``metrics.csv``,
and emits:

  1. A PER-RUN table   (one row per run dir).
  2. A PER-FOLD table  (runs grouped by the fold their config points at).
  3. A TOTAL summary   (everything pooled).

Optionally dumps the long per-lipid table and the per-run/per-fold tables to CSV.

Usage
-----
    python maldi/per_lipid_report.py /path/to/experiment_batch_14
    python maldi/per_lipid_report.py /path/to/out --csv-dir ./report_out
    python maldi/per_lipid_report.py /path/to/out --sort test_r2 --metric test_r2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Per-lipid metric columns we summarise. Anything missing in a given run is
# simply skipped for that run.
METRIC_COLS = ["test_rmse_z", "test_corr", "test_r2", "mean_pred_std_z", "fit_sec"]


def derive_fold(run_dir: Path, config: dict | None) -> str:
    """Best-effort fold label for a run.

    Priority:
      1. config['slices_dataset_file'] basename stem  (e.g. 'fold_3')
      2. config['exp_prefix'] / leading FOLD-* token of exp_name
      3. the run-dir name prefix before '-manifold' / '-euclidean'
    """
    if config:
        sf = config.get("slices_dataset_file")
        if sf:
            return Path(str(sf)).stem  # 'fold_3', 'difficult', ...
        pref = config.get("exp_prefix")
        if pref:
            return str(pref)
        exp = config.get("exp_name")
        if exp:
            return _prefix_from_exp(str(exp))
    return _prefix_from_exp(run_dir.name)


def _prefix_from_exp(name: str) -> str:
    for sep in ("-manifold", "-euclidean"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name


def load_run(run_dir: Path) -> dict | None:
    """Load one run dir into a record, or None if it has no usable metrics."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_path)
    except Exception as ex:  # noqa: BLE001 - report and skip
        print(f"  ! skipping {run_dir.name}: unreadable metrics.csv ({ex})",
              file=sys.stderr)
        return None
    if df.empty:
        return None

    config = _load_json(run_dir / "config.json")
    summary = _load_json(run_dir / "summary.json")

    return {
        "run": run_dir.name,
        "fold": derive_fold(run_dir, config),
        "kernel_family": (config or {}).get("kernel_family")
        or (summary or {}).get("kernel_family", "?"),
        "failed": (run_dir / "FAILED.txt").exists(),
        "n_lipids": int(len(df)),
        "wall_time_sec": float((summary or {}).get("wall_time_sec", float("nan"))),
        "df": df,
    }


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def summarise(df: pd.DataFrame) -> dict:
    """Mean + median for each metric column present in df."""
    out: dict[str, float] = {}
    for col in METRIC_COLS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            out[f"{col}_median"] = float(np.nanmedian(vals)) if len(vals) else float("nan")
    return out


def build_tables(runs: list[dict], metric: str):
    # ---- long per-lipid table (one row per lipid, tagged with run+fold) ----
    long_parts = []
    for r in runs:
        d = r["df"].copy()
        d.insert(0, "run", r["run"])
        d.insert(1, "fold", r["fold"])
        d.insert(2, "kernel_family", r["kernel_family"])
        long_parts.append(d)
    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    # ---- per-run table ----
    per_run_rows = []
    for r in runs:
        row = {
            "run": r["run"],
            "fold": r["fold"],
            "kernel_family": r["kernel_family"],
            "failed": r["failed"],
            "n_lipids": r["n_lipids"],
            "wall_time_sec": r["wall_time_sec"],
        }
        row.update(summarise(r["df"]))
        per_run_rows.append(row)
    per_run = pd.DataFrame(per_run_rows)

    # ---- per-fold table (pool all lipids across runs of that fold) ----
    per_fold_rows = []
    if not long_df.empty:
        for fold, g in long_df.groupby("fold"):
            row = {
                "fold": fold,
                "n_runs": g["run"].nunique(),
                "n_lipids": int(len(g)),
            }
            row.update(summarise(g))
            per_fold_rows.append(row)
    per_fold = pd.DataFrame(per_fold_rows)

    # Sort per-fold by chosen metric median (descending for corr/r2, else as-is)
    sort_col = f"{metric}_median"
    if not per_fold.empty and sort_col in per_fold.columns:
        ascending = metric.endswith("rmse_z") or metric == "fit_sec"
        per_fold = per_fold.sort_values(sort_col, ascending=ascending)

    return long_df, per_run, per_fold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path,
                    help="Directory containing the per-run output dirs "
                         "(e.g. .../experiment_batch_14).")
    ap.add_argument("--metric", default="test_r2",
                    choices=METRIC_COLS,
                    help="Metric used to sort the per-fold table (default test_r2).")
    ap.add_argument("--sort", default="fold",
                    help="Column to sort the per-run table by (default 'fold').")
    ap.add_argument("--csv-dir", type=Path, default=None,
                    help="If set, write per_lipid_long.csv / per_run.csv / "
                         "per_fold.csv here.")
    ap.add_argument("--max-rows", type=int, default=200,
                    help="Cap rows printed in the per-run table (default 200).")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    # Any directory containing a metrics.csv is a run. rglob so nested layouts
    # (e.g. an extra fold subdir on S3) are handled too.
    run_dirs = sorted({p.parent for p in root.rglob("metrics.csv")})
    if not run_dirs:
        print(f"No metrics.csv found anywhere under {root}", file=sys.stderr)
        return 1

    runs = [r for r in (load_run(d) for d in run_dirs) if r is not None]
    if not runs:
        print(f"Found metrics.csv files under {root} but none were usable.",
              file=sys.stderr)
        return 1

    long_df, per_run, per_fold = build_tables(runs, args.metric)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    n_failed = int(per_run["failed"].sum())
    print("=" * 90)
    print(f"PER-LIPID GP REPORT  —  root: {root}")
    print(f"  runs: {len(runs)}   folds: {per_fold.shape[0]}   "
          f"lipid-rows: {len(long_df)}   failed runs: {n_failed}")
    print("=" * 90)

    # ---- per-fold ----
    print("\n### PER-FOLD SUMMARY (lipids pooled across that fold's runs)")
    if per_fold.empty:
        print("  (none)")
    else:
        print(per_fold.to_string(index=False))

    # ---- total ----
    print("\n### TOTAL SUMMARY (all lipids pooled)")
    total = {"n_runs": len(runs), "n_folds": per_fold.shape[0],
             "n_lipids": int(len(long_df))}
    total.update(summarise(long_df))
    total_df = pd.DataFrame([total])
    print(total_df.to_string(index=False))

    # ---- per-run ----
    print("\n### PER-RUN SUMMARY")
    pr = per_run
    if args.sort in pr.columns:
        ascending = not (args.sort.endswith("_mean") or args.sort.endswith("_median")) \
            or args.sort.endswith("rmse_z_mean") or args.sort.endswith("rmse_z_median")
        pr = pr.sort_values(args.sort, ascending=ascending, kind="stable")
    with pd.option_context("display.max_rows", args.max_rows):
        print(pr.head(args.max_rows).to_string(index=False))
        if len(pr) > args.max_rows:
            print(f"  ... {len(pr) - args.max_rows} more runs (raise --max-rows)")

    # ---- optional CSV dump ----
    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(args.csv_dir / "per_lipid_long.csv", index=False)
        per_run.to_csv(args.csv_dir / "per_run.csv", index=False)
        per_fold.to_csv(args.csv_dir / "per_fold.csv", index=False)
        total_df.to_csv(args.csv_dir / "total.csv", index=False)
        print(f"\nWrote CSVs to {args.csv_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
