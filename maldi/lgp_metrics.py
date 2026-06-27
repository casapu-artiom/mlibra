#!/usr/bin/env python
"""Shared per-lipid metric computation + metrics.csv cache for the whole-brain
latent-GP runs (lgp / manifold / baselines).

A run dir holds ``<split>/predictions.npy`` and ``<split>/true_values.npy`` —
(n, P) arrays in ORIGINAL (un-log, un-norm) scale. This module computes, per
output column (lipid):

    r2    1 - SSE/SST              corr  Pearson correlation
    rmse  sqrt(mean (y-yhat)^2)    mae   mean |y-yhat|

and caches the table as a ``metrics.csv`` next to the run (``metrics.csv`` for
the test split, ``metrics_<split>.csv`` otherwise), so the experiment writes it
at the end of a run and ``lgp_report.py`` reads it instead of recomputing.

Public API
----------
    metrics_csv_path(run_dir, split)            -> Path
    per_lipid_metrics(pred, true, chunk_rows)   -> DataFrame   (from arrays)
    write_metrics(run_dir, split, ...)          -> DataFrame   (compute + cache)
    load_or_compute(run_dir, split, ..., force) -> DataFrame   (cache or compute+cache)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Metric columns and which ones are lower-is-better (for downstream sorting).
METRIC_COLS = ["r2", "corr", "rmse", "mae"]
LOWER_IS_BETTER = {"rmse", "mae"}
DEFAULT_CHUNK_ROWS = 200_000


def metrics_csv_path(run_dir, split: str = "test") -> Path:
    """Where the per-lipid metrics table for `split` lives (test -> metrics.csv)."""
    run_dir = Path(run_dir)
    return run_dir / ("metrics.csv" if split == "test" else f"metrics_{split}.csv")


def per_lipid_metrics(pred, true, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> pd.DataFrame:
    """Per-column r2/corr/rmse/mae from two (n, P) arrays (ndarray or memmap).

    Accumulates sufficient statistics over row-chunks so the full arrays are
    never promoted to float64 at once. NaN/inf entries are excluded per (row,
    column) — the per-column count `n` reflects only the finite pairs used.
    """
    Y, P = true, pred
    if getattr(P, "shape", None) != getattr(Y, "shape", None) or Y.ndim != 2:
        raise ValueError(
            f"shape mismatch pred{getattr(P, 'shape', None)} vs true{getattr(Y, 'shape', None)}")

    n, p = Y.shape
    cnt = np.zeros(p)
    Sy = np.zeros(p); Sp = np.zeros(p)
    Syy = np.zeros(p); Spp = np.zeros(p); Syp = np.zeros(p)
    Se = np.zeros(p); Sae = np.zeros(p)

    for s in range(0, n, chunk_rows):
        e = min(n, s + chunk_rows)
        y = np.asarray(Y[s:e], dtype=np.float64)
        q = np.asarray(P[s:e], dtype=np.float64)
        m = np.isfinite(y) & np.isfinite(q)
        y = np.where(m, y, 0.0)
        q = np.where(m, q, 0.0)
        cnt += m.sum(0)
        Sy += y.sum(0); Sp += q.sum(0)
        Syy += (y * y).sum(0); Spp += (q * q).sum(0); Syp += (y * q).sum(0)
        d = y - q
        Se += (d * d).sum(0); Sae += np.abs(d).sum(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        nn = np.where(cnt > 0, cnt, np.nan)
        rmse = np.sqrt(Se / nn)
        mae = Sae / nn
        sst = Syy - Sy * Sy / nn          # total sum of squares of y (about its mean)
        cov = Syp - Sy * Sp / nn
        var_p = Spp - Sp * Sp / nn
        corr = cov / np.sqrt(sst * var_p)
        r2 = 1.0 - Se / sst

    return pd.DataFrame({
        "lipid": np.arange(p),
        "n": cnt.astype(np.int64),
        "r2": r2, "corr": corr, "rmse": rmse, "mae": mae,
    })


def _metrics_from_files(run_dir, split, chunk_rows) -> pd.DataFrame | None:
    run_dir = Path(run_dir)
    pred = run_dir / split / "predictions.npy"
    true = run_dir / split / "true_values.npy"
    if not (pred.exists() and true.exists()):
        return None
    P = np.load(pred, mmap_mode="r")
    Y = np.load(true, mmap_mode="r")
    return per_lipid_metrics(P, Y, chunk_rows)


def write_metrics(run_dir, split: str = "test", *, pred=None, true=None,
                  lipid_names=None, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> pd.DataFrame | None:
    """Compute per-lipid metrics — from the given in-memory arrays if provided,
    else from the saved ``<split>/{predictions,true_values}.npy`` — and write the
    ``metrics.csv`` cache. Returns the DataFrame, or None if no data was found."""
    if pred is not None and true is not None:
        df = per_lipid_metrics(np.asarray(pred), np.asarray(true), chunk_rows)
    else:
        df = _metrics_from_files(run_dir, split, chunk_rows)
    if df is None:
        return None
    if lipid_names is not None and len(lipid_names) == len(df):
        df.insert(1, "lipid_name", [str(x) for x in lipid_names])
    out = metrics_csv_path(run_dir, split)
    try:
        df.to_csv(out, index=False)
    except Exception as ex:  # noqa: BLE001
        print(f"  ! could not write {out} ({ex})", file=sys.stderr)
    return df


def load_or_compute(run_dir, split: str = "test", *, chunk_rows: int = DEFAULT_CHUNK_ROWS,
                    lipid_names=None, force: bool = False) -> pd.DataFrame | None:
    """Return per-lipid metrics: read the cached ``metrics.csv`` if it exists
    (and not ``force``), otherwise compute from the npy arrays AND write the
    cache so subsequent reads are free."""
    csv = metrics_csv_path(run_dir, split)
    if csv.exists() and not force:
        try:
            return pd.read_csv(csv)
        except Exception:  # noqa: BLE001 - corrupt cache: fall through and recompute
            pass
    return write_metrics(run_dir, split, lipid_names=lipid_names, chunk_rows=chunk_rows)
