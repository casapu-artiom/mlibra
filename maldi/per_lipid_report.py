#!/usr/bin/env python
"""
Aggregate report over per-lipid GP runs produced by ``run_lgp_per_lipid.sh``
(submitted to the cloud via ``submit/run_submit_baselines.sh``).

Each run lands in ``<root>/<exp_name>/`` with:

    config.json    # the full args dict (records the fold via slices_dataset_file)
    metrics.csv    # one row per lipid: test_rmse_z, test_corr, test_r2, ...
    summary.json   # the run's own aggregate
    FAILED.txt     # present only if the run diverged

This script walks one or more ``<root>`` dirs, finds every directory that has a
``metrics.csv`` (runs from all roots are pooled; a ``source`` column tags each
when >1 root is given), and emits:

  1. A PER-RUN table   (one row per run dir).
  2. A PER-FOLD table  (runs grouped by the fold their config points at).
  3. A TOTAL summary   (everything pooled).

Optionally dumps the long per-lipid table and the per-run/per-fold tables to CSV.

parcelgp runs
-------------
Runs from ``parcelgp/run_parcel.sh`` are ordinary per-lipid runs with an extra
``--parcel-field``, and the parcel factor wraps the SAME kernel — so they report
``kernel_family='euclidean'`` and used to be indistinguishable from the baseline
they ablate. Their kernel_family is now suffixed ``+parcel`` (so the arms group
separately everywhere), and a ``parcel`` column showing the field/rank they used
appears in the per-run table whenever such a run is loaded.

Comparison with the whole-brain ("full") LGP models
---------------------------------------------------
Pass ``--full-lgp-root`` (one or more roots of whole-brain runs, i.e. dirs
holding ``<split>/predictions.npy`` as scored by ``lgp_report.py``) to fold those
models in for a head-to-head, RESTRICTED to the lipids the per-lipid runs
trained (joined on ``lipid_name``). They then appear:
  * as extra ``[full] <family>`` rows in the PER-LIPID x PER-MODEL table,
  * in a FULL-LGP vs PER-LIPID leaderboard (pooled over trained lipids), and
  * in an UPLIFT table: how much each full model adds ON TOP OF the best single
    per-lipid GP for the same lipid (mean/median delta + win-rate) — the point of
    the whole comparison.
Metrics compared are r2, corr and a standardized rmse: the cached whole-brain
``rmse`` is in original units, but the per-lipid ``test_rmse_z`` is the RMSE on
the standardized target, which equals sqrt(1 - r2), so it is derived from r2 for
the full models (rather than left blank). The per-run / per-fold / total tables
stay per-lipid-only (unchanged).

Usage
-----
    python maldi/per_lipid_report.py /path/to/experiment_batch_14
    python maldi/per_lipid_report.py out_a out_b out_c          # multiple roots pooled
    python maldi/per_lipid_report.py /path/to/out --csv-dir ./report_out
    python maldi/per_lipid_report.py /path/to/out --sort test_r2 --metric test_r2
    # compare the per-lipid GPs against the whole-brain LGP/manifold models:
    python maldi/per_lipid_report.py /path/to/per_lipid_out \
        --full-lgp-root /path/to/whole_brain_out \
        --full-lgp-lipid-names-file .../available_lipids.npy
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Per-lipid metric columns we summarise. Anything missing in a given run is
# simply skipped for that run.
METRIC_COLS = ["test_rmse_z", "test_corr", "test_r2", "mean_pred_std_z", "fit_sec"]

# Candidate columns that identify a lipid within a metrics.csv, in priority order.
LIPID_ID_COLS = ["lipid_name", "slug", "lipid_global_idx"]

def _nan_agg(fn, vals) -> float:
    """Apply a nan-reduction (nanmean/median/min/max) without the RuntimeWarning
    an all-NaN or empty slice raises — full-LGP rows have no ``test_rmse_z``, so
    that column is legitimately all-NaN for them."""
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
    return float(fn(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


# Whole-brain ("full") LGP metric columns (from lgp_metrics) mapped onto the
# per-lipid schema, so the two families land in the SAME columns. Only r2 and
# corr are directly comparable: the per-lipid rmse is z-scored (`test_rmse_z`)
# while the whole-brain `rmse` is in original units, so it is deliberately NOT
# carried over (it would compare apples to oranges).
FULL_LGP_METRIC_MAP = {"r2": "test_r2", "corr": "test_corr"}

# The per-lipid runs' kernel_family is always a real kernel (euclidean / manifold /
# eigenmap / spectral). The whole-brain ("full") models are NOT per-lipid kernels,
# so for those the kernel_family column instead carries a human-readable
# description of the model itself, keyed off the lgp_report family label.
_FULL_LGP_DESC = {
    "lgp": "whole-brain Euclidean (Matern) latent GP",
    "manifold": "whole-brain Riemann manifold latent GP (inducing-point)",
    "spectral": "whole-brain Riemann manifold latent GP (spectral / weight-space eigenbasis)",
}
_SOTA_DESC = {
    "ntf": "Neural Transcriptomic Field (hash-grid INR)",
    "spa3d": "spatial-pattern GCN (SPE + z-aware)",
    "deepspatial": "DeepSpatial optimal-transport",
}


def full_lgp_description(family: str) -> str:
    """Readable description of a whole-brain model, from its lgp_report family
    label (lgp / manifold / spectral / gplfr-<base> / sota-<method> /
    baseline-<model>, optionally with a '+parcel' suffix). Falls back to the
    family string for anything unrecognised."""
    if family.endswith("+parcel"):
        return (full_lgp_description(family[: -len("+parcel")])
                + " + reference-only parcel factor")
    if family in _FULL_LGP_DESC:
        return _FULL_LGP_DESC[family]
    if family.startswith("gplfr"):
        base = family.split("-", 1)[1] if "-" in family else None
        return f"GP latent-factor regression ({base} base)" if base \
            else "GP latent-factor regression"
    if family.startswith("sota-"):
        method = family.split("-", 1)[1]
        return _SOTA_DESC.get(method, f"SOTA model ({method})")
    if family.startswith("baseline"):
        model = family.split("-", 1)[1] if "-" in family else "baseline"
        return f"{model} baseline"
    return family


def _lipid_col(df: pd.DataFrame) -> str | None:
    """Return the column identifying a lipid, or None if none is present."""
    for col in LIPID_ID_COLS:
        if col in df.columns:
            return col
    return None


def derive_parcel(run_dir: Path, config: dict | None) -> str:
    """The parcelgp factor the run carried, ``''`` if it carried none.

    A parcelgp run (``parcelgp/run_parcel.sh`` -> ``--parcel-field``) wraps the
    parcel factor around the SAME kernel, so it still records
    ``kernel_family='euclidean'``: nothing tells it apart from the baseline it
    ablates unless the parcel field is read out explicitly. The label is the
    field file's stem (which carries its build parameters) plus the embedding
    rank.
    """
    field = (config or {}).get("parcel_field")
    if not field:
        # Runs synced without config.json: the runner puts the same settings in
        # exp_name, so the directory name still shows it is a parcel run. Only
        # the field is read back — config.py appends n_pixels to the directory
        # name, so '-r8' + '10' is indistinguishable from a rank of 810.
        m = re.search(r"-parcel([^-]+)", run_dir.name, re.IGNORECASE)
        return m.group(1) if m else ""
    bits = [Path(str(field)).stem]
    if (config or {}).get("parcel_rank") is not None:
        bits.append(f"r{config['parcel_rank']}")
    if not (config or {}).get("parcel_per_task", True):  # --parcel-shared-B
        bits.append("sharedB")
    return "/".join(bits)


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


# Leading fold token in a run/exp name, e.g. 'FOLD-3-', 'fold_3-', 'difficult-'.
_FOLD_PREFIX_RE = re.compile(r"^(fold[-_]?\d+|difficult)[-_]", re.IGNORECASE)
# The same token ANYWHERE in the name, with the separator on either side. Some
# runners (maldi/run_final.sh, parcelgp/run_latent.sh) put the splits-file stem
# in the MIDDLE of exp_name as well, so stripping only the leading token leaves
# every fold of one configuration as a model of its own.
_FOLD_TOKEN_RE = re.compile(r"[-_]?\b(fold[-_]?\d+|difficult)\b[-_]?", re.IGNORECASE)


def derive_model(run_dir: Path, config: dict | None) -> str:
    """Full model-configuration label, stable across folds.

    This is the run/exp name with EVERY fold token stripped, wherever it sits, so
    the same configuration scored under different folds collapses to one model.
    Prefers config['exp_name'] (the canonical config string) over the dir name.
    """
    name = str((config or {}).get("exp_name") or run_dir.name)
    return _FOLD_TOKEN_RE.sub("-", name).strip("-_")


def load_run(run_dir: Path, source: str = "") -> dict | None:
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

    # A parcelgp run reports the base kernel_family and would pool with the very
    # run it ablates; the '+parcel' suffix keeps the two arms apart.
    parcel = derive_parcel(run_dir, config)
    kernel_family = (config or {}).get("kernel_family") \
        or (summary or {}).get("kernel_family", "?")

    return {
        "run": run_dir.name,
        "source": source,
        "fold": derive_fold(run_dir, config),
        "model": derive_model(run_dir, config),
        "kernel_family": f"{kernel_family}+parcel" if parcel else kernel_family,
        "parcel": parcel,
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


def summarise_across_folds(g: pd.DataFrame) -> dict:
    """Mean / median / min / max (range) for each metric across a group's rows.

    Used for the per-lipid x per-model aggregation, where each row is the same
    lipid scored under one fold, so these stats describe spread across folds.
    """
    out: dict[str, float] = {}
    for col in METRIC_COLS:
        if col in g.columns:
            out[f"{col}_mean"] = _nan_agg(np.nanmean, g[col])
            out[f"{col}_median"] = _nan_agg(np.nanmedian, g[col])
            out[f"{col}_min"] = _nan_agg(np.nanmin, g[col])
            out[f"{col}_max"] = _nan_agg(np.nanmax, g[col])
    return out


def build_per_lipid_model(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (lipid, full model configuration), aggregating across folds.

    The model key is the full config string (run name minus the fold token), so
    each row describes how one lipid scores under one specific configuration,
    pooled over whatever folds ran it. Reports mean, median and the min..max
    range for each metric, plus how many folds contributed.
    """
    if long_df.empty:
        return pd.DataFrame()
    lipid_col = _lipid_col(long_df)
    if lipid_col is None:
        return pd.DataFrame()

    rows = []
    for (lipid, model), g in long_df.groupby([lipid_col, "model"], dropna=False):
        row = {
            "lipid": lipid,
            "model": model,
            "kernel_family": g["kernel_family"].iloc[0],
            "n_folds": g["fold"].nunique(),
            "n_runs": g["run"].nunique(),
        }
        row.update(summarise_across_folds(g))
        rows.append(row)
    out = pd.DataFrame(rows)

    sort_col = f"{metric}_median"
    if not out.empty and sort_col in out.columns:
        ascending = metric.endswith("rmse_z") or metric == "fit_sec"
        out = out.sort_values(["lipid", sort_col], ascending=[True, ascending],
                              kind="stable")
    return out


def _long_from_runs(runs: list[dict], multi_source: bool = False) -> pd.DataFrame:
    """Stack every run's per-lipid df into one long table, tagged with the run's
    identity columns (run / fold / model / kernel_family, + source if pooled)."""
    long_parts = []
    for r in runs:
        d = r["df"].copy()
        d.insert(0, "run", r["run"])
        d.insert(1, "fold", r["fold"])
        d.insert(2, "model", r["model"])
        d.insert(3, "kernel_family", r["kernel_family"])
        if multi_source:
            d.insert(1, "source", r.get("source", ""))
        long_parts.append(d)
    return pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()


def build_tables(runs: list[dict], metric: str):
    # Only surface the `source` (root) column when runs span >1 root, else it's
    # constant clutter. Same for the parcel column: only when a parcelgp run is in.
    multi_source = len({r.get("source", "") for r in runs}) > 1
    any_parcel = any(r.get("parcel") for r in runs)

    # ---- long per-lipid table (one row per lipid, tagged with run+fold) ----
    long_df = _long_from_runs(runs, multi_source)

    # ---- per-run table ----
    per_run_rows = []
    for r in runs:
        row = {"run": r["run"]}
        if multi_source:
            row["source"] = r.get("source", "")
        row.update({
            "fold": r["fold"],
            "kernel_family": r["kernel_family"],
            "failed": r["failed"],
            **({"parcel": r.get("parcel") or "-"} if any_parcel else {}),
            "n_lipids": r["n_lipids"],
            "wall_time_sec": r["wall_time_sec"],
        })
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

    # ---- per-lipid x per-model table (aggregated across folds) ----
    per_lipid_model = build_per_lipid_model(long_df, metric)

    return long_df, per_run, per_fold, per_lipid_model


# ---------------------------------------------------------------------------
# Optional: whole-brain ("full") LGP models, restricted to the trained lipids
# ---------------------------------------------------------------------------
def load_full_lgp_run(run_dir: Path, split: str, chunk_rows: int,
                      lipid_names: np.ndarray | None, trained_lipids: set[str],
                      force: bool, family_filter: str | None) -> dict | None:
    """Load one whole-brain LGP run and shape it like a per-lipid run record,
    but restricted to the lipids that the per-lipid runs actually trained.

    Reuses ``lgp_metrics.load_or_compute`` (the same per-lipid r2/corr/rmse/mae
    the whole-brain report scores) and ``lgp_report``'s fold/family derivation,
    then renames the comparable columns onto the per-lipid schema. Returns None
    if the run has no usable metrics, no lipid names to join on, or none of its
    lipids overlap the trained set.
    """
    # Imported lazily so per_lipid_report has no hard dependency on the
    # whole-brain report unless --full-lgp-root is used.
    from lgp_metrics import load_or_compute as wb_load_or_compute
    from lgp_report import (derive_family as wb_family, derive_fold as wb_fold,
                            derive_model as wb_model, _load_args as wb_load_args)

    df = wb_load_or_compute(run_dir, split, chunk_rows=chunk_rows,
                            lipid_names=lipid_names, force=force)
    if df is None or df.empty:
        return None

    args = wb_load_args(run_dir)
    family = wb_family(run_dir, args)
    # The full CONFIG label (exp_name minus the fold token), so each distinct
    # full model gets its OWN line -- two configs of the same family (e.g. two
    # manifold variants) stay separate instead of collapsing into one row.
    model_cfg = wb_model(run_dir, args)
    if family_filter and family_filter.lower() not in family.lower():
        return None

    # Need lipid names to join against the trained set. Prefer the cached
    # column; else label from --full-lgp-lipid-names-file if it lines up.
    if "lipid_name" not in df.columns and lipid_names is not None and len(lipid_names) == len(df):
        df = df.copy()
        df.insert(1, "lipid_name", [str(x) for x in lipid_names])
    if "lipid_name" not in df.columns:
        print(f"  ! {run_dir.name}: no lipid_name (pass --full-lgp-lipid-names-file); skipping",
              file=sys.stderr)
        return None

    df = df[df["lipid_name"].astype(str).isin(trained_lipids)].copy()
    if df.empty:
        return None

    # Map the comparable whole-brain metrics onto the per-lipid column names.
    df = df.rename(columns=FULL_LGP_METRIC_MAP)
    # The per-lipid `test_rmse_z` is the RMSE on the STANDARDIZED target. On the
    # whole-brain side r2 = 1 - SSE/SST (SST about the test mean), so the
    # standardized RMSE is exactly sqrt(1 - r2) -- derive it so the rmse column
    # is populated and comparable (rather than left blank because the cached
    # whole-brain `rmse` is in original units).
    if "test_r2" in df.columns:
        df["test_rmse_z"] = np.sqrt(np.clip(1.0 - df["test_r2"], 0.0, None))
    keep = ["lipid_name", *FULL_LGP_METRIC_MAP.values(), "test_rmse_z"]
    df = df[[c for c in keep if c in df.columns]]

    return {
        "run": run_dir.name,
        "fold": wb_fold(run_dir, args),
        # One line per full-model CONFIG (the '[full] ' prefix also keeps it from
        # colliding with a per-lipid config of the same name).
        "model": f"[full] {model_cfg}",
        # Not a per-lipid kernel: describe the model family instead of a kernel.
        "kernel_family": full_lgp_description(family),
        "failed": (run_dir / "FAILED.txt").exists(),
        "n_lipids": int(len(df)),
        "wall_time_sec": float("nan"),
        "df": df,
        "is_full": True,
    }


def load_full_lgp_runs(roots: list[Path], split: str, chunk_rows: int,
                       lipid_names: np.ndarray | None, trained_lipids: set[str],
                       force: bool, family_filter: str | None) -> list[dict]:
    """Discover + load every whole-brain run under `roots` (a run dir is any dir
    holding ``<split>/predictions.npy``), restricted to the trained lipids."""
    run_dirs: set[Path] = set()
    for root in roots:
        for p in root.rglob(f"{split}/predictions.npy"):
            run_dirs.add(p.parent.parent)
    out = []
    for d in sorted(run_dirs):
        r = load_full_lgp_run(d, split, chunk_rows, lipid_names, trained_lipids,
                              force, family_filter)
        if r is not None:
            print(f"  full-lgp scored {r['model']:22s} {r['run']} "
                  f"({r['n_lipids']} trained lipids)", file=sys.stderr)
            out.append(r)
    return out


def build_model_comparison(combined_long: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Leaderboard over the trained lipids: one row per model (per-lipid configs
    AND ``[full] …`` whole-brain families), pooled over lipids/folds, on the
    comparable metrics (test_r2 / test_corr). Marks which rows are full models."""
    if combined_long.empty:
        return pd.DataFrame()
    lipid_col = _lipid_col(combined_long)
    rows = []
    for model, g in combined_long.groupby("model", dropna=False):
        row = {
            "model": model,
            "kernel_family": g["kernel_family"].iloc[0],
            "is_full": str(model).startswith("[full] "),
            "n_lipids": g[lipid_col].nunique() if lipid_col else int(len(g)),
            "n_folds": g["fold"].nunique(),
            "n_runs": g["run"].nunique(),
        }
        for col in ("test_r2", "test_corr", "test_rmse_z"):
            if col in g.columns:
                row[f"{col}_mean"] = _nan_agg(np.nanmean, g[col])
                row[f"{col}_median"] = _nan_agg(np.nanmedian, g[col])
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_col = f"{metric}_median"
    if not out.empty and sort_col in out.columns:
        ascending = metric.endswith("rmse_z") or metric == "fit_sec"
        out = out.sort_values(sort_col, ascending=ascending, kind="stable")
    return out


def build_uplift(per_lipid_model: pd.DataFrame) -> pd.DataFrame:
    """How much each ``[full] …`` model adds ON TOP OF a single per-lipid GP.

    For every trained lipid the baseline is the BEST single per-lipid GP for that
    lipid (the max median test_r2 / test_corr over all per-lipid configs) — a
    conservative bar, so a positive uplift means the full model beats the best
    single GP, not just an average one. Each full model is then compared to that
    baseline per lipid and aggregated: median full vs baseline, the mean/median
    delta, and the win-rate (fraction of lipids where the full model wins).
    """
    if per_lipid_model.empty or "test_r2_median" not in per_lipid_model.columns:
        return pd.DataFrame()
    df = per_lipid_model.copy()
    df["_is_full"] = df["model"].astype(str).str.startswith("[full] ")
    per, full = df[~df["_is_full"]], df[df["_is_full"]]
    if per.empty or full.empty:
        return pd.DataFrame()

    # Best single per-lipid GP per lipid (best achievable by ANY one GP config).
    agg = {"base_r2": ("test_r2_median", "max")}
    if "test_corr_median" in df.columns:
        agg["base_corr"] = ("test_corr_median", "max")
    base = per.groupby("lipid").agg(**agg).reset_index()

    rows = []
    for model, g in full.groupby("model"):
        m = g.merge(base, on="lipid", how="inner")
        if m.empty:
            continue
        d_r2 = m["test_r2_median"] - m["base_r2"]
        row = {
            "full_model": model,
            "kernel_family": g["kernel_family"].iloc[0],
            "n_lipids": int(len(m)),
            "r2_full_median": float(m["test_r2_median"].median()),
            "r2_best_single_median": float(m["base_r2"].median()),
            "d_r2_mean": float(d_r2.mean()),
            "d_r2_median": float(d_r2.median()),
            "r2_win_rate": float((d_r2 > 0).mean()),
        }
        if "test_corr_median" in m.columns and "base_corr" in m.columns:
            d_corr = m["test_corr_median"] - m["base_corr"]
            row.update({
                "d_corr_mean": float(d_corr.mean()),
                "d_corr_median": float(d_corr.median()),
                "corr_win_rate": float((d_corr > 0).mean()),
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("d_r2_median", ascending=False, kind="stable")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+",
                    help="One or more directories containing the per-run output dirs "
                         "(e.g. .../experiment_batch_14). Runs from all roots are "
                         "pooled (a 'source' column tags each when >1 root is given).")
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
    # ---- optional: compare against the whole-brain ("full") LGP models -----
    ap.add_argument("--full-lgp-root", type=Path, nargs="+", default=None,
                    metavar="ROOT",
                    help="One or more roots of whole-brain LGP runs "
                         "(dirs holding <split>/predictions.npy, as scored by "
                         "lgp_report.py). They are added to the per-lipid x per-model "
                         "table and a FULL-vs-PER-LIPID comparison, restricted to the "
                         "lipids the per-lipid runs trained. Only r2/corr are compared "
                         "(the whole-brain rmse is original-unit, not z-scored).")
    ap.add_argument("--full-lgp-lipid-names-file", type=Path, default=None,
                    help="Optional .npy of lipid names to label the whole-brain output "
                         "columns (needed to join by lipid_name unless the run's cached "
                         "metrics.csv already carries lipid_name).")
    ap.add_argument("--full-lgp-split", default="test", choices=["test", "train"],
                    help="Which whole-brain split to score (default test).")
    ap.add_argument("--full-lgp-family", default=None,
                    help="Only include whole-brain runs whose family contains this "
                         "substring (e.g. lgp, manifold, gplfr).")
    ap.add_argument("--full-lgp-force", action="store_true",
                    help="Recompute whole-brain metrics from the npy arrays even if a "
                         "cached metrics.csv exists.")
    ap.add_argument("--full-lgp-chunk-rows", type=int, default=200_000,
                    help="Row block size for the streaming whole-brain metric recompute.")
    args = ap.parse_args()

    roots: list[Path] = args.roots
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        print("ERROR: not a directory: " + ", ".join(str(m) for m in missing),
              file=sys.stderr)
        return 1

    # Any directory containing a metrics.csv is a run. rglob so nested layouts
    # (e.g. an extra fold subdir on S3) are handled too. Discover across every
    # root; a run's `source` is the first root it's found under (dedup by full
    # path, so the same dir under two roots is still scored once).
    run_dir_to_source: dict[Path, str] = {}
    for root in roots:
        for p in root.rglob("metrics.csv"):
            run_dir_to_source.setdefault(p.parent, root.name)
    run_dirs = sorted(run_dir_to_source)
    if not run_dirs:
        roots_str = ", ".join(str(r) for r in roots)
        print(f"No metrics.csv found anywhere under {roots_str}", file=sys.stderr)
        return 1

    runs = [r for r in (load_run(d, run_dir_to_source[d]) for d in run_dirs)
            if r is not None]
    if not runs:
        roots_str = ", ".join(str(r) for r in roots)
        print(f"Found metrics.csv files under {roots_str} but none were usable.",
              file=sys.stderr)
        return 1

    long_df, per_run, per_fold, per_lipid_model = build_tables(runs, args.metric)

    # ---- optional: fold in the whole-brain ("full") LGP models --------------
    # Restricted to the lipids the per-lipid runs trained (join on lipid_name).
    # These are added to the per-lipid x per-model table and drive a dedicated
    # FULL-vs-PER-LIPID comparison; per-run / per-fold / total stay per-lipid.
    model_comparison = pd.DataFrame()
    uplift = pd.DataFrame()
    n_full_runs = 0
    if args.full_lgp_root:
        lipid_col = _lipid_col(long_df)
        if lipid_col != "lipid_name":
            print("  ! --full-lgp-root needs a 'lipid_name' column on the per-lipid "
                  f"runs to join on (found '{lipid_col}'); skipping full-LGP comparison.",
                  file=sys.stderr)
        else:
            trained_lipids = {str(x) for x in long_df["lipid_name"].dropna().unique()}
            full_lipid_names = None
            if args.full_lgp_lipid_names_file and args.full_lgp_lipid_names_file.exists():
                try:
                    full_lipid_names = np.load(args.full_lgp_lipid_names_file,
                                               allow_pickle=True).ravel()
                except Exception as ex:  # noqa: BLE001
                    print(f"  ! could not read --full-lgp-lipid-names-file ({ex})",
                          file=sys.stderr)
            full_runs = load_full_lgp_runs(
                args.full_lgp_root, args.full_lgp_split, args.full_lgp_chunk_rows,
                full_lipid_names, trained_lipids, args.full_lgp_force,
                args.full_lgp_family)
            n_full_runs = len(full_runs)
            if full_runs:
                combined_long = pd.concat([long_df, _long_from_runs(full_runs)],
                                          ignore_index=True)
                per_lipid_model = build_per_lipid_model(combined_long, args.metric)
                model_comparison = build_model_comparison(combined_long, args.metric)
                uplift = build_uplift(per_lipid_model)
            else:
                print("  ! no whole-brain runs overlapped the trained lipids.",
                      file=sys.stderr)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    n_failed = int(per_run["failed"].sum())
    roots_str = ", ".join(str(r) for r in roots)
    print("=" * 90)
    print(f"PER-LIPID GP REPORT  —  roots: {roots_str}")
    print(f"  runs: {len(runs)}   folds: {per_fold.shape[0]}   "
          f"lipid-rows: {len(long_df)}   failed runs: {n_failed}")
    if args.full_lgp_root:
        print(f"  full-lgp: {n_full_runs} whole-brain run(s) folded in "
              f"(trained lipids only; r2/corr comparison)")
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

    # ---- full-LGP vs per-lipid comparison (trained lipids only) ----
    if args.full_lgp_root and not model_comparison.empty:
        print("\n### FULL-LGP vs PER-LIPID (pooled over the trained lipids only; "
              "'[full] …' = whole-brain models)")
        with pd.option_context("display.max_rows", args.max_rows):
            print(model_comparison.to_string(index=False))

    # ---- uplift: what each full model adds over the best single per-lipid GP ----
    if args.full_lgp_root and not uplift.empty:
        print("\n### UPLIFT — full model minus BEST single per-lipid GP, per lipid "
              "(d_* = full - best-single; win_rate = fraction of lipids the full model wins)")
        with pd.option_context("display.max_rows", args.max_rows):
            print(uplift.to_string(index=False))

    # ---- per-lipid x per-model (aggregated across folds) ----
    print("\n### PER-LIPID x PER-MODEL SUMMARY (aggregated across folds: "
          "mean / median / min..max)")
    if per_lipid_model.empty:
        print("  (none)")
    else:
        with pd.option_context("display.max_rows", args.max_rows):
            print(per_lipid_model.head(args.max_rows).to_string(index=False))
            if len(per_lipid_model) > args.max_rows:
                print(f"  ... {len(per_lipid_model) - args.max_rows} more "
                      f"(lipid, model) rows (raise --max-rows)")

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
        per_lipid_model.to_csv(args.csv_dir / "per_lipid_model.csv", index=False)
        total_df.to_csv(args.csv_dir / "total.csv", index=False)
        if not model_comparison.empty:
            model_comparison.to_csv(args.csv_dir / "full_vs_per_lipid.csv", index=False)
        if not uplift.empty:
            uplift.to_csv(args.csv_dir / "full_lgp_uplift.csv", index=False)
        print(f"\nWrote CSVs to {args.csv_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
