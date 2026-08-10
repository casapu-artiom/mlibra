#!/usr/bin/env python
"""
Aggregate report over WHOLE-BRAIN latent-GP AND baseline runs — the sibling of
``per_lipid_report.py``. It scores any run dir that holds ``<split>/predictions.npy``
(+ ``true_values.npy``), so it is model-agnostic; the ``family`` label just groups
them. Recognised families:

    * lgp             — ``lgp_experiment.py``            (Euclidean latent GP)
    * manifold        — ``lgp_manifold_experiment.py``   (Riemann inducing-point GP)
    * spectral        — ``spectral_lgp_manifold_experiment.py`` (weight-space spectral GP)
    * gplfr-<base>    — ``run_sota.sh MODEL=gplfr`` -> ``sota/gplfr_experiment.py``, split by the
                        latent-GP kernel: gplfr-euclidean / gplfr-riemann / gplfr-spectral
    * sota-<method>   — the SOTA 3D-reconstruction papers from ``run_sota.sh``:
                        sota-ntf (Neural Transcriptomic Field), sota-spa3d (SPE + z-aware
                        GCN), sota-deepspatial (faithful transport driver). The shared
                        ``sota-`` prefix lets ``--family sota`` select them together.
    * baseline-<model>— ``experiment_baselines.py``: baseline-mean / -linear /
                        -xgboost / -mlp / -mlp_eigen / -gcn / -gcn_faiss
                        (bottleneck runs -> baseline-bottleneck-<model>)
    * <base>+parcel   — ``local_run/run_parcel_lgp.sh`` -> ``other_experiments/parcelgp/lgp_parcel_experiment.py``:
                        the same model with the reference-only parcel factor in the
                        kernel (lgp+parcel, manifold+parcel, ...). The suffix keeps
                        the parcel arm out of its own baseline's row — without it the
                        run's exp_name still says LGP and the ablation pools with the
                        thing it ablates. ``--family parcel`` selects them all.

Unlike the per-lipid pipeline, these runs do NOT write a ``metrics.csv``. Each run
dir instead holds:

    args.npy                 # the full args dict (fold, kernel, model, ...)
    test/predictions.npy     # (n_test, P)  un-log, un-norm (original scale)
    test/true_values.npy     # (n_test, P)
    train/...                # same, for the training split

So this script RECOMPUTES per-lipid metrics from those arrays (chunked, so a
700k x 173 pair never blows up memory) and then, exactly like
``per_lipid_report.py``, emits:

  1. A PER-FOLD table          (lipids pooled across that fold's runs).
  2. A TOTAL summary           (everything pooled).
  3. A PER-LIPID x PER-MODEL    table (aggregated across folds).
  4. A PER-RUN table           (one row per run dir).

A ``parcel`` column (the field / rank the run used, ``-`` for runs without one)
appears in the per-run table whenever a parcelgp run is loaded.

Per-lipid metrics (computed per output column, then averaged per run):
    r2    1 - SSE/SST              (higher better; can be negative)
    corr  Pearson corr            (higher better)
    rmse  sqrt(mean (y-yhat)^2)   (lower better; ORIGINAL units)
    mae   mean |y-yhat|           (lower better)

Multiple roots may be passed; runs from each are pooled, and a ``source`` column
tags which root each run came from.

Usage
-----
    python reports/lgp_report.py /home/casap/mlibra/output
    python reports/lgp_report.py out_a out_b out_c            # multiple roots pooled
    python reports/lgp_report.py /path/to/out --split test --metric r2
    python reports/lgp_report.py /path/to/out --fold fold_3               # one fold
    python reports/lgp_report.py /path/to/out --fold fold_2 difficult     # several
    python reports/lgp_report.py /path/to/out --family gplfr --csv-dir ./lgp_report_out
    python reports/lgp_report.py /path/to/out --lipid-names-file .../available_lipids.npy
"""
from __future__ import annotations

# --- repo path bootstrap (this file moved out of maldi/) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_), str(_REPO_ / "maldi"), str(_REPO_ / "manifold"), str(_REPO_ / "reports"),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lgp_metrics import METRIC_COLS, LOWER_IS_BETTER, load_or_compute  # shared metric I/O

# Leading fold token, e.g. 'FOLD-3-', 'fold_3-', 'difficult-'.
_FOLD_PREFIX_RE = re.compile(r"^(fold[-_]?\d+|difficult)[-_]", re.IGNORECASE)
# The same token ANYWHERE in the name, with the separator on either side. The
# LGP-family runners (local_run/run_lgp.sh, local_run/run_parcel_lgp.sh) build
# '<EXP_PREFIX>-LGP-<splits-file-stem>-d5-...', so the fold appears a second time
# in the MIDDLE of exp_name — stripping only the leading token leaves every fold
# of one configuration as a model of its own, which is not what the per-model
# tables are for.
_FOLD_TOKEN_RE = re.compile(r"[-_]?\b(fold[-_]?\d+|difficult)\b[-_]?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Run identity (fold / family / model label) — best-effort from args.npy + name
# ---------------------------------------------------------------------------
def _load_args(run_dir: Path) -> dict | None:
    p = run_dir / "args.npy"
    if not p.exists():
        return None
    try:
        a = np.load(p, allow_pickle=True)
        a = a.item() if getattr(a, "shape", None) == () else a
        return a if isinstance(a, dict) else None
    except Exception:  # noqa: BLE001
        return None


def derive_fold(run_dir: Path, args: dict | None) -> str:
    """fold_2 / fold_3 / difficult ... preferring the splits file the run used."""
    if args:
        sf = args.get("slices_dataset_file")
        if sf:
            return Path(str(sf)).stem
        exp = args.get("exp_name")
        if exp:
            m = _FOLD_PREFIX_RE.match(str(exp))
            if m:
                return m.group(1)
    m = _FOLD_PREFIX_RE.match(run_dir.name) or _FOLD_TOKEN_RE.search(run_dir.name)
    return m.group(1) if m else "?"


def derive_parcel(run_dir: Path, args: dict | None) -> str:
    """The parcelgp factor the run carried, ``''`` if it carried none.

    A parcelgp run (``local_run/run_parcel_lgp.sh`` -> ``--parcel-field``) wraps the
    parcel factor around the SAME kernel, so its exp_name still says LGP and its
    args still say matern: nothing tells it apart from the baseline it ablates
    unless the parcel field is read out explicitly. The label is the field file's
    stem (which carries its build parameters) plus the embedding rank.
    """
    field = (args or {}).get("parcel_field")
    if not field:
        # Runs synced without args.npy: the entrypoint appends the same settings
        # to exp_name, so the directory name still shows it is a parcel run. Only
        # the field is read back — config.py appends n_pixels to the directory
        # name, so '-r8' + '10' is indistinguishable from a rank of 810.
        m = re.search(r"-parcel([^-]+)", run_dir.name, re.IGNORECASE)
        return m.group(1) if m else ""
    bits = [Path(str(field)).stem]
    if (args or {}).get("parcel_rank") is not None:
        bits.append(f"r{args['parcel_rank']}")
    if not (args or {}).get("parcel_per_task", True):  # --parcel-shared-B
        bits.append("sharedB")
    return "/".join(bits)


def derive_family(run_dir: Path, args: dict | None) -> str:
    """Coarse model family, stable across folds:
    lgp | manifold | spectral | gplfr-<base> | baseline-<model>, with a
    ``+parcel`` suffix when the run carried a parcelgp ``--parcel-field`` (else
    the parcel arm would land in the family row of the run it ablates)."""
    name = str((args or {}).get("exp_name") or run_dir.name).upper()
    parcel = "+parcel" if derive_parcel(run_dir, args) else ""
    model = (args or {}).get("model")
    base_gp = (args or {}).get("base_gp")
    if "BASELINE" in name:
        tag = "baseline"
        if "BOTTLENECK" in name:
            tag += "-bottleneck"
        if model:
            tag += f"-{model}"
        return tag + parcel
    if "GPLFR" in name:
        # Split by the latent-GP kernel (euclidean / riemann / spectral) so the
        # kernels are comparable side by side rather than pooled into one "gplfr".
        return (f"gplfr-{base_gp}" if base_gp else "gplfr") + parcel
    # SOTA 3D-reconstruction papers (run_sota.sh). All share a `sota-` prefix so
    # `--family sota` selects them together, but stay split by method.
    if "DEEPSPATIAL" in name:
        return "sota-deepspatial" + parcel
    if "SOTA" in name or model in {"ntf", "spa3d"}:
        # ntf / spa3d go through run_sota.py; the `model` field IS the method.
        return (f"sota-{model}" if model else "sota") + parcel
    if "MANIFOLD" in name:
        return "manifold" + parcel
    if "SPECTRAL" in name:
        return "spectral" + parcel
    if "LGPALL" in name or "LGP" in name:
        return "lgp" + parcel
    return str((args or {}).get("kernel", "?")) + parcel


def derive_model(run_dir: Path, args: dict | None) -> str:
    """Full configuration label (exp_name minus EVERY fold token, wherever it
    sits), so the same config scored under different folds collapses to one
    model instead of one model per fold."""
    name = str((args or {}).get("exp_name") or run_dir.name)
    return _FOLD_TOKEN_RE.sub("-", name).strip("-_")


# ---------------------------------------------------------------------------
# One run -> per-lipid metric table (cached metrics.csv if present, else recompute)
# ---------------------------------------------------------------------------
def load_run(run_dir: Path, split: str, chunk_rows: int,
             lipid_names: np.ndarray | None, force: bool,
             source: str = "") -> dict | None:
    # Read the cached metrics.csv if it exists (and not --force); otherwise
    # recompute from the npy arrays AND write the cache. Shared with the
    # experiments, which write the same metrics.csv at the end of a run.
    df = load_or_compute(run_dir, split, chunk_rows=chunk_rows,
                         lipid_names=lipid_names, force=force)
    if df is None or df.empty:
        return None

    args = _load_args(run_dir)
    if lipid_names is not None and "lipid_name" not in df.columns and len(lipid_names) == len(df):
        df.insert(1, "lipid_name", [str(x) for x in lipid_names])
    n_col = "n" if "n" in df.columns else ("n_test" if "n_test" in df.columns else None)

    return {
        "run": run_dir.name,
        "source": source,
        "fold": derive_fold(run_dir, args),
        "family": derive_family(run_dir, args),
        "model": derive_model(run_dir, args),
        "parcel": derive_parcel(run_dir, args),
        "failed": (run_dir / "FAILED.txt").exists(),
        "n_lipids": int(len(df)),
        "n_test": int(df[n_col].max()) if n_col else 0,
        "df": df,
    }


# ---------------------------------------------------------------------------
# Aggregation (mirrors per_lipid_report.py)
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    for col in METRIC_COLS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            out[f"{col}_median"] = float(np.nanmedian(vals)) if len(vals) else float("nan")
    return out


def summarise_across_folds(g: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    for col in METRIC_COLS:
        if col in g.columns:
            vals = pd.to_numeric(g[col], errors="coerce")
            out[f"{col}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            out[f"{col}_median"] = float(np.nanmedian(vals)) if len(vals) else float("nan")
            out[f"{col}_min"] = float(np.nanmin(vals)) if len(vals) else float("nan")
            out[f"{col}_max"] = float(np.nanmax(vals)) if len(vals) else float("nan")
    return out


def _lipid_key(df: pd.DataFrame) -> str:
    return "lipid_name" if "lipid_name" in df.columns else "lipid"


def build_per_lipid_model(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    key = _lipid_key(long_df)
    rows = []
    for (lipid, model), g in long_df.groupby([key, "model"], dropna=False):
        row = {"lipid": lipid, "model": model, "family": g["family"].iloc[0],
               "n_folds": g["fold"].nunique(), "n_runs": g["run"].nunique()}
        row.update(summarise_across_folds(g))
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_col = f"{metric}_median"
    if not out.empty and sort_col in out.columns:
        out = out.sort_values(["lipid", sort_col],
                              ascending=[True, metric in LOWER_IS_BETTER], kind="stable")
    return out


def build_tables(runs: list[dict], metric: str):
    # Only surface the `source` (root) column when runs span >1 root, else it's
    # constant clutter.
    multi_source = len({r.get("source", "") for r in runs}) > 1
    # Same idea for the parcel column: only shown when a parcelgp run is loaded.
    any_parcel = any(r.get("parcel") for r in runs)

    long_parts = []
    for r in runs:
        d = r["df"].copy()
        d.insert(0, "run", r["run"])
        d.insert(1, "fold", r["fold"])
        d.insert(2, "family", r["family"])
        d.insert(3, "model", r["model"])
        if multi_source:
            d.insert(1, "source", r.get("source", ""))
        long_parts.append(d)
    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    per_run_rows = []
    for r in runs:
        row = {"run": r["run"]}
        if multi_source:
            row["source"] = r.get("source", "")
        row.update({"fold": r["fold"], "family": r["family"],
                    "failed": r["failed"], "n_lipids": r["n_lipids"], "n_test": r["n_test"]})
        if any_parcel:
            row["parcel"] = r.get("parcel") or "-"
        row.update(summarise(r["df"]))
        per_run_rows.append(row)
    per_run = pd.DataFrame(per_run_rows)

    per_fold_rows = []
    if not long_df.empty:
        for (fold, family), g in long_df.groupby(["fold", "family"]):
            row = {"fold": fold, "family": family,
                   "n_runs": g["run"].nunique(), "n_lipids": int(len(g))}
            row.update(summarise(g))
            per_fold_rows.append(row)
    per_fold = pd.DataFrame(per_fold_rows)
    sort_col = f"{metric}_median"
    if not per_fold.empty and sort_col in per_fold.columns:
        per_fold = per_fold.sort_values(["fold", sort_col],
                                        ascending=[True, metric in LOWER_IS_BETTER])

    per_lipid_model = build_per_lipid_model(long_df, metric)
    return long_df, per_run, per_fold, per_lipid_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+",
                    help="One or more directories containing per-run output dirs. "
                         "Runs from all roots are pooled (a 'source' column tags each).")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="Which split's arrays to score (default test).")
    ap.add_argument("--metric", default="r2", choices=METRIC_COLS,
                    help="Metric used to sort the per-fold / per-lipid tables (default r2).")
    ap.add_argument("--family", default=None,
                    help="Only include runs whose family contains this substring "
                         "(e.g. manifold, lgp, baseline).")
    ap.add_argument("--fold", nargs="+", default=None, metavar="FOLD",
                    help="Only include runs whose fold contains one of these substrings "
                         "(e.g. --fold fold_2 fold_3, or --fold difficult). The fold label "
                         "is the run's slices_dataset_file stem, so '3' matches 'fold_3'.")
    ap.add_argument("--sort", default="fold",
                    help="Column to sort the per-run table by (default 'fold').")
    ap.add_argument("--lipid-names-file", type=Path, default=None,
                    help="Optional .npy of lipid names to label columns "
                         "(must match the number of output columns).")
    ap.add_argument("--chunk-rows", type=int, default=200_000,
                    help="Row block size for the streaming metric accumulation.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute metrics from the npy arrays even if a cached "
                         "metrics.csv exists (and overwrite the cache).")
    ap.add_argument("--csv-dir", type=Path, default=None,
                    help="If set, write per_lipid_long.csv / per_run.csv / per_fold.csv / "
                         "per_lipid_model.csv / total.csv here.")
    ap.add_argument("--max-rows", type=int, default=200,
                    help="Cap rows printed in the per-run / per-lipid tables (default 200).")
    args = ap.parse_args()

    roots: list[Path] = args.roots
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        print("ERROR: not a directory: " + ", ".join(str(m) for m in missing), file=sys.stderr)
        return 1

    lipid_names = None
    if args.lipid_names_file and args.lipid_names_file.exists():
        try:
            lipid_names = np.load(args.lipid_names_file, allow_pickle=True).ravel()
        except Exception as ex:  # noqa: BLE001
            print(f"  ! could not read --lipid-names-file ({ex})", file=sys.stderr)

    # A run dir is any dir holding <split>/predictions.npy (and its sibling truth).
    # Discover across every root; a run's `source` is the first root it's found under
    # (dedup by full path, so the same dir under two roots is still scored once).
    run_dir_to_source: dict[Path, str] = {}
    for root in roots:
        for p in root.rglob(f"{args.split}/predictions.npy"):
            run_dir_to_source.setdefault(p.parent.parent, root.name)
    run_dirs = sorted(run_dir_to_source)
    if not run_dirs:
        roots_str = ", ".join(str(r) for r in roots)
        print(f"No {args.split}/predictions.npy found anywhere under {roots_str}", file=sys.stderr)
        return 1

    fold_pats = [f.lower() for f in args.fold] if args.fold else None

    runs = []
    skipped_folds: set[str] = set()
    for d in run_dirs:
        # The fold is derivable from args.npy alone, so filter BEFORE load_run —
        # that skips the (expensive) metric recompute for folds we don't want.
        run_args = _load_args(d)
        fold = derive_fold(d, run_args)
        if fold_pats and not any(p in fold.lower() for p in fold_pats):
            skipped_folds.add(fold)
            continue
        r = load_run(d, args.split, args.chunk_rows, lipid_names, args.force,
                     source=run_dir_to_source[d])
        if r is None:
            continue
        if args.family and args.family.lower() not in r["family"].lower():
            continue
        print(f"  scored {r['family']:18s} {r['run']}", file=sys.stderr)
        runs.append(r)
    if not runs:
        roots_str = ", ".join(str(r) for r in roots)
        filters = []
        if args.family:
            filters.append(f"family~='{args.family}'")
        if args.fold:
            filters.append(f"fold~='{' | '.join(args.fold)}'")
        print(f"Found {args.split} arrays under {roots_str} but none were usable"
              + (f" for {', '.join(filters)}." if filters else "."), file=sys.stderr)
        if skipped_folds:
            print(f"  folds present but filtered out: {', '.join(sorted(skipped_folds))}",
                  file=sys.stderr)
        return 1

    long_df, per_run, per_fold, per_lipid_model = build_tables(runs, args.metric)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    n_failed = int(per_run["failed"].sum())
    roots_str = ", ".join(str(r) for r in roots)
    print("=" * 100)
    print(f"WHOLE-BRAIN LGP REPORT  —  roots: {roots_str}   split: {args.split}")
    print(f"  runs: {len(runs)}   folds: {long_df['fold'].nunique()}   "
          f"families: {long_df['family'].nunique()}   lipid-rows: {len(long_df)}   "
          f"failed: {n_failed}")
    if args.fold or args.family:
        bits = []
        if args.fold:
            bits.append(f"fold~={' | '.join(args.fold)}")
        if args.family:
            bits.append(f"family~={args.family}")
        kept = ", ".join(sorted(long_df["fold"].unique()))
        print(f"  filtered: {'   '.join(bits)}   -> folds kept: {kept}"
              + (f"   (skipped: {', '.join(sorted(skipped_folds))})" if skipped_folds else ""))
    print("=" * 100)

    print("\n### PER-FOLD x FAMILY SUMMARY (lipids pooled)")
    print("  (none)" if per_fold.empty else per_fold.to_string(index=False))

    print("\n### TOTAL SUMMARY (all lipids pooled)")
    total = {"n_runs": len(runs), "n_folds": long_df["fold"].nunique(),
             "n_lipids": int(len(long_df))}
    total.update(summarise(long_df))
    total_df = pd.DataFrame([total])
    print(total_df.to_string(index=False))

    print("\n### PER-LIPID x PER-MODEL SUMMARY (aggregated across folds: mean / median / min..max)")
    if per_lipid_model.empty:
        print("  (none)")
    else:
        with pd.option_context("display.max_rows", args.max_rows):
            print(per_lipid_model.head(args.max_rows).to_string(index=False))
            if len(per_lipid_model) > args.max_rows:
                print(f"  ... {len(per_lipid_model) - args.max_rows} more (lipid, model) rows "
                      f"(raise --max-rows)")

    print("\n### PER-RUN SUMMARY")
    pr = per_run
    if args.sort in pr.columns:
        ascending = args.sort in {"fold", "family", "run"} or any(
            args.sort == f"{m}_mean" or args.sort == f"{m}_median" for m in LOWER_IS_BETTER)
        pr = pr.sort_values(args.sort, ascending=ascending, kind="stable")
    with pd.option_context("display.max_rows", args.max_rows):
        print(pr.head(args.max_rows).to_string(index=False))
        if len(pr) > args.max_rows:
            print(f"  ... {len(pr) - args.max_rows} more runs (raise --max-rows)")

    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(args.csv_dir / "per_lipid_long.csv", index=False)
        per_run.to_csv(args.csv_dir / "per_run.csv", index=False)
        per_fold.to_csv(args.csv_dir / "per_fold.csv", index=False)
        per_lipid_model.to_csv(args.csv_dir / "per_lipid_model.csv", index=False)
        total_df.to_csv(args.csv_dir / "total.csv", index=False)
        print(f"\nWrote CSVs to {args.csv_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
