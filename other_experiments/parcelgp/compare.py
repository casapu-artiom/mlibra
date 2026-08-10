"""Paired ablation over finished per-lipid runs.

    python -m other_experiments.parcelgp.compare --baseline base=RUN_DIR --run parcel=RUN_DIR [...]

Reads each run's ``metrics.csv`` (one row per lipid), joins on the lipid slug, and
compares every run to the baseline **per lipid**.

Why paired, and why it matters more than it sounds: lipid-to-lipid spread in corr
is roughly 0.2, while the effect we are hunting is on the order of 0.01. Comparing
two runs by their mean corr buries a real effect under between-lipid variance that
is IDENTICAL in both arms and therefore pure noise for this question. Differencing
lipid-by-lipid cancels it exactly. The same 173 lipids, folds and seed appear in
both arms (``run_parcel_per_lipid.sh`` guarantees it), so the pairing is valid by
construction.

Reported per run:

  ``delta``      mean over lipids of (run - baseline). The effect size.
  ``ci95``       percentile bootstrap over LIPIDS. Lipids are the independent
                 unit here -- voxels within a lipid are not.
  ``win``        fraction of lipids that improved. A small mean delta with a 90%
                 win rate is a real, uniform effect; a large mean delta with a 55%
                 win rate is a handful of lipids moving and should not be believed.
  ``p_sign``     two-sided sign test on that win rate. Makes no assumption about
                 the shape of the per-lipid deltas, which are usually skewed.

A run that is missing lipids (crashed batch, partial resume) is intersected down
to the lipids it shares with the baseline, and the count is reported so a run that
quietly lost half its panel is visible rather than silently favoured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"rmse", "mae"}


def load_metrics(path, metric: str) -> pd.Series:
    """(slug -> metric) for one run directory."""
    f = Path(path) / "metrics.csv"
    if not f.exists():
        raise FileNotFoundError(f"no metrics.csv in {path}")
    df = pd.read_csv(f)
    key = "slug" if "slug" in df.columns else df.columns[0]
    if metric not in df.columns:
        raise KeyError(f"{f} has no column {metric!r}; has {list(df.columns)}")
    s = df.set_index(key)[metric].astype(float)
    return s[~s.index.duplicated(keep="first")].dropna()


def paired_stats(base: np.ndarray, run: np.ndarray, higher_is_better: bool,
                 n_boot: int = 10_000, seed: int = 0) -> dict:
    d = run - base
    if not higher_is_better:
        d = -d
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n_boot, d.size))
    boot = d[idx].mean(1)
    wins = int((d > 0).sum())
    n = d.size
    # Two-sided sign test against p=0.5, normal approximation (n is ~50-173 here,
    # comfortably in range; exact binomial would differ in the 3rd decimal).
    z = abs(wins - n / 2) / np.sqrt(n / 4) if n else 0.0
    from math import erfc, sqrt
    return {
        "n": n,
        "delta": float(d.mean()),
        "ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        "win": wins / n if n else float("nan"),
        "p_sign": float(erfc(z / sqrt(2))),
        "per_lipid": d,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", required=True, metavar="NAME=DIR",
                   help="The arm everything is compared against.")
    p.add_argument("--run", action="append", default=[], metavar="NAME=DIR",
                   help="An arm to compare. Repeatable.")
    p.add_argument("--metric", default="corr", choices=["corr", "r2", "rmse", "mae"])
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=0,
                   help="Also list the N lipids that moved most (either way).")
    p.add_argument("--out", default=None, help="Write the table as json.")
    args = p.parse_args(argv)

    def split(spec):
        name, _, path = spec.partition("=")
        return (name, path) if path else (Path(name).name, name)

    b_name, b_dir = split(args.baseline)
    base = load_metrics(b_dir, args.metric)
    higher = args.metric not in LOWER_IS_BETTER

    rows, detail = [], {}
    for spec in args.run:
        name, d = split(spec)
        run = load_metrics(d, args.metric)
        shared = base.index.intersection(run.index)
        st = paired_stats(base.loc[shared].to_numpy(), run.loc[shared].to_numpy(),
                          higher, args.n_boot, args.seed)
        detail[name] = st.pop("per_lipid")
        rows.append({"run": name, "base_mean": float(base.loc[shared].mean()),
                     "run_mean": float(run.loc[shared].mean()),
                     "missing": int(len(base) - len(shared)), **st})

    w = max([len(r["run"]) for r in rows] + [8])
    print(f"\nmetric: {args.metric}   baseline: {b_name}  "
          f"(mean {base.mean():.4f} over {len(base)} lipids)\n")
    print(f"{'run':<{w}}{'n':>5}{'mean':>9}{'delta':>9}{'ci95':>20}{'win':>7}{'p':>9}")
    print("-" * (w + 59))
    for r in rows:
        ci = f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]"
        flag = "" if r["missing"] == 0 else f"  ({r['missing']} lipids missing)"
        print(f"{r['run']:<{w}}{r['n']:>5}{r['run_mean']:>9.4f}{r['delta']:>+9.4f}"
              f"{ci:>20}{r['win']:>6.0%}{r['p_sign']:>9.2g}{flag}")
    print("\nA CI that excludes 0 AND a win rate near 100% is a real effect; a "
          "significant p with a win rate near 50% is a few lipids, not a trend.\n")

    if args.top:
        for name, d in detail.items():
            shared = base.index.intersection(load_metrics(dict(
                map(split, args.run))[name], args.metric).index)
            order = np.argsort(-np.abs(d))[:args.top]
            print(f"{name}: largest per-lipid changes")
            for i in order:
                print(f"    {shared[i]:<40}{d[i]:+.4f}")
            print()

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
