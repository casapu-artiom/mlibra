#!/usr/bin/env python3
"""Per-family comparison grids: one row per model, one column block per lipid.

Regenerates `combined_baselines.png`, `combined_sota.png` and `combined_gps.png`
-- the figures that compare every model within a family on the same two lipids --
plus `combined_figure.png`, the all-models single page, via `--groups figure`.
Each row is a model; each lipid contributes three columns: an axial slice, a
sagittal slice, and the true-vs-predicted scatter lifted out of that run's
`renders/<lipid>_diagnostics.png`.

The colour scale is the point of the figure. vmin/vmax are the 2nd/98th
percentile POOLED OVER EVERY MODEL for a given lipid, so a row that looks washed
out really is washed out -- brightness is comparable down a column and across
the three files. That pooling is computed over all models in GROUPS, not just
the groups being written, so `--groups gps` produces panels on the same scale as
a full run rather than silently rescaling to its own subset.

Finding the runs
----------------
Two ways, because the runs live differently in the two places:

  --root <dir>            (default) the staged tree submit/download_models.sh
                          builds: <root>/<model>/<RUN>/volume*/...

  --from-report <csv>     a report CSV (reports/output/report_all/per_run.csv).
                          S3 has no per-model directory and no notion of a
                          "winner" -- but the report does: `source` is the batch
                          dir and `run` is the run dir, so <s3-root>/<source>/
                          <run> IS the path. Winners are derived, not declared:
                          runs are grouped by method and config, ranked on the
                          mean of --rank-metric across folds, and the best
                          config per method wins. Verified to reproduce the
                          hardcoded WINNERS table in submit/download_models.sh
                          exactly, and it additionally picks up methods added
                          since that table was written.

Usage:
    # locally, against the staged winners tree
    python reports/figures/combined_panels.py
    python reports/figures/combined_panels.py --groups gps
    python reports/figures/combined_panels.py --lipids "PA 36:1" "LPC 22:6"

    # on the cluster, straight off the S3 mount
    python reports/figures/combined_panels.py \\
        --from-report reports/output/report_all/per_run.csv \\
        --s3-root /s3/mlibra/mlibra-data/artiom --fold 2 \\
        --out-dir /myhome/mlibra/figures --scale-json /myhome/mlibra/scale.json

    # see which run each row resolves to, and its score, without reading a volume
    python reports/figures/combined_panels.py --from-report ... --fold 2 --dry-run

Volumes are ~2.5 GB dense (~150 MB sparse) and the shared colour scale needs
every model, so a cold run over FUSE reads all of them. --scale-json writes the
pooled vmin/vmax on the first run and reuses it after, which is what makes
`--groups` cheap: with a cached scale only the selected models are loaded, and
the result is identical to a full run.

Provenance: originally a throwaway written into a session scratchpad on
2026-07-26 and never saved; the PNGs in this directory came from it. Restored
here so the figures can be reproduced -- appearance is unchanged from those
originals (verified pixel-identical). EUCLID is absent from GROUPS because it
postdates them; add it with --models once its runs are staged.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # never touch a display
import matplotlib.pyplot as plt

ROOT = "/home/casap/mlibra/output/winners"
S3_ROOT = "/s3/mlibra/mlibra-data/artiom"
DEFAULT_LIPIDS = ("Hex2Cer 40:1;O2", "LPC 22:6")

# 55% along each axis -- the same cuts as the Axial Z=290 / Sagittal X=250 panels
# in maldi/render_lipid_volumes.py, so these line up with the per-run renders.
ZF, XF = 0.55, 0.55

GROUPS = {
    "baselines": [("mean", "Mean"), ("linear", "Linear"), ("xgboost", "XGBoost"),
                  ("mlp", "MLP"), ("mlp_eigen", "MLP-eigen"), ("gcn", "GCN")],
    "sota": [("ntf", "NTF"), ("spa3d", "Spa3D"), ("deepspatial", "DeepSpatial"),
             ("gplfr", "GPLFR")],
    "gps": [("lgp", "Euclidean GP"), ("manifold", "Manifold GP"),
            ("spectral", "Spectral GP")],
}

# Run names carry the method; the report's `family` column does not (all six
# baselines share family="baseline"). Order matters -- MLP_EIGEN must be tested
# before MLP, or every mlp_eigen run is claimed by mlp.
METHOD_PATTERNS = (
    ("mean", r"BASELINES-MEAN-"),
    ("linear", r"BASELINES-LINEAR-"),
    ("xgboost", r"BASELINES-XGBOOST-"),
    ("mlp_eigen", r"BASELINES-MLP_EIGEN-"),
    ("mlp", r"BASELINES-MLP-"),
    ("gcn", r"BASELINES-GCN"),
    ("euclid", r"BASELINES-EUCLID-"),
    ("ntf", r"SOTA-NTF-"),
    ("spa3d", r"SOTA-SPA3D-"),
    ("deepspatial", r"DEEPSPATIAL-"),
    ("gplfr", r"GPLFR-"),
    ("lgp", r"LGPALL-"),
    ("manifold", r"MANIFOLD-"),
    ("spectral", r"SPECTRAL-"),
)

# The original single-page version: every model in one figure, same layout and
# the same shared scale, under its original name (combined_figure.png). Not in
# the default set -- `--groups figure` asks for it, `--groups baselines sota gps
# figure` gets all four.
ALL_GROUPS = dict(GROUPS)
ALL_GROUPS["figure"] = [m for g in GROUPS.values() for m in g]

COLS = ("axial", "sagittal", "scatter")
COL_TITLES = {"axial": "axial", "sagittal": "sagittal", "scatter": "true vs pred"}

# Where in a diagnostics PNG the true-vs-predicted scatter sits: bottom-left
# quadrant, starting below the suptitle. Fractions of the image, so it survives
# a dpi change in the upstream renderer but not a panel reshuffle.
DIAG_CROP = (0.02, 0.565, 0.505, 1.0)

# Pattern-major: a dense volume/ anywhere beats a sparse one, matching the
# original ordering when a model has more than one staged run dir.
VOL_PATTERNS = ("volume/{lip}_volume.npy",
                "volume_sparse/{lip}_volume_sparse.npy",
                "volume/{lip}_volume_*.npy")          # cross-mouse runs
DIAG_PATTERNS = ("renders/{lip}_diagnostics.png",
                 "renders/*/{lip}_diagnostics.png")


# ---------------------------------------------------------------------------
# Resolving which run dir backs each row
# ---------------------------------------------------------------------------

def method_of(run: str) -> str | None:
    for name, pat in METHOD_PATTERNS:
        if re.search(pat, run):
            return name
    return None


def winners_from_report(csv_path, fold, s3_root, metric="corr_mean"):
    """{method: (run_dir, config, score)} -- best config per method, from a report.

    Ranks on the mean of `metric` ACROSS FOLDS, so the winner is the config that
    generalises, not the one that got a lucky fold; then returns that config's
    run dir for the fold actually being drawn.
    """
    d = pd.read_csv(csv_path)
    if "failed" in d.columns:
        d = d[~d["failed"].astype(bool)]
    d = d.copy()
    d["method"] = d["run"].map(method_of)
    unmatched = sorted(set(d[d["method"].isna()]["run"]))
    d = d.dropna(subset=["method"])
    d["config"] = d["run"].str.replace(r"^FOLD-\d+-", "", regex=True)
    # Drop +-inf as well as NaN: lgp_metrics can emit +-inf for a lipid whose
    # prediction came out constant, and a mean carrying inf would rank that
    # config first or last regardless of how it actually did.
    v = pd.to_numeric(d[metric], errors="coerce")
    d[metric] = v.where(np.isfinite(v))

    g = (d.groupby(["method", "config", "source"], as_index=False)
           .agg(score=(metric, "mean"), n_folds=("fold", "nunique")))
    best = (g.sort_values("score", ascending=False)
              .groupby("method", as_index=False).first())

    out = {}
    for _, r in best.iterrows():
        run = f"FOLD-{fold}-{r['config']}"
        out[r["method"]] = (os.path.join(s3_root, r["source"], run),
                            r["config"], float(r["score"]))
    return out, unmatched


def resolve_dirs(models, args, winners):
    """model key -> list of candidate run directories."""
    dirs = {}
    for mkey, _ in models:
        if winners is not None:
            hit = winners.get(mkey)
            dirs[mkey] = [hit[0]] if hit else []
        else:
            dirs[mkey] = sorted(glob.glob(os.path.join(args.root, mkey, "*")))
    return dirs


def _first(dirs, patterns, lip) -> str | None:
    for pat in patterns:
        for d in dirs:
            hits = sorted(glob.glob(os.path.join(d, pat.format(lip=lip))))
            if hits:
                return hits[0]
    return None


def vol_path(dirs, lip):
    """First reconstruction we can find for a lipid, whatever form it took."""
    return _first(dirs, VOL_PATTERNS, lip)


def diag_path(dirs, lip):
    return _first(dirs, DIAG_PATTERNS, lip)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def autotrim(im, thresh: int = 248, pad: int = 6):
    """Crop the white margin off a cropped-out subplot."""
    a = np.asarray(im.convert("RGB"))
    mask = (a < thresh).any(2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(im.width, xs.max() + pad), min(im.height, ys.max() + pad)))


def scatter_img(dirs, lip):
    p = diag_path(dirs, lip)
    if not p:
        return None
    im = Image.open(p)
    w, h = im.size
    l, t, r, b = DIAG_CROP
    return autotrim(im.crop((int(l * w), int(t * h), int(r * w), int(b * h))))


def load_slices(dirs_by_model, models, lipids, rng, n_sample=200_000):
    """One pass over the volumes: keep the two slices, and a subsample per lipid.

    The volumes are up to ~2.5 GB each, so nothing is retained beyond the two 2D
    cuts and the sample the shared colour scale needs.
    """
    slices, samples, missing = {}, {lip: [] for lip in lipids}, []
    for model, _ in models:
        for lip in lipids:
            p = vol_path(dirs_by_model.get(model, []), lip)
            if not p:
                slices[(model, lip)] = None
                missing.append((model, lip))
                continue
            v = np.load(p)
            z, _, x = v.shape
            slices[(model, lip)] = (v[int(z * ZF), :, :], v[:, :, int(x * XF)])
            fin = v[np.isfinite(v)]
            if fin.size:
                samples[lip].append(
                    rng.choice(fin, min(n_sample, fin.size), replace=False))
            del v
    return slices, samples, missing


def shared_scale(samples, lipids):
    """Per-lipid (vmin, vmax) at the 2nd/98th percentile, pooled over all models."""
    scale = {}
    for lip in lipids:
        if not samples[lip]:
            scale[lip] = (0.0, 1.0)
            continue
        pool = np.concatenate(samples[lip])
        scale[lip] = (float(np.percentile(pool, 2)), float(np.percentile(pool, 98)))
    return scale


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def make_figure(dirs_by_model, models, lipids, slices, scale, out_path, dpi=130):
    nrow, ncol = len(models), len(lipids) * len(COLS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.3, nrow * 1.9),
                             squeeze=False)
    for r, (mkey, mname) in enumerate(models):
        for li, lip in enumerate(lipids):
            for ci, what in enumerate(COLS):
                c = li * len(COLS) + ci
                ax = axes[r, c]
                _bare(ax)
                if what == "scatter":
                    img = scatter_img(dirs_by_model.get(mkey, []), lip)
                    if img is not None:
                        ax.imshow(img)
                    else:
                        ax.text(.5, .5, "n/a", ha="center", va="center",
                                color="0.6", transform=ax.transAxes)
                else:
                    sl = slices.get((mkey, lip))
                    if sl is not None:
                        vmin, vmax = scale[lip]
                        ax.set_facecolor("white")
                        ax.imshow(sl[0 if what == "axial" else 1], cmap="inferno",
                                  vmin=vmin, vmax=vmax, interpolation="nearest")
                    else:
                        ax.text(.5, .5, "n/a", ha="center", va="center",
                                color="0.6", transform=ax.transAxes)
                if c == 0:
                    ax.set_ylabel(mname, rotation=0, ha="right", va="center",
                                  fontsize=12, labelpad=14)
                if r == 0:
                    ax.set_title(COL_TITLES[what], fontsize=10)

    plt.tight_layout(rect=[0.015, 0, 1, 0.95])
    # Lipid headers span their own three columns. Positions come from the laid-out
    # axes, so this has to run after tight_layout.
    for li, lip in enumerate(lipids):
        x0 = axes[0, li * len(COLS)].get_position().x0
        x1 = axes[0, li * len(COLS) + len(COLS) - 1].get_position().x1
        fig.text((x0 + x1) / 2, 0.99, lip, ha="center", va="top",
                 fontsize=14, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT,
                    help="Staged tree with one subdirectory per model.")
    ap.add_argument("--from-report", default=None,
                    help="Report CSV (per_run.csv) to derive winners from. "
                         "Run dirs become <s3-root>/<source>/FOLD-<fold>-<config>.")
    ap.add_argument("--s3-root", default=S3_ROOT,
                    help="Prefix the report's `source` column hangs off.")
    ap.add_argument("--fold", type=int, default=2,
                    help="Which fold's run dir to draw (--from-report only).")
    ap.add_argument("--rank-metric", default="corr_mean",
                    help="Report column the winner is chosen on (mean across folds).")
    ap.add_argument("--groups", nargs="+", choices=tuple(ALL_GROUPS), default=tuple(GROUPS),
                    help="Default: the three split figures. Add 'figure' for the "
                         "all-models single page (combined_figure.png).")
    ap.add_argument("--models", nargs="+", default=None, metavar="KEY=LABEL",
                    help="Override the rows for a one-off figure, e.g. "
                         "--models euclid=EUCLID lgp='Euclidean GP'. Writes "
                         "combined_custom.png.")
    ap.add_argument("--lipids", nargs="+", default=list(DEFAULT_LIPIDS))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    ap.add_argument("--scale-json", default=None,
                    help="Cache the pooled vmin/vmax here. Reused if it exists, "
                         "which lets --groups load only the models it draws.")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds the colour-scale subsample, so runs are reproducible.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what each row resolves to; read no volumes.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    winners, unmatched = None, []
    if args.from_report:
        winners, unmatched = winners_from_report(
            args.from_report, args.fold, args.s3_root, args.rank_metric)
        print(f"winners from {args.from_report} (ranked on mean {args.rank_metric}"
              f" across folds, drawing fold {args.fold}):")
        for m, (d, cfg, sc) in sorted(winners.items(), key=lambda kv: -kv[1][2]):
            print(f"  {m:<12} {sc:>7.4f}  {cfg[:60]}")
        if unmatched:
            print(f"  ! {len(unmatched)} run(s) matched no method pattern, ignored: "
                  f"{unmatched[:3]}")

    if args.models:
        groups = {"custom": [tuple(m.split("=", 1)) if "=" in m else (m, m)
                             for m in args.models]}
        selected = ["custom"]
    else:
        groups, selected = ALL_GROUPS, list(args.groups)

    all_models = list(dict.fromkeys(m for g in groups.values() for m in g))
    draw_models = [m for k in selected for m in groups[k]]

    # A cached scale is what makes a subset cheap: without it the pooled scale
    # needs every model's volume, so drawing one group would still read all of
    # them (or, worse, silently rescale to the subset).
    cached = None
    if args.scale_json and Path(args.scale_json).exists():
        cached = {k: tuple(v) for k, v in json.load(open(args.scale_json)).items()}
        if all(lip in cached for lip in args.lipids):
            print(f"scale: reusing {args.scale_json}")
        else:
            print(f"scale: {args.scale_json} lacks these lipids, recomputing")
            cached = None
    scale_models = draw_models if cached else all_models
    dirs_by_model = resolve_dirs(scale_models, args, winners)

    if args.dry_run:
        for mkey, mname in scale_models:
            dirs = dirs_by_model.get(mkey, [])
            v = vol_path(dirs, args.lipids[0]) if dirs else None
            d = diag_path(dirs, args.lipids[0]) if dirs else None
            where = dirs[0] if dirs else "UNRESOLVED"
            print(f"  {mname:<14} volume={'OK ' if v else 'MISSING'} "
                  f"scatter={'OK ' if d else 'MISSING'}  {where}")
        return 0

    rng = np.random.default_rng(args.seed)
    slices, samples, missing = load_slices(dirs_by_model, scale_models,
                                           args.lipids, rng)
    scale = cached or shared_scale(samples, args.lipids)
    print("shared scales:", {k: (round(a, 6), round(b, 6)) for k, (a, b) in scale.items()})
    if missing:
        print(f"missing volumes ({len(missing)}), drawn as 'n/a': "
              + ", ".join(f"{m}/{l}" for m, l in missing))
    if args.scale_json and not cached:
        Path(args.scale_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({k: list(v) for k, v in scale.items()}, open(args.scale_json, "w"))
        print("wrote", args.scale_json)

    for key in selected:
        out = make_figure(dirs_by_model, groups[key], args.lipids, slices, scale,
                          out_dir / f"combined_{key}.png", dpi=args.dpi)
        print("wrote", out, Image.open(out).size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
