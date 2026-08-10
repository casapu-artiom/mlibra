#!/usr/bin/env python
"""Render the MISSING per-lipid volumes + scatter diagnostics for every model
in a batch directory.

A "batch directory" (e.g. ``experiment_batch_16`` / ``experiment_batch_long_cv``)
holds one sub-directory per trained model. This script walks it, finds every
model, and for each renders whatever is missing under ``<model>/renders/``:

    <model>/renders/<name>_multi_panel.png    # <- the "lipid render" (volume)
    <model>/renders/<name>_diagnostics.png    # <- the "scatterplot"
    <model>/renders/<name>_error_slice.png    # <- one MALDI section, error heatmap

It understands BOTH on-disk layouts the pipeline produces:

  A. Per-lipid GP (``lgp_experiment_per_lipid.py``)
     ``<model>/config.json``, ``graph_meta.npz``, ``lipid_names.json``,
     ``predictions/<slug>/{graph_pred_raw,test_pred_z,test_pred_raw,
     test_true_z}.npy``. The whole-brain prediction is scattered back into the
     template grid to make each volume; the scatter is de-standardized from the
     saved z / raw arrays. Fully self-contained (config.json carries the
     reference file, stride, log-transform, ...).

  B. Joint LGP (``lgp_experiment.py`` / ``lgp_manifold_experiment.py``)
     ``<model>/volume[/_diffusion]/<name>_volume[...].npy`` (per-lipid volumes
     already written), ``<model>/test/{predictions,true_values}.npy`` (held-out
     test points, already in raw de-standardized units, columns ordered by
     ``selected_lipids_names``). These runs DON'T persist a config.json, so:
       * volumes render with the reference template as anatomy background if
         ``--reference-file`` is given (else without a background — still fine);
       * the scatter needs the ordered lipid-name list to map a lipid name to
         its test-array column — pass ``--available-lipids-file`` (the same
         ``*_available_lipids.npy`` the experiment was trained with). Without it,
         Layout-B scatterplots are skipped (volumes still render).

In all cases the render filenames are identical, so already-rendered lipids are
skipped (unless ``--force``) — safe to re-run over a batch still trickling in.

Which lipids to render is chosen with ``--lipids`` / ``--lipids-file`` (names,
like the training script); omit both to render every lipid found on disk.

Usage::

    python render_missing.py experiment_batch_16 --lipids "PC 34:1" "PE 38:4"
    python render_missing.py experiment_batch_16 --lipids-file my_lipids.txt
    python render_missing.py experiment_batch_16 --lipids "PC 34:1" --only scatter
    python render_missing.py experiment_batch_16 --dry-run
    python render_missing.py <batch> --reference-file /s3/.../reference_image.npy \
                                     --available-lipids-file /s3/.../..._available_lipids.npy
"""
from __future__ import annotations

# --- repo path bootstrap (this file moved out of maldi/) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ = _Path(__file__).resolve().parents[2]
for _p in (str(_REPO_), str(_REPO_ / "maldi"), str(_REPO_ / "manifold"),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

from render_lipid_volumes import (
    render_selected_lipids, render_lipid_diagnostics, render_error_slice,
)

log = logging.getLogger("render_missing")

# Per-lipid volume files ``<name>_volume{suffix}.npy`` — but NOT the byte-scaled
# ``<name>_volume255{suffix}.npy`` companions written alongside them.
_VOLUME_RE = re.compile(r"^(?P<name>.+)_volume(?P<suffix>(?!255)[^.]*)\.npy$")


# --- helpers mirrored from lgp_experiment_per_lipid.py (kept identical) -------
def safe_filename(name: str) -> str:
    """'PA 36:1 PA 38:4' -> 'PA_36-1_PA_38-4'  (matches the training pipeline)."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace(":", "-"))
    return s.strip("_")


def graph_meta_voxel_layout(graph_meta, stride, template_full):
    """Map graph nodes -> integer voxel indices + pick the render volume shape
    and anatomy background. Mirrors ``_graph_meta_voxel_layout`` in
    ``lgp_experiment_per_lipid.py``: euclidean runs store full-res
    ``node_voxel_idx``; manifold runs store the strided ``reference_nodes_z``
    and render into the strided template.

    Returns (positions:int64 (N,3), out_shape:tuple, anatomy:np.ndarray).
    """
    if "node_voxel_idx" in graph_meta:
        positions = graph_meta["node_voxel_idx"].astype(np.int64)
        out_shape = tuple(int(s) for s in graph_meta["template_shape"])
        anatomy = template_full
    else:
        mm = (graph_meta["reference_nodes_z"] * graph_meta["coord_std"]
              + graph_meta["coord_mean"])
        scale = float(graph_meta["voxel_scale_mm"])   # strided scale
        positions = (mm / scale).round().astype(np.int64)
        anatomy = template_full[::stride, ::stride, ::stride]
        out_shape = anatomy.shape
    positions = positions.copy()
    for ax in range(3):
        np.clip(positions[:, ax], 0, out_shape[ax] - 1, out=positions[:, ax])
    return positions, out_shape, anatomy


def recover_affine(pred_z: np.ndarray, pred_raw: np.ndarray):
    """Recover (cs, cm) from the exact affine relation
    ``pred_raw = cs*pred_z + cm``. Returns (cs, cm) or None if pred_z is
    degenerate (constant), in which case the scatter can't be de-standardized."""
    z = np.asarray(pred_z, dtype=np.float64).ravel()
    r = np.asarray(pred_raw, dtype=np.float64).ravel()
    m = np.isfinite(z) & np.isfinite(r)
    z, r = z[m], r[m]
    if z.size < 2 or np.ptp(z) < 1e-12:
        return None
    cs, cm = np.polyfit(z, r, 1)
    return float(cs), float(cm)


def load_names_list(path: str | None):
    """Ordered lipid-name list (column order of the joint-LGP test arrays),
    from an ``*_available_lipids.npy`` file. None if unavailable."""
    if not path or not Path(path).exists():
        return None
    try:
        return [str(n) for n in np.load(path, allow_pickle=True)]
    except (OSError, ValueError) as ex:
        log.warning(f"couldn't read available-lipids file {path}: {ex}")
        return None


def load_joint_coords(model: Path):
    """Row-aligned coordinates for the joint-LGP test points, from
    ``test/coordinates_pixel_index.pth`` (integer voxel index, preferred) or
    ``test/coordinates.pth`` (physical mm). Returns (coords:(N,3), axis_labels)
    or (None, None) if neither is present."""
    for fn, labels in (("coordinates_pixel_index.pth", ("x_index", "y_index", "z_index")),
                       ("coordinates.pth", ("xccf", "yccf", "zccf"))):
        p = model / "test" / fn
        if p.exists():
            try:
                import torch
                t = torch.load(p, map_location="cpu")
                arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
                return np.asarray(arr, dtype=np.float64), labels
            except Exception as ex:   # noqa: BLE001
                log.warning(f"{model.name}: couldn't read test/{fn} ({ex}).")
    return None, None


def load_lipid_selection(lipids, lipids_file) -> set[str] | None:
    """The set of lipid names to render, from ``--lipids`` and/or
    ``--lipids-file``. Returns None when neither is given (= render every lipid).

    Lipid names contain spaces (e.g. 'PC 34:1'), which don't survive shell
    word-splitting — use ``--lipids-file`` (one name per line, '#' comments) for
    those, exactly like ``lgp_experiment_per_lipid.py``."""
    wanted: set[str] = set()
    if lipids:
        wanted.update(str(x) for x in lipids)
    if lipids_file:
        fp = Path(lipids_file)
        if not fp.exists():
            log.error(f"--lipids-file {lipids_file} not found.")
        else:
            for line in fp.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    wanted.add(line)
    return wanted or None


# --- model discovery ----------------------------------------------------------
def classify(p: Path) -> str | None:
    """Return 'perlipid', 'joint', or None for a candidate directory."""
    if (p / "graph_meta.npz").exists() and (p / "predictions").is_dir():
        return "perlipid"
    if (p / "test" / "predictions.npy").exists():
        return "joint"
    for d in p.iterdir() if p.is_dir() else []:
        if d.is_dir() and d.name.startswith("volume") and \
                any(_VOLUME_RE.match(f.name) for f in d.glob("*_volume*.npy")):
            return "joint"
    return None


def find_models(batch_dir: Path) -> list[tuple[Path, str]]:
    """Every model dir under (and including) the batch dir, with its layout.
    A per-lipid model wins classification before its ``renders/volume`` folder
    could be mistaken for a joint model's volume dir (that folder has no
    ``*_volume*.npy`` at its own top level anyway)."""
    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    candidates = [batch_dir] + [d for d in batch_dir.rglob("*") if d.is_dir()]
    for p in candidates:
        # Don't treat the pipeline's own sub-folders as models.
        if p.name in {"predictions", "renders", "volume", "volume_diffusion",
                      "test", "train", "checkpoints"}:
            continue
        kind = classify(p)
        if kind and p not in seen:
            seen.add(p)
            out.append((p, kind))
    out.sort(key=lambda t: str(t[0]))
    return out


# --- Layout A: per-lipid GP ---------------------------------------------------
def process_perlipid(model: Path, only: str, force: bool, max_lipids: int,
                     dry_run: bool, wanted: set[str] | None,
                     reference_override: str | None = None) -> dict:
    stats = {"model": model.name, "kind": "perlipid", "volumes": 0,
             "scatters": 0, "errors": 0, "skipped": 0, "no_pred": 0}

    args = json.loads((model / "config.json").read_text())
    # config.json stores the reference path from the training host (often an
    # /s3/... cluster path that doesn't exist locally); --reference-file lets you
    # point at a local copy without editing every config.json.
    ref_file = reference_override or args.get("reference_file")
    template_full = (np.load(ref_file)
                     if ref_file and Path(ref_file).exists() else None)
    if template_full is None:
        log.warning(f"{model.name}: reference_file '{ref_file}' missing; "
                    f"volumes will be skipped (pass --reference-file to override).")
    stride = int(args.get("stride", 4))
    log_transform = bool(args.get("log_transform", False))

    graph_meta = np.load(model / "graph_meta.npz")
    predictions_root = model / "predictions"
    render_dir = model / "renders"
    volume_dir = render_dir / "volume"

    slug_to_name: dict[str, str] = {}
    names_path = model / "lipid_names.json"
    if names_path.exists():
        try:
            for n in json.loads(names_path.read_text()):
                slug_to_name[safe_filename(str(n))] = str(n)
        except (json.JSONDecodeError, OSError) as ex:
            log.warning(f"{model.name}: couldn't read lipid_names.json ({ex})")

    positions = out_shape = anatomy = None
    if template_full is not None:
        positions, out_shape, anatomy = graph_meta_voxel_layout(
            graph_meta, stride, template_full)

    do_volume = only in ("both", "volume")
    do_scatter = only in ("both", "scatter")
    to_render_volumes: list[str] = []
    n_done = 0

    for lip_dir in sorted(p for p in predictions_root.iterdir() if p.is_dir()):
        if max_lipids and n_done >= max_lipids:
            break
        name = slug_to_name.get(lip_dir.name, lip_dir.name)
        if wanted is not None and name not in wanted:
            continue
        if not (lip_dir / "graph_pred_raw.npy").exists():
            stats["no_pred"] += 1
            continue
        touched = False

        panel_png = render_dir / f"{name}_multi_panel.png"
        if do_volume and (force or not panel_png.exists()) and positions is not None:
            vals = np.load(lip_dir / "graph_pred_raw.npy").astype(np.float32)
            if vals.shape[0] != positions.shape[0]:
                log.warning(f"{model.name}/{name}: {vals.shape[0]} node preds vs "
                            f"{positions.shape[0]} nodes; skipping volume.")
            else:
                if not dry_run:
                    volume_dir.mkdir(parents=True, exist_ok=True)
                    vol = np.full(out_shape, np.nan, dtype=np.float32)
                    vol[positions[:, 0], positions[:, 1], positions[:, 2]] = vals
                    np.save(volume_dir / f"{name}_volume.npy", vol)
                to_render_volumes.append(name)
                stats["volumes"] += 1
                touched = True

        diag_png = render_dir / f"{name}_diagnostics.png"
        err_png = render_dir / f"{name}_error_slice.png"
        need_diag = do_scatter and (force or not diag_png.exists())
        need_err = do_scatter and (force or not err_png.exists())
        if need_diag or need_err:
            tz, pz, pr = (lip_dir / "test_true_z.npy",
                          lip_dir / "test_pred_z.npy",
                          lip_dir / "test_pred_raw.npy")
            if tz.exists() and pz.exists() and pr.exists():
                pred_raw = np.load(pr).astype(np.float64)
                aff = recover_affine(np.load(pz), pred_raw)
                if aff is None:
                    log.warning(f"{model.name}/{name}: degenerate preds; "
                                f"skipping scatter/error.")
                else:
                    cs, cm = aff
                    true_v = np.load(tz).astype(np.float64) * cs + cm
                    pred_v = pred_raw
                    if log_transform:
                        true_v, pred_v = np.exp(true_v) - 1e-10, np.exp(pred_v) - 1e-10
                    if need_diag:
                        if not dry_run:
                            _safe_diag(true_v, pred_v, name, diag_png, model)
                        stats["scatters"] += 1
                        touched = True
                    coords_f = lip_dir / "test_coords_mm.npy"
                    if need_err and coords_f.exists():
                        if not dry_run:
                            _safe_err(np.load(coords_f), true_v, pred_v, name,
                                      err_png, model,
                                      labels=("xccf", "yccf", "zccf"))
                        stats["errors"] += 1
                        touched = True

        if touched:
            n_done += 1
        else:
            stats["skipped"] += 1

    _flush_volumes(to_render_volumes, anatomy, volume_dir, render_dir,
                   suffix="", force=force, dry_run=dry_run)
    return stats


# --- Layout B: joint LGP ------------------------------------------------------
def process_joint(model: Path, only: str, force: bool, max_lipids: int,
                  dry_run: bool, template_full, names_list,
                  wanted: set[str] | None) -> dict:
    stats = {"model": model.name, "kind": "joint", "volumes": 0,
             "scatters": 0, "errors": 0, "skipped": 0, "no_pred": 0}
    render_dir = model / "renders"
    do_volume = only in ("both", "volume")
    do_scatter = only in ("both", "scatter")

    # Discover per-lipid volumes across every volume* dir; a (name, suffix) pair
    # points at the file render_selected_lipids will read.
    vol_entries: dict[str, tuple[Path, str]] = {}   # name -> (volume_dir, suffix)
    for d in sorted(model.glob("volume*")):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*_volume*.npy")):
            m = _VOLUME_RE.match(f.name)
            if m:
                vol_entries.setdefault(m["name"], (d, m["suffix"]))

    # Held-out test arrays (raw units, columns ordered by names_list).
    true_te = pred_te = None
    tp = model / "test" / "predictions.npy"
    tt = model / "test" / "true_values.npy"
    if tp.exists() and tt.exists():
        pred_te, true_te = np.load(tp), np.load(tt)

    # Volumes and scatterplots are INDEPENDENT here: the joint LGP only writes a
    # curated SUBSET of lipids as volume/*_volume.npy, but test/*.npy holds EVERY
    # lipid (columns ordered by names_list). So a lipid can get a scatter with no
    # volume, and vice-versa — don't gate one on the other.
    name_to_col = ({n: i for i, n in enumerate(names_list)}
                   if names_list is not None else {})
    can_scatter = bool(name_to_col) and true_te is not None and pred_te is not None
    scatterable = {n for n, i in name_to_col.items()
                   if can_scatter and i < true_te.shape[1]}

    if wanted is not None:
        candidates = sorted(wanted)
    else:
        # Everything we could possibly render: reconstructed volumes + every
        # lipid present in the test arrays.
        candidates = sorted(set(vol_entries) | scatterable)

    # Volumes to (re)render, grouped by (volume_dir, suffix) for one batched call.
    groups: dict[tuple[Path, str], list[str]] = {}
    scatter_names: list[str] = []
    errslice_names: list[str] = []
    n_done = 0
    for name in candidates:
        if max_lipids and n_done >= max_lipids:
            break
        touched = False
        if do_volume and name in vol_entries:
            panel_png = render_dir / f"{name}_multi_panel.png"
            if force or not panel_png.exists():
                groups.setdefault(vol_entries[name], []).append(name)
                stats["volumes"] += 1
                touched = True
        if do_scatter and name in scatterable:
            if force or not (render_dir / f"{name}_diagnostics.png").exists():
                scatter_names.append(name)
                touched = True
            if force or not (render_dir / f"{name}_error_slice.png").exists():
                errslice_names.append(name)
                touched = True
        if touched:
            n_done += 1
        else:
            stats["skipped"] += 1

    # Volume PNGs.
    for (vol_dir, suffix), names in groups.items():
        _flush_volumes(names, template_full, vol_dir, render_dir,
                       suffix=suffix, force=force, dry_run=dry_run)

    # Scatter PNGs (names already known to map to a test column via `scatterable`).
    for name in scatter_names:
        if not dry_run:
            gi = name_to_col[name]
            _safe_diag(true_te[:, gi], pred_te[:, gi], name,
                       render_dir / f"{name}_diagnostics.png", model)
        stats["scatters"] += 1

    # Error-slice PNGs — need the row-aligned test coordinates.
    if errslice_names:
        coords, coord_labels = load_joint_coords(model)
        if coords is None:
            log.warning(f"{model.name}: no test/coordinates[_pixel_index].pth; "
                        f"skipping {len(errslice_names)} error-slice(s).")
        else:
            for name in errslice_names:
                if not dry_run:
                    gi = name_to_col[name]
                    _safe_err(coords, true_te[:, gi], pred_te[:, gi], name,
                              render_dir / f"{name}_error_slice.png", model,
                              labels=coord_labels)
                stats["errors"] += 1

    # Explain a common surprise: scatter requested but nothing was mappable.
    if do_scatter and not scatter_names and not can_scatter:
        if true_te is None:
            log.warning(f"{model.name}: no test/{{predictions,true_values}}.npy — "
                        f"can't make scatterplots.")
        else:
            log.warning(f"{model.name}: --available-lipids-file not given, can't "
                        f"map lipids to test columns; skipping scatterplots.")
    elif do_scatter and wanted is not None:
        missing = [n for n in wanted if n not in scatterable]
        if missing and can_scatter:
            log.warning(f"{model.name}: requested lipid(s) not in test columns: "
                        f"{missing}")
    return stats


# --- shared rendering helpers -------------------------------------------------
def _safe_diag(true_v, pred_v, name, out_path, model):
    try:
        render_lipid_diagnostics(np.asarray(true_v), np.asarray(pred_v),
                                 str(name), out_path)
    except Exception as ex:   # noqa: BLE001 — a plot hiccup must not abort the run
        log.warning(f"{model.name}/{name}: diagnostics failed ({ex}).")


def _safe_err(coords, true_v, pred_v, name, out_path, model,
              labels=("x", "y", "z")):
    try:
        render_error_slice(np.asarray(coords), np.asarray(true_v),
                           np.asarray(pred_v), str(name), out_path,
                           axis_labels=labels)
    except Exception as ex:   # noqa: BLE001 — a plot hiccup must not abort the run
        log.warning(f"{model.name}/{name}: error-slice failed ({ex}).")


def _flush_volumes(names, anatomy, volume_dir, render_dir, suffix, force, dry_run):
    """Render the multi-panel PNGs for ``names`` from ``volume_dir``. Mirrors
    the training pipeline's call; render_selected_lipids skips any PNG that
    already exists, so --force clears them first to actually redraw."""
    if not names or dry_run:
        return
    if force:
        for name in names:
            p = render_dir / f"{name}_multi_panel.png"
            if p.exists():
                p.unlink()
    render_selected_lipids(
        template_volume=anatomy,          # None -> render without anatomy overlay
        volume_dir=volume_dir,
        output_dir=render_dir,
        selected_lipids_names=list(names),
        lipid_names=list(names),
        suffix=suffix,
        n_rotation_frames=10,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_dir", type=Path,
                   help="Directory holding the model sub-dirs (e.g. "
                        "experiment_batch_16).")
    p.add_argument("--lipids", nargs="+", default=None,
                   help="Only render these lipids (by name). NOTE: names with "
                        "spaces (e.g. 'PC 34:1') must be quoted, or won't survive "
                        "shell word-splitting — use --lipids-file for those. "
                        "Default: every lipid found on disk.")
    p.add_argument("--lipids-file", default=None,
                   help="Text file with one lipid name per line ('#' comments "
                        "and blank lines ignored) — the right way to pass names "
                        "with spaces. Combined with --lipids if both are given.")
    p.add_argument("--only", choices=["both", "volume", "scatter"],
                   default="both",
                   help="Render only volumes, only scatter diagnostics, or both.")
    p.add_argument("--force", "--overwrite", dest="force", action="store_true",
                   help="Overwrite: re-render even lipids that already have a PNG "
                        "(stale multi-panel PNGs are deleted first so they redraw).")
    p.add_argument("--max-lipids", type=int, default=0,
                   help="Cap on lipids rendered PER MODEL (0 = all missing).")
    p.add_argument("--reference-file", default=None,
                   help="Anatomy template (reference_image.npy). Layout-B volumes "
                        "use it as the background. For Layout-A (per-lipid) it "
                        "OVERRIDES the reference_file baked into each model's "
                        "config.json — use it when that path is a cluster path "
                        "(e.g. /s3/...) that doesn't exist on this machine.")
    p.add_argument("--available-lipids-file", default=None,
                   help="Ordered lipid-name .npy (the *_available_lipids.npy the "
                        "joint LGP was trained with). Required for Layout-B "
                        "scatterplots (maps lipid name -> test-array column).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be rendered without writing PNGs.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s", stream=sys.stdout)

    batch_dir = args.batch_dir
    if not batch_dir.is_dir():
        log.error(f"{batch_dir} is not a directory.")
        return 1

    models = find_models(batch_dir)
    if not models:
        log.error(f"No models found under {batch_dir} (looked for per-lipid "
                  f"graph_meta.npz+predictions/ and joint volume*/ or test/).")
        return 1

    template_full = (np.load(args.reference_file)
                     if args.reference_file and Path(args.reference_file).exists()
                     else None)
    if args.reference_file and template_full is None:
        log.warning(f"--reference-file {args.reference_file} not found; "
                    f"Layout-B volumes will render without an anatomy background.")
    names_list = load_names_list(args.available_lipids_file)
    wanted = load_lipid_selection(args.lipids, args.lipids_file)
    if wanted is not None:
        log.info(f"Restricting to {len(wanted)} lipid(s): {sorted(wanted)}")

    log.info(f"Found {len(models)} model(s) under {batch_dir}:")
    for m, kind in models:
        rel = m.relative_to(batch_dir) if m != batch_dir else Path(".")
        log.info(f"  - [{kind:8s}] {rel}")

    tag = "[dry-run] " if args.dry_run else ""
    totals = {"volumes": 0, "scatters": 0, "errors": 0, "skipped": 0}
    for m, kind in models:
        if kind == "perlipid":
            s = process_perlipid(m, args.only, args.force, args.max_lipids,
                                 args.dry_run, wanted,
                                 reference_override=args.reference_file)
        else:
            s = process_joint(m, args.only, args.force, args.max_lipids,
                              args.dry_run, template_full, names_list, wanted)
        log.info(f"{tag}{s['model']} ({s['kind']}): {s['volumes']} volume(s), "
                 f"{s['scatters']} scatter(s), {s['errors']} error-slice(s), "
                 f"{s['skipped']} up-to-date, {s['no_pred']} without predictions.")
        for k in totals:
            totals[k] += s[k]

    log.info(f"{tag}DONE — {totals['volumes']} volume render(s), "
             f"{totals['scatters']} scatter(s), {totals['errors']} error-slice(s) "
             f"across {len(models)} model(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
