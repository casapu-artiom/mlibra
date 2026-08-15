#!/usr/bin/env python
"""Visualize the RAW MALDI dataset over the reference template.

No training, no GP, no kernels — this just reads the measured MALDI
voxels straight from the parquet and shows them in napari on top of the
CCF reference volume, so you can eyeball where the actual data lives.

Two render modes (``--render``):

  * ``volume`` (default, FAST) — each mouse (the ``Sample`` column) is
    rasterized into a dense 3D volume and shown as an Image/Labels layer.
    napari uploads a volume to the GPU as a single texture, so a filled
    brain costs the same whether it holds 100K or 2M voxels — orders of
    magnitude faster than a scatter layer, both on first open and on
    every rotate/toggle. The baked volumes are cached to disk (keyed by
    the parquet + all render params), so repeat launches skip even the
    rasterization and just ``np.load`` the volumes.

  * ``points`` (SLOW, exact) — the original scatter viewer: every mouse
    is its own napari Points layer of one dot per measured voxel. Kept
    for when you need the true scatter look / ``--point-size`` control.

Either way every mouse becomes its OWN layer, so napari's per-layer
visibility checkbox is exactly "a checkbox for each mouse". Within a
mouse, points/voxels are coloured by ``Section`` (the slice index) so the
individual coronal slices stay distinguishable.

Layers added:

  0. reference template volume (faint anatomy background)
  1. (optional) anatomical annotations
  2..N  one layer per mouse (Sample), one checkbox each

Coordinate handling mirrors ``visualize_lipid_gp.py``: mm coords are
converted to display-volume voxel indices via ``--template-voxel-scale``
(0.025 mm for the Allen 25 µm atlas), with the same ``--axis-order`` /
``--flip-axes`` knobs in case the scatter comes up rotated/mirrored
relative to the template.

Examples
--------
    # every mouse, coloured by slice (fast volume mode, cached)
    python manifold/viz/visualize_raw_data.py \
        --maldi-file  /path/maindata_minimal.parquet \
        --reference-file /path/reference_image.npy

    # only the two atlas brains, one flat colour per mouse
    python manifold/viz/visualize_raw_data.py ... \
        --samples ReferenceAtlas SecondAtlas --color-by sample

    # colour every voxel by a raw lipid intensity instead of slice
    python manifold/viz/visualize_raw_data.py ... --color-by lipid --lipid "PC 38:1"

    # the old exact scatter viewer
    python manifold/viz/visualize_raw_data.py ... --render points --point-size 2
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
import hashlib
import json
import logging
import os

import numpy as np
import matplotlib


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Show the raw MALDI dataset over the reference template, "
                    "one toggleable layer per mouse.",
    )
    p.add_argument("--maldi-file", required=True,
                   help="Path to the MALDI parquet (the same file the "
                        "experiments read; must have xccf/yccf/zccf plus "
                        "Sample and Section columns).")
    p.add_argument("--reference-file", required=True,
                   help="Path to the CCF reference volume (.npy) used as "
                        "the anatomical background.")
    p.add_argument("--annotations-file", default=None,
                   help="Optional anatomical-labels volume (.npy).")
    p.add_argument("--samples", nargs="*", default=None,
                   help="Optional subset of mouse (Sample) names to show. "
                        "Default: all mice found in the parquet.")
    p.add_argument("--coord-cols", nargs=3,
                   default=["xccf", "yccf", "zccf"],
                   metavar=("XCOL", "YCOL", "ZCOL"),
                   help="Coordinate column names in the parquet.")
    p.add_argument("--sample-col", default="Sample",
                   help="Column identifying the mouse. Default 'Sample'.")
    p.add_argument("--section-col", default="Section",
                   help="Column identifying the slice. Default 'Section'.")
    p.add_argument("--color-by", choices=["section", "sample", "lipid"],
                   default="section",
                   help="'section' (default): colour by slice index within "
                        "each mouse (see individual slices). "
                        "'sample': one flat colour per mouse. "
                        "'lipid': colour by --lipid's raw intensity (magma); "
                        "needs --lipid.")
    p.add_argument("--lipid", default=None,
                   help="Lipid column to colour by when --color-by lipid.")
    # --- precomputed lipid volume overlay(s) ------------------------------
    p.add_argument("--lipid-volume", nargs="*", default=None,
                   metavar="NPY",
                   help="One or more dense lipid volumes (.npy) to overlay as "
                        "extra Image layers, e.g. a GP prediction "
                        "'<lipid>_volume.npy'. Must be on the template grid "
                        "(same shape as --reference-file). Rendered with an "
                        "alpha-ramped colormap so the NaN/low background is "
                        "transparent. Independent of --render / --color-by.")
    p.add_argument("--lipid-volume-name", nargs="*", default=None,
                   metavar="NAME",
                   help="Optional display name(s) for --lipid-volume layers "
                        "(defaults to each file's stem).")
    p.add_argument("--lipid-volume-colormap", default="magma",
                   help="Colormap for --lipid-volume overlays. Default magma.")
    # --- render mode ------------------------------------------------------
    p.add_argument("--render", choices=["volume", "points"], default="volume",
                   help="'volume' (default, FAST): rasterize each mouse to a "
                        "3D volume (Image/Labels), cached to disk. "
                        "'points' (SLOW): the exact scatter viewer, one dot "
                        "per voxel (honours --point-size).")
    p.add_argument("--volume-downsample", type=int, default=1,
                   help="Volume mode only: integer factor to shrink each "
                        "baked volume by (aligned back to the template via a "
                        "per-layer scale). 2 cuts memory ~8x; the sparse "
                        "MALDI voxels still read fine. Default 1 (full res).")
    p.add_argument("--cache-dir", default=None,
                   help="Volume mode only: where to store baked-volume "
                        "caches. Default: '<maldi-file dir>/.rawviz_cache'.")
    p.add_argument("--no-cache", action="store_true",
                   help="Volume mode only: never read or write the disk "
                        "cache (always rasterize fresh).")
    p.add_argument("--point-size", type=float, default=2.0,
                   help="points mode only: napari Points size.")
    p.add_argument("--max-points-per-sample", type=int, default=0,
                   help="Randomly subsample each mouse to at most this many "
                        "points/voxels (0 = no cap). Big brains can be >1M "
                        "voxels.")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Display gamma for the lipid colour mapping.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for --max-points-per-sample subsampling.")
    # --- coordinate alignment (mirrors visualize_lipid_gp.py) -------------
    p.add_argument("--axis-order", nargs=3, type=int, default=[0, 1, 2],
                   metavar=("A0", "A1", "A2"),
                   help="Permutation of (xccf, yccf, zccf) → napari volume "
                        "axes. Default '0 1 2' (identity) is right for the "
                        "Allen CCF convention (xccf=AP, yccf=DV, zccf=LR; "
                        "template shape (AP, DV, LR)). Try '1 0 2', '0 2 1', "
                        "etc. if the scatter looks rotated 90°.")
    p.add_argument("--flip-axes", nargs="*", type=int, default=[],
                   metavar="AXIS",
                   help="OUTPUT (post-permutation) axes to mirror, e.g. "
                        "'--flip-axes 2' flips the last axis only.")
    p.add_argument("--template-voxel-scale", type=float, default=0.025,
                   help="Physical voxel size (mm) of the displayed reference "
                        "volume. Default 0.025 = Allen 25 µm atlas.")
    return vars(p.parse_args())


# =============================================================================
# Coordinate conversion (kept local so this viewer has no cross-file deps)
# =============================================================================
def mm_to_voxel_idx(coords_mm: np.ndarray,
                    voxel_scale: float = 0.025,
                    axis_order: tuple = (0, 1, 2),
                    flip_axes: tuple = (),
                    flip_ref_max: np.ndarray | None = None) -> np.ndarray:
    """Convert mm coordinates → integer voxel indices, applying a
    permutation (axis_order) and optional per-axis flip.

    Identical convention to ``visualize_lipid_gp.mm_to_voxel_idx``. When
    ``flip_ref_max`` is given (per output-axis max index), it is used as
    the mirror reference so every mouse flips about the SAME plane — using
    each layer's own max would misalign mice that don't span the full
    volume.
    """
    idx = (coords_mm / voxel_scale).round().astype(np.int32)
    permuted = idx[:, list(axis_order)]
    if flip_axes:
        flipped = permuted.copy()
        for ax in flip_axes:
            ref = (permuted[:, ax].max() if flip_ref_max is None
                   else flip_ref_max[ax])
            flipped[:, ax] = ref - permuted[:, ax]
        return flipped
    return permuted


# =============================================================================
# Colour helpers
# =============================================================================
def colors_sequential(values, cmap_name="magma", gamma=0.5,
                      vmin=None, vmax=None):
    """Map values → RGBA via a sequential colormap (percentile-pinned by
    the caller). Mirrors visualize_lipid_gp for a consistent look."""
    lo = float(values.min()) if vmin is None else float(vmin)
    hi = float(values.max()) if vmax is None else float(vmax)
    if hi > lo:
        norm = np.clip((values - lo) / (hi - lo), 0, 1) ** float(gamma)
    else:
        norm = np.zeros_like(values, dtype=np.float32)
    return matplotlib.colormaps[cmap_name](norm)


def section_colors(sections: np.ndarray, global_sections: np.ndarray):
    """One distinct hue per Section, from a categorical colormap. Colours
    are keyed to the GLOBAL ordered set of section ids so the same slice
    number reads as the same hue across every mouse."""
    order = {s: i for i, s in enumerate(global_sections)}
    cmap = matplotlib.colormaps["tab20"]
    idx = np.array([order[s] for s in sections])
    return cmap(np.mod(idx, 20) / 19.0)


def section_color_dict(global_sections):
    """{label int -> RGBA} for a Labels layer, so voxel value = (section
    ordinal + 1) reads as the SAME tab20 hue section_colors() would give
    it. Label 0 / background stays transparent."""
    cmap = matplotlib.colormaps["tab20"]
    d = {None: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    for i in range(len(global_sections)):
        d[i + 1] = np.asarray(cmap((i % 20) / 19.0), dtype=np.float32)
    return d


# =============================================================================
# Rasterization (points → dense cropped volume)
# =============================================================================
def rasterize(vloc: np.ndarray, shape, values: np.ndarray, dtype):
    """Scatter ``values`` into a dense volume of ``shape`` at the local
    voxel coords ``vloc`` (already offset to the crop origin and, if
    downsampling, already divided). Voxel collisions keep the MAX value
    (arbitrary but stable). Returns the dense volume."""
    vol = np.zeros(shape, dtype=dtype)
    v = np.round(vloc).astype(np.int64)
    shp = np.asarray(shape)
    inb = np.all((v >= 0) & (v < shp), axis=1)
    v = v[inb]
    vals = np.asarray(values)[inb].astype(dtype, copy=False)
    if v.size == 0:
        return vol
    flat = np.ravel_multi_index((v[:, 0], v[:, 1], v[:, 2]), shape)
    np.maximum.at(vol.reshape(-1), flat, vals)
    return vol


# =============================================================================
# Disk cache for baked volumes
# =============================================================================
def cache_key(args, samples, template_shape):
    """Stable hash of everything that changes the baked volumes, incl. the
    parquet's mtime+size so an edited dataset invalidates the cache."""
    try:
        st = os.stat(args["maldi_file"])
        stamp = (int(st.st_mtime), int(st.st_size))
    except OSError:
        stamp = (0, 0)
    payload = {
        "stamp": stamp,
        "samples": list(map(str, samples)),
        "color_by": args["color_by"],
        "lipid": args["lipid"],
        "coord_cols": args["coord_cols"],
        "section_col": args["section_col"],
        "axis_order": list(args["axis_order"]),
        "flip_axes": list(args["flip_axes"] or []),
        "voxel_scale": args["template_voxel_scale"],
        "downsample": int(args["volume_downsample"]),
        "cap": int(args["max_points_per_sample"]),
        "seed": int(args["seed"]),
        "template_shape": list(map(int, template_shape)),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def cache_path(args, key):
    d = args["cache_dir"] or os.path.join(
        os.path.dirname(os.path.abspath(args["maldi_file"])), ".rawviz_cache")
    return os.path.join(d, f"rawviz_{args['color_by']}_{key}.npz")


def load_cache(path, log):
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"].item()))
        vols = [z[f"vol_{i}"] for i in range(len(meta["layers"]))]
        offs = [z[f"off_{i}"] for i in range(len(meta["layers"]))]
        log.info(f"  loaded baked volumes from cache: {path}")
        return meta, vols, offs
    except Exception as e:  # noqa: BLE001
        log.warning(f"  cache unreadable ({e}); re-baking.")
        return None


def save_cache(path, meta, vols, offs, log):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"meta": np.array(json.dumps(meta))}
    for i, (v, o) in enumerate(zip(vols, offs)):
        payload[f"vol_{i}"] = v
        payload[f"off_{i}"] = np.asarray(o, dtype=np.int64)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)  # ~97% zeros → tiny on disk
    os.replace(tmp, path)
    log.info(f"  wrote baked-volume cache: {path}")


# =============================================================================
# Volume mode: bake per-mouse volumes (from cache or fresh)
# =============================================================================
def bake_volumes(args, df, samples, global_sections, lip_vmin, lip_vmax,
                 voxel_scale, axis_order, flip_axes, flip_ref_max,
                 template_shape, log):
    """Return (meta, vols, offsets). Each vol is a dense cropped volume for
    one mouse; offsets[i] is its origin (world voxel coords) for translate.
    Uses the disk cache when possible."""
    key = cache_key(args, samples, template_shape)
    path = cache_path(args, key)
    if not args["no_cache"]:
        hit = load_cache(path, log)
        if hit is not None:
            return hit

    log.info("  rasterizing per-mouse volumes …")
    xcol, ycol, zcol = args["coord_cols"]
    sample_col = args["sample_col"]
    section_col = args["section_col"]
    d = max(1, int(args["volume_downsample"]))
    cap = int(args["max_points_per_sample"])
    rng = np.random.default_rng(int(args["seed"]))
    order = {s: i for i, s in enumerate(global_sections)}

    vols, offs, layers = [], [], []
    for s in samples:
        sub = df[df[sample_col] == s]
        if cap > 0 and len(sub) > cap:
            keep = rng.choice(len(sub), size=cap, replace=False)
            sub = sub.iloc[keep]
        coords_mm = sub[[xcol, ycol, zcol]].to_numpy(dtype=np.float32)
        voxel = mm_to_voxel_idx(coords_mm, voxel_scale=voxel_scale,
                                axis_order=axis_order, flip_axes=flip_axes,
                                flip_ref_max=flip_ref_max)
        # clip to the template box so translate/scale line up with points
        voxel = np.clip(voxel, 0, np.asarray(template_shape) - 1)
        origin = voxel.min(axis=0)
        vloc = (voxel - origin) // d
        shape = tuple(int(x) for x in (vloc.max(axis=0) + 1))
        n_sec = int(sub[section_col].nunique())

        if args["color_by"] == "sample":
            vol = rasterize(vloc, shape,
                            np.ones(len(sub), np.uint8), np.uint8)
        elif args["color_by"] == "lipid":
            vals = sub[args["lipid"]].to_numpy(dtype=np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            vol = rasterize(vloc, shape, vals, np.float32)
        else:  # section → label = ordinal + 1
            labels = np.fromiter((order[x] + 1 for x in
                                  sub[section_col].to_numpy()),
                                 dtype=np.int64, count=len(sub))
            dt = np.uint16 if len(global_sections) < 65535 else np.int32
            vol = rasterize(vloc, shape, labels, dt)

        vols.append(vol)
        offs.append(np.asarray(origin, dtype=np.int64))
        layers.append({"sample": str(s), "n_pts": int(len(sub)),
                       "n_sec": n_sec})
        log.info(f"  + {s}: {len(sub):,} voxels, bbox {shape}, "
                 f"{n_sec} slices")

    meta = {"color_by": args["color_by"], "downsample": d,
            "lipid": args["lipid"], "lip_vmin": lip_vmin,
            "lip_vmax": lip_vmax,
            "global_sections": list(map(int, global_sections)),
            "layers": layers}
    if not args["no_cache"]:
        try:
            save_cache(path, meta, vols, offs, log)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  could not write cache ({e}); continuing.")
    return meta, vols, offs


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("rawviz")
    args = parse_args()

    import pandas as pd
    import napari

    xcol, ycol, zcol = args["coord_cols"]
    sample_col = args["sample_col"]
    section_col = args["section_col"]

    read_cols = [xcol, ycol, zcol, sample_col, section_col]
    if args["color_by"] == "lipid":
        if not args["lipid"]:
            raise SystemExit("--color-by lipid requires --lipid <name>")
        read_cols.append(args["lipid"])

    log.info(f"Reading {read_cols} from {args['maldi_file']} …")
    df = pd.read_parquet(args["maldi_file"], columns=read_cols)
    log.info(f"  {len(df):,} rows total")

    all_samples = list(pd.unique(df[sample_col]))
    if args["samples"]:
        wanted = set(args["samples"])
        missing = wanted - set(map(str, all_samples)) - set(all_samples)
        if missing:
            log.warning(f"  requested samples not in data: {sorted(missing)}")
        samples = [s for s in all_samples if s in wanted or str(s) in wanted]
    else:
        samples = all_samples
    log.info(f"  showing {len(samples)} mouse layer(s): {samples}")

    # Global ordered section set → stable, shared per-slice colours.
    global_sections = sorted(pd.unique(df[section_col]))

    # Global lipid colour range so intensity is comparable across mice.
    lip_vmin = lip_vmax = None
    if args["color_by"] == "lipid":
        lv = df[args["lipid"]].to_numpy(dtype=np.float32)
        lv = lv[np.isfinite(lv)]
        lip_vmin = float(np.percentile(lv, 1.0))
        lip_vmax = float(np.percentile(lv, 99.0))
        log.info(f"  lipid '{args['lipid']}' p1-p99 range: "
                 f"[{lip_vmin:.3g}, {lip_vmax:.3g}]")

    axis_order = tuple(args["axis_order"])
    flip_axes = tuple(args["flip_axes"] or [])
    voxel_scale = float(args["template_voxel_scale"])

    template = np.load(args["reference_file"])
    # Shared mirror reference: max output-axis index the template supports,
    # so all mice flip about the same plane (not their own local max).
    tmpl_perm_shape = np.array(template.shape)[list(axis_order)]
    flip_ref_max = (tmpl_perm_shape - 1) if flip_axes else None

    log.info("Opening napari …")
    viewer = napari.Viewer()
    viewer.title = "raw MALDI data over reference template"

    viewer.add_image(
        template, name="0 reference (atlas)",
        rendering="attenuated_mip", colormap="gray",
        opacity=0.55, blending="additive", visible=True,
    )
    if args["annotations_file"]:
        try:
            ann = np.load(args["annotations_file"])
            max_lbl = int(ann.max())
            if max_lbl < 256:
                ann = ann.astype(np.uint8)
            elif max_lbl < 65_536:
                ann = ann.astype(np.uint16)
            viewer.add_labels(
                np.ascontiguousarray(ann), name="1 annotations (atlas)",
                opacity=0.3, blending="translucent", visible=False,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"Could not load annotations: {e}")

    if args["render"] == "volume":
        add_volume_layers(args, viewer, df, samples, global_sections,
                          lip_vmin, lip_vmax, voxel_scale, axis_order,
                          flip_axes, flip_ref_max, template.shape, log)
    else:
        add_points_layers(args, viewer, df, samples, global_sections,
                          lip_vmin, lip_vmax, voxel_scale, axis_order,
                          flip_axes, flip_ref_max, log)

    if args["lipid_volume"]:
        add_lipid_volume_layers(args, viewer, template.shape, log)

    log.info("Ready. Toggle a mouse on/off via its checkbox in the "
             "layer list on the left.")
    napari.run()


def add_volume_layers(args, viewer, df, samples, global_sections,
                      lip_vmin, lip_vmax, voxel_scale, axis_order,
                      flip_axes, flip_ref_max, template_shape, log):
    """FAST path: bake (or load) per-mouse volumes and attach them as
    Image/Labels layers, translated/scaled back onto the template."""
    from napari.utils import Colormap
    from napari.utils.colormaps import DirectLabelColormap

    meta, vols, offs = bake_volumes(
        args, df, samples, global_sections, lip_vmin, lip_vmax,
        voxel_scale, axis_order, flip_axes, flip_ref_max,
        template_shape, log)

    d = int(meta["downsample"])
    scale = (d, d, d)
    color_by = meta["color_by"]
    sample_cmap = matplotlib.colormaps["tab10"]
    sec_cmap = (DirectLabelColormap(color_dict=section_color_dict(
        meta["global_sections"])) if color_by == "section" else None)

    for si, (vol, off, lyr) in enumerate(zip(vols, offs, meta["layers"])):
        if not vol.any():
            continue
        name = (f"{lyr['sample']}  ({lyr['n_sec']} slices, "
                f"{lyr['n_pts']:,} vox)")
        translate = tuple(int(x) for x in off)
        if color_by == "sample":
            col = sample_cmap(si % 10)
            cmap = Colormap([[0, 0, 0, 0], [col[0], col[1], col[2], 1.0]],
                            name=f"mouse{si}")
            viewer.add_image(
                vol, name=name, colormap=cmap, contrast_limits=[0, 1],
                rendering="mip", opacity=0.9, blending="translucent",
                translate=translate, scale=scale, visible=True)
        elif color_by == "lipid":
            viewer.add_image(
                vol, name=name, colormap="magma",
                contrast_limits=[meta["lip_vmin"], meta["lip_vmax"]],
                gamma=float(args["gamma"]), rendering="mip",
                opacity=0.9, blending="translucent",
                translate=translate, scale=scale, visible=True)
        else:  # section
            viewer.add_labels(
                vol, name=name, colormap=sec_cmap,
                opacity=0.9, blending="translucent",
                translate=translate, scale=scale, visible=True)


def add_lipid_volume_layers(args, viewer, template_shape, log):
    """Overlay one or more precomputed dense lipid volumes (.npy on the
    template grid) as Image layers. NaN/low background is made transparent
    via an alpha-ramped colormap so only the signal reads over the anatomy."""
    from napari.utils import Colormap

    # magma (or chosen cmap) with alpha ramping 0→1 over the low end, so the
    # NaN-mapped background and near-zero voxels stay see-through.
    base = matplotlib.colormaps[args["lipid_volume_colormap"]](
        np.linspace(0, 1, 256))
    base[:, 3] = np.clip(np.linspace(0, 1, 256) / 0.15, 0, 1)
    overlay_cmap = Colormap(base, name="lipidvol")

    paths = args["lipid_volume"]
    names = args["lipid_volume_name"] or []
    for i, path in enumerate(paths):
        if not os.path.isfile(path):
            log.warning(f"  lipid-volume not found, skipping: {path}")
            continue
        vol = np.load(path).astype(np.float32)
        if tuple(vol.shape) != tuple(template_shape):
            log.warning(f"  lipid-volume shape {vol.shape} != template "
                        f"{tuple(template_shape)}; skipping: {path}")
            continue
        finite = vol[np.isfinite(vol)]
        if finite.size == 0:
            log.warning(f"  lipid-volume all non-finite; skipping: {path}")
            continue
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.0))
        if hi <= lo:
            hi = lo + 1e-6
        # NaN → lo so it normalizes to 0 → fully transparent under the ramp.
        vol = np.where(np.isfinite(vol), vol, lo)
        name = names[i] if i < len(names) else \
            os.path.basename(path).replace("_volume255.npy", "") \
                                  .replace("_volume.npy", "") \
                                  .replace(".npy", "")
        viewer.add_image(
            vol, name=f"lipid-vol: {name}", colormap=overlay_cmap,
            contrast_limits=[lo, hi], gamma=float(args["gamma"]),
            rendering="mip", opacity=0.9, blending="translucent",
            visible=True)
        log.info(f"  + lipid-volume '{name}': p1-p99 [{lo:.3g}, {hi:.3g}]")


def add_points_layers(args, viewer, df, samples, global_sections,
                      lip_vmin, lip_vmax, voxel_scale, axis_order,
                      flip_axes, flip_ref_max, log):
    """SLOW path: the original one-dot-per-voxel scatter viewer."""
    sample_col = args["sample_col"]
    section_col = args["section_col"]
    xcol, ycol, zcol = args["coord_cols"]
    rng = np.random.default_rng(int(args["seed"]))
    cap = int(args["max_points_per_sample"])
    sample_cmap = matplotlib.colormaps["tab10"]

    for si, s in enumerate(samples):
        sub = df[df[sample_col] == s]
        if cap > 0 and len(sub) > cap:
            keep = rng.choice(len(sub), size=cap, replace=False)
            sub = sub.iloc[keep]
        coords_mm = sub[[xcol, ycol, zcol]].to_numpy(dtype=np.float32)
        voxel = mm_to_voxel_idx(coords_mm, voxel_scale=voxel_scale,
                                axis_order=axis_order, flip_axes=flip_axes,
                                flip_ref_max=flip_ref_max).astype(np.float32)
        n_sec = sub[section_col].nunique()

        if args["color_by"] == "sample":
            face = np.tile(sample_cmap(si % 10), (len(sub), 1))
        elif args["color_by"] == "lipid":
            vals = sub[args["lipid"]].to_numpy(dtype=np.float32)
            face = colors_sequential(vals, "magma", args["gamma"],
                                     vmin=lip_vmin, vmax=lip_vmax)
        else:  # section
            face = section_colors(sub[section_col].to_numpy(),
                                  global_sections)

        viewer.add_points(
            voxel, name=f"{s}  ({n_sec} slices, {len(sub):,} pts)",
            size=float(args["point_size"]), face_color=face,
            opacity=0.9, blending="translucent", visible=True,
        )
        log.info(f"  + {s}: {len(sub):,} pts across {n_sec} slices")


if __name__ == "__main__":
    main()
