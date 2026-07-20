#!/usr/bin/env python
"""Visualize the RAW MALDI dataset over the reference template.

No training, no GP, no kernels — this just reads the measured MALDI
voxels straight from the parquet and scatters them in napari on top of
the CCF reference volume, so you can eyeball where the actual data lives.

Every mouse (the ``Sample`` column) becomes its OWN napari Points layer,
so napari's built-in per-layer visibility checkbox in the layer list is
exactly "a checkbox for each mouse" — tick/untick a mouse to show/hide
all of its slices. Within a mouse, points are coloured by ``Section``
(the slice index) so the individual coronal slices are distinguishable.

Layers added:

  0. reference template volume (faint anatomy background)
  1. (optional) anatomical annotations
  2..N  one Points layer per mouse (Sample), one checkbox each

Coordinate handling mirrors ``visualize_lipid_gp.py``: mm coords are
converted to display-volume voxel indices via ``--template-voxel-scale``
(0.025 mm for the Allen 25 µm atlas), with the same ``--axis-order`` /
``--flip-axes`` knobs in case the scatter comes up rotated/mirrored
relative to the template.

Examples
--------
    # every mouse, coloured by slice
    python maldi/visualize_raw_data.py \
        --maldi-file  /path/maindata_minimal.parquet \
        --reference-file /path/reference_image.npy

    # only the two atlas brains, one flat colour per mouse
    python maldi/visualize_raw_data.py ... \
        --samples ReferenceAtlas SecondAtlas --color-by sample

    # colour every point by a raw lipid intensity instead of slice
    python maldi/visualize_raw_data.py ... --color-by lipid --lipid "PC 38:1"
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Scatter the raw MALDI dataset over the reference "
                    "template, one toggleable layer per mouse.",
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
                   help="'section' (default): colour points by slice index "
                        "within each mouse (see individual slices). "
                        "'sample': one flat colour per mouse. "
                        "'lipid': colour every point by --lipid's raw "
                        "intensity (magma); needs --lipid.")
    p.add_argument("--lipid", default=None,
                   help="Lipid column to colour by when --color-by lipid.")
    p.add_argument("--point-size", type=float, default=2.0,
                   help="napari Points size.")
    p.add_argument("--max-points-per-sample", type=int, default=0,
                   help="Randomly subsample each mouse to at most this many "
                        "points (0 = no cap). Big brains can be >1M voxels; "
                        "napari slows down when recolouring huge layers.")
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

    rng = np.random.default_rng(int(args["seed"]))
    cap = int(args["max_points_per_sample"])
    # Distinct flat colour per mouse (used when --color-by sample).
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

    log.info("Ready. Toggle a mouse on/off via its checkbox in the "
             "layer list on the left.")
    napari.run()


if __name__ == "__main__":
    main()
