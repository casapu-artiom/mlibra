#!/usr/bin/env python
"""Look at a built parcellation on the reference brain.

The parcellation is the one thing in this package that is decided before any
lipid is read, so the first question about a new field is always visual: are the
parcels contiguous blobs that follow anatomy, or is it speckle that happens to
score well? ``parcelgp.validate`` answers the second question numerically; this
answers the first.

Two modes, same colours:

  **napari** (default) -- the strided reference volume as a greyscale image with
  the parcel ids on top as a napari ``Labels`` layer, so hovering reads out the
  parcel under the cursor and the usual napari controls (3D, opacity, slice
  sliders) all work. Extra layers for the border shell, the membership
  confidence, and the relative distance-to-border field.

  **montage** (``--montage out.png``) -- a grid of evenly spaced sections,
  reference underneath, parcels tinted on top, borders drawn. Headless, and the
  right thing for putting in a report or diffing two builds side by side.

``--stats`` prints the per-parcel size table and the build metadata without
opening anything.

    python -m other_experiments.parcelgp.view_parcels --field .../parcels/full_k128_sw3.0_s2_t5.npz
    python -m other_experiments.parcelgp.view_parcels --field ... --montage /tmp/parcels.png --n-slices 12
    python -m other_experiments.parcelgp.view_parcels --field ... --stats
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .field import ParcelField
from .viz import border_mask, label_volume, parcel_colors, reference_subvolume

log = logging.getLogger("parcelgp.view_parcels")


# --------------------------------------------------------------------------- #
# volumes
# --------------------------------------------------------------------------- #
def _mask_volume(field, mask: np.ndarray) -> np.ndarray:
    """Paint a per-node boolean back onto the strided grid."""
    vol = np.zeros(field.volume_shape, bool)
    vol[tuple(field.node_vox.T)] = mask
    return vol


def _node_volume(field, values: np.ndarray, background=np.nan) -> np.ndarray:
    vol = np.full(field.volume_shape, background, np.float32)
    vol[tuple(field.node_vox.T)] = values.astype(np.float32)
    return vol


def tissue_extent(field, axis: int) -> tuple[int, int]:
    """(first, last) index along ``axis`` that contains any tissue node."""
    v = field.node_vox[:, axis]
    return int(v.min()), int(v.max())


# --------------------------------------------------------------------------- #
# montage
# --------------------------------------------------------------------------- #
def blended_slice(ref_sl: np.ndarray, lab_sl: np.ndarray, brd_sl: np.ndarray,
                  colors: np.ndarray, alpha: float, ref_max: float,
                  border_rgb=(0.0, 0.0, 0.0)) -> np.ndarray:
    """One (H, W, 3) RGB frame: reference in grey, parcels tinted, borders drawn."""
    g = np.clip(ref_sl.astype(np.float32) / max(ref_max, 1e-6), 0, 1)
    rgb = np.repeat(g[..., None], 3, axis=2)
    tissue = lab_sl >= 0
    if tissue.any():
        col = colors[lab_sl[tissue], :3]
        rgb[tissue] = (1.0 - alpha) * rgb[tissue] + alpha * col
    rgb[brd_sl] = np.asarray(border_rgb, np.float32)
    return rgb


def montage(field, ref_sub, out_path, n_slices=9, axis=0, alpha=0.55,
            ncols=0, dpi=140, border_rgb=(0, 0, 0), title=None):
    """Grid of evenly spaced sections through the parcellation. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab_vol = label_volume(field)
    brd_vol = _mask_volume(field, border_mask(field))
    colors = parcel_colors(field.n_parcels)
    lo, hi = tissue_extent(field, axis)
    # Skip the extreme ends: the first and last few sections are a handful of
    # voxels and read as empty panels.
    idx = np.linspace(lo, hi, int(n_slices) + 2)[1:-1].round().astype(int)
    ref_max = float(np.percentile(ref_sub[ref_sub > 0], 99.5)) if (ref_sub > 0).any() else 1.0

    ncols = int(ncols) if ncols else int(np.ceil(np.sqrt(len(idx))))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        sl = [slice(None)] * 3; sl[axis] = i
        rgb = blended_slice(ref_sub[tuple(sl)], lab_vol[tuple(sl)],
                            brd_vol[tuple(sl)], colors, alpha, ref_max, border_rgb)
        ax.imshow(np.transpose(rgb, (1, 0, 2)) if axis == 2 else rgb,
                  interpolation="nearest")
        n_here = int((lab_vol[tuple(sl)] >= 0).sum())
        k_here = int(np.unique(lab_vol[tuple(sl)][lab_vol[tuple(sl)] >= 0]).size)
        ax.set_title(f"{'ADL'[axis]}={i}  ({k_here} parcels, {n_here:,} nodes)",
                     fontsize=8)
    fig.suptitle(title or (f"{Path(field.meta.get('reference_file', '?')).name} — "
                           f"{field.n_parcels} parcels, features="
                           f"{field.meta.get('features')}, "
                           f"spatial_weight={field.meta.get('spatial_weight')}, "
                           f"stride={field.meta.get('stride')}"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log.info("wrote %s (%d sections along axis %d)", out_path, len(idx), axis)
    return out_path


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def format_stats(field) -> str:
    counts = np.bincount(field.labels, minlength=field.n_parcels)
    vox_mm3 = float(field.meta.get("voxel_scale_mm", 0.1)) ** 3
    order = np.argsort(-counts)
    lines = [
        f"field        : {field.n_parcels} parcels over {field.labels.size:,} nodes",
        f"built from   : {field.meta.get('reference_file')}",
        f"grid         : stride {field.meta.get('stride')} "
        f"({field.meta.get('voxel_scale_mm')} mm), threshold {field.meta.get('threshold')}, "
        f"shape {field.volume_shape}",
        f"features     : {field.meta.get('features')} "
        f"(v{field.meta.get('feature_version')}), spatial_weight "
        f"{field.meta.get('spatial_weight')}, symmetric {field.meta.get('symmetric')}, "
        f"normalize_blocks {field.meta.get('normalize_blocks')}",
        f"contiguity   : dissolved {field.meta.get('dissolved_frac', 0):.1%} of nodes, "
        f"parcels {field.meta.get('parcels_before')} -> {field.meta.get('parcels_after')}",
        f"memberships  : top-{field.mem_idx.shape[1]}, argmax agrees with label on "
        f"{field.meta.get('membership_argmax_agreement', float('nan')):.1%} of nodes",
        f"border       : mean {field.meta.get('d_border_mm_mean', float('nan')):.3f} mm, "
        f"{field.meta.get('border_adjacent_frac', float('nan')):.1%} of nodes border-adjacent",
        "",
        f"{'parcel':>7} {'nodes':>10} {'mm^3':>9} {'share':>7}   "
        f"{'mean top-1 membership':>21}",
    ]
    top1 = field.mem_val[:, 0]
    for k in order:
        sel = field.labels == k
        lines.append(f"{k:>7d} {counts[k]:>10,d} {counts[k] * vox_mm3:>9.2f} "
                     f"{counts[k] / max(field.labels.size, 1):>6.2%}   "
                     f"{float(top1[sel].mean()) if sel.any() else float('nan'):>21.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# napari
# --------------------------------------------------------------------------- #
def launch(field, ref_sub, ndisplay=2):
    import napari

    lab_vol = label_volume(field)
    scale = (float(field.meta.get("voxel_scale_mm", 1.0)),) * 3
    viewer = napari.Viewer(
        title=f"parcelgp — {field.n_parcels} parcels "
              f"({field.meta.get('features')}, stride {field.meta.get('stride')})",
        ndisplay=ndisplay)

    viewer.add_image(ref_sub, name="reference", colormap="gray", scale=scale,
                     contrast_limits=(0, float(np.percentile(ref_sub, 99.5)) or 1.0))
    # +1 so 0 means background: napari's Labels layer treats 0 as "not a label"
    # and renders it transparent, which is exactly what non-tissue should be.
    lab_layer = viewer.add_labels(np.where(lab_vol >= 0, lab_vol + 1, 0).astype(np.int32),
                                  name="parcels", scale=scale, opacity=0.55)
    # napari picks its own label colours; override with ours so the parcel ids
    # look the same here as in the montage and the explorer.
    colors = parcel_colors(field.n_parcels, alpha=1.0)
    try:
        from napari.utils import DirectLabelColormap
        cmap = {i + 1: colors[i] for i in range(field.n_parcels)}
        cmap[None] = np.array([0, 0, 0, 0], np.float32)
        lab_layer.colormap = DirectLabelColormap(color_dict=cmap)
    except Exception as e:                       # older napari: keep its default
        log.info("using napari's default label colours (%s)", e)

    viewer.add_labels(_mask_volume(field, border_mask(field)).astype(np.uint8),
                      name="parcel borders", scale=scale, opacity=0.9, visible=False)
    viewer.add_image(_node_volume(field, field.mem_val[:, 0]), name="top-1 membership",
                     colormap="magma", scale=scale, contrast_limits=(0.25, 1.0),
                     visible=False)
    viewer.add_image(_node_volume(field, field.d_border_rel), name="d_border_rel",
                     colormap="viridis", scale=scale, contrast_limits=(0, 1),
                     visible=False)

    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QTextEdit
    box = QTextEdit(); box.setReadOnly(True)
    mono = QFont("Monospace"); mono.setStyleHint(QFont.TypeWriter); mono.setPointSize(9)
    box.setFont(mono); box.setMinimumWidth(520)
    box.setPlainText(format_stats(field))
    viewer.window.add_dock_widget(box, name="parcellation", area="right")
    napari.run()


# --------------------------------------------------------------------------- #
def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--field", required=True, help="A built parcel field .npz.")
    p.add_argument("--reference-file", default=None,
                   help="reference_image.npy for the greyscale underlay. "
                        "Defaults to the file recorded in the field's metadata.")
    p.add_argument("--montage", default=None,
                   help="Write a PNG grid of sections instead of opening napari.")
    p.add_argument("--n-slices", type=int, default=9)
    p.add_argument("--axis", type=int, choices=(0, 1, 2), default=0,
                   help="Slicing axis: 0=AP (coronal), 1=DV, 2=LR (sagittal).")
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Parcel tint opacity over the reference.")
    p.add_argument("--ncols", type=int, default=0, help="Montage columns (0 = square).")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--ndisplay", type=int, choices=(2, 3), default=2,
                   help="napari: start in 2D slice view or 3D.")
    p.add_argument("--stats", action="store_true",
                   help="Print the parcel table and exit (no window, no PNG).")
    a = p.parse_args(argv)

    field = ParcelField.load(a.field)
    if a.stats:
        print(format_stats(field))
        return

    ref_file = a.reference_file or field.meta.get("reference_file")
    if not ref_file or not Path(ref_file).exists():
        raise SystemExit(
            f"reference image not found ({ref_file!r}). The field records the file "
            f"it was built from; pass --reference-file if it has moved.")
    ref_sub = reference_subvolume(ref_file, field)

    if a.montage:
        montage(field, ref_sub, a.montage, n_slices=a.n_slices, axis=a.axis,
                alpha=a.alpha, ncols=a.ncols, dpi=a.dpi)
        return
    launch(field, ref_sub, ndisplay=a.ndisplay)


if __name__ == "__main__":
    main()
