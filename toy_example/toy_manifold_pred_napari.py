#!/usr/bin/env python
# encoding: utf-8
"""
toy_manifold_pred_napari.py
===========================

Render GP predictions on the folded manifold in napari. Loads the .npz dumped by
toy_manifold_gp_compare.py (--dump-predictions) and shows, on the same 3D point
cloud, toggleable layers:

  - true signal
  - <method> prediction           (one per GP method, same color scale as truth)
  - <method> |error|              (abs error vs truth, magma)
  - test points (truth)           (marks where held-out metrics were computed)

Truth and all predictions share a symmetric color scale so they're directly
comparable; flip between "true signal" and a method's prediction to see where the
reconstruction is right or wrong (the fold is where euclidean predictions go bad).

Run on a machine with a display:
    pip install "napari[all]"
    python toy_manifold_pred_napari.py preds_geodesic.npz
"""
from __future__ import annotations
import argparse
import numpy as np


def colormap_rgba(vals, cmap="coolwarm", vmin=None, vmax=None):
    import matplotlib
    try:
        cmap_obj = matplotlib.colormaps[cmap]          # matplotlib >= 3.7
    except Exception:
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(cmap)
    vmin = float(np.min(vals)) if vmin is None else vmin
    vmax = float(np.max(vals)) if vmax is None else vmax
    v = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return cmap_obj(v)


def _no_border(layer):
    for attr in ("border_width", "edge_width"):
        try:
            setattr(layer, attr, 0.0)
        except Exception:
            pass
    # some napari builds keep a visible outline unless the relative flag is off
    for flag in ("border_width_is_relative", "edge_width_is_relative"):
        try:
            setattr(layer, flag, False)
            setattr(layer, flag.replace("_is_relative", ""), 0.0)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="predictions .npz from toy_manifold_gp_compare.py")
    ap.add_argument("--point-size", type=float, default=0.08)
    args = ap.parse_args()

    import napari

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"]; y = d["y_true"]
    names = [str(n) for n in d["names"]]
    preds = d["preds"]                                  # (M, N)
    test_idx = d["test_idx"]
    pts = X[:, [1, 0, 2]]                               # height-first, cosmetic

    # Scale colors to the TRUE signal's robust range. Using the max over
    # predictions too lets a broken/mean-reverted layer's outliers blow up the
    # scale and wash out every layer (including truth), so scope it to y.
    amp = float(np.nanpercentile(np.abs(y), 99)) or 1.0
    vmin, vmax = -amp, amp
    err_max = float(np.nanpercentile(np.abs(preds - y[None, :]), 99)) or 1.0

    viewer = napari.Viewer(ndisplay=3, title=f"predictions: {args.npz}")
    layers = []

    lay_true = viewer.add_points(pts, name="true signal", size=args.point_size,
                                 face_color=colormap_rgba(y, "coolwarm", vmin, vmax),
                                 shading="none")
    layers.append(lay_true)

    for m, name in enumerate(names):
        short = (name.replace("manifold (", "").replace("faiss_atlas_weighted", "atlas")
                     .replace(")", "").replace(" (3D Matern", "").replace(" ", "_"))
        lp = viewer.add_points(pts, name=f"pred: {short}", size=args.point_size,
                               visible=(m == 0),
                               face_color=colormap_rgba(preds[m], "coolwarm", vmin, vmax),
                               shading="none")
        le = viewer.add_points(pts, name=f"|err|: {short}", size=args.point_size,
                               visible=False,
                               face_color=colormap_rgba(np.abs(preds[m] - y), "magma",
                                                         0.0, err_max),
                               shading="none")
        layers += [lp, le]

    # mark the held-out test points (colored by truth) so you can see where eval ran
    lay_test = viewer.add_points(pts[test_idx], name="test points (truth)",
                                 size=args.point_size * 1.6, visible=False,
                                 face_color=colormap_rgba(y[test_idx], "coolwarm", vmin, vmax),
                                 shading="none")
    layers.append(lay_test)

    for l in layers:
        _no_border(l)

    print("Toggle 'true signal' vs each 'pred:' layer. The fold is where the "
          "euclidean prediction flips sign relative to truth; check '|err|:' layers.")
    napari.run()


if __name__ == "__main__":
    main()