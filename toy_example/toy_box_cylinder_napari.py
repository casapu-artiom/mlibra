#!/usr/bin/env python
# encoding: utf-8
"""
toy_box_cylinder_napari.py
==========================

Interactive napari viewer for the box+cylinder toy manifold.  Shows, on the
same 3D point cloud, toggleable layers:

  - signal              : antiphase geodesic target f(θ)
  - surface type        : cylinder (blue) vs box wall (orange) -- the two "layers"
  - manifold kernel     : K_man(source, .)  the spectral graph kernel from a chosen
                          source point; with --inflation > 1 it respects the gap
                          between the cylinder and the box wall
  - euclidean kernel    : K_euc(source, .) = exp(-d_3D^2 / 2 l^2)  for contrast
  - euclidean distance  : ||x - x_source||_3D
  - surface distance    : unrolled-surface distance from the source (approximate)

CLICK any data layer to move the source; the kernel and distance layers
recompute live.  This makes the fold obvious: the Euclidean kernel/distance leaks
across the 3D gap to the OPPOSITE surface, which carries an antiphase signal
value.  With --inflation 50, the manifold kernel is contained on the source's
own surface.

Key demonstration:
  python toy_box_cylinder_napari.py               # euclidean graph (leaks)
  python toy_box_cylinder_napari.py --inflation 50  # atlas prior (respects gap)

Try clicking on a cylinder point and watching the Euclidean kernel light up the
NEARBY BOX WALL (opposite signal!).  Then enable the manifold (atlas) kernel
and watch it stay on the cylinder.

Run on a machine with a display:
    pip install "napari[all]" scikit-learn scipy matplotlib
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

from toy_box_cylinder import make_box_cylinder, LABEL_CYL, LABEL_BOX


# =============================================================================
# Spectral kernel builder (numpy-only, no torch)
# =============================================================================

def spectral_features(coords, labels=None, inflation=1.0, graphbandwidth=0.0,
                      k=15, n_modes=300, nu=2, lengthscale=16.0):
    """Build manifold-kernel eigenbasis from a kNN graph on coords.

    With inflation > 1 and labels provided, cross-label edges (cylinder ↔ box)
    are penalised so the kernel respects the surface gap.

    graphbandwidth MUST be computed from original (un-inflated) distances; if
    the bandwidth were recomputed after inflation the inflation factor cancels.
    Auto bw = median original edge distance.
    """
    D = kneighbors_graph(coords, k, mode="distance")
    D = D.maximum(D.T).tocoo()
    bw = (graphbandwidth if graphbandwidth and graphbandwidth > 0
          else float(np.median(D.data)))
    d2 = D.data ** 2
    cross = np.zeros(len(d2), dtype=bool)
    if labels is not None and inflation and inflation != 1.0:
        cross = labels[D.row] != labels[D.col]
        d2 = d2 * np.where(cross, float(inflation), 1.0)
    w = np.exp(-d2 / (4.0 * bw * bw))
    N = coords.shape[0]
    W = sparse.coo_matrix((w, (D.row, D.col)), shape=(N, N)).tocsr()
    W = 0.5 * (W + W.T)
    deg  = np.asarray(W.sum(1)).ravel()
    dinv = sparse.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L    = sparse.identity(N) - dinv @ W @ dinv
    L    = 0.5 * (L + L.T)
    vals, vecs = eigsh(L, k=n_modes + 1, sigma=0, which="LM")
    o = np.argsort(vals)
    vals = np.clip(vals[o], 0.0, None); vecs = vecs[:, o]
    sd = (2.0 * nu / lengthscale ** 2 + vals) ** (-nu)
    info = dict(
        bw=bw,
        w_within=float(np.median(w[~cross])) if (~cross).any() else float("nan"),
        w_cross =float(np.median(w[cross]))  if cross.any()  else float("nan"),
        n_cross_edges=int(cross.sum()),
    )
    return vecs * np.sqrt(sd), info            # (N, M) scaled features


# =============================================================================
# Helpers
# =============================================================================

def colormap_rgba(vals, cmap="viridis", vmin=None, vmax=None):
    import matplotlib
    try:
        cmap_obj = matplotlib.colormaps[cmap]
    except Exception:
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(cmap)
    vmin = float(np.min(vals)) if vmin is None else vmin
    vmax = float(np.max(vals)) if vmax is None else vmax
    v = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    return cmap_obj(v)


def _no_border(layer, blending=None):
    for attr in ("border_width", "edge_width"):
        try:
            setattr(layer, attr, 0.0)
        except Exception:
            pass
    try:
        layer.antialiasing = 0.0
    except Exception:
        pass
    if blending is not None:
        try:
            layer.blending = blending
        except Exception:
            pass


# =============================================================================
# Surface distance (approximate)
# =============================================================================

def surface_dist_from(intr, i):
    """Euclidean distance in intrinsic (arc, z) coords.

    On the cylinder surface:  arc = θ * R  (unrolled circumference)
    On the box faces:         arc = θ * L  (larger scale)
    This approximates the surface geodesic within each surface; the
    cross-surface component is not the true geodesic (which must go around
    the cylinder cap) but gives a qualitative indication of surface distance.
    """
    return np.linalg.norm(intr - intr[i], axis=1)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Geometry
    ap.add_argument("--n",               type=int,   default=8_000)
    ap.add_argument("--box-half",        type=float, default=4.0)
    ap.add_argument("--cyl-radius",      type=float, default=3.8,
                    help="cylinder radius; radial gap = box_half - cyl_radius")
    ap.add_argument("--cyl-half-height", type=float, default=3.0)
    ap.add_argument("--mode",            choices=["surface", "volume"],
                    default="surface",
                    help="surface: data on the surfaces (fold present); "
                         "volume: data fills the box (no fold)")
    ap.add_argument("--signal",          choices=["geodesic", "ambient"],
                    default="geodesic",
                    help="geodesic: antiphase cylinder/box signal; "
                         "ambient: smooth 3D function")
    ap.add_argument("--n-azimuthal-cycles", type=int, default=2)
    # Manifold kernel
    ap.add_argument("--inflation",       type=float, default=1.0,
                    help="cross-surface edge inflation (atlas prior). "
                         "1.0 = plain Euclidean kNN (kernel leaks across gap); "
                         "> 1 respects the cylinder/box gap. Try 50.")
    ap.add_argument("--graphbandwidth",  type=float, default=0.0,
                    help="affinity bandwidth; 0 => auto (median edge distance). "
                         "MUST be fixed from original distances for inflation to work.")
    ap.add_argument("--knn-k",           type=int,   default=15)
    ap.add_argument("--num-modes",       type=int,   default=200)
    ap.add_argument("--nu",              type=int,   default=2)
    ap.add_argument("--lengthscale",     type=float, default=16.0,
                    help="kernel lengthscale on Laplacian eigval scale (~[0,2]); "
                         "16 = smooth global kernel, 1 = near-delta kernel")
    ap.add_argument("--euc-lengthscale", type=float, default=0.0,
                    help="Euclidean kernel lengthscale; 0 => auto (5x NN spacing)")
    # Display
    ap.add_argument("--point-size",      type=float, default=0.0,
                    help="point diameter in data units; 0 => auto (1.3x median NN)")
    ap.add_argument("--blending",        default="opaque",
                    choices=["opaque", "translucent", "additive"])
    ap.add_argument("--seed",            type=int,   default=0)
    args = ap.parse_args()

    import napari

    data = make_box_cylinder(
        n=args.n, box_half=args.box_half, cyl_radius=args.cyl_radius,
        cyl_half_height=args.cyl_half_height, mode=args.mode,
        signal_kind=args.signal, n_azimuthal_cycles=args.n_azimuthal_cycles,
        seed=args.seed)
    X, intr, sig, labels = (data["X"], data["intrinsic"],
                             data["signal"], data["labels"])

    # napari convention: first axis = Z (up); put the cylinder axis (z) first
    pts = X[:, [2, 0, 1]]

    # auto point size from NN spacing
    if args.point_size <= 0 or args.euc_lengthscale <= 0:
        from sklearn.neighbors import NearestNeighbors as _NN
        _dd, _ = _NN(n_neighbors=2).fit(X).kneighbors(X)
        nn1 = float(np.median(_dd[:, 1]))
        if args.point_size <= 0:
            args.point_size = 1.3 * nn1
        if args.euc_lengthscale <= 0:
            args.euc_lengthscale = 5.0 * nn1
    print(f"point_size={args.point_size:.4g}  euc_lengthscale={args.euc_lengthscale:.4g}")

    # build spectral features
    feats, ginfo = spectral_features(
        X, labels=labels, inflation=args.inflation,
        graphbandwidth=args.graphbandwidth, k=args.knn_k,
        n_modes=args.num_modes, nu=args.nu, lengthscale=args.lengthscale)

    print(f"graph bandwidth bw={ginfo['bw']:.4g}  "
          f"n_cross_edges={ginfo['n_cross_edges']}  "
          f"w_within={ginfo['w_within']:.3g}  w_cross={ginfo['w_cross']:.3g}")
    if args.inflation > 1.0 and ginfo["w_cross"] > 0.1 * ginfo["w_within"]:
        print(f"WARNING: cross-surface weight {ginfo['w_cross']:.3g} is not << within "
              f"{ginfo['w_within']:.3g}. Atlas prior not cleanly separating surfaces. "
              f"Try lower --graphbandwidth or larger --inflation.")
    if ginfo["n_cross_edges"] == 0 and args.inflation > 1.0:
        print("INFO: 0 cross-surface KNN edges — the gap is larger than the kNN "
              "neighbourhood radius.  Inflation has no edges to penalise.  "
              "The plain and atlas kernels will be identical here; the Euclidean "
              "GLOBAL KERNEL still reaches across the gap (that's the point).")

    state = {"src": int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))}

    def manifold_kernel_from(i):
        kv = feats @ feats[i]
        return kv / (kv[i] + 1e-12)

    def euclidean_kernel_from(i):
        e = np.linalg.norm(X - X[i], axis=1)
        return np.exp(-0.5 * (e / args.euc_lengthscale) ** 2)

    # --- surface-type colours -------------------------------------------------
    # blue = cylinder (LABEL_CYL=0), orange = box (LABEL_BOX=1)
    SURF_COLORS = np.array([[0.2, 0.5, 0.9, 1.0],   # cylinder: blue
                             [1.0, 0.6, 0.1, 1.0]])  # box: orange
    face_colors_label = SURF_COLORS[labels]

    viewer = napari.Viewer(
        ndisplay=3,
        title=f"box+cylinder  mode={args.mode}  inflation={args.inflation:g}")

    # --- static layers --------------------------------------------------------
    lay_sig = viewer.add_points(
        pts, name="signal", size=args.point_size, shading="none",
        face_color=colormap_rgba(sig, "coolwarm"))
    lay_lbl = viewer.add_points(
        pts, name="surface type (blue=cyl, orange=box)",
        size=args.point_size, shading="none", visible=False,
        face_color=face_colors_label)

    # --- dynamic layers -------------------------------------------------------
    lay_mk = viewer.add_points(pts, name="manifold kernel (src)",
                               size=args.point_size, visible=False, shading="none")
    lay_ek = viewer.add_points(pts, name="euclidean kernel (src)",
                               size=args.point_size, visible=False, shading="none")
    lay_ed = viewer.add_points(pts, name="euclidean distance (src)",
                               size=args.point_size, visible=False, shading="none")
    lay_sd = viewer.add_points(pts, name="surface distance (src, approx)",
                               size=args.point_size, visible=False, shading="none")
    lay_src = viewer.add_points(
        pts[state["src"]][None], name="source",
        size=args.point_size * 4, face_color="lime", shading="none")

    all_layers = [lay_sig, lay_lbl, lay_mk, lay_ek, lay_ed, lay_sd, lay_src]
    for lyr in all_layers:
        _no_border(lyr, blending=args.blending)

    def refresh():
        i = state["src"]
        lay_mk.face_color = colormap_rgba(manifold_kernel_from(i), "viridis", 0, 1)
        lay_ek.face_color = colormap_rgba(euclidean_kernel_from(i), "viridis", 0, 1)
        lay_ed.face_color = colormap_rgba(
            np.linalg.norm(X - X[i], axis=1), "magma")
        lay_sd.face_color = colormap_rgba(surface_dist_from(intr, i), "magma")
        lay_src.data = pts[i][None]
        for lyr in (lay_mk, lay_ek, lay_ed, lay_sd, lay_src):
            _no_border(lyr, blending=args.blending)
            lyr.refresh()
        surf = "cylinder" if labels[i] == LABEL_CYL else "box wall"
        print(f"  source #{i} on {surf}  θ={np.degrees(data['theta'][i]):+.1f}°  "
              f"r={data['r_xy'][i]:.3g}  z={X[i,2]:.3g}  "
              f"signal={sig[i]:+.3f}")

    def make_click(layer):
        def _cb(layer, event):
            idx = layer.get_value(
                event.position, view_direction=event.view_direction,
                dims_displayed=event.dims_displayed, world=True)
            if idx is not None:
                state["src"] = int(idx)
                refresh()
        return _cb

    for lyr in (lay_sig, lay_lbl, lay_mk, lay_ek, lay_ed, lay_sd):
        lyr.mouse_drag_callbacks.append(make_click(lyr))

    refresh()

    print("\nTip: click a cylinder point (blue in 'surface type' layer).  "
          "Toggle 'euclidean kernel' -- it lights up the NEARBY BOX WALL "
          "(orange surface, OPPOSITE signal).  "
          "Then toggle 'manifold kernel' and watch it stay on the cylinder.\n"
          "Run with --inflation 50 to make the manifold kernel respect the gap.")
    napari.run()


if __name__ == "__main__":
    main()
