#!/usr/bin/env python
# encoding: utf-8
"""
toy_manifold_napari.py
======================

Interactive napari viewer for the folded toy manifold. Shows, on the same 3D
point cloud, several layers you can toggle:

  - signal              : the geodesic-dependent value function f(s)
  - manifold kernel     : K_man(source, .) -- the spectral graph kernel from a
                          chosen source point (the "2D kernel" over the surface)
  - euclidean kernel    : K_euc(source, .) = exp(-d_3D^2 / 2 l^2)  for contrast
  - euclidean distance  : ||x - x_source||_3D
  - geodesic distance   : true intrinsic distance from the source

CLICK any point (on the "pick source" layer) to move the source; the kernel and
distance layers recompute live. This makes the fold obvious: the euclidean
kernel/distance "leak" across the fold to the wrong layer, while the manifold
kernel and geodesic distance stay on the correct sheet.

Run on a machine with a display:
    pip install "napari[all]" scikit-learn scipy matplotlib
    python toy_manifold_napari.py                      # plain euclidean kNN graph
    python toy_manifold_napari.py --inflation 50       # + atlas prior (respects fold)

The manifold kernel's graph is always built on EUCLIDEAN coords (what you have on
real data). --inflation applies the faiss_atlas_weighted atlas prior: cross-layer
edges are down-weighted so the kernel stops bridging the fold. With --inflation 1
(default) the kernel bridges the fold and looks like the euclidean kernel; raise it
to make the kernel respect the fold. See the --graphbandwidth note: the bandwidth
must be fixed and not too large or the inflation has no effect.
"""
from __future__ import annotations
import argparse
import inspect
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

from toy_manifold import make_folded_manifold, geodesic_from, euclidean_from


def gen_manifold(signal="geodesic", **kw):
    """Call make_folded_manifold tolerant of its exact signature (see gp_compare)."""
    params = inspect.signature(make_folded_manifold).parameters
    if "signal_kind" in params:
        kw["signal_kind"] = signal
    elif signal != "geodesic":
        raise SystemExit("This make_folded_manifold has no signal_kind: only the "
                         "geodesic signal is available. Re-add signal_kind to use "
                         "--signal euclidean.")
    return make_folded_manifold(**{k: v for k, v in kw.items() if k in params})


# --------------------------------------------------------------------------- #
# numpy-only eigenbasis + scaled spectral features (no torch needed here)
# --------------------------------------------------------------------------- #
def spectral_features(coords, labels=None, inflation=1.0, graphbandwidth=0.0,
                      k=15, n_modes=300, nu=2, lengthscale=16.0):
    """Build the manifold-kernel eigenbasis from a kNN graph on EUCLIDEAN coords
    (what you actually have on real data). Fold-awareness comes from the atlas
    prior: cross-`label` edges have their squared distance multiplied by
    `inflation` (the faiss_atlas_weighted mechanism), which weakens those edges.

    CRITICAL (graphbandwidth): the affinity is w = exp(-F*d^2 / (4*bw^2)). The
    bandwidth `bw` must be FIXED and computed from the ORIGINAL distances -- if it
    were recomputed from the inflated distances, F would cancel and the prior would
    do nothing. And if `bw` is too LARGE, F*d^2/(4*bw^2) is tiny even on cross
    edges so w_cross ~ w_within ~ 1 and the prior again can't separate layers.
    Auto bw = median original edge distance keeps within-edge weights ~O(1) and,
    with inflation, drives cross-edge weights toward 0."""
    D = kneighbors_graph(coords, k, mode="distance"); D = D.maximum(D.T).tocoo()
    bw = graphbandwidth if (graphbandwidth and graphbandwidth > 0) \
        else float(np.median(D.data))                 # FIXED, from ORIGINAL distances
    d2 = D.data ** 2
    if labels is not None and inflation and inflation != 1.0:
        cross = labels[D.row] != labels[D.col]
        d2 = d2 * np.where(cross, inflation, 1.0)
    else:
        cross = np.zeros(d2.shape[0], dtype=bool)
    w = np.exp(-d2 / (4 * bw * bw))
    N = coords.shape[0]
    W = sparse.coo_matrix((w, (D.row, D.col)), shape=(N, N)).tocsr(); W = 0.5 * (W + W.T)
    deg = np.asarray(W.sum(1)).ravel()
    dinv = sparse.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sparse.identity(N) - dinv @ W @ dinv; L = 0.5 * (L + L.T)
    vals, vecs = eigsh(L, k=n_modes + 1, sigma=0, which="LM")
    o = np.argsort(vals); vals = np.clip(vals[o], 0, None); vecs = vecs[:, o]
    sd = (2 * nu / lengthscale ** 2 + vals) ** (-nu)   # Matern spectral density
    info = dict(
        bw=bw,
        w_within=float(np.median(w[~cross])) if (~cross).any() else float("nan"),
        w_cross=float(np.median(w[cross])) if cross.any() else float("nan"),
    )
    return vecs * np.sqrt(sd), info                    # (N, M) scaled features


def colormap_rgba(vals, cmap="viridis", vmin=None, vmax=None):
    import matplotlib
    try:
        cmap_obj = matplotlib.colormaps[cmap]          # matplotlib >= 3.7
    except Exception:
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(cmap)
    vmin = np.min(vals) if vmin is None else vmin
    vmax = np.max(vals) if vmax is None else vmax
    v = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return cmap_obj(v)


def _no_border(layer):
    """Remove point outlines across napari versions (border_* new, edge_* old)."""
    for attr in ("border_width", "edge_width"):
        try:
            setattr(layer, attr, 0.0)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--cycles-per-turn", type=float, default=1.5)
    ap.add_argument("--signal", choices=["geodesic", "euclidean"], default="geodesic")
    ap.add_argument("--shell-cycles", type=float, default=4.0)
    ap.add_argument("--n-turns", type=float, default=3.5)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=6.0)
    ap.add_argument("--thickness", type=float, default=0.0,
                    help="sheet thickness (isotropic noise). MUST stay well below "
                         "--gap or the fold smears into a blob; >gap/3 is flagged.")
    ap.add_argument("--inflation", type=float, default=1.0,
                    help="cross-layer edge inflation (faiss_atlas_weighted). 1.0 = "
                         "plain euclidean kNN (manifold kernel bridges the fold); "
                         ">1 applies the atlas prior so the kernel respects the fold.")
    ap.add_argument("--graphbandwidth", type=float, default=0.1,
                    help="affinity bandwidth (FIXED). 0 => auto = median edge "
                         "distance. Too large => inflation can't separate layers.")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--num-modes", type=int, default=300)
    ap.add_argument("--nu", type=int, default=2)
    ap.add_argument("--lengthscale", type=float, default=16.0,
                    help="kernel lengthscale on the normalized-Laplacian scale "
                         "(eigvals in [0,2]); ~16 gives a smooth geodesic-tracking "
                         "kernel, ~1 gives a near-delta kernel")
    ap.add_argument("--euc-lengthscale", type=float, default=1.0)
    ap.add_argument("--point-size", type=float, default=0.08)
    args = ap.parse_args()

    import napari

    if args.thickness > args.gap / 3:
        print(f"WARNING: thickness={args.thickness} is large vs gap={args.gap}. "
              f"Layers will smear together and the fold (the point of the toy) "
              f"degrades. Keep thickness <~ {args.gap/4:.3f}, or raise --gap to "
              f"keep the layers separated (e.g. --gap {args.thickness*4:.2f}).")

    data = gen_manifold(signal=args.signal, n=args.n, gap=args.gap,
                        cycles_per_turn=args.cycles_per_turn,
                        shell_cycles=args.shell_cycles,
                        n_turns=args.n_turns, r0=args.r0,
                        height=args.height, thickness=args.thickness)
    X, intr, sig = data["X"], data["intrinsic"], data["signal"]
    # napari expects (z, y, x)-ish; we just feed the 3 columns as-is.
    pts = X[:, [1, 0, 2]]  # put height first for a nicer default camera; cosmetic

    # graph is built on EUCLIDEAN coords (what you have on real data); the atlas
    # prior (cross-layer inflation) is what makes the kernel respect the fold.
    labels = np.floor(data["t"] / (2 * np.pi)).astype(np.int64)   # roll layer = 'region'
    feats, ginfo = spectral_features(
        X, labels=labels, inflation=args.inflation, graphbandwidth=args.graphbandwidth,
        k=args.knn_k, n_modes=args.num_modes, nu=args.nu, lengthscale=args.lengthscale)
    print(f"graph bandwidth bw={ginfo['bw']:.4g}  |  median edge weight  "
          f"within-layer={ginfo['w_within']:.3g}  cross-layer={ginfo['w_cross']:.3g}")
    if args.inflation > 1.0 and ginfo["w_cross"] > 0.1 * ginfo["w_within"]:
        print(f"WARNING: cross-layer weight {ginfo['w_cross']:.3g} is not << within-layer "
              f"{ginfo['w_within']:.3g} — the atlas prior isn't separating the layers. "
              f"bw={ginfo['bw']:.4g} is likely too large for inflation={args.inflation:g}; "
              f"lower --graphbandwidth (or leave it 0 for auto) or raise --inflation.")

    state = {"src": int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))}

    def manifold_kernel_from(i):
        kv = feats @ feats[i]
        return kv / (kv[i] + 1e-12)                        # normalize so K(src,src)=1

    def euclidean_kernel_from(i):
        e = euclidean_from(X, i)
        return np.exp(-0.5 * (e / args.euc_lengthscale) ** 2)

    viewer = napari.Viewer(ndisplay=3,
                           title=f"folded manifold (euclidean graph, inflation x{args.inflation:g})")

    # static: signal
    lay_sig = viewer.add_points(pts, name="signal", size=args.point_size,
                                face_color=colormap_rgba(sig, "coolwarm"), shading="none")

    # dynamic layers (recomputed on click)
    lay_mk = viewer.add_points(pts, name="manifold kernel (from src)",
                               size=args.point_size, visible=False, shading="none")
    lay_ek = viewer.add_points(pts, name="euclidean kernel (from src)",
                               size=args.point_size, visible=False, shading="none")
    lay_ed = viewer.add_points(pts, name="euclidean distance", size=args.point_size,
                               visible=False, shading="none")
    lay_gd = viewer.add_points(pts, name="geodesic distance", size=args.point_size,
                               visible=False, shading="none")
    # source marker (small layer, drawn on top)
    lay_src = viewer.add_points(pts[state["src"]][None], name="source",
                                size=args.point_size * 4, face_color="lime",
                                shading="none")

    data_layers = [lay_sig, lay_mk, lay_ek, lay_ed, lay_gd, lay_src]
    for l in data_layers:
        _no_border(l)                                  # clean filled dots, no gray rings

    def refresh():
        i = state["src"]
        lay_mk.face_color = colormap_rgba(manifold_kernel_from(i), "viridis", 0, 1)
        lay_ek.face_color = colormap_rgba(euclidean_kernel_from(i), "viridis", 0, 1)
        lay_ed.face_color = colormap_rgba(euclidean_from(X, i), "magma")
        lay_gd.face_color = colormap_rgba(geodesic_from(intr, i), "magma")
        lay_src.data = pts[i][None]
        for l in (lay_mk, lay_ek, lay_ed, lay_gd, lay_src):
            _no_border(l); l.refresh()

    def make_click(layer):
        def _on_click(layer, event):
            idx = layer.get_value(event.position, view_direction=event.view_direction,
                                  dims_displayed=event.dims_displayed, world=True)
            if idx is not None:
                state["src"] = int(idx)
                refresh()
        return _on_click

    # attach to every data layer so clicking works whichever layer is active
    for l in (lay_sig, lay_mk, lay_ek, lay_ed, lay_gd):
        l.mouse_drag_callbacks.append(make_click(l))

    refresh()
    print("Tip: show one layer at a time. Toggle 'manifold kernel' vs 'euclidean "
          "kernel' and watch the euclidean one leak across the fold to the wrong layer.")
    napari.run()


if __name__ == "__main__":
    main()