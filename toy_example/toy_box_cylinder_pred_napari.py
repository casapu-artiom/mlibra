#!/usr/bin/env python
# encoding: utf-8
"""
toy_box_cylinder_pred_napari.py
================================

Visualise GP predictions and the manifold eigenbasis for the box+cylinder toy.
Loads the .npz written by toy_box_cylinder_gp_compare.py --dump-predictions.

Layers (all on the same 3D point cloud):

  Prediction layers:
    true signal                  ground truth
    pred: <method>               GP prediction per method (shared ± scale)
    |err|: <method>              absolute error vs truth (magma)
    test points (truth)          test split only, truth colour
    surface type                 cylinder (blue) vs box wall (orange)

  Kernel / distance layers (CLICK any layer to move the source):
    manifold kernel: plain       K_man from the plain eigenbasis
    manifold kernel: atlas       K_man from the atlas eigenbasis
    euclidean kernel             K_euc = exp(-d^2 / 2 l^2)
    euclidean distance           ||x - x_src||_3D
    surface distance             distance in unrolled intrinsic coords (approx)

  Eigenvector layers:
    eigvec 0..N (atlas/plain)    the Laplacian modes the kernel is built on

Run on a machine with a display:
    pip install "napari[all]" scikit-learn scipy matplotlib
    python toy_box_cylinder_gp_compare.py --mode surface \\
        --dump-predictions preds_boxcyl
    python toy_box_cylinder_pred_napari.py preds_boxcyl_surface.npz
"""
from __future__ import annotations
import argparse
import numpy as np

from toy_box_cylinder import LABEL_CYL, LABEL_BOX


def colormap_rgba(vals, cmap="coolwarm", vmin=None, vmax=None):
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
    for flag in ("border_width_is_relative", "edge_width_is_relative"):
        try:
            setattr(layer, flag, False)
            setattr(layer, flag.replace("_is_relative", ""), 0.0)
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


def spectral_density(eigval, nu, ell):
    return (2.0 * nu / (ell ** 2) + np.clip(eigval, 0.0, None)) ** (-nu)


def features_from_eig(eigvec, eigval, nu, ell):
    return eigvec * np.sqrt(spectral_density(eigval, nu, ell))   # (N, M)


def recompute_eigenbasis(X, labels, inflation, k, n_modes, graphbandwidth):
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import kneighbors_graph
    D = kneighbors_graph(X, k, mode="distance"); D = D.maximum(D.T).tocoo()
    bw = graphbandwidth if graphbandwidth and graphbandwidth > 0 else float(np.median(D.data))
    d2 = D.data ** 2
    if labels is not None and inflation and inflation != 1.0:
        cross = labels[D.row] != labels[D.col]; d2 = d2 * np.where(cross, inflation, 1.0)
    w = np.exp(-d2 / (4 * bw * bw))
    N = X.shape[0]
    W = sparse.coo_matrix((w, (D.row, D.col)), shape=(N, N)).tocsr(); W = 0.5 * (W + W.T)
    deg = np.asarray(W.sum(1)).ravel(); dinv = sparse.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sparse.identity(N) - dinv @ W @ dinv; L = 0.5 * (L + L.T)
    vals, vecs = eigsh(L, k=n_modes + 1, sigma=0, which="LM")
    o = np.argsort(vals)
    return vecs[:, o].astype(np.float32), np.clip(vals[o], 0.0, None).astype(np.float32)


def load_eigenbases(d, args, X, labels):
    eig = {}
    if "eig_vec" in d.files:
        names = [str(n) for n in d["eig_names"]]
        nu = float(d["eig_nu"])
        for i, en in enumerate(names):
            eig[en] = (d["eig_vec"][i], d["eig_val"][i], float(d["eig_ell"][i]), nu)
    else:
        ev, ea = recompute_eigenbasis(X, labels, args.inflation, args.knn_k,
                                      args.num_modes, args.graphbandwidth)
        key = "atlas" if (labels is not None and args.inflation != 1.0) else "plain"
        eig[key] = (ev, ea, args.lengthscale, args.nu)
    return eig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help=".npz from toy_box_cylinder_gp_compare.py "
                                "--dump-predictions")
    ap.add_argument("--point-size",      type=float, default=0.0,
                    help="point diameter in data units; 0 => auto (1.3x median NN)")
    ap.add_argument("--blending",        default="opaque",
                    choices=["opaque", "translucent", "additive"])
    ap.add_argument("--euc-lengthscale", type=float, default=0.0,
                    help="euclidean-kernel lengthscale; 0 => auto (5x NN spacing)")
    ap.add_argument("--n-eig",           type=int,   default=8,
                    help="# eigenvectors to cycle through with prev/next buttons")
    ap.add_argument("--kernel",          default="atlas",
                    help="which eigenbasis to use for eigvec layers (plain or atlas)")
    # fallback eigenbasis (only used when the dump lacks eig_vec)
    ap.add_argument("--inflation",       type=float, default=50.0)
    ap.add_argument("--knn-k",           type=int,   default=15)
    ap.add_argument("--num-modes",       type=int,   default=200)
    ap.add_argument("--nu",              type=int,   default=2)
    ap.add_argument("--lengthscale",     type=float, default=1.0)
    ap.add_argument("--graphbandwidth",  type=float, default=0.0)
    args = ap.parse_args()

    import napari

    d = np.load(args.npz, allow_pickle=True)
    X      = d["X"]
    y      = d["y_true"]
    names  = [str(n) for n in d["names"]]
    preds  = d["preds"]                          # (M, N)
    test_idx = d["test_idx"]
    intr   = d["intrinsic"] if "intrinsic" in d.files else None
    labels = d["labels"]    if "labels" in d.files   else None
    mode   = str(d["mode"])  if "mode"   in d.files   else "surface"

    # napari: z-axis (cylinder axis) first
    pts = X[:, [2, 0, 1]]

    # auto sizes
    nn1 = None
    if args.point_size <= 0 or args.euc_lengthscale <= 0:
        from sklearn.neighbors import NearestNeighbors
        _dd, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
        nn1 = float(np.median(_dd[:, 1]))
    psize = args.point_size if args.point_size > 0 else 1.3 * nn1
    print(f"point size = {psize:.4g}  (override with --point-size)")
    euc_ell = args.euc_lengthscale if args.euc_lengthscale > 0 else 5.0 * nn1

    eig   = load_eigenbases(d, args, X, labels)
    feats = {en: features_from_eig(ev, ea, nu_, ell_)
             for en, (ev, ea, ell_, nu_) in eig.items()}

    amp     = float(np.nanpercentile(np.abs(y), 99)) or 1.0
    vmin, vmax = -amp, amp
    err_max = float(np.nanpercentile(np.abs(preds - y[None, :]), 99)) or 1.0

    viewer = napari.Viewer(ndisplay=3,
                           title=f"box+cyl predictions  mode={mode}  {args.npz}")
    layers = []

    # ---- static layers -------------------------------------------------------

    # surface type (blue = cylinder, orange = box)
    if labels is not None:
        SURF_COLORS = np.array([[0.2, 0.5, 0.9, 1.0],
                                 [1.0, 0.6, 0.1, 1.0]])
        lay_lbl = viewer.add_points(
            pts, name="surface type (blue=cyl, orange=box)",
            size=psize, shading="none", visible=True,
            face_color=SURF_COLORS[labels])
        layers.append(lay_lbl)

    # true signal
    lay_true = viewer.add_points(
        pts, name="true signal", size=psize,
        face_color=colormap_rgba(y, "coolwarm", vmin, vmax), shading="none")
    layers.append(lay_true)

    # per-method prediction + error
    for m, name in enumerate(names):
        short = (name.replace("manifold (", "").replace("faiss_atlas_weighted", "atlas")
                     .replace(")", "").replace(" (3D Matern", "").replace(" ", "_")
                     .replace("atlas_x", "atlasx"))
        lp = viewer.add_points(
            pts, name=f"pred: {short}", size=psize,
            visible=(m == 0),
            face_color=colormap_rgba(preds[m], "coolwarm", vmin, vmax), shading="none")
        le = viewer.add_points(
            pts, name=f"|err|: {short}", size=psize, visible=False,
            face_color=colormap_rgba(np.abs(preds[m] - y), "magma", 0.0, err_max),
            shading="none")
        layers += [lp, le]

    lay_test = viewer.add_points(
        pts[test_idx], name="test points (truth)",
        size=psize * 1.6, visible=False,
        face_color=colormap_rgba(y[test_idx], "coolwarm", vmin, vmax), shading="none")
    layers.append(lay_test)

    # ---- dynamic kernel / distance layers ------------------------------------
    state = {"src": int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))}

    def man_kernel_from(en, i):
        F = feats[en]; kv = F @ F[i]
        return kv / (kv[i] + 1e-12)

    def euc_kernel_from(i):
        e = np.linalg.norm(X - X[i], axis=1)
        return np.exp(-0.5 * (e / euc_ell) ** 2)

    mk_layers = {}
    for en in feats:
        mk_layers[en] = viewer.add_points(
            pts, name=f"manifold kernel: {en} (src)",
            size=psize, visible=False, shading="none")
    lay_ek = viewer.add_points(pts, name="euclidean kernel (src)",
                               size=psize, visible=False, shading="none")
    lay_ed = viewer.add_points(pts, name="euclidean distance (src)",
                               size=psize, visible=False, shading="none")
    lay_sd = None
    if intr is not None:
        lay_sd = viewer.add_points(pts, name="surface distance (src, approx)",
                                   size=psize, visible=False, shading="none")
    lay_src = viewer.add_points(
        pts[state["src"]][None], name="source",
        size=psize * 4, face_color="lime", shading="none")
    dyn = list(mk_layers.values()) + [lay_ek, lay_ed]
    if lay_sd is not None:
        dyn.append(lay_sd)
    layers += dyn + [lay_src]

    # ---- single eigenvector layer (prev/next buttons cycle through eigvecs) --
    key = args.kernel if args.kernel in eig else next(iter(eig))
    ev_show = eig[key][0]
    n_eig_avail = min(args.n_eig, ev_show.shape[1])
    eig_nav = {"idx": 0}

    def _eig_rgba(q):
        a = float(np.nanpercentile(np.abs(ev_show[:, q]), 99)) or 1.0
        return colormap_rgba(ev_show[:, q], "coolwarm", -a, a)

    lay_eig = viewer.add_points(
        pts, name=f"eigvec 0 / {n_eig_avail - 1} ({key})",
        size=psize, visible=False,
        face_color=_eig_rgba(0), shading="none")
    layers.append(lay_eig)

    for lyr in layers:
        _no_border(lyr, blending=args.blending)

    def refresh():
        i = state["src"]
        for en in feats:
            mk_layers[en].face_color = colormap_rgba(man_kernel_from(en, i), "viridis", 0, 1)
        lay_ek.face_color = colormap_rgba(euc_kernel_from(i), "viridis", 0, 1)
        lay_ed.face_color = colormap_rgba(np.linalg.norm(X - X[i], axis=1), "magma")
        if lay_sd is not None and intr is not None:
            lay_sd.face_color = colormap_rgba(
                np.linalg.norm(intr - intr[i], axis=1), "magma")
        lay_src.data = pts[i][None]
        surf = "cylinder" if (labels is not None and labels[i] == LABEL_CYL) else "box/volume"
        print(f"  source #{i} on {surf}  signal={y[i]:+.3f}")
        for lyr in dyn + [lay_src]:
            _no_border(lyr, blending=args.blending); lyr.refresh()

    def make_click(layer):
        def _cb(layer, event):
            idx = layer.get_value(
                event.position, view_direction=event.view_direction,
                dims_displayed=event.dims_displayed, world=True)
            if idx is not None:
                state["src"] = int(idx); refresh()
        return _cb

    for lyr in layers:
        if lyr is not lay_src:
            lyr.mouse_drag_callbacks.append(make_click(lyr))

    refresh()

    # ---- navigation widgets (magicgui) -------------------------------------
    from magicgui.widgets import Container, PushButton, Label as MLabel

    # eigenvector prev / next
    eig_lbl = MLabel(value=f"0 / {n_eig_avail - 1}  [{key}]")
    prev_e  = PushButton(text="◄ prev eigvec")
    next_e  = PushButton(text="next eigvec ►")

    def _go_eig(delta):
        eig_nav["idx"] = (eig_nav["idx"] + delta) % n_eig_avail
        q = eig_nav["idx"]
        lay_eig.face_color = _eig_rgba(q)
        lay_eig.name = f"eigvec {q} / {n_eig_avail - 1} ({key})"
        _no_border(lay_eig, blending=args.blending)
        lay_eig.refresh()
        eig_lbl.value = f"{q} / {n_eig_avail - 1}  [{key}]"

    prev_e.clicked.connect(lambda: _go_eig(-1))
    next_e.clicked.connect(lambda: _go_eig(+1))
    viewer.window.add_dock_widget(
        Container(widgets=[prev_e, eig_lbl, next_e], layout="horizontal"),
        name="Eigenvectors", area="left")

    # source node prev / next  (~100 pre-selected random nodes)
    _rng_src  = np.random.default_rng(42)
    src_nodes = _rng_src.choice(len(X), size=min(100, len(X)), replace=False)
    src_nav   = {"idx": 0}
    state["src"] = int(src_nodes[0])
    refresh()  # re-render with the first nav source

    src_lbl = MLabel(value=f"0 / {len(src_nodes) - 1}  [node {src_nodes[0]}]")
    prev_s  = PushButton(text="◄ prev source")
    next_s  = PushButton(text="next source ►")

    def _go_src(delta):
        src_nav["idx"] = (src_nav["idx"] + delta) % len(src_nodes)
        state["src"] = int(src_nodes[src_nav["idx"]])
        refresh()
        src_lbl.value = (f"{src_nav['idx']} / {len(src_nodes) - 1}"
                         f"  [node {state['src']}]")

    prev_s.clicked.connect(lambda: _go_src(-1))
    next_s.clicked.connect(lambda: _go_src(+1))
    viewer.window.add_dock_widget(
        Container(widgets=[prev_s, src_lbl, next_s], layout="horizontal"),
        name="Source node", area="left")

    print("Use buttons (left dock) or click any layer to set the kernel source.\n"
          "Eigvec buttons cycle through graph-Laplacian eigenmodes (◄/► Eigenvectors).\n"
          "Source buttons step through 100 pre-selected random nodes (◄/► Source node).\n"
          "Toggle 'manifold kernel: atlas' vs 'euclidean kernel' to see the fold effect.")
    napari.run()


if __name__ == "__main__":
    main()
