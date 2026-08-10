#!/usr/bin/env python
# encoding: utf-8
"""
toy_gp_explorer_napari.py
=========================

Interactive napari explorer for the manifold-GP toy examples.

Pick a geometry (swiss roll / box+cylinder), a signal function, and generation +
GP hyper-parameters from the dock on the right, then press **Fit GP** to run the
whole comparison (euclidean Matern / manifold plain / manifold atlas / oracle) on
the GPU and drop the predictions, per-method error, and a clickable manifold-vs-
euclidean kernel view onto the 3-D cloud.

Design notes matching the request:
  * Defaults to the swiss roll.
  * A geometry dropdown switches to the box+cylinder (its parameter panel swaps in).
  * A signal dropdown + per-geometry parameter widgets.
  * Nothing expensive runs until you click **Fit GP** (a cheap **Preview geometry**
    button just shows the coloured cloud).
  * **Save default (this geometry)** writes the current control values to
    toy_gp_explorer_defaults.json (keyed per geometry) next to this script, so each
    geometry reopens from its own saved state. Switching geometry in-session also
    remembers unsaved edits per geometry.

Run on a machine with a display + GPU:
    pip install "napari[all]" scikit-learn scipy matplotlib
    python toy_gp_explorer_napari.py

Shares the real manifold_gp stack (NearestNeighbors -> inflate_cross_region_edges
-> GraphLaplacianOperator -> LaplacianEigensolver -> RiemannMaternKernel) with
toy_manifold_gp_compare.py / toy_box_cylinder_gp_compare.py.
"""
from __future__ import annotations
import json
import os
import tempfile
import traceback

import numpy as np
import torch
import gpytorch

from toy_manifold import make_folded_manifold
from toy_box_cylinder import make_box_cylinder, LABEL_CYL, LABEL_BOX

from manifold_gp.utils.nearest_neighbors import NearestNeighbors
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

torch.set_default_dtype(torch.float32)

DEFAULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "toy_gp_explorer_defaults.json")

SWISS_SIGNALS = ["geodesic", "ambient_x", "ambient_grid", "radial"]
BOX_SIGNALS = ["geodesic", "ambient"]

# Drawing every kNN edge of a large cloud swamps the canvas; each edge group is
# randomly subsampled down to this many segments for display only.
MAX_EDGES_DISPLAY = 40_000

# =============================================================================
# manifold_gp kernel builder (shared with the *_gp_compare.py scripts)
# =============================================================================

def standardize(X):
    """The exact input transform the kernels see (global isotropic rescale)."""
    return ((X - X.mean(0)) / X.std()).astype(np.float32)


def build_graph_edges(Xs, labels, k, device="cpu"):
    """Build the SAME kNN graph the manifold kernels use, for visualisation.

    Returns (edge_index (2,E) int64, edge_d2 (E,) float, cross (E,) bool, bw).
    `edge_d2` are squared distances in standardized units and `bw` is the
    auto graph bandwidth, so the Gaussian edge weight is exp(-d2 / (4 bw^2)) --
    matching GraphLaplacianOperator.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    coords = torch.as_tensor(Xs, dtype=torch.float32, device=dev).contiguous()
    edge_index, edge_value = NearestNeighbors(coords).graph(k)
    ei = edge_index.detach().cpu().numpy().astype(np.int64)
    d2 = edge_value.detach().cpu().numpy().astype(np.float64)
    cross = labels[ei[0]] != labels[ei[1]]
    bw = float(np.sqrt(np.median(d2)))
    return ei, d2, cross, bw


def edge_vectors(disp, ei):
    """(2,E) edge index -> napari Vectors array (E, 2, 3) of [origin, delta]."""
    p0 = disp[ei[0]].astype(np.float32)
    p1 = disp[ei[1]].astype(np.float32)
    return np.stack([p0, p1 - p0], axis=1)


def subsample_edges(mask, cap=MAX_EDGES_DISPLAY, seed=0):
    idx = np.flatnonzero(mask)
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=cap, replace=False))
    return idx


def build_riemann_kernel(coords_np, labels, inflation, k, n_modes, nu,
                         lap_norm, graphbandwidth, device):
    coords = torch.as_tensor(coords_np, dtype=torch.float32,
                             device=device).contiguous()
    knn = NearestNeighbors(coords)
    edge_index, edge_value = knn.graph(k)

    bw = (graphbandwidth if graphbandwidth and graphbandwidth > 0
          else float(np.sqrt(np.median(edge_value.detach().cpu().numpy()))))

    if inflation and inflation != 1.0:
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, labels,
            inflation=inflation, treat_zero_as_cross=False,
        )

    lap = GraphLaplacianOperator(
        edge_value, edge_index, coords.shape[0],
        torch.tensor(bw, device=device), lap_norm, True,
    )
    solver = LaplacianEigensolver(
        num_modes=n_modes,
        backend="cupy" if device.type == "cuda" else "scipy",
        ncv_min=max(1500, 3 * n_modes + 20), verbose=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        eigval, eigvec = solver.compute_or_load(
            lap, cache_dir=tmp, key="explorer",
            graphbandwidth=bw, laplacian_normalization=lap_norm,
            force_recompute=True, device=device,
        )

    kernel = RiemannMaternKernel(
        nu=nu, knn=knn, edge_index=edge_index, edge_value=edge_value,
        eigval=eigval, eigvec=eigvec, nearest_neighbors=k, num_modes=n_modes,
        bump_scale=0.1, bump_decay=0.01, laplacian_normalization=lap_norm,
        graphbandwidth_init=bw,
    ).to(device)
    kernel.eval()
    for name, p in kernel.named_parameters():
        if "graphbandwidth" in name:
            p.requires_grad_(False)
    return kernel, coords


# =============================================================================
# GP models (exact + sparse variational), shared with the compare scripts
# =============================================================================

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def fit_gp(base_kernel, train_x, train_y, iters=100, lr=0.01, noise=1e-2):
    dev = train_x.device
    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    lik.noise = noise
    model = ExactGP(train_x, train_y, lik, base_kernel).to(dev)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    with gpytorch.settings.max_cholesky_size(1_000_000):
        for _ in range(iters):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik


class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, base_kernel, learn_inducing):
        vd = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        vs = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=learn_inducing)
        super().__init__(vs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def fit_svgp(base_kernel, train_x, train_y, num_inducing, learn_inducing,
             iters=100, lr=0.1, noise=1e-2):
    dev = train_x.device
    m = min(num_inducing, train_x.size(0))
    sel = torch.randperm(train_x.size(0), device=dev)[:m]
    inducing = train_x[sel].clone()
    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    lik.noise = noise
    model = SVGP(inducing, base_kernel, learn_inducing).to(dev)
    model.train(); lik.train()
    opt = torch.optim.Adam(list(model.parameters()) + list(lik.parameters()), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=train_y.size(0))
    with gpytorch.settings.max_cholesky_size(1_000_000):
        for _ in range(iters):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik


def predict_gp(model, lik, x):
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(1_000_000), \
            gpytorch.settings.fast_pred_var(False):
        return lik(model(x)).mean


def metrics(pred, true):
    from scipy.stats import pearsonr, spearmanr
    p = pred.detach().cpu().numpy(); t = true.detach().cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    pear = float(pearsonr(p, t)[0])
    spear = float(spearmanr(p, t).correlation)
    return rmse, pear, spear


def suggest_lengthscale(eigval, eigvec, y_centered, nu, q=0.9):
    c = eigvec.t() @ y_centered
    E = c ** 2
    cum = torch.cumsum(E, 0) / E.sum().clamp(min=1e-30)
    idx = int((cum < q).sum().clamp(max=eigval.numel() - 1))
    lam_q = float(eigval[idx].clamp(min=1e-6))
    return float((2 * nu / lam_q) ** 0.5), lam_q


def _ell_of(kernel):
    try:
        return float(kernel.lengthscale.reshape(-1)[0].item())
    except Exception:
        return float("nan")


# =============================================================================
# Data generation (both geometries) -> a common bundle
# =============================================================================

def generate_data(cfg):
    """Return X, intrinsic, y, labels, display_pts for the chosen geometry."""
    geom = cfg["geometry"]
    if geom == "swiss roll":
        data = make_folded_manifold(
            n=cfg["n"], n_turns=cfg["n_turns"], gap=cfg["gap"], r0=cfg["r0"],
            height=cfg["height"], cycles_per_turn=cfg["cycles_per_turn"],
            thickness=cfg["thickness"], seed=cfg["seed"],
            signal_kind=cfg["signal"], wavelength_gaps=cfg["wavelength_gaps"])
        X, intr, y = data["X"], data["intrinsic"], data["signal"]
        labels = np.floor(data["t"] / (2 * np.pi)).astype(np.int64)  # roll layer
        disp = X[:, [1, 0, 2]]                                        # height-up
    else:  # box cylinder
        data = make_box_cylinder(
            n=cfg["n"], box_half=cfg["box_half"], cyl_radius=cfg["cyl_radius"],
            cyl_half_height=cfg["cyl_half_height"], mode=cfg["mode"],
            signal_kind=cfg["signal"], n_azimuthal_cycles=cfg["n_azimuthal_cycles"],
            wavelength_gaps=cfg["wavelength_gaps"], seed=cfg["seed"])
        X, intr, y, labels = (data["X"], data["intrinsic"],
                              data["signal"], data["labels"])
        disp = X
    return X, intr, y.astype(np.float64), labels, disp.astype(np.float32)


# =============================================================================
# Full fit: run the four methods, return everything napari needs (numpy only)
# =============================================================================

def run_fit(cfg, log=print):
    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])

    X, intr, y, labels, disp = generate_data(cfg)
    n = X.shape[0]

    Xs = standardize(X)
    Is = ((intr - intr.mean(0)) / (intr.std() + 1e-8)).astype(np.float32)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    Xs_t = torch.as_tensor(Xs, device=device)
    Is_t = torch.as_tensor(Is, device=device)

    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(n)
    ntr = int(cfg["train_frac"] * n)
    tr, te = perm[:ntr], perm[ntr:]
    tr_t = torch.as_tensor(tr, device=device)
    te_t = torch.as_tensor(te, device=device)
    yc = yt - yt.mean()

    log(f"N={n} train={ntr} test={len(te)} device={device} "
        f"modes={cfg['num_modes']} k={cfg['knn_k']} nu={cfg['nu']} "
        f"inflation={cfg['inflation']} "
        f"infer={'variational' if cfg['variational'] else 'exact'}")

    results, preds_full, hparams = {}, {}, {}
    atlas_eig = {}   # eigenbasis of the atlas kernel, for the clickable kernel view

    def evaluate(name, kernel, inputs, is_manifold=False):
        if cfg["variational"]:
            m_ind = (min(cfg["num_inducing"], cfg["num_modes"])
                     if is_manifold else cfg["num_inducing"])
            model, lik = fit_svgp(kernel, inputs[tr_t], yt[tr_t], m_ind,
                                  learn_inducing=(not is_manifold),
                                  iters=cfg["iters"])
        else:
            model, lik = fit_gp(kernel, inputs[tr_t], yt[tr_t], iters=cfg["iters"])
        results[name] = metrics(predict_gp(model, lik, inputs[te_t]), yt[te_t])
        hparams[name] = _ell_of(kernel)
        preds_full[name] = predict_gp(model, lik, inputs).detach().cpu().numpy()
        log(f"  [{name}] test_rmse={results[name][0]:.4f} "
            f"pearson={results[name][1]:.4f} ell={_ell_of(kernel):.3g}")

    def init_manifold_ell(kernel):
        ell0, _ = suggest_lengthscale(kernel.eigval, kernel.eigvec, yc, cfg["nu"])
        kernel.lengthscale = ell0
        return kernel

    # 1. euclidean
    evaluate("euclidean",
             gpytorch.kernels.MaternKernel(nu=cfg["nu"] + 0.5).to(device), Xs_t)

    # 2. manifold plain
    k_plain, coords = build_riemann_kernel(
        Xs, labels, 1.0, cfg["knn_k"], cfg["num_modes"], cfg["nu"],
        cfg["laplacian_norm"], cfg["graphbandwidth"], device)
    evaluate("manifold plain", init_manifold_ell(k_plain), coords, is_manifold=True)

    # 3. manifold atlas (inflate cross-region edges)
    k_atlas, coords = build_riemann_kernel(
        Xs, labels, cfg["inflation"], cfg["knn_k"], cfg["num_modes"], cfg["nu"],
        cfg["laplacian_norm"], cfg["graphbandwidth"], device)
    evaluate("manifold atlas", init_manifold_ell(k_atlas), coords, is_manifold=True)
    atlas_eig = dict(
        eigvec=k_atlas.eigvec.detach().cpu().numpy().astype(np.float32),
        eigval=k_atlas.eigval.detach().cpu().numpy().astype(np.float32),
        ell=_ell_of(k_atlas), nu=cfg["nu"])

    # 4. oracle (intrinsic coords)
    evaluate("oracle", gpytorch.kernels.MaternKernel(nu=cfg["nu"] + 0.5).to(device),
             Is_t)

    return dict(
        X=X.astype(np.float32), disp=disp, y=y.astype(np.float32),
        intr=intr.astype(np.float32), labels=labels, test_idx=te,
        preds=preds_full, results=results, hparams=hparams,
        atlas_eig=atlas_eig,
        method_order=["euclidean", "manifold plain", "manifold atlas", "oracle"],
    )


# =============================================================================
# napari helpers
# =============================================================================

def colormap_rgba(vals, cmap="coolwarm", vmin=None, vmax=None):
    import matplotlib
    try:
        cmap_obj = matplotlib.colormaps[cmap]
    except Exception:
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(cmap)
    vmin = float(np.nanmin(vals)) if vmin is None else vmin
    vmax = float(np.nanmax(vals)) if vmax is None else vmax
    v = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return cmap_obj(v)


def log_rgba(vals, cmap="viridis", decades=6.0):
    """Colour a kernel slice on a LOG scale, 1 at the source down to 10^-decades.

    A Matern/heat kernel spans tens of decades over a cloud this size, so a
    linear 0..1 map puts >99.9% of the points in the bottom colour and the
    layer reads as one flat colour. Log makes the decay itself visible.
    """
    v = np.clip(np.asarray(vals, dtype=np.float64), 0.0, None)
    vmax = float(np.nanmax(v)) or 1.0
    lg = np.log10(np.clip(v / vmax, 10.0 ** -decades, 1.0))
    return colormap_rgba(lg, cmap, -decades, 0.0)


def matern_kernel(r, ell, nu):
    """Matern correlation at radius r — the nu values gpytorch supports."""
    x = np.abs(np.asarray(r, dtype=np.float64)) / max(float(ell), 1e-12)
    if nu <= 0.5:
        return np.exp(-x)
    if nu <= 1.5:
        z = np.sqrt(3.0) * x
        return (1.0 + z) * np.exp(-z)
    z = np.sqrt(5.0) * x
    return (1.0 + z + z * z / 3.0) * np.exp(-z)


def _no_border(layer, blending="opaque"):
    """Strip the marker outline off a Points layer and set its blending.

    Only Points get the width treatment. On a **Vectors** layer `edge_width` is
    the width of the drawn line ITSELF, not an outline around it -- zeroing it
    collapses every segment to degenerate geometry and the graph renders black.
    """
    if type(layer).__name__ != "Vectors":
        for attr in ("border_width", "edge_width"):
            try:
                setattr(layer, attr, 0.0)
            except Exception:
                pass
        try:
            layer.antialiasing = 0.0
        except Exception:
            pass
    try:
        layer.blending = blending
    except Exception:
        pass


def auto_point_size(X):
    from sklearn.neighbors import NearestNeighbors as SkNN
    dd, _ = SkNN(n_neighbors=2).fit(X).kneighbors(X)
    return 1.3 * float(np.median(dd[:, 1]))


# =============================================================================
# The app
# =============================================================================

class Explorer:
    def __init__(self, viewer):
        self.viewer = viewer
        self.state = {"src": 0}
        self.dyn = {}         # clickable kernel-source layers
        self.feats = None     # atlas feature map (N, M) for the manifold kernel
        self.fit = None       # last run_fit result
        self._graph_note = ""  # kNN-graph diagnostics from the last build
        self._session_state = {}          # geometry -> cfg edited this session
        self._prev_geometry = None
        self._file_defaults = self._load_defaults_file()   # geometry -> saved cfg
        self._build_controls()
        self._on_geometry_changed()       # applies the startup geometry's defaults

    # ---- controls ---------------------------------------------------------
    def _build_controls(self):
        from magicgui.widgets import (ComboBox, SpinBox, FloatSpinBox, CheckBox,
                                       PushButton, Label, Container)

        cuda = torch.cuda.is_available()

        self.geometry = ComboBox(label="geometry",
                                 choices=["swiss roll", "box cylinder"],
                                 value="swiss roll")
        self.signal = ComboBox(label="signal", choices=SWISS_SIGNALS, value="geodesic")
        self.n = SpinBox(label="n points", min=500, max=200_000, step=500, value=8000)
        self.seed = SpinBox(label="seed", min=0, max=9999, value=0)

        # swiss-roll-only
        self.n_turns = FloatSpinBox(label="n_turns", min=0.5, max=12, step=0.5, value=3.5)
        self.gap = FloatSpinBox(label="gap", min=0.05, max=10, step=0.05, value=1.0)
        self.r0 = FloatSpinBox(label="r0", min=0.1, max=10, step=0.1, value=1.0)
        self.height = FloatSpinBox(label="height", min=1, max=50, step=1, value=8.0)
        self.cycles_per_turn = FloatSpinBox(label="cycles/turn", min=0.5, max=6,
                                            step=0.5, value=1.5)
        self.thickness = FloatSpinBox(label="thickness", min=0.0, max=20, step=0.5,
                                      value=0.0)
        self.swiss_box = Container(
            widgets=[self.n_turns, self.gap, self.r0, self.height,
                     self.cycles_per_turn, self.thickness],
            label="swiss roll")

        # box-cylinder-only
        self.box_half = FloatSpinBox(label="box_half", min=1, max=20, step=0.5, value=4.0)
        self.cyl_radius = FloatSpinBox(label="cyl_radius", min=0.5, max=20, step=0.1,
                                       value=3.7)
        self.cyl_half_height = FloatSpinBox(label="cyl_half_h", min=0.5, max=20,
                                            step=0.1, value=3.7)
        self.mode = ComboBox(label="mode", choices=["surface", "volume"], value="surface")
        self.n_azimuthal_cycles = SpinBox(label="azim cycles", min=1, max=12, value=2)
        self.box_box = Container(
            widgets=[self.box_half, self.cyl_radius, self.cyl_half_height,
                     self.mode, self.n_azimuthal_cycles],
            label="box cylinder")

        # shared signal param
        self.wavelength_gaps = FloatSpinBox(label="wavelength_gaps", min=0.5, max=20,
                                            step=0.5, value=3.0)
        self.draw_graph = CheckBox(label="draw kNN graph", value=True)

        # GP hyper-params
        self.knn_k = SpinBox(label="knn_k", min=3, max=60, value=15)
        self.num_modes = SpinBox(label="num_modes", min=10, max=2000, step=10, value=200)
        self.nu = SpinBox(label="nu", min=1, max=5, value=2)
        self.inflation = FloatSpinBox(label="inflation", min=1, max=1000, step=1,
                                      value=50.0)
        self.laplacian_norm = ComboBox(label="lap_norm",
                                       choices=["randomwalk", "symmetric"],
                                       value="randomwalk")
        self.graphbandwidth = FloatSpinBox(label="graphbw (0=auto)", min=0.0, max=10,
                                           step=0.01, value=0.0)
        self.train_frac = FloatSpinBox(label="train_frac", min=0.05, max=0.95,
                                       step=0.05, value=0.5)
        self.iters = SpinBox(label="iters", min=10, max=1000, step=10, value=100)
        self.variational = CheckBox(label="variational (SVGP)", value=False)
        self.num_inducing = SpinBox(label="num_inducing", min=16, max=2000, step=16,
                                    value=256)
        self.device = ComboBox(label="device",
                               choices=(["cuda", "cpu"] if cuda else ["cpu"]),
                               value=("cuda" if cuda else "cpu"))
        gp_box = Container(
            widgets=[self.knn_k, self.num_modes, self.nu, self.inflation,
                     self.laplacian_norm, self.graphbandwidth, self.train_frac,
                     self.iters, self.variational, self.num_inducing, self.device],
            label="GP hyper-parameters")

        # buttons + status
        self.preview_btn = PushButton(text="Preview geometry")
        self.fit_btn = PushButton(text="▶ Fit GP")
        self.save_btn = PushButton(text="Save default (this geometry)")
        self.status = Label(value="Ready. Pick params, then Fit GP.")
        try:
            self.status.native.setWordWrap(True)
        except Exception:
            pass

        # kernel-highlight navigation (enabled after a fit). Navigation only moves
        # the source marker; the kernel is (re)computed only on Compute highlight.
        self.pt_label = Label(value="kernel source: (fit first)")
        self.prev_pt = PushButton(text="◄ prev test pt")
        self.next_pt = PushButton(text="next test pt ►")
        self.compute_btn = PushButton(text="✦ Compute highlight")
        for b in (self.prev_pt, self.next_pt, self.compute_btn):
            b.enabled = False
        kern_box = Container(
            widgets=[self.pt_label, self.prev_pt, self.next_pt, self.compute_btn],
            label="kernel highlight")

        self.geometry.changed.connect(self._on_geometry_changed)
        self.preview_btn.clicked.connect(self.on_preview)
        self.fit_btn.clicked.connect(self.on_fit)
        self.save_btn.clicked.connect(self.on_save_default)
        self.prev_pt.clicked.connect(lambda: self._nav_test(-1))
        self.next_pt.clicked.connect(lambda: self._nav_test(+1))
        # explicit lambda: psygnal would otherwise feed the button's own value
        # into the new `show` parameter and silently suppress the layer.
        self.compute_btn.clicked.connect(lambda: self.on_compute_highlight(show=True))

        self._geom_row = Container(widgets=[self.geometry, self.signal, self.n,
                                            self.seed, self.wavelength_gaps,
                                            self.draw_graph],
                                   label="geometry / signal")

        self.panel = Container(
            widgets=[self._geom_row, self.swiss_box, self.box_box, gp_box,
                     self.preview_btn, self.fit_btn, self.save_btn, kern_box,
                     self.status],
            layout="vertical")

        from qtpy.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel.native)
        self.viewer.window.add_dock_widget(scroll, name="controls", area="right")

    # ---- config <-> widgets ----------------------------------------------
    WIDGET_KEYS = [
        "geometry", "signal", "n", "seed", "wavelength_gaps",
        "n_turns", "gap", "r0", "height", "cycles_per_turn", "thickness",
        "box_half", "cyl_radius", "cyl_half_height", "mode", "n_azimuthal_cycles",
        "knn_k", "num_modes", "nu", "inflation", "laplacian_norm", "graphbandwidth",
        "train_frac", "iters", "variational", "num_inducing", "device",
        "draw_graph",
    ]

    def collect_cfg(self):
        return {k: getattr(self, k).value for k in self.WIDGET_KEYS}

    def _load_defaults_file(self):
        """Return {geometry: cfg}. Migrates the old flat single-cfg format."""
        if not os.path.exists(DEFAULTS_PATH):
            return {}
        try:
            with open(DEFAULTS_PATH) as f:
                data = json.load(f)
        except Exception as e:
            print(f"could not read defaults: {e}")
            return {}
        if "geometry" in data:                       # old flat format -> key by geom
            return {data.get("geometry", "swiss roll"): data}
        print(f"loaded per-geometry defaults from {DEFAULTS_PATH} "
              f"({', '.join(data.keys())})")
        return data

    def _apply_params(self, params):
        """Set widgets from a saved cfg (geometry itself is left untouched)."""
        for k, v in params.items():
            if k == "geometry":
                continue
            self._set_widget(k, v)

    def _set_widget(self, key, value):
        w = getattr(self, key, None)
        if w is None:
            return
        try:
            if hasattr(w, "choices") and value not in list(w.choices):
                return
            w.value = value
        except Exception:
            pass

    def on_save_default(self):
        geom = self.geometry.value
        cfg = self.collect_cfg()
        self._file_defaults[geom] = cfg
        self._session_state[geom] = cfg
        try:
            with open(DEFAULTS_PATH, "w") as f:
                json.dump(self._file_defaults, f, indent=2)
            self._status(f"Saved '{geom}' defaults -> {os.path.basename(DEFAULTS_PATH)}")
        except Exception as e:
            self._status(f"Save failed: {e}")

    # ---- geometry switching ----------------------------------------------
    def _sync_signal_choices(self):
        geom = self.geometry.value
        choices = SWISS_SIGNALS if geom == "swiss roll" else BOX_SIGNALS
        cur = self.signal.value
        self.signal.choices = choices
        self.signal.value = cur if cur in choices else choices[0]

    def _on_geometry_changed(self, *_):
        geom = self.geometry.value
        # remember the outgoing geometry's current edits for this session
        if self._prev_geometry is not None and self._prev_geometry != geom:
            self._session_state[self._prev_geometry] = self.collect_cfg()
        is_swiss = (geom == "swiss roll")
        self.swiss_box.visible = is_swiss
        self.box_box.visible = not is_swiss
        self._sync_signal_choices()
        # restore this geometry's params: in-session edits first, else saved file
        # defaults; if neither exists keep the built-in widget values as-is.
        params = self._session_state.get(geom) or self._file_defaults.get(geom)
        if params:
            self._apply_params(params)
        self._prev_geometry = geom

    # ---- status -----------------------------------------------------------
    def _status(self, msg):
        self.status.value = msg
        print(msg)

    # ---- preview (cheap, synchronous) ------------------------------------
    def on_preview(self):
        cfg = self.collect_cfg()
        try:
            X, intr, y, labels, disp = generate_data(cfg)
        except Exception as e:
            self._status(f"generate failed: {e}")
            traceback.print_exc()
            return
        psize = auto_point_size(X)
        self._clear_layers()
        self._add_signal_layers(disp, y, labels, psize)
        self._add_graph_layers(disp, X, labels, cfg, psize)
        self.fit = None
        self._status(f"Preview: {geom_desc(cfg)}  N={X.shape[0]}  "
                     f"(press Fit GP to run the GPs)\n{self._graph_note}")

    # ---- fit (expensive, threaded) ---------------------------------------
    def on_fit(self):
        from napari.qt.threading import thread_worker
        cfg = self.collect_cfg()
        self.fit_btn.enabled = False
        self.preview_btn.enabled = False
        self._status(f"Fitting {geom_desc(cfg)} ... (see console for progress)")

        @thread_worker
        def _work():
            return run_fit(cfg, log=print)

        worker = _work()
        worker.returned.connect(lambda res: self._on_fit_done(cfg, res))
        worker.errored.connect(self._on_fit_error)
        worker.start()

    def _on_fit_error(self, exc):
        self.fit_btn.enabled = True
        self.preview_btn.enabled = True
        self._status(f"Fit failed: {exc}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    def _on_fit_done(self, cfg, res):
        self.fit = res
        self.fit_btn.enabled = True
        self.preview_btn.enabled = True
        self._rebuild_all_layers(cfg, res)
        self._status(self._metrics_table(cfg, res))

    # ---- layer building ---------------------------------------------------
    def _clear_layers(self):
        for layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)
        self.dyn = {}

    def _add_signal_layers(self, disp, y, labels, psize):
        amp = float(np.nanpercentile(np.abs(y), 99)) or 1.0
        lt = self.viewer.add_points(disp, name="true signal", size=psize,
                                    face_color=colormap_rgba(y, "coolwarm", -amp, amp),
                                    shading="none")
        _no_border(lt)
        ll = self.viewer.add_points(disp, name="region labels", size=psize,
                                    visible=False,
                                    face_color=colormap_rgba(labels.astype(float), "tab10"),
                                    shading="none")
        _no_border(ll)

    def _add_graph_layers(self, disp, X, labels, cfg, psize):
        """Draw the kNN graph the manifold kernels are built on.

        Three Vectors layers (all off by default): within-region edges, the
        cross-region edges that the atlas inflation targets, and every edge
        coloured by its atlas weight exp(-inflation*d^2/(4 bw^2)) -- i.e. what
        the atlas kernel actually lets heat flow along. If the shortcut layer
        is dense, the graph is not the manifold and 'manifold plain' is being
        built on a Laplacian of the ambient solid, not the sheet.
        """
        self._graph_note = ""
        if not self.draw_graph.value:
            return
        try:
            ei, d2, cross, bw = build_graph_edges(
                standardize(X), labels, cfg["knn_k"], cfg["device"])
        except Exception as e:
            self._graph_note = f"graph layer failed: {e}"
            traceback.print_exc()
            return

        ew = max(psize * 0.12, 1e-3)
        for mask, color, name in (
                (~cross, "#8a8a8a", "graph: within-region edges"),
                (cross, "#ff2d2d", "graph: cross-region edges (shortcuts)")):
            idx = subsample_edges(mask)
            if not len(idx):
                continue
            lg = self.viewer.add_vectors(
                edge_vectors(disp, ei[:, idx]), name=name, visible=False,
                edge_width=ew, edge_color=color, vector_style="line", length=1.0)
            _no_border(lg, blending="translucent")

        # every edge, coloured by the weight the ATLAS kernel gives it
        infl = np.where(cross, float(cfg["inflation"]), 1.0)
        w_atlas = np.exp(-infl * d2 / (4.0 * bw ** 2 + 1e-12))
        idx = subsample_edges(np.ones(len(d2), bool), seed=1)
        lw = self.viewer.add_vectors(
            edge_vectors(disp, ei[:, idx]), name="graph: atlas edge weight",
            visible=False, edge_width=ew, vector_style="line", length=1.0,
            edge_color=colormap_rgba(w_atlas[idx], "viridis", 0.0, 1.0))
        _no_border(lw, blending="translucent")

        w_plain = np.exp(-d2 / (4.0 * bw ** 2 + 1e-12))
        self._graph_note = (
            f"graph: {len(d2)} edges, {100 * cross.mean():.1f}% cross-region; "
            f"bw={bw:.3g}; mean w cross {w_plain[cross].mean():.3g} -> "
            f"{w_atlas[cross].mean():.3g} under atlas"
            if cross.any() else
            f"graph: {len(d2)} edges, 0% cross-region; bw={bw:.3g}")
        print(self._graph_note)

    def _rebuild_all_layers(self, cfg, res):
        disp, y, labels = res["disp"], res["y"], res["labels"]
        X, intr = res["X"], res["intr"]
        psize = auto_point_size(X)
        self._clear_layers()
        self._add_signal_layers(disp, y, labels, psize)
        self._add_graph_layers(disp, X, labels, cfg, psize)

        amp = float(np.nanpercentile(np.abs(y), 99)) or 1.0
        preds = res["preds"]
        err_max = float(np.nanpercentile(
            np.abs(np.stack(list(preds.values())) - y[None]), 99)) or 1.0

        for i, name in enumerate(res["method_order"]):
            p = preds[name]
            lp = self.viewer.add_points(
                disp, name=f"pred: {name}", size=psize, visible=(i == 0),
                face_color=colormap_rgba(p, "coolwarm", -amp, amp), shading="none")
            _no_border(lp)
            le = self.viewer.add_points(
                disp, name=f"|err|: {name}", size=psize, visible=False,
                face_color=colormap_rgba(np.abs(p - y), "magma", 0.0, err_max),
                shading="none")
            _no_border(le)

        lt = self.viewer.add_points(
            disp[res["test_idx"]], name="test points (truth)", size=psize * 1.6,
            visible=False,
            face_color=colormap_rgba(y[res["test_idx"]], "coolwarm", -amp, amp),
            shading="none")
        _no_border(lt)

        # clickable kernel-from-source layers (atlas manifold vs euclidean)
        eig = res["atlas_eig"]
        S = (2.0 * eig["nu"] / (eig["ell"] ** 2) + np.clip(eig["eigval"], 0, None)) \
            ** (-eig["nu"])
        self.feats = eig["eigvec"] * np.sqrt(S)          # (N, M)
        self._X_kern = X
        self._intr_kern = intr
        # Render the euclidean kernel the GP ACTUALLY fitted: its learned
        # lengthscale, in the standardized space it was learned in. The old
        # "5 NN spacings" heuristic was ~an order of magnitude too tight, which
        # drove the whole cloud to ~1e-25 and flattened the layer.
        self._Xs_kern = standardize(X)
        self._euc_ell = float(res["hparams"]["euclidean"])
        self._euc_nu = cfg["nu"] + 0.5
        self.state["src"] = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))

        # NB: give these an explicit starting colour. Created bare, napari
        # defaults them to opaque white, so toggling one on before pressing
        # Compute shows a solid white cloud that looks like a broken layer.
        blank = np.tile(np.array([0.28, 0.28, 0.32, 1.0]), (len(disp), 1))
        self.dyn["man"] = self.viewer.add_points(
            disp, name="manifold kernel: atlas (src)", size=psize, visible=False,
            face_color=blank.copy(), shading="none")
        self.dyn["euc"] = self.viewer.add_points(
            disp, name="euclidean kernel (src)", size=psize, visible=False,
            face_color=blank.copy(), shading="none")
        self.dyn["geo"] = self.viewer.add_points(
            disp, name="geodesic distance (src)", size=psize, visible=False,
            face_color=blank.copy(), shading="none")
        self.dyn["src"] = self.viewer.add_points(
            disp[self.state["src"]][None], name="source", size=psize * 4,
            face_color="lime", shading="none")
        self._disp_kern = disp
        for l in self.dyn.values():
            _no_border(l)

        # test-point navigation list (subsampled so prev/next is manageable)
        te = res["test_idx"].astype(np.int64)
        if len(te) > 200:
            te = np.random.default_rng(0).choice(te, size=200, replace=False)
        self._nav_nodes = np.sort(te)
        self._nav_i = 0
        for b in (self.prev_pt, self.next_pt, self.compute_btn):
            b.enabled = True

        # clicking any layer moves the source marker (cheap); the kernel is only
        # (re)computed when Compute highlight is pressed.
        # Points layers only: a Vectors layer's get_value returns an EDGE index,
        # which would move the source marker to an unrelated node.
        import napari.layers
        for layer in list(self.viewer.layers):
            if layer is not self.dyn["src"] and isinstance(layer, napari.layers.Points):
                layer.mouse_drag_callbacks.append(self._make_click(layer))
        self._move_source(int(self._nav_nodes[0]))
        # Populate the kernel layers now (cheap) so they are never shown blank;
        # show=False keeps the default visible layer as 'pred: euclidean'.
        self.on_compute_highlight(show=False)

    def _make_click(self, layer):
        def _on_click(layer, event):
            idx = layer.get_value(event.position,
                                  view_direction=event.view_direction,
                                  dims_displayed=event.dims_displayed, world=True)
            if idx is not None:
                self._move_source(int(idx), from_nav=False)
        return _on_click

    # ---- kernel-highlight navigation / compute ---------------------------
    def _move_source(self, i, from_nav=True):
        """Move the source marker only. Cheap: no kernel recompute."""
        self.state["src"] = int(i)
        if not from_nav:
            self._nav_i = -1        # clicked point is not a nav index
        if "src" in self.dyn:
            self.dyn["src"].data = self._disp_kern[i][None]
            _no_border(self.dyn["src"]); self.dyn["src"].refresh()
        self._update_pt_label()

    def _nav_test(self, delta):
        if self.feats is None or not len(getattr(self, "_nav_nodes", [])):
            return
        n = len(self._nav_nodes)
        self._nav_i = (self._nav_i + delta) % n if self._nav_i >= 0 else 0
        self._move_source(int(self._nav_nodes[self._nav_i]))

    def _update_pt_label(self):
        i = self.state["src"]
        n = len(getattr(self, "_nav_nodes", []))
        if n and self._nav_i >= 0 and int(self._nav_nodes[self._nav_i]) == i:
            where = f"test pt {self._nav_i}/{n - 1}"
        else:
            where = "clicked"
        self.pt_label.value = f"kernel source: {where}  node {i}  — press Compute"

    def on_compute_highlight(self, show=True):
        """Actually compute the kernel-from-source highlight for the current node.

        Both kernels are drawn on the SAME log colour scale (1 at the source,
        floor at 10^-DECADES) so 'manifold vs euclidean' is a like-for-like read.
        """
        if self.feats is None:
            self._status("Fit a GP first, then Compute highlight.")
            return
        decades = 6.0
        i = self.state["src"]
        kv = self.feats @ self.feats[i]
        man = kv / (kv[i] + 1e-12)
        self.dyn["man"].face_color = log_rgba(man, "viridis", decades)
        euc_d = np.linalg.norm(self._Xs_kern - self._Xs_kern[i], axis=1)
        euc = matern_kernel(euc_d, self._euc_ell, self._euc_nu)
        self.dyn["euc"].face_color = log_rgba(euc, "viridis", decades)
        geo_d = np.linalg.norm(self._intr_kern - self._intr_kern[i], axis=1)
        self.dyn["geo"].face_color = colormap_rgba(geo_d, "magma")
        if show:
            self.dyn["man"].visible = True
        self.dyn["src"].visible = True
        for l in self.dyn.values():
            _no_border(l); l.refresh()
        frac = lambda a: float(np.mean(a > 10.0 ** -decades))
        self._status(
            f"Highlight at node {i} (log scale, {decades:g} decades; "
            f"euclidean ell={self._euc_ell:.3g} nu={self._euc_nu:g}). "
            f"support: manifold {100 * frac(man):.1f}% vs euclidean "
            f"{100 * frac(euc):.1f}% of points above the floor.")

    def _metrics_table(self, cfg, res):
        lines = [f"{geom_desc(cfg)}  |  N={res['X'].shape[0]}  "
                 f"infer={'SVGP' if cfg['variational'] else 'exact'}",
                 f"{'method':<16}{'RMSE':>8}{'Pears':>7}{'Spear':>7}"]
        for name, (r, p, s) in sorted(res["results"].items(), key=lambda kv: kv[1][0]):
            lines.append(f"{name:<16}{r:>8.4f}{p:>7.3f}{s:>7.3f}")
        euc = res["results"]["euclidean"][0]
        atl = res["results"]["manifold atlas"][0]
        verdict = "manifold WINS" if atl < euc else "euclidean wins"
        lines.append(f"atlas/euclidean = {atl/euc:.2f}x  -> {verdict}")
        if self._graph_note:
            lines.append(self._graph_note)
        return "\n".join(lines)


def geom_desc(cfg):
    if cfg["geometry"] == "swiss roll":
        return (f"swiss roll [{cfg['signal']}] gap={cfg['gap']:g} "
                f"thick={cfg['thickness']:g} cyc/turn={cfg['cycles_per_turn']:g}")
    return (f"box cylinder [{cfg['signal']}/{cfg['mode']}] "
            f"gap={cfg['box_half'] - cfg['cyl_radius']:g}")


def main():
    import napari
    viewer = napari.Viewer(ndisplay=3, title="toy GP explorer")
    Explorer(viewer)
    print("Controls dock on the right. Preview geometry is cheap; Fit GP runs the "
          "four-method comparison on the GPU.\n"
          "After a fit, toggle 'pred:' / '|err|:' layers. Use the 'kernel highlight' "
          "panel: ◄/► step through test points (or click any layer to place the "
          "source), then press 'Compute highlight' to render the manifold-atlas vs "
          "euclidean kernel from that point.")
    napari.run()


if __name__ == "__main__":
    main()
