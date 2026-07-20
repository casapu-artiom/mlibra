#!/usr/bin/env python
"""Manual kernel-fitting explorer — Euclidean vs Manifold, side by side.

A fork of ``maldi_kernel_explorer.py`` aimed at ONE question: hand-tune (or load
from a trained run) a Euclidean and a Manifold kernel, hit **Refit**, and read
off exactly the numbers the analysis notebook produces —

  * a per-cell COVARIANCE view: drop a test point, see k(test, ·) for BOTH
    kernels painted over the brain (what neighbourhood each kernel "considers");
  * PER-LIPID held-out R² for both, with the count of lipids on which manifold
    beats euclidean and by how much;
  * an overall SUMMARY (R² all / interior / near-border), and
  * a COVARIANCE-vs-BORDER table (does the kernel reproduce the drop the data
    shows across a region boundary?).

All the numeric machinery is reused from ``analysis/graph_analysis_playroom.py``
(pooled 0.2 mm cells, held-out prediction, border tables) so this file is only
the wiring + UI.

Two ways to specify each kernel
-------------------------------
  * MANUAL   -- set the params on the CLI or in the GUI controls.
  * TRAINED  -- ``--manifold-run <dir>`` reads that run's config.json and its
                LEARNED lengthscale out of model.pth.

Eigenvectors
------------
  If a manifold config's eigenpairs are not cached the tool, by default, prints
  the ``compute_eigenvectors.py`` command and stops (never solves silently).
  Pass ``--compute-if-missing`` (or tick the GUI box) to solve on the GPU once
  and write it into the shared pipeline cache.

Headless
--------
  python maldi/manifold_vs_euclidean_explorer.py --self-test [--compute-if-missing]
runs the entire numeric pipeline and prints the report, no display needed.

Launch the GUI
--------------
  python maldi/manifold_vs_euclidean_explorer.py \
      --manifold knn_method=faiss_atlas_weighted,inflation=50,num_modes=1000,nu=2,lengthscale=5 \
      --euclidean nu=0.5,lengthscale=2,border_downweight=0.1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis"))

import graph_analysis_playroom as G
from graph_analysis_playroom import (
    CellPool, KernelSpec, make_kernel, implied_rho, prize,
    per_lipid_r2, compare_two, lengthscale_health,
)

EDG = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0])


# ===========================================================================
# CLI spec parsing
# ===========================================================================
def _coerce(v: str):
    try:
        return int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError:
        return v

def parse_kv(s: str) -> dict:
    return {k: _coerce(v) for k, v in (tok.split("=", 1) for tok in s.split(",") if tok)}

# "5"/"15" are the LBAE-shipped volumes: partial cuts that leave 72%/57% of the
# brain in the root (997) catch-all. "d5"/"d7" are true CCF depth-cuts built by
# maldi/download_bg_atlas.py --max-depth: 175/476 regions, root down to 0.7%.
# Their brain mask is IoU 0.948 vs level_15 (brainglobe CCF vs LBAE's ccf_2017).
ATLASES = {"5":  "/home/casap/mlibra/mlibra_data/level_5annot.npy",
           "15": "/home/casap/mlibra/mlibra_data/level_15annot.npy",
           "d5": "/home/casap/mlibra/mlibra_data/ccf_depth5annot.npy",
           "d7": "/home/casap/mlibra/mlibra_data/ccf_depth7annot.npy"}

ATLAS_CHOICES = list(ATLASES)

def atlas_path(which) -> str:
    return ATLASES[str(which).replace("level_", "").replace("annot", "").strip()]

def atlas_key(annotations) -> str:
    """Reverse of atlas_path: the ATLASES key for a loaded annotation path."""
    return next((k for k, v in ATLASES.items() if v == str(annotations)),
                ATLAS_CHOICES[0])

def euclidean_spec(kv: dict) -> KernelSpec:
    return KernelSpec.euclidean(nu=kv.get("nu", 0.5),
                                lengthscale=kv.get("lengthscale", 2.0),
                                outputscale=kv.get("outputscale", 1.0),
                                border_downweight=kv.get("border_downweight", 0.0))

def manifold_spec(kv: dict) -> KernelSpec:
    kv = dict(kv); kv.setdefault("kind", "manifold")
    return KernelSpec(**kv)


# ===========================================================================
# The numeric core  --  used by BOTH the GUI and --self-test
# ===========================================================================
def analyze(pool: CellPool, eu: KernelSpec, man: KernelSpec, allow_solve=False):
    """Everything the notebook computes, for one (euclidean, manifold) pair."""
    specs = [eu, man]

    # (0) is the manifold lengthscale even identifiable?
    health = lengthscale_health(man, pool, allow_solve=allow_solve)

    # (1) the kernel-free prize
    pz = prize(pool, verbose=False)

    # (2) per-lipid held-out R²  ->  summary + comparison (ONE fit, reused)
    pl = per_lipid_r2(pool, specs, allow_solve=allow_solve)
    summary = _summary(pool, pl, specs)
    cmp = compare_two(pl, man.name, eu.name)

    # (3) covariance-vs-border + inside-region tables (implied correlation)
    border = _binned_drop(pool, [("EMPIRICAL", pool.S)] +
                          [(s.name, implied_rho(s, pool, allow_solve=allow_solve)) for s in specs])
    inside = _binned_inside(pool, [("EMPIRICAL", pool.S)] +
                            [(s.name, implied_rho(s, pool, allow_solve=allow_solve)) for s in specs])

    return dict(health=health, prize=pz, summary=summary, compare=cmp,
                border=border, inside=inside, per_lipid=pl, eu=eu, man=man,
                atlas=Path(pool.annotations).stem)


def _summary(pool, pl, specs):
    """Overall R² split all / interior / near-border, from the ALREADY-FITTED
    per-lipid predictions (no refit), plus the region-mean floor."""
    Z = pool.Z.astype(np.float64)
    te, FAR, NEAR = pl["test"], pl["FAR"], pl["NEAR"]
    def R2(pred, truth): return 1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean(0)) ** 2).sum()
    def MSE(pred, truth): return float(((truth - pred) ** 2).mean())   # z-units²

    rows = {}
    for s in specs:
        pr = pl["preds"][s.name]                     # (n_test, n_lipids), reused
        rows[s.name] = dict(all=R2(pr, Z[te]), mse=MSE(pr, Z[te]),
                            interior=R2(pr[FAR[te]], Z[te][FAR[te]]),
                            near=R2(pr[NEAR[te]], Z[te][NEAR[te]]),
                            alpha=pl["alpha"][s.name])
    mu = {r: Z[pl["tr"]][pool.region[pl["tr"]] == r].mean(0)
          for r in np.unique(pool.region[pl["tr"]])}
    base = np.stack([mu.get(r, Z[pl["tr"]].mean(0)) for r in pool.region[te]])
    rows["region mean (floor)"] = dict(
        all=R2(base, Z[te]), mse=MSE(base, Z[te]),
        interior=R2(base[FAR[te]], Z[te][FAR[te]]),
        near=R2(base[NEAR[te]], Z[te][NEAR[te]]), alpha=np.nan)
    rows["_n_test"] = len(te)
    return rows


def _binned_drop(pool, curves):
    out = {}
    for nm, v in curves:
        r, ref = [], None
        for i in range(len(EDG) - 1):
            m = (pool.d >= EDG[i]) & (pool.d < EDG[i + 1])
            a, c = v[m & ~pool.cross].mean(), v[m & pool.cross].mean()
            ref = abs(a) if ref is None else ref
            r.append(100 * (a - c) / a if abs(a) > 0.05 * ref else np.nan)
        out[nm] = np.array(r)
    return out

def _binned_inside(pool, curves):
    out = {}
    for nm, v in curves:
        r = np.array([v[(pool.d >= EDG[i]) & (pool.d < EDG[i + 1]) & ~pool.cross].mean()
                      for i in range(len(EDG) - 1)])
        out[nm] = r / r[0]
    return out


# ===========================================================================
# Text report  (shared by --self-test and the GUI results panel)
# ===========================================================================
def format_report(res: dict) -> str:
    eu, man = res["eu"], res["man"]
    L = []
    L.append(f"EUCLIDEAN : {eu.name}")
    L.append(f"MANIFOLD  : {man.name}")
    L.append("")
    h = res["health"]
    L.append(f"lengthscale health : {'OK' if h['ok'] else 'PROBLEM'}")
    L.append(f"   {h['msg']}")
    L.append("")
    pz = res["prize"]
    L.append(f"THE PRIZE (no kernel fitted): the border is worth "
             f"{100*pz['border']:.2f}% of Var(S)  "
             f"(distance alone: {100*pz['distance']:.1f}%)")
    L.append("")
    s = res["summary"]; n = s.pop("_n_test", 0)
    L.append(f"SUMMARY — held-out on {res['per_lipid']['r2'][eu.name].size} lipids, {n} test cells "
             f"| atlas = {res.get('atlas', '?')}")
    L.append(f"   {'model':<38}{'R² all':>9}{'MSE':>8}{'R² inter':>10}{'R² near':>9}")
    for nm, r in s.items():
        L.append(f"   {nm[:37]:<38}{r['all']:>9.3f}{r['mse']:>8.3f}"
                 f"{r['interior']:>10.3f}{r['near']:>9.3f}")
    L.append("   (MSE in z-units²; lower = better)")
    L.append("")
    c = res["compare"]
    L.append(f"PER-LIPID — manifold vs euclidean  ({c['n']} lipids)")
    L.append(f"   manifold WINS on {c['n_a_wins']}/{c['n']} lipids "
             f"({100*c['n_a_wins']/max(c['n'],1):.0f}%)")
    L.append(f"   mean ΔR² (manifold − euclidean) = {c['mean_delta']:+.4f}   "
             f"median = {c['median_delta']:+.4f}")
    L.append(f"   mean R²: manifold {c['mean_a']:.3f}  vs  euclidean {c['mean_b']:.3f}")
    L.append(f"   biggest manifold WINS : " +
             ", ".join(f"{nm}({d:+.3f})" for nm, d in c['top_wins'][:4]))
    L.append(f"   biggest manifold LOSSES: " +
             ", ".join(f"{nm}({d:+.3f})" for nm, d in c['top_losses'][:4]))
    L.append("")
    L.append("COVARIANCE vs THE BORDER — relative drop cross/same at matched distance (%)")
    hdr = "".join(f"{f'{EDG[i]}-{EDG[i+1]}':>8}" for i in range(len(EDG) - 1))
    L.append(f"   {'':<26}{hdr}")
    for nm, r in res["border"].items():
        L.append(f"   {nm[:25]:<26}" + "".join(
            f"{x:>8.1f}" if np.isfinite(x) else f"{'--':>8}" for x in r))
    L.append("   (a Euclidean kernel MUST give ~0; the data shows ~15-25%)")
    L.append("")
    L.append("INSIDE A REGION — same-region correlogram, normalised to 1.0 at 0.3 mm")
    L.append(f"   {'':<26}{hdr}")
    for nm, r in res["inside"].items():
        L.append(f"   {nm[:25]:<26}" + "".join(f"{x:>8.3f}" for x in r))
    L.append("   (a kernel decaying FASTER than EMPIRICAL is losing the within-region tail)")
    return "\n".join(L)


# ===========================================================================
# MALDI slices  —  one sample's sections, projected to flat 2D, snapped to nodes
# ===========================================================================
def _project_section_2d(coords_mm):
    """PCA-project a (tilted) section's mm coords onto their best-fit plane -> (M,2)."""
    c = coords_mm - coords_mm.mean(0, keepdims=True)
    if c.shape[0] < 3:
        return c[:, :2].copy()
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


class MaldiSlices:
    """One wild-type sample's MALDI voxels, snapped to template nodes and grouped
    by Section. Each section is laid out as a flat 2D tile; ``layout`` stacks the
    sections along a scroll axis so napari's slider walks through them."""
    def __init__(self, pool, sample, gap_mm=1.0):
        import pandas as pd
        import maldi_kernel_explorer as mke
        df = pd.read_parquet(G.DATA["maldi"], columns=["xccf", "yccf", "zccf", "Section"],
                             filters=[("Sample", "==", sample)])
        coords = df[["xccf", "yccf", "zccf"]].to_numpy(np.float32)
        node, valid = mke.snap_points_to_nodes(coords, pool.node_mm, (0, 1, 2), 1.0)
        sec = df["Section"].to_numpy()
        self.sample = sample
        self.sections = sorted(np.unique(sec[valid]).tolist())
        # per-section: (xy 2D, node index of each voxel)
        self.by_section = {}
        for s in self.sections:
            m = valid & (sec == s)
            xy = _project_section_2d(coords[m].astype(np.float64)).astype(np.float32)
            self.by_section[s] = (xy, node[m].astype(np.int64))
        # point size so voxels TILE the slice (area / N), robust across sections
        s0 = self.sections[len(self.sections) // 2]
        xy0 = self.by_section[s0][0]
        if len(xy0) > 3:
            span = xy0.max(0) - xy0.min(0)
            area = float(max(span[0], 1e-6) * max(span[1], 1e-6))
            self.spacing = float(np.sqrt(area / max(len(xy0), 1)))
        else:
            self.spacing = 0.1

    def layer_data(self, section):
        """(points (M,2) in-plane, node index of each voxel) for a 2D napari view.

        NOTE: pure 2D — no section axis. Stacking sections on axis 0 makes napari's
        dims-slicing keep a stale ``_indices_view`` when the data is swapped, which
        crashes on section change. 2D points sidestep that entirely."""
        xy, nodes = self.by_section[section]
        return xy.astype(np.float32), nodes


def region_rgba(pool):
    """Per-node RGBA of the atlas region it belongs to (len N, stable colours).

    Reuses ``maldi_kernel_explorer.region_color_table`` — one colour per region id,
    hues spread by the golden ratio so adjacent ids stay distinct. Follows the
    pool's CURRENT atlas, so it re-colours on a level_5 <-> level_15 switch.
    root was dissolved by the pool, so every node is a real region (zero_is_region)."""
    from maldi_kernel_explorer import region_color_table
    code, rgba = region_color_table(pool.region_full, zero_is_region=True)
    return rgba[code]                                          # (N, 4)


def border_nodes(pool, both_sides=False):
    """Boolean mask (len N): template nodes touching a region border, from the pool's
    CURRENT atlas (region_full). Vectorised 6-neighbour label comparison on the
    strided grid, so it follows the level_5 / level_15 choice.

    ``both_sides=False`` (default) marks only the LOWER-index side of each interface,
    giving a ONE-node-thick outline; marking both sides doubles the drawn width (each
    border node paints its whole ~0.1 mm cell, so both sides reads as a ~0.2 mm band)."""
    shape = pool.sub_volume.shape
    lab = np.zeros(shape, np.int64)
    lab[tuple(pool.node_idx.T)] = pool.region_full + 1          # 0 = non-tissue
    border = np.zeros(shape, bool)
    for ax in range(3):
        sl_a = [slice(None)] * 3; sl_a[ax] = slice(0, shape[ax] - 1)
        sl_b = [slice(None)] * 3; sl_b[ax] = slice(1, shape[ax])
        a, b = lab[tuple(sl_a)], lab[tuple(sl_b)]
        diff = (a > 0) & (b > 0) & (a != b)                    # two different real regions
        border[tuple(sl_a)] |= diff
        if both_sides:
            border[tuple(sl_b)] |= diff
    return border[tuple(pool.node_idx.T)]


def predict_field(pool, spec, lipid, slice_nodes, allow_solve=False, n_train=1500, seed=0):
    """GP predict one lipid at the slice's voxels under `spec`.

    Fits on a random set of covered template nodes (noise tuned on a val split),
    predicts at the slice nodes. Returns ``(pred, measured, err)`` arrays over the
    slice nodes (measured/err are NaN where the lipid was not measured there)."""
    z, covered = G.node_lipid_field(pool, lipid, verbose=False)
    cov_idx = np.where(covered)[0]
    rng = np.random.default_rng(seed)
    tr = rng.choice(cov_idx, min(n_train, len(cov_idx)), replace=False)
    va = rng.choice(np.setdiff1d(cov_idx, tr), min(500, len(cov_idx) - len(tr)), replace=False)
    ytr = z[tr].astype(np.float64)
    Ktt = G.node_gram(spec, pool, tr, tr, allow_solve=allow_solve)
    Kvt = G.node_gram(spec, pool, va, tr, allow_solve=allow_solve)
    d = np.trace(Ktt) / len(tr)
    best = (-9e9, 0.1)
    for al in [1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
        pv = Kvt @ np.linalg.solve(Ktt + al * d * np.eye(len(tr)), ytr)
        r2 = 1 - ((z[va] - pv) ** 2).sum() / ((z[va] - z[va].mean()) ** 2).sum()
        if r2 > best[0]: best = (r2, al)
    sol = np.linalg.solve(Ktt + best[1] * d * np.eye(len(tr)), ytr)
    pred = G.node_gram(spec, pool, slice_nodes, tr, allow_solve=allow_solve) @ sol
    meas = z[slice_nodes].astype(np.float64).copy()
    meas[~covered[slice_nodes]] = np.nan
    err = pred - meas
    return pred, meas, err


# ===========================================================================
# GUI  —  2D MALDI slices, click a test point, covariance heatmap
# ===========================================================================
def _heat_colors(vals, gamma=0.45, cmap="inferno", fade=True, alpha_floor=0.15):
    """Strong sequential heatmap: robust-normalise, gamma-boost the high end, and
    (fade=True) modulate OPACITY by the value so low-covariance points recede."""
    import matplotlib
    v = np.asarray(vals, np.float64)
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99.5)
    hi = hi if hi > lo else lo + 1e-9
    n = np.clip((v - lo) / (hi - lo), 0, 1) ** gamma
    rgba = matplotlib.colormaps[cmap](np.nan_to_num(n))
    if fade:
        rgba[:, 3] = alpha_floor + (1.0 - alpha_floor) * np.nan_to_num(n)
    return rgba


def _diverging_colors(vals, cmap="coolwarm", vlim=None):
    """Signed error heatmap centred at 0; |value| drives opacity (0 = transparent)."""
    import matplotlib
    v = np.asarray(vals, np.float64)
    m = np.isfinite(v)
    lim = vlim if vlim is not None else (np.nanpercentile(np.abs(v[m]), 98) if m.any() else 1.0)
    lim = lim if lim > 0 else 1.0
    n = np.clip(v / (2 * lim) + 0.5, 0, 1)
    rgba = matplotlib.colormaps[cmap](np.nan_to_num(n))
    rgba[:, 3] = np.where(m, np.clip(np.abs(v) / lim, 0, 1), 0.0)   # unmeasured -> invisible
    return rgba


def launch(pool, eu0, man0, allow_solve, sample="ReferenceAtlas"):
    import napari
    from magicgui import magicgui
    from qtpy.QtWidgets import QTextEdit
    from qtpy.QtGui import QFont

    bmask_box = [border_nodes(pool)]                    # (N,) region-border nodes (mutable)
    rgba_box = [region_rgba(pool)]                      # (N,4) per-node region colour
    viewer = napari.Viewer(title="Euclidean vs Manifold — MALDI kernel playroom",
                           ndisplay=2)
    slices_cache = {}
    def get_slices(smp):
        if smp not in slices_cache:
            slices_cache[smp] = MaldiSlices(pool, smp)
        return slices_cache[smp]

    slices = get_slices(sample)
    state = dict(eu=eu0, man=man0, which="manifold", slices=slices, ptsz=0.05,
                 section=slices.sections[len(slices.sections) // 2],
                 lipid=pool.lipids[0], test_node=None, cur_pts=None, cur_nodes=None)
    state["ptsz"] = max(slices.spacing * 1.4, 0.03)

    report = QTextEdit(); report.setReadOnly(True)
    mono = QFont("Monospace"); mono.setStyleHint(QFont.TypeWriter); mono.setPointSize(9)
    report.setFont(mono); report.setMinimumWidth(560)
    viewer.window.add_dock_widget(report, name="analysis", area="right")

    pts, nodes = slices.layer_data(state["section"])     # (M,2)
    state["cur_pts"], state["cur_nodes"] = pts, nodes
    state["test_node"] = int(nodes[len(nodes) // 2])
    blank = lambda n: np.tile([0, 0, 0, 0.0], (n, 1))
    ptsz = state["ptsz"]

    # layer stack (bottom -> top) — all pure 2D points
    cov_layer = viewer.add_points(pts, name="covariance heatmap", size=ptsz,
                                  face_color=np.tile([0.15, 0.15, 0.18, 1.0], (len(pts), 1)),
                                  border_width=0, opacity=1.0)
    true_layer = viewer.add_points(pts, name="true lipid", size=ptsz,
                                   face_color=blank(len(pts)), border_width=0, visible=False)
    pred_layer = viewer.add_points(pts, name="predicted lipid", size=ptsz,
                                   face_color=blank(len(pts)), border_width=0, visible=False)
    err_layer = viewer.add_points(pts, name="held-out error", size=ptsz,
                                  face_color=blank(len(pts)), border_width=0, visible=False)
    reg_layer = viewer.add_points(pts, name="regions (atlas)", size=ptsz,
                                  face_color=rgba_box[0][nodes], border_width=0,
                                  opacity=0.85, visible=False)
    brd_layer = viewer.add_points(pts[bmask_box[0][nodes]], name="region borders",
                                  size=ptsz * 0.7, face_color="#39ff14",
                                  border_width=0, opacity=0.9, visible=True)
    star = viewer.add_points(pts[len(pts) // 2][None], name="★ test point", size=ptsz * 6,
                             face_color=np.array([[0.1, 1.0, 0.9, 1.0]]),
                             border_color="white", border_width=0.12, symbol="star",
                             opacity=1.0)
    data_layers = (cov_layer, true_layer, pred_layer, err_layer, reg_layer)

    def _spec():
        return state["eu"] if state["which"] == "euclidean" else state["man"]

    def repaint_heatmap():
        try:
            kv = G.node_covariance(_spec(), pool, state["test_node"],
                                   state["cur_nodes"], allow_solve=allow_solve)
        except FileNotFoundError as e:
            report.setPlainText("EIGENPAIRS NOT CACHED\n\n" + str(e) +
                                "\n\nTick 'compute if missing' + Refit.")
            return
        cov_layer.face_color = _heat_colors(kv)          # colour + opacity fade
        cov_layer.refresh()
        j = int(np.where(state["cur_nodes"] == state["test_node"])[0][0]) \
            if state["test_node"] in state["cur_nodes"] else len(state["cur_pts"]) // 2
        star.data = state["cur_pts"][j][None]
        star.refresh()

    def repaint_prediction():
        """True / predicted / error layers — one shared GP fit (only if any is shown)."""
        if not (true_layer.visible or pred_layer.visible or err_layer.visible):
            return
        try:
            pred, meas, err = predict_field(pool, _spec(), state["lipid"],
                                            state["cur_nodes"], allow_solve=allow_solve)
        except FileNotFoundError:
            return
        vlim = np.nanpercentile(np.abs(np.r_[pred, meas]), 98)
        true_layer.face_color = _diverging_colors(meas, cmap="viridis", vlim=vlim)
        pred_layer.face_color = _diverging_colors(pred, cmap="viridis", vlim=vlim)
        err_layer.face_color = _diverging_colors(err)     # coolwarm, centred at 0

    def refresh_borders():
        m = bmask_box[0][state["cur_nodes"]]
        brd_layer.data = state["cur_pts"][m] if m.any() else np.zeros((0, 2), np.float32)
        reg_layer.face_color = rgba_box[0][state["cur_nodes"]]   # one colour per region
        reg_layer.refresh()

    def load_section(sec):
        state["section"] = sec
        pts, nodes = state["slices"].layer_data(sec)
        # swap data + a matching-length neutral colour TOGETHER, so no layer is ever
        # left with a face_color of the wrong length (the source of the crash).
        for ly in data_layers:
            ly.data = pts
            ly.face_color = np.tile([0.15, 0.15, 0.18, 1.0], (len(pts), 1))
            ly.size = state["ptsz"]
        state["cur_pts"], state["cur_nodes"] = pts, nodes
        if state["test_node"] not in nodes:
            state["test_node"] = int(nodes[len(nodes) // 2])
        refresh_borders(); repaint_heatmap(); repaint_prediction()

    def on_click(v, event):
        # VIEWER-level (fires regardless of active layer — a per-layer callback only
        # fires when that exact layer is selected, which is why clicking felt dead).
        # Generator form so we can tell a CLICK from a pan-DRAG and only act on a click.
        if event.type != "mouse_press":
            return
        p0 = np.asarray(event.position, np.float64)
        moved = False
        yield
        while event.type == "mouse_move":
            if np.linalg.norm(np.asarray(event.position, np.float64) - p0) > 0.2:  # mm
                moved = True
            yield
        if moved:
            return                                                # it was a pan, ignore
        pos = p0[-2:]                                             # world (row, col)
        j = int(np.argmin(((state["cur_pts"] - pos) ** 2).sum(1)))
        state["test_node"] = int(state["cur_nodes"][j])
        repaint_heatmap()
    viewer.mouse_drag_callbacks.append(on_click)

    def refit():
        report.setPlainText("… refitting analysis (a few seconds) …")
        try:
            res = analyze(pool, state["eu"], state["man"], allow_solve=allow_solve)
        except FileNotFoundError as e:
            report.setPlainText("EIGENPAIRS NOT CACHED\n\n" + str(e) +
                                "\n\nTick 'compute if missing' + Refit.")
            return
        report.setPlainText(format_report(res))
        refresh_borders(); repaint_heatmap(); repaint_prediction()

    # ---- navigation: pick a MOUSE and a SLICE (live — both are cheap) ----
    # (the ATLAS lives in the REFIT panel instead: switching it re-labels regions and
    #  re-runs the whole analysis, so it should happen on a button press, not live.)
    from magicgui.widgets import ComboBox, Container
    cur_atlas = atlas_key(pool.annotations)
    samp_w = ComboBox(label="mouse", choices=G.SAMPLES,
                      value=sample if sample in G.SAMPLES else G.SAMPLES[0])
    sec_w = ComboBox(label="slice", choices=slices.sections, value=state["section"])

    def on_section(sec):
        if sec is not None:
            load_section(sec)

    def on_sample(smp):
        sl = get_slices(smp)
        state["slices"] = sl
        state["ptsz"] = max(sl.spacing * 1.4, 0.03)
        sec = sl.sections[len(sl.sections) // 2]
        sec_w.changed.disconnect(on_section)             # avoid double-fire while resetting
        sec_w.choices = sl.sections
        sec_w.value = sec
        sec_w.changed.connect(on_section)
        load_section(sec)

    def apply_atlas(which):
        """Re-label regions from a different Allen level. Cheap (no re-pooling), but it
        changes every border/region-dependent number, so it runs on the REFIT press."""
        if atlas_path(which) == pool.annotations:
            return
        pool.use_atlas(atlas_path(which), verbose=False)   # recompute region labels
        bmask_box[0] = border_nodes(pool)                  # borders follow the new atlas
        rgba_box[0] = region_rgba(pool)                    # ... and so do the region colours

    samp_w.changed.connect(on_sample)
    sec_w.changed.connect(on_section)
    viewer.window.add_dock_widget(Container(widgets=[samp_w, sec_w]),
                                  name="mouse / slice", area="left")

    @magicgui(
        call_button="◤ REFIT + REPAINT ◢",
        atlas={"label": "atlas level (applies on REFIT)", "choices": ATLAS_CHOICES},
        show={"label": "covariance / pred shows", "choices": ["manifold", "euclidean"]},
        lipid={"label": "lipid (true/pred/error)", "choices": pool.lipids},
        show_true={"label": "layer: true lipid"},
        show_pred={"label": "layer: predicted lipid"},
        show_error={"label": "layer: held-out error"},
        show_regions={"label": "layer: regions (atlas)"},
        eu_nu={"label": "Euclid ν", "choices": [0.5, 1.5, 2.5]},
        eu_ls={"label": "Euclid ℓ (mm)", "widget_type": "FloatSpinBox", "min": 0.1, "max": 20, "step": 0.1},
        eu_os={"label": "Euclid outputscale", "widget_type": "FloatSpinBox", "min": 0.01, "max": 100, "step": 0.1},
        eu_border={"label": "Euclid border down-weight", "widget_type": "FloatSpinBox", "min": 0, "max": 1, "step": 0.05},
        man_method={"label": "Manifold graph", "choices": ["faiss", "faiss_atlas_weighted"]},
        man_infl={"label": "cross-region ×", "widget_type": "FloatSpinBox", "min": 1, "max": 500, "step": 1},
        man_prune={"label": "prune fraction", "widget_type": "FloatSpinBox", "min": 0, "max": 1, "step": 0.05},
        man_denoise={"label": "denoise passes", "widget_type": "SpinBox", "min": 0, "max": 5},
        man_modes={"label": "num modes", "widget_type": "SpinBox", "min": 20, "max": 6000, "step": 100},
        man_bw={"label": "graph bandwidth", "widget_type": "FloatSpinBox", "min": 0.01, "max": 1, "step": 0.01},
        man_nu={"label": "Manifold ν", "choices": [1, 2, 3]},
        man_ls={"label": "Manifold ℓ", "widget_type": "FloatSpinBox", "min": 0.1, "max": 20, "step": 0.1},
        man_os={"label": "Manifold outputscale", "widget_type": "FloatSpinBox", "min": 0.01, "max": 100, "step": 0.1},
        compute={"label": "compute eigvecs if missing"},
    )
    def controls(atlas=cur_atlas, show="manifold", lipid=pool.lipids[0],
                 show_true=False, show_pred=False, show_error=False, show_regions=False,
                 eu_nu=eu0.nu, eu_ls=eu0.lengthscale, eu_os=eu0.outputscale,
                 eu_border=eu0.border_downweight,
                 man_method=man0.knn_method, man_infl=man0.inflation, man_prune=man0.prune,
                 man_denoise=man0.denoise, man_modes=man0.num_modes, man_bw=man0.bandwidth,
                 man_nu=int(man0.nu), man_ls=man0.lengthscale, man_os=man0.outputscale,
                 compute=allow_solve):
        nonlocal allow_solve
        allow_solve = bool(compute)
        apply_atlas(atlas)              # BEFORE the specs: man_ann reads pool.annotations
        state["which"] = show; state["lipid"] = lipid
        true_layer.visible = bool(show_true)
        pred_layer.visible = bool(show_pred)
        err_layer.visible = bool(show_error)
        reg_layer.visible = bool(show_regions)
        state["eu"] = KernelSpec.euclidean(nu=eu_nu, lengthscale=eu_ls, outputscale=eu_os,
                                           border_downweight=eu_border)
        # keep the manifold graph's atlas in sync with the analysis atlas (pool)
        man_ann = pool.annotations if man_method == "faiss_atlas_weighted" else man0.annotations
        state["man"] = replace(man0, knn_method=man_method, inflation=man_infl, prune=man_prune,
                               denoise=man_denoise, num_modes=man_modes, bandwidth=man_bw,
                               nu=man_nu, lengthscale=man_ls, outputscale=man_os,
                               annotations=man_ann, name=None)
        refit()

    viewer.window.add_dock_widget(controls, name="kernel controls", area="right")
    refit()
    napari.run()


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euclidean", default="nu=0.5,lengthscale=2.0",
                    help="comma-separated k=v (nu, lengthscale, border_downweight)")
    ap.add_argument("--manifold",
                    default="knn_method=faiss_atlas_weighted,inflation=50,num_modes=1000,nu=2,lengthscale=5",
                    help="comma-separated k=v graph+kernel params")
    ap.add_argument("--manifold-run", default=None,
                    help="a trained run dir -> use its config + LEARNED lengthscale for the manifold kernel")
    ap.add_argument("--sample", default="ReferenceAtlas",
                    help="which wild-type brain's MALDI slices to show in the GUI")
    ap.add_argument("--atlas", choices=ATLAS_CHOICES, default="15",
                    help="Annotation defining regions/borders: 5/15 = LBAE levels "
                         "(root catch-all covers 72%%/57%% of brain), d5/d7 = CCF "
                         "depth-cuts with 175/476 regions and no root catch-all")
    ap.add_argument("--anchors", type=int, default=3000)
    ap.add_argument("--cell", type=int, default=2)
    ap.add_argument("--compute-if-missing", action="store_true",
                    help="solve missing eigenpairs on the GPU (one-time) instead of stopping")
    ap.add_argument("--self-test", action="store_true",
                    help="run the numeric pipeline headless and print the report")
    ap.add_argument("--dump-slice", nargs="?", const="__mid__", default=None,
                    help="headless: build the MALDI slices and node-covariance for one "
                         "section (no napari), print stats. Optional section id.")
    a = ap.parse_args()

    pool = CellPool(cell=a.cell, n_anchors=a.anchors, annotations=atlas_path(a.atlas))
    eu = euclidean_spec(parse_kv(a.euclidean))
    man = (KernelSpec.from_run(a.manifold_run) if a.manifold_run
           else manifold_spec(parse_kv(a.manifold)))

    if a.self_test:
        try:
            res = analyze(pool, eu, man, allow_solve=a.compute_if_missing)
        except FileNotFoundError as e:
            print(str(e))
            print("\nRe-run with --compute-if-missing to solve this on the GPU (one-time).")
            sys.exit(1)
        print("\n" + "=" * 90)
        print(format_report(res))
        print("=" * 90)
        return

    if a.dump_slice is not None:
        sl = MaldiSlices(pool, a.sample)
        sec = sl.sections[len(sl.sections) // 2] if a.dump_slice == "__mid__" else \
            type(sl.sections[0])(a.dump_slice)
        pts, nodes = sl.layer_data(sec)
        print(f"sample {a.sample!r}: {len(sl.sections)} sections, spacing {sl.spacing:.3f} mm")
        print(f"section {sec}: {len(nodes):,} voxels snapped to {len(np.unique(nodes)):,} nodes")
        tn = int(nodes[len(nodes) // 2])
        bm = border_nodes(pool)
        print(f"border layer: {int(bm[nodes].sum()):,}/{len(nodes):,} slice voxels on a region border")
        for spec, tag in [(eu, "euclidean"), (man, "manifold")]:
            try:
                kv = G.node_covariance(spec, pool, tn, nodes, allow_solve=a.compute_if_missing)
                col = _heat_colors(kv)   # raw k tiny for manifold; heatmap percentile-normalises
                pred, meas, err = predict_field(pool, spec, pool.lipids[0], nodes,
                                                allow_solve=a.compute_if_missing)
                fin = np.isfinite(err)
                r2 = 1 - (np.nansum(err ** 2) /
                          np.nansum((meas - np.nanmean(meas)) ** 2))
                print(f"  {tag:<10} os={spec.outputscale:g}  cov k[{kv.min():+.1e},{kv.max():.1e}] "
                      f"contrast {col[:,0].std():.3f} alpha[{col[:,3].min():.2f},{col[:,3].max():.2f}] | "
                      f"{pool.lipids[0]}: true[{np.nanmin(meas):+.1f},{np.nanmax(meas):+.1f}] "
                      f"pred[{pred.min():+.1f},{pred.max():+.1f}] |err|med {np.nanmedian(np.abs(err)):.2f} "
                      f"R²={r2:.3f} ({fin.sum():,} meas)")
            except FileNotFoundError as e:
                print(f"  {tag:<10} {str(e).splitlines()[1]}")
        return

    launch(pool, eu, man, allow_solve=a.compute_if_missing, sample=a.sample)


if __name__ == "__main__":
    main()
