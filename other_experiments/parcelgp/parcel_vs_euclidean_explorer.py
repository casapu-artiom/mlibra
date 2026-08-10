#!/usr/bin/env python
"""Euclidean vs parcel kernel, side by side, on real MALDI slices.

The parcel factor exists to say one thing a stationary kernel cannot: *these two
voxels are close but on opposite sides of a boundary, do not smooth between
them*. This tool shows whether it actually says it. Pick a mouse and a section,
click a voxel, and the same section is drawn twice —

    LEFT   k_base(test, .)                    the Euclidean Matern alone
    RIGHT  k_base(test, .) * exp(-|z-z'|^2/2) the same kernel times the factor

— with the parcel borders drawn on both panels, so the question "does the right
panel stop at a border the left panel walks straight through?" is answered by
looking. The right panel can also show the FACTOR alone (in [0, 1], the pure
parcel geometry with the distance decay divided out) or the signed difference.

Where the parcel embedding comes from
-------------------------------------
``k_parcel = k_base * exp(-||m(x)^T B - m(x')^T B||^2 / 2)``, and ``B`` is the
only learned part. Two ways to fill it in:

  * **untrained** (default) -- ``B = strength * I``, i.e. the embedding is the
    membership vector itself. Deep inside two different parcels the multiplier
    is ``exp(-strength^2)``, and ``strength=0`` is an exact no-op. This shows the
    partition's geometry honestly: nothing has been fitted to lipids.
  * **trained** (``--run <dir> --run-lipid <name>``) -- the ``B`` a per-lipid run
    actually learned, together with that lipid's learned ARD lengthscale and
    outputscale, so both panels reproduce the deployed model rather than a
    hand-tuned lookalike. Omit ``--run-lipid`` once to list the run's lipids.

The measured lipid layer is there to referee: if the right panel's boundary is
real, the lipid should change across it.

    python -m other_experiments.parcelgp.parcel_vs_euclidean_explorer --field .../full_k128_...npz
    python -m other_experiments.parcelgp.parcel_vs_euclidean_explorer --field ... \
        --run /home/casap/mlibra/output/parcel/FOLD-2-...-parcel... \
        --run-lipid "PC 35:1 PE 38:1"
    python -m other_experiments.parcelgp.parcel_vs_euclidean_explorer --field ... --dump /tmp/k.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .field import ParcelField
from .viz import (available_lipids, border_mask, diverging_colors, euclidean_row,
                  heat_colors, list_samples, load_trained_parcel, MaldiSections,
                  parcel_colors, parcel_embedding, parcel_factor_row)

log = logging.getLogger("parcelgp.explorer")

#: What the right-hand panel can show. The left is always the base kernel.
RIGHT_MODES = ("parcel kernel", "parcel factor", "difference")


# --------------------------------------------------------------------------- #
# the numeric core — shared by the GUI and the headless --dump
# --------------------------------------------------------------------------- #
class SliceKernels:
    """Kernel rows for one section, evaluated at its unique template nodes.

    A section holds ~4 MALDI voxels per 0.05 mm node, and the kernel is a
    function of the node, so everything is computed once per unique node and
    gathered back to voxels. That is a 4x saving on the parcel embedding, which
    is the only part big enough to notice (dense memberships are K-wide).
    """

    def __init__(self, field, nodes: np.ndarray):
        self.field = field
        self.uniq, self.inv = np.unique(np.asarray(nodes), return_inverse=True)
        self.coords = field.node_coords[self.uniq].astype(np.float64)
        self.labels = field.labels[self.uniq]
        self._Z = None
        self._Z_key = None
        self._Z_B = None            # strong ref: keeps id(B) from being recycled

    def embedding(self, B, strength: float) -> np.ndarray:
        key = (id(B) if B is not None else None, float(strength))
        if self._Z_key != key:
            self._Z = parcel_embedding(self.field, self.uniq, B, strength)
            self._Z_key, self._Z_B = key, B
        return self._Z

    def rows(self, test_node: int, lengthscale, outputscale, nu, B, strength):
        """``(k_base, factor)`` at every voxel, given the test node.

        ``k_parcel`` is the product; keeping the two apart is what lets the GUI
        show the factor on its own, which is the only view where the parcel
        geometry is visible without the distance decay on top of it.
        """
        j = int(np.searchsorted(self.uniq, int(test_node)))
        if j >= self.uniq.size or self.uniq[j] != int(test_node):
            raise ValueError(f"node {test_node} is not in this section")
        k_base = euclidean_row(self.coords, self.coords[j], lengthscale,
                               outputscale, nu)
        factor = parcel_factor_row(self.embedding(B, strength), j)
        return k_base[self.inv], factor[self.inv], j


def panel_values(k_base, factor, mode: str):
    """``(values, kind)`` for the right-hand panel under ``mode``."""
    if mode == "parcel kernel":
        return k_base * factor, "heat"
    if mode == "parcel factor":
        return factor, "factor"
    if mode == "difference":
        return k_base * factor - k_base, "diverging"
    raise ValueError(f"unknown right-panel mode {mode!r}")


def _weighted_std(y: np.ndarray, w: np.ndarray) -> float:
    """Standard deviation of ``y`` under weights ``w`` (NaNs excluded).

    Applied to a kernel row this is *how chemically heterogeneous the tissue is
    that this kernel smooths over* — the quantity the parcel factor is supposed
    to reduce. It is the honest form of the question: comparing each neighbour
    against the test voxel's own value instead would be dominated by that single
    voxel's ~19% measurement noise (see other_experiments/parcelgp/README.md), which is the same
    for both kernels but swamps the difference between them.
    """
    m = np.isfinite(y) & (w > 0)
    if m.sum() < 10:
        return float("nan")
    yy, ww = y[m].astype(np.float64), w[m].astype(np.float64)
    ww = ww / ww.sum()
    mu = float((ww * yy).sum())
    return float(np.sqrt((ww * (yy - mu) ** 2).sum()))


def summarize(field, sk: SliceKernels, test_node: int, k_base, factor,
              nodes, lipid_vals=None, B=None, strength=1.0,
              lengthscale=None, outputscale=1.0, nu=2.5, trained=None) -> str:
    """The text panel: what the factor is doing, in numbers rather than colour."""
    lab = field.labels[np.asarray(nodes)]
    t_lab = int(field.labels[int(test_node)])
    same = lab == t_lab
    k_par = k_base * factor
    mem_i = field.mem_idx[int(test_node)]
    mem_v = field.mem_val[int(test_node)]

    def eff(k):
        """Effective neighbourhood size, in voxels: sum(k) / max(k).

        Scale-free (the outputscale cancels), and it is the number the two
        kernels should differ on — the factor can only ever shrink it.
        """
        m = float(np.max(k))
        return float(np.sum(k) / m) if m > 0 else float("nan")

    out = [
        f"test node    : {int(test_node)}  parcel {t_lab}  "
        f"(d_border_rel {float(field.d_border_rel[int(test_node)]):.2f}, "
        f"nearest other parcel {int(field.nearest_other[int(test_node)])})",
        f"memberships  : " + ", ".join(f"p{int(i)}={float(v):.2f}"
                                       for i, v in zip(mem_i, mem_v)),
        f"this section : {len(nodes):,} voxels, {sk.uniq.size:,} nodes, "
        f"{int(same.sum()):,} ({same.mean():.1%}) in the test point's parcel",
        "",
        f"kernel       : Matern nu={nu}, outputscale={outputscale:.4g}, "
        f"lengthscale={np.array2string(np.broadcast_to(np.asarray(lengthscale, float).ravel(), (3,)), precision=3)}"
        f" (standardized units)",
        f"parcel B     : " + (f"TRAINED on {trained.lipid!r} — rank {B.shape[1]}, "
                              f"mean |B_k| {float(np.linalg.norm(B, axis=1).mean()):.3f}"
                              if trained is not None and B is not None else
                              f"untrained identity, strength {strength:g} "
                              f"(cross-parcel multiplier {np.exp(-strength ** 2):.3f})"),
        "",
        f"{'':<22}{'euclidean':>12}{'parcel':>12}{'ratio':>9}",
        f"{'effective #voxels':<22}{eff(k_base):>12.1f}{eff(k_par):>12.1f}"
        f"{eff(k_par) / max(eff(k_base), 1e-30):>9.3f}",
    ]
    if same.any() and (~same).any():
        out += [
            f"{'mean k, same parcel':<22}{k_base[same].mean():>12.4g}"
            f"{k_par[same].mean():>12.4g}"
            f"{k_par[same].mean() / max(k_base[same].mean(), 1e-30):>9.3f}",
            f"{'mean k, other parcel':<22}{k_base[~same].mean():>12.4g}"
            f"{k_par[~same].mean():>12.4g}"
            f"{k_par[~same].mean() / max(k_base[~same].mean(), 1e-30):>9.3f}",
            "",
            f"mean factor  : {factor[same].mean():.3f} within parcel, "
            f"{factor[~same].mean():.3f} across a border  "
            f"(1.000 = the factor is doing nothing)",
        ]
    else:
        out += ["", "the whole section is one parcel — nothing to separate here"]

    if lipid_vals is not None:
        y = np.asarray(lipid_vals, np.float64)
        flat = np.ones_like(k_base)
        out += [
            "",
            f"k-weighted SD of the measured lipid over the slice — how "
            f"chemically mixed each",
            f"kernel's neighbourhood is (lower is better; the whole section is "
            f"the baseline):",
            f"{'':<22}{'euclidean':>12}{'parcel':>12}{'section':>12}",
            f"{'SD of ' + str(len(y)) + ' voxels':<22}"
            f"{_weighted_std(y, k_base):>12.4f}{_weighted_std(y, k_par):>12.4f}"
            f"{_weighted_std(y, flat):>12.4f}",
        ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# headless dump
# --------------------------------------------------------------------------- #
def dump(field, sections: MaldiSections, out_path, section=None, test_node=None,
         lengthscale=1.0, outputscale=1.0, nu=2.5, B=None, strength=1.0,
         lipid=None, dpi=140, trained=None):
    """Write a panel figure for one section, and print the same summary the GUI shows.

    Panels: the parcels, both kernel rows, the factor alone, and — when ``lipid``
    is given — the measured lipid, so the figure carries its own referee.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sec = section if section is not None else sections.sections[len(sections.sections) // 2]
    xy, nodes, rows = sections.layer_data(sec)
    sk = SliceKernels(field, nodes)
    bm_slice = border_mask(field)[nodes]
    if test_node is None:
        # Default to a border voxel near the middle of the section: the border is
        # where the two kernels are supposed to disagree, and the middle keeps
        # the neighbourhood from running off the edge of the tissue.
        d = ((xy - xy.mean(0)) ** 2).sum(1)
        cand = np.where(bm_slice)[0] if bm_slice.any() else np.arange(len(xy))
        test_node = int(nodes[cand[int(np.argmin(d[cand]))]])
    k_base, factor, _ = sk.rows(test_node, lengthscale, outputscale, nu, B, strength)

    y = sections.lipid(lipid)[rows] if lipid else None
    print(summarize(field, sk, test_node, k_base, factor, nodes, y, B, strength,
                    lengthscale, outputscale, nu, trained))

    colors = parcel_colors(field.n_parcels)
    tj = int(np.argmax(nodes == test_node))
    panels = [
        ("parcels + borders", colors[field.labels[nodes]]),
        ("euclidean   k(test, ·)", heat_colors(k_base)),
        ("parcel   k(test, ·)", heat_colors(k_base * factor)),
        ("parcel factor   exp(−|Δz|²/2)",
         heat_colors(factor, vlim=(0.0, 1.0), cmap="viridis", fade=False, gamma=1.0)),
    ]
    if y is not None:
        panels.append((f"measured {lipid}", diverging_colors(y, cmap="viridis")))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6),
                             squeeze=False, facecolor="#101014")
    # Marker area in points^2 that makes the voxels tile: one voxel is `spacing`
    # mm wide, and the panel maps its data span across ~4.2 inches of figure.
    span = float(xy[:, 1].max() - xy[:, 1].min()) or 1.0
    s = max((sections.spacing * (4.2 * 72.0) / span) ** 2, 0.5)
    for i, (ax, (title, rgba)) in enumerate(zip(axes[0], panels)):
        ax.set_facecolor("#101014")
        ax.scatter(xy[:, 1], xy[:, 0], c=rgba, s=s, marker="s", linewidths=0)
        # Borders go on every panel EXCEPT the parcel-colour one, whose own
        # colours already delineate them, and at close to the base marker size:
        # anything much smaller is sub-pixel at this voxel density and vanishes.
        if bm_slice.any() and i > 0:
            ax.scatter(xy[bm_slice, 1], xy[bm_slice, 0], c="#39ff14", s=s * 0.8,
                       marker="s", linewidths=0, alpha=0.75, zorder=3)
        ax.scatter([xy[tj, 1]], [xy[tj, 0]], marker="*", s=280, c="#19fce6",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_title(title, fontsize=10, color="white")
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{sections.sample} section {sec} — {field.n_parcels} parcels, "
                 + (f"trained B ({trained.lipid})" if trained is not None and B is not None
                    else f"untrained B, strength {strength:g}"),
                 fontsize=11, color="white")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("wrote %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def launch(field, maldi_file, sample, lipids, kernel0, sections0=None,
           trained=None, max_snap_mm=0.5):
    import napari
    from magicgui import magicgui
    from magicgui.widgets import ComboBox, Container
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QTextEdit

    bmask = border_mask(field)
    pcolors = parcel_colors(field.n_parcels)
    viewer = napari.Viewer(title="parcelgp — euclidean vs parcel kernel", ndisplay=2)

    cache = {sample: sections0} if sections0 is not None else {}

    def get_sections(smp):
        if smp not in cache:
            cache[smp] = MaldiSections(field, maldi_file, smp, max_snap_mm)
        return cache[smp]

    secs = get_sections(sample)
    st = dict(sections=secs, section=secs.sections[len(secs.sections) // 2],
              right="parcel kernel", lipid=lipids[0] if lipids else None,
              show_lipid=False, **kernel0)

    report = QTextEdit(); report.setReadOnly(True)
    mono = QFont("Monospace"); mono.setStyleHint(QFont.TypeWriter); mono.setPointSize(9)
    report.setFont(mono); report.setMinimumWidth(560)
    viewer.window.add_dock_widget(report, name="what the factor is doing", area="right")

    def panel_layout(xy):
        """Left panel at the data's own coordinates, right panel shifted across.

        One canvas, two copies of the section: a side-by-side comparison beats
        toggling layers, because the difference between the two kernels is
        usually a change of SHAPE near a border, which a toggle hides."""
        span = float(xy[:, 1].max() - xy[:, 1].min())
        dx = span * 1.12 + 4 * max(span * 0.01, 1e-3)
        return dx, xy + np.array([0.0, dx], np.float32)

    xy, nodes, rows = secs.layer_data(st["section"])
    dx, xy_r = panel_layout(xy)
    st.update(xy=xy, xy_r=xy_r, nodes=nodes, rows=rows, dx=dx,
              sk=SliceKernels(field, nodes), test_node=int(nodes[len(nodes) // 2]))
    ptsz = max(secs.spacing * 1.4, 0.03)
    st["ptsz"] = ptsz
    # `both` duplicates a per-voxel VALUE for the two panels; `both_pts` places
    # the two copies — the second one shifted across. Duplicating the positions
    # with `both` instead would stack the overlays on the left panel only.
    both = lambda a: np.concatenate([a, a], axis=0)
    both_pts = lambda: np.concatenate([st["xy"], st["xy_r"]])
    grey = lambda n: np.tile([0.14, 0.14, 0.17, 1.0], (n, 1))

    parc_layer = viewer.add_points(both_pts(), name="parcels", size=ptsz,
                                   face_color=both(pcolors[field.labels[nodes]]),
                                   border_width=0, opacity=0.85, visible=False)
    left_layer = viewer.add_points(xy, name="k euclidean (left)", size=ptsz,
                                   face_color=grey(len(xy)), border_width=0)
    right_layer = viewer.add_points(xy_r, name="k parcel (right)", size=ptsz,
                                    face_color=grey(len(xy)), border_width=0)
    lip_layer = viewer.add_points(both_pts(), name="measured lipid", size=ptsz,
                                  face_color=np.zeros((2 * len(xy), 4), np.float32),
                                  border_width=0, visible=False)
    brd_layer = viewer.add_points(both_pts()[both(bmask[nodes])], name="parcel borders",
                                  size=ptsz * 0.7, face_color="#39ff14",
                                  border_width=0, opacity=0.95)
    star = viewer.add_points(np.stack([xy[0], xy_r[0]]), name="* test point",
                             size=ptsz * 6, face_color=np.array([[0.1, 1.0, 0.9, 1.0]]),
                             border_color="white", border_width=0.12, symbol="star")

    def repaint():
        sk = st["sk"]
        try:
            k_base, factor, _ = sk.rows(st["test_node"], st["lengthscale"],
                                        st["outputscale"], st["nu"],
                                        st["B"], st["strength"])
        except ValueError as e:
            report.setPlainText(str(e))
            return
        left_layer.face_color = heat_colors(k_base)
        vals, kind = panel_values(k_base, factor, st["right"])
        if kind == "diverging":
            right_layer.face_color = diverging_colors(vals)
        elif kind == "factor":
            # Fixed 0..1 scale, no gamma, no opacity fade: the factor IS the
            # picture here, and fading it by value would hide the borders.
            right_layer.face_color = heat_colors(vals, vlim=(0.0, 1.0), cmap="viridis",
                                                 fade=False, gamma=1.0)
        else:
            # SAME absolute scale as the left panel, so the right panel losing
            # brightness across a border is a real drop in covariance and not a
            # per-panel renormalisation artefact.
            lo, hi = np.nanpercentile(k_base, 1), np.nanpercentile(k_base, 99.5)
            right_layer.face_color = heat_colors(vals, vlim=(lo, hi))
        right_layer.name = f"{st['right']} (right)"
        left_layer.refresh(); right_layer.refresh()

        j = int(np.argmax(st["nodes"] == st["test_node"]))
        star.data = np.stack([st["xy"][j], st["xy_r"][j]])
        star.refresh()

        y = None
        if st["show_lipid"] and st["lipid"]:
            y = st["sections"].lipid(st["lipid"])[st["rows"]]
            lip_layer.face_color = both(diverging_colors(y, cmap="viridis"))
        report.setPlainText(summarize(
            field, sk, st["test_node"], k_base, factor, st["nodes"], y,
            st["B"], st["strength"], st["lengthscale"], st["outputscale"],
            st["nu"], trained))

    def load_section(sec):
        st["section"] = sec
        xy, nodes, rows = st["sections"].layer_data(sec)
        dx, xy_r = panel_layout(xy)
        st.update(xy=xy, xy_r=xy_r, nodes=nodes, rows=rows, dx=dx,
                  sk=SliceKernels(field, nodes))
        if st["test_node"] not in nodes:
            st["test_node"] = int(nodes[len(nodes) // 2])
        # Swap data AND a matching-length colour together: a layer left holding a
        # face_color of the previous section's length is the classic napari crash.
        for ly, pts in ((left_layer, xy), (right_layer, xy_r)):
            ly.data = pts
            ly.face_color = grey(len(pts))
            ly.size = st["ptsz"]
        parc_layer.data = both_pts()
        parc_layer.face_color = both(pcolors[field.labels[nodes]])
        parc_layer.size = st["ptsz"]
        lip_layer.data = both_pts()
        lip_layer.face_color = np.zeros((2 * len(xy), 4), np.float32)
        lip_layer.size = st["ptsz"]
        m = both(bmask[nodes])
        brd_layer.data = both_pts()[m] if m.any() else np.zeros((0, 2), np.float32)
        brd_layer.size = st["ptsz"] * 0.7
        repaint()
        viewer.reset_view()

    def on_click(v, event):
        # Viewer-level and generator-form: a per-layer callback only fires when
        # that exact layer is selected, and the generator lets a pan-DRAG be told
        # apart from a click so dragging the canvas does not move the test point.
        if event.type != "mouse_press":
            return
        p0 = np.asarray(event.position, np.float64)
        moved = False
        yield
        while event.type == "mouse_move":
            if np.linalg.norm(np.asarray(event.position, np.float64) - p0) > 0.2:
                moved = True
            yield
        if moved:
            return
        pos = p0[-2:]
        # Either panel picks the same voxel: match against BOTH copies and fold
        # the right one's offset away with a modulo. Deciding which panel was
        # clicked from a distance threshold instead would misfire along the seam
        # between them, which is exactly where you click to compare.
        n = len(st["xy"])
        j = int(np.argmin(((both_pts() - pos) ** 2).sum(1))) % n
        st["test_node"] = int(st["nodes"][j])
        repaint()
    viewer.mouse_drag_callbacks.append(on_click)

    samp_w = ComboBox(label="mouse", choices=list_samples(maldi_file), value=sample)
    sec_w = ComboBox(label="section", choices=secs.sections, value=st["section"])

    def on_section(sec):
        if sec is not None:
            load_section(sec)

    def on_sample(smp):
        sl = get_sections(smp)
        st["sections"] = sl
        st["ptsz"] = max(sl.spacing * 1.4, 0.03)
        sec_w.changed.disconnect(on_section)     # don't double-fire while resetting
        sec_w.choices = sl.sections
        sec_w.value = sl.sections[len(sl.sections) // 2]
        sec_w.changed.connect(on_section)
        load_section(sec_w.value)

    samp_w.changed.connect(on_sample)
    sec_w.changed.connect(on_section)
    viewer.window.add_dock_widget(Container(widgets=[samp_w, sec_w]),
                                  name="mouse / section", area="left")

    ls0 = np.broadcast_to(np.asarray(kernel0["lengthscale"], float).ravel(), (3,))
    trained_ok = trained is not None and trained.B is not None

    @magicgui(
        call_button="◤ REPAINT ◢",
        right={"label": "right panel shows", "choices": list(RIGHT_MODES)},
        use_trained={"label": "use trained B" + ("" if trained_ok else " (none loaded)"),
                     "widget_type": "CheckBox"},
        strength={"label": "untrained B strength", "widget_type": "FloatSpinBox",
                  "min": 0.0, "max": 8.0, "step": 0.1},
        nu={"label": "Matern nu", "choices": [0.5, 1.5, 2.5]},
        ls_ap={"label": "lengthscale AP", "widget_type": "FloatSpinBox",
               "min": 0.001, "max": 50.0, "step": 1e-4},
        ls_dv={"label": "lengthscale DV", "widget_type": "FloatSpinBox",
               "min": 0.001, "max": 50.0, "step": 1e-4},
        ls_lr={"label": "lengthscale LR", "widget_type": "FloatSpinBox",
               "min": 0.001, "max": 50.0, "step": 1e-4},
        outputscale={"label": "outputscale", "widget_type": "FloatSpinBox",
                     "min": 1e-4, "max": 1000.0, "step": 1e-4},
        lipid={"label": "lipid", "choices": lipids or [""]},
        show_lipid={"label": "layer: measured lipid"},
        show_parcels={"label": "layer: parcels"},
        show_borders={"label": "layer: parcel borders"},
    )
    def controls(right=st["right"], use_trained=trained_ok,
                 strength=kernel0["strength"], nu=kernel0["nu"],
                 ls_ap=float(ls0[0]), ls_dv=float(ls0[1]), ls_lr=float(ls0[2]),
                 outputscale=kernel0["outputscale"],
                 lipid=(lipids[0] if lipids else ""), show_lipid=False,
                 show_parcels=False, show_borders=True):
        st["right"] = right
        st["strength"] = float(strength)
        st["B"] = trained.B if (use_trained and trained_ok) else None
        st["nu"] = float(nu)
        st["lengthscale"] = np.array([ls_ap, ls_dv, ls_lr], np.float64)
        st["outputscale"] = float(outputscale)
        st["lipid"] = lipid or None
        st["show_lipid"] = bool(show_lipid)
        lip_layer.visible = bool(show_lipid)
        parc_layer.visible = bool(show_parcels)
        brd_layer.visible = bool(show_borders)
        repaint()

    viewer.window.add_dock_widget(controls, name="kernel controls", area="right")
    repaint()
    viewer.reset_view()
    napari.run()


# --------------------------------------------------------------------------- #
def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--field", default=None,
                   help="A built parcel field .npz. Required for everything "
                        "except --list-samples.")
    p.add_argument("--maldi-file",
                   default="/home/casap/mlibra/mlibra_data/maindata_minimal.parquet")
    p.add_argument("--available-lipids-file", default=None,
                   help="maindata_minimal_available_lipids.npy — restricts the "
                        "lipid dropdown to the modelled set.")
    p.add_argument("--sample", default="ReferenceAtlas",
                   help="which brain's MALDI sections to show (--list-samples).")
    p.add_argument("--list-samples", action="store_true")
    p.add_argument("--max-snap-mm", type=float, default=0.5,
                   help="drop voxels further than this from any template node.")

    p.add_argument("--run", default=None,
                   help="a per-lipid run dir -> use its LEARNED B, ARD lengthscale "
                        "and outputscale instead of the manual settings.")
    p.add_argument("--run-lipid", default=None,
                   help="which lipid of --run to load (each has its own B). Omit "
                        "once to list the run's lipids.")

    p.add_argument("--nu", type=float, default=2.5, choices=(0.5, 1.5, 2.5))
    p.add_argument("--lengthscale", type=float, nargs="+", default=[0.15],
                   help="one value (isotropic) or three (ARD, AP DV LR), in "
                        "STANDARDIZED units — the field's coord_std is mm per unit.")
    p.add_argument("--outputscale", type=float, default=1.0)
    p.add_argument("--parcel-strength", type=float, default=1.5,
                   help="untrained B = strength * I. Cross-parcel covariance "
                        "multiplier is exp(-strength^2); 0 is an exact no-op.")

    p.add_argument("--dump", default=None,
                   help="write a PNG for one section and exit (no napari).")
    p.add_argument("--section", default=None, help="--dump: which section (default: middle).")
    p.add_argument("--test-node", type=int, default=None,
                   help="--dump: the node k(test, .) is centred on. Default is a "
                        "border voxel near the middle of the section; pass the id "
                        "printed in the summary to reproduce a figure exactly.")
    p.add_argument("--lipid", default=None, help="--dump: also draw this lipid.")
    p.add_argument("--dpi", type=int, default=140)
    a = p.parse_args(argv)

    if a.list_samples:
        print("\n".join(list_samples(a.maldi_file)))
        return
    if not a.field:
        raise SystemExit("--field is required (build one with `python -m other_experiments.parcelgp.build`)")

    field = ParcelField.load(a.field)
    log.info("field: %d parcels over %d nodes (%s, stride %s, spatial_weight %s)",
             field.n_parcels, field.labels.size, field.meta.get("features"),
             field.meta.get("stride"), field.meta.get("spatial_weight"))

    trained = None
    kernel0 = dict(nu=float(a.nu), lengthscale=np.asarray(a.lengthscale, float),
                   outputscale=float(a.outputscale), B=None,
                   strength=float(a.parcel_strength))
    if a.run:
        trained = load_trained_parcel(a.run, a.run_lipid)
        if trained.parcel_field and Path(trained.parcel_field).resolve() != Path(a.field).resolve():
            log.warning("the run was trained against %s but --field is %s; a B "
                        "from a different build indexes different parcels",
                        trained.parcel_field, a.field)
        kernel0.update(nu=trained.nu, lengthscale=trained.lengthscale,
                       outputscale=trained.outputscale, B=trained.B)
        log.info("trained %r: nu=%g outputscale=%.4g lengthscale=%s", trained.lipid,
                 trained.nu, trained.outputscale,
                 np.array2string(trained.lengthscale, precision=3))
        if trained.B is None:
            # The BASELINE arm of run_parcel_per_lipid.sh has no parcel factor to load. Its
            # learned lengthscale/outputscale are still worth having — that IS the
            # deployed Euclidean kernel — so keep them and let the right panel
            # fall back to the untrained identity B.
            log.info("this run has no parcel factor (it is the BASELINE arm): the "
                     "left panel is its real kernel, the right panel uses the "
                     "untrained B = %g * I", kernel0["strength"])

    lipids = [l for l in available_lipids(a.maldi_file, a.available_lipids_file)]
    if trained is not None and trained.lipid in lipids:      # default the dropdown
        lipids = [trained.lipid] + [l for l in lipids if l != trained.lipid]

    if a.dump:
        secs = MaldiSections(field, a.maldi_file, a.sample, a.max_snap_mm)
        sec = None
        if a.section is not None:
            # Section ids come out of the parquet as floats; accept "12" too.
            match = [s for s in secs.sections if str(s) == a.section
                     or np.isclose(float(s), float(a.section))]
            if not match:
                raise SystemExit(f"section {a.section!r} is not in {a.sample}; "
                                 f"available: {secs.sections}")
            sec = match[0]
        dump(field, secs, a.dump, section=sec, test_node=a.test_node,
             lengthscale=kernel0["lengthscale"], outputscale=kernel0["outputscale"],
             nu=kernel0["nu"], B=kernel0["B"], strength=kernel0["strength"],
             lipid=a.lipid or (trained.lipid if trained else None),
             dpi=a.dpi, trained=trained)
        return

    launch(field, a.maldi_file, a.sample, lipids, kernel0, trained=trained,
           max_snap_mm=a.max_snap_mm)


if __name__ == "__main__":
    main()
