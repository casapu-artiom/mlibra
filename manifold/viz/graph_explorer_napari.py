#!/usr/bin/env python
# encoding: utf-8
"""Interactive napari explorer for the FAISS manifold graph over the brain.

Replaces the notebook widget: the full-resolution reference template is the image
layer, the graph nodes are a Points layer, and native napari pan/zoom/slice is the
"movable ROI". Click anywhere on the brain to drop an ROI there — the app draws the
graph nodes and edges inside a 3-D box around the click, with **edge width and
colour proportional to the Gaussian edge weight** w = exp(-d^2 / 2 sigma^2).

Three graph flavours (dock panel):
  * pure FAISS          -- plain Euclidean k-NN.
  * inflate xN          -- soft anatomical prior: cross-annotation edges inflated.
  * cutoff <mm>         -- hard-drop edges longer than a cutoff, remove orphans.

The dock panel also prints debug info on Rebuild: connected components, isolated
points, and the MALDI-voxel -> nearest-node distance distribution.

Background image is FULL RESOLUTION; the graph stays on the strided nodes. A
strided node at grid index i maps to full-res voxel i*STRIDE (since
sub_volume = REF[::STRIDE, ::STRIDE, ::STRIDE]).

Run:
    python manifold/viz/graph_explorer_napari.py
    python manifold/viz/graph_explorer_napari.py --self-test    # headless logic check
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
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# --- repo imports (script lives in manifold/viz/; root holds manifold_gp,
# maldi/ holds utils) -- see the path bootstrap at the top of this file -------
_HERE = Path(__file__).resolve().parent

from utils import coord_norm_from_reference, maldi_voxels_standardized  # noqa: E402
from manifold_gp.utils.nearest_neighbors import (  # noqa: E402
    NearestNeighbors, resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.anatomical_knn import (  # noqa: E402
    inflate_cross_region_edges, labels_for_nodes_from_template_clustering,
)


# ============================================================================
# Graph construction (mirrors the notebook helpers)
# ============================================================================
class GraphExplorer:
    def __init__(self, ref, annot, coord_mean, coord_std, maldi_std,
                 stride, knn_k, nlist, nprobe, device, cluster_seed=0):
        self.REF = ref
        self.ANNOT = annot
        self.coord_mean = coord_mean
        self.coord_std = coord_std
        self.coord_std_mm = float(coord_std)
        self.maldi_std = maldi_std
        self.stride = stride
        self.knn_k = knn_k
        self.nlist_spec = nlist
        self.nprobe_spec = nprobe
        self.device = device
        self.cluster_seed = cluster_seed
        self._topo_cache = {}
        self._cluster_cache = {}
        self.bundle = None

    # ---- node set + faiss topology (cached by threshold) -------------------
    def _topology(self, threshold):
        key = (threshold, self.stride, self.knn_k)
        if key in self._topo_cache:
            return self._topo_cache[key]
        sub = self.REF[::self.stride, ::self.stride, ::self.stride]
        sub_atlas = self.ANNOT[::self.stride, ::self.stride, ::self.stride]
        zc, yc, xc = np.where(sub > threshold)
        node_vox = np.stack([zc, yc, xc], axis=1).astype(np.int32)
        coords_mm = node_vox.astype(np.float32) * (self.stride * 0.025)
        coords_std = (torch.from_numpy(coords_mm) - self.coord_mean) / self.coord_std
        node_labels = sub_atlas[zc, yc, xc].astype(np.int64)
        x = coords_std.contiguous().to(self.device)
        nlist = resolve_nlist(self.nlist_spec, x.shape[0])
        nprobe = resolve_nprobe(self.nprobe_spec, nlist)
        knn = NearestNeighbors(x, nlist=nlist)
        ei, ev = knn.graph(self.knn_k, symmetric=True, nprobe=nprobe)
        topo = dict(
            sub_volume=sub, node_vox=node_vox, coords_std=coords_std,
            node_labels=node_labels,
            edge_index=ei.cpu().numpy().astype(np.int64),
            edge_sqdist=ev.cpu().numpy().astype(np.float64),
            N=int(x.shape[0]),
        )
        self._topo_cache[key] = topo
        return topo

    @staticmethod
    def _weights(sqdist, gb):
        return np.exp(-sqdist / (2.0 * gb * gb))

    def _cluster_labels(self, threshold, sub, cluster_k, spatial_weight):
        """Data-driven, lipid-free node labels from k-means over the template
        (cached). Same routine the `faiss_cluster_weighted` pipeline uses."""
        key = (threshold, self.stride, int(cluster_k), round(float(spatial_weight), 3),
               self.cluster_seed)
        if key not in self._cluster_cache:
            self._cluster_cache[key] = labels_for_nodes_from_template_clustering(
                sub, threshold, n_clusters=int(cluster_k),
                spatial_weight=float(spatial_weight),
                fit_subsample=40000, seed=self.cluster_seed)
        return self._cluster_cache[key]

    def build(self, threshold, graph_type, gb, inflation, cutoff_mm,
              cluster_k=64, cluster_spatial_weight=1.0):
        """Build the requested graph flavour into self.bundle."""
        t = self._topology(threshold)
        b = dict(t)
        b["gb"] = gb
        b["weights"] = self._weights(b["edge_sqdist"], gb)
        b["node_keep"] = np.arange(b["N"])
        b["name"] = f"faiss thr={threshold} k={self.knn_k}"

        if graph_type in ("inflate", "cluster"):
            if graph_type == "inflate":
                labels = b["node_labels"]              # atlas annotations
                treat_zero = True
                tag = f"inflate(atlas) x{inflation:g} thr={threshold}"
            else:
                labels = self._cluster_labels(threshold, t["sub_volume"],
                                              cluster_k, cluster_spatial_weight)
                treat_zero = False                     # cluster id 0 is a real region
                tag = (f"cluster k{int(cluster_k)} sw{cluster_spatial_weight:g} "
                       f"x{inflation:g} thr={threshold}")
                b["node_labels_cluster"] = labels
            ei = torch.from_numpy(b["edge_index"])
            ev = torch.from_numpy(b["edge_sqdist"].astype(np.float32))
            _, ev2, info = inflate_cross_region_edges(
                ei, ev, labels, inflation=inflation, treat_zero_as_cross=treat_zero)
            b["edge_sqdist"] = ev2.numpy().astype(np.float64)
            b["weights"] = self._weights(b["edge_sqdist"], gb)
            b["name"] = tag
            b["inflation_info"] = info
        elif graph_type == "cutoff":
            b = self._apply_cutoff(b, cutoff_mm)

        self.bundle = b
        return b

    def _apply_cutoff(self, bundle, cutoff_mm):
        cut2 = (cutoff_mm / self.coord_std_mm) ** 2
        ei, ev, w = bundle["edge_index"], bundle["edge_sqdist"], bundle["weights"]
        keep_e = ev <= cut2
        src, dst = ei[0, keep_e], ei[1, keep_e]
        N = bundle["N"]
        A = coo_matrix((np.ones(src.shape[0]), (src, dst)), shape=(N, N))
        ncomp, labels = connected_components(A, directed=False, connection="weak")
        _, counts = np.unique(labels, return_counts=True)
        giant = int(np.argmax(counts))
        keep_n = labels == giant
        remap = -np.ones(N, np.int64); remap[keep_n] = np.arange(int(keep_n.sum()))
        e_in = keep_n[src] & keep_n[dst]
        b = dict(bundle)
        b["node_vox"] = bundle["node_vox"][keep_n]
        b["coords_std"] = bundle["coords_std"][keep_n]
        b["node_labels"] = bundle["node_labels"][keep_n]
        b["edge_index"] = np.stack([remap[src[e_in]], remap[dst[e_in]]])
        b["edge_sqdist"] = ev[keep_e][e_in]
        b["weights"] = w[keep_e][e_in]
        b["N"] = int(keep_n.sum())
        b["node_keep"] = np.flatnonzero(keep_n)
        b["name"] = f"cutoff {cutoff_mm:g}mm thr(before)"
        b["cutoff_stats"] = dict(cutoff_mm=cutoff_mm, ncomp_before=int(ncomp),
                                 removed=int((~keep_n).sum()), N_after=int(keep_n.sum()),
                                 edges_kept_frac=float(keep_e.mean()))
        return b

    # ---- coordinate helpers ------------------------------------------------
    def node_fullres(self, bundle=None):
        """Node positions in full-res (z,y,x) voxel coords for napari."""
        b = bundle or self.bundle
        return b["node_vox"].astype(np.float32) * self.stride

    def maldi_fullres(self):
        """MALDI voxels in full-res (z,y,x) voxel coords."""
        mm = self.maldi_std * self.coord_std + self.coord_mean
        return (mm * 40.0).numpy()

    # ---- debug -------------------------------------------------------------
    def debug_text(self, bundle=None):
        b = bundle or self.bundle
        N, ei, w = b["N"], b["edge_index"], b["weights"]
        live = w > 0
        A = coo_matrix((np.ones(int(live.sum())), (ei[0, live], ei[1, live])), shape=(N, N))
        ncomp, labels = connected_components(A, directed=False, connection="weak")
        _, counts = np.unique(labels, return_counts=True)
        n_single = int((counts == 1).sum())
        # MALDI -> nearest node
        x = b["coords_std"].to(self.device).contiguous()
        nlist = resolve_nlist(self.nlist_spec, x.shape[0])
        nprobe = resolve_nprobe(self.nprobe_spec, nlist)
        knn = NearestNeighbors(x, nlist=nlist)
        q = self.maldi_std.to(self.device).contiguous()
        ev, _ = knn.search(q, 1, nprobe=nprobe)
        d_mm = np.sqrt(np.clip(ev[:, 0].cpu().numpy(), 0, None)) * self.coord_std_mm
        p = np.percentile(d_mm, [50, 95, 99, 99.9, 100])
        lines = [
            f"{b['name']}",
            f"N nodes: {N:,}   edges: {ei.shape[1]:,}",
            f"components: {int(ncomp)}   isolated pts: {n_single}",
            f"MALDI->node mm: med={p[0]:.3f} p95={p[1]:.3f} "
            f"p99={p[2]:.3f} p99.9={p[3]:.3f} max={p[4]:.3f}",
        ]
        if "inflation_info" in b:
            i = b["inflation_info"]
            lines.append(f"inflated: {i['n_cross']:,}/{i['n_total']:,} "
                         f"({100*i['frac_cross']:.1f}%) x{i['inflation']:g}")
        if "cutoff_stats" in b:
            c = b["cutoff_stats"]
            lines.append(f"cutoff {c['cutoff_mm']:g}mm: {100*c['edges_kept_frac']:.1f}% "
                         f"edges kept, {c['ncomp_before']} comps -> removed "
                         f"{c['removed']} -> N={c['N_after']:,}")
        return "\n".join(lines)

    # ---- ROI edge/line extraction around a click ---------------------------
    def roi_lines(self, center_vox_ss, half, slice_axis=None, slice_tol=0,
                  min_w=1e-3, max_edges=4000):
        """Edges inside a strided box centred on `center_vox_ss` (strided grid
        coords). If `slice_axis` is given (2-D napari view), the box is thin
        (±slice_tol) along that axis so every returned node/edge lies in the
        current slice and is visible in 2-D; otherwise it is a full 3-D box.
        Returns (node_pts_fullres, line_segments_fullres, weights); edges are
        randomly subsampled to `max_edges` so the Shapes layer stays responsive."""
        b = self.bundle
        nv = b["node_vox"]; ei = b["edge_index"]; w = b["weights"]; N = b["N"]
        c = np.asarray(center_vox_ss)
        if slice_axis is None:
            in_box = np.all(np.abs(nv - c) <= half, axis=1)
        else:
            halves = np.array([half, half, half]); halves[slice_axis] = slice_tol
            in_box = np.all(np.abs(nv - c) <= halves, axis=1)
        node_ids = np.flatnonzero(in_box)
        mask = np.zeros(N, bool); mask[node_ids] = True
        em = np.flatnonzero(mask[ei[0]] & mask[ei[1]] & (w > min_w))
        a, bb = ei[0, em], ei[1, em]
        key = np.minimum(a, bb).astype(np.int64) * N + np.maximum(a, bb)
        em = em[np.unique(key, return_index=True)[1]]
        if max_edges and em.size > max_edges:
            em = np.random.default_rng(0).choice(em, max_edges, replace=False)
        fr = nv.astype(np.float32) * self.stride
        segs = np.stack([fr[ei[0, em]], fr[ei[1, em]]], axis=1)   # (E,2,3)
        return fr[node_ids], segs, w[em]


# ============================================================================
# Rendering helpers
# ============================================================================
def colormap_rgba(vals, cmap="plasma", vmin=None, vmax=None):
    import matplotlib
    cmap_obj = matplotlib.colormaps[cmap]
    vals = np.asarray(vals, float)
    vmin = float(np.min(vals)) if vmin is None else vmin
    vmax = float(np.max(vals)) if vmax is None else vmax
    v = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return cmap_obj(v)


# ============================================================================
# Data loading
# ============================================================================
def load_data(args):
    logging.info("Loading reference + annotations ...")
    ref = np.load(args.reference_file)
    annot = np.load(args.annotations_file)
    coord_mean, coord_std = coord_norm_from_reference(ref, voxel_per_mm=40.0)
    logging.info(f"reference {ref.shape}  coord_std={float(coord_std):.4f} mm")
    logging.info("Loading MALDI voxels ...")
    maldi_full = maldi_voxels_standardized(args.maldi_file, None, coord_mean, coord_std)
    if maldi_full.shape[0] > args.maldi_max:
        sel = np.random.default_rng(0).choice(maldi_full.shape[0], args.maldi_max, replace=False)
        maldi_std = maldi_full[sel].contiguous()
    else:
        maldi_std = maldi_full.contiguous()
    logging.info(f"MALDI voxels: {maldi_full.shape[0]:,} -> using {maldi_std.shape[0]:,}")
    return ref, annot, coord_mean, coord_std, maldi_std


# ============================================================================
# Main
# ============================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-file", default="/home/casap/mlibra/mlibra_data/reference_image.npy")
    ap.add_argument("--annotations-file", default="/home/casap/mlibra/mlibra_data/level_15annot.npy")
    ap.add_argument("--maldi-file", default="/home/casap/mlibra/mlibra_data/maindata_minimal.parquet")
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--graphbandwidth", type=float, default=0.05, help="sigma in standardized units")
    ap.add_argument("--roi-half", type=int, default=4,
                    help="ROI box half-size in strided (node) units; window is "
                         "(2*half+1) nodes across, ~0.1mm per step at stride 4")
    ap.add_argument("--edge-width", type=float, default=6.0, help="line width = scale * weight")
    ap.add_argument("--node-size", type=float, default=1.5,
                    help="point diameter in full-res voxels (node spacing = stride; "
                         "keep below it so nodes stay distinct)")
    ap.add_argument("--max-roi-edges", type=int, default=4000,
                    help="cap on edges drawn per ROI (subsampled) to keep napari responsive")
    ap.add_argument("--cluster-k", type=int, default=64,
                    help="(cluster) number of template k-means regions")
    ap.add_argument("--cluster-spatial-weight", type=float, default=1.0,
                    help="(cluster) weight on z-scored coords; higher = more contiguous")
    ap.add_argument("--cluster-seed", type=int, default=0)
    ap.add_argument("--n-list", default="sqrt")
    ap.add_argument("--n-probe", default="sqrt")
    ap.add_argument("--maldi-max", type=int, default=150_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--self-test", action="store_true",
                    help="Build graph + one ROI headlessly, print debug, exit (no GUI).")
    return ap


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    ref, annot, coord_mean, coord_std, maldi_std = load_data(args)

    gx = GraphExplorer(ref, annot, coord_mean, coord_std, maldi_std,
                       stride=args.stride, knn_k=args.knn_k,
                       nlist=args.n_list, nprobe=args.n_probe, device=args.device,
                       cluster_seed=args.cluster_seed)

    # ---- self-test: exercise the whole pipeline headlessly -----------------
    if args.self_test:
        for gt in ("faiss", "inflate", "cluster", "cutoff"):
            b = gx.build(args.threshold, gt, args.graphbandwidth,
                         10.0, 0.15, cluster_k=args.cluster_k,
                         cluster_spatial_weight=args.cluster_spatial_weight)
            print("\n" + gx.debug_text(b))
            c = np.median(b["node_vox"], axis=0).astype(int)
            pts, segs, wts = gx.roi_lines(c, args.roi_half, max_edges=args.max_roi_edges)
            print(f"  ROI @ {tuple(c)} half={args.roi_half}: "
                  f"{len(pts)} nodes, {len(segs)} edges "
                  f"(w range {wts.min():.3g}..{wts.max():.3g})")
        print("\nself-test OK")
        return

    # ---- napari GUI --------------------------------------------------------
    import napari
    from magicgui import magicgui

    gx.build(args.threshold, "faiss", args.graphbandwidth, 10.0, 0.15,
             cluster_k=args.cluster_k, cluster_spatial_weight=args.cluster_spatial_weight)

    viewer = napari.Viewer(title="Manifold graph explorer")
    viewer.add_image(ref, name="reference", colormap="gray", blending="additive")
    nodes_layer = viewer.add_points(
        gx.node_fullres(), name="graph nodes", size=args.node_size,
        face_color="#1f77b4", opacity=0.35, shading="none")
    maldi_layer = viewer.add_points(
        gx.maldi_fullres(), name="MALDI voxels", size=max(args.node_size, 2.0),
        face_color="#2ca02c", opacity=0.6, visible=False, shading="none",
        out_of_slice_display=True)
    roi_nodes = viewer.add_points(
        np.empty((0, 3)), name="ROI nodes", size=args.node_size, face_color="#00e5ff",
        shading="none", out_of_slice_display=True)
    roi_edges = viewer.add_shapes(name="ROI edges", ndim=3, edge_color="#ff3d7f")

    def _slice_axis():
        """Non-displayed axis in 2-D view (so the ROI stays in the current slice);
        None in 3-D view (full 3-D box)."""
        nd = [a for a in range(3) if a not in tuple(viewer.dims.displayed)]
        return nd[0] if (viewer.dims.ndisplay == 2 and nd) else None

    def draw_roi(center_fullres, slice_axis):
        c_ss = np.round(np.asarray(center_fullres) / gx.stride).astype(int)
        pts, segs, wts = gx.roi_lines(
            c_ss, PANEL.roi_half.value, slice_axis=slice_axis,
            max_edges=args.max_roi_edges)
        roi_nodes.data = pts if len(pts) else np.empty((0, 3))
        if len(pts):
            roi_nodes.size = PANEL.node_size.value   # keep size after data swap
        roi_edges.data = []
        if len(segs):
            widths = np.clip(PANEL.edge_width.value * wts, 0.4, None)
            colors = colormap_rgba(wts, "plasma", 0.0, float(wts.max()))
            roi_edges.add(list(segs), shape_type="line",
                          edge_width=list(widths), edge_color=colors)
        viewer.status = (f"ROI @ (z,y,x)_strided={tuple(int(v) for v in c_ss)}: "
                         f"{len(pts)} nodes, {len(segs)} edges")

    def on_click(layer, event):
        pos = np.asarray(event.position)[-3:]   # (z,y,x) world coords
        draw_roi(pos, _slice_axis())

    # Attach to every layer so a click registers whichever layer is selected,
    # and make the (pan-zoom) nodes layer active so clicks aren't captured by
    # the Shapes layer's editing tools.
    for lyr in list(viewer.layers):
        lyr.mouse_drag_callbacks.append(on_click)
    viewer.layers.selection.active = nodes_layer

    @magicgui(
        call_button="Rebuild graph",
        graph_type={"choices": ["faiss", "inflate", "cluster", "cutoff"]},
        threshold={"widget_type": "SpinBox", "min": 1, "max": 60},
        bandwidth={"widget_type": "FloatSpinBox", "min": 0.005, "max": 0.5, "step": 0.005},
        inflation={"widget_type": "FloatSpinBox", "min": 1.0, "max": 1000.0, "step": 1.0},
        cutoff_mm={"widget_type": "FloatSpinBox", "min": 0.10, "max": 0.6, "step": 0.005},
        cluster_k={"widget_type": "SpinBox", "min": 2, "max": 512},
        cluster_spatial_weight={"widget_type": "FloatSpinBox", "min": 0.0, "max": 10.0, "step": 0.25},
        roi_half={"widget_type": "SpinBox", "min": 2, "max": 60},
        edge_width={"widget_type": "FloatSpinBox", "min": 0.5, "max": 40.0, "step": 0.5},
        node_size={"widget_type": "FloatSpinBox", "min": 0.3, "max": 10.0, "step": 0.3},
        show_maldi={"widget_type": "CheckBox"},
        info={"widget_type": "TextEdit", "label": "debug"},
    )
    def PANEL(graph_type="faiss", threshold=args.threshold, bandwidth=args.graphbandwidth,
              inflation=10.0, cutoff_mm=0.15, cluster_k=args.cluster_k,
              cluster_spatial_weight=args.cluster_spatial_weight, roi_half=args.roi_half,
              edge_width=args.edge_width, node_size=args.node_size,
              show_maldi=False, info=""):
        gx.build(threshold, graph_type, bandwidth, inflation, cutoff_mm,
                 cluster_k=cluster_k, cluster_spatial_weight=cluster_spatial_weight)
        nodes_layer.data = gx.node_fullres()
        nodes_layer.size = node_size
        roi_nodes.size = node_size
        maldi_layer.visible = show_maldi
        roi_nodes.data = np.empty((0, 3))
        roi_edges.data = []
        PANEL.info.value = gx.debug_text()
        viewer.status = "graph rebuilt — click on the brain to drop an ROI"

    @PANEL.node_size.changed.connect
    def _live_node_size(v):
        for lyr in (nodes_layer, roi_nodes, maldi_layer):
            if len(lyr.data):
                lyr.size = v

    @PANEL.show_maldi.changed.connect
    def _live_show_maldi(v):
        maldi_layer.visible = bool(v)

    PANEL.info.value = gx.debug_text()
    viewer.window.add_dock_widget(PANEL, area="right", name="graph controls")
    print("Navigate to a slice, then CLICK on the brain to drop an ROI in the "
          "current slice. 'ROI nodes' (cyan) and 'ROI edges' (pink/yellow, "
          "width & colour = edge weight) fill in. Switch to 3-D (the cube icon, "
          "bottom-left) for a full 3-D box ROI.")
    napari.run()


if __name__ == "__main__":
    main()
