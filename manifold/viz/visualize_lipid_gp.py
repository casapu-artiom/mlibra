#!/usr/bin/env python
"""Visualize per-lipid GP predictions from disk.

Reads the output directory of ``per_lipid_gp_experiment.py`` (the new
training pipeline) and renders the predictions in napari. NO training
happens here — everything is loaded from .npy files. This makes
visualisation cheap (just file IO + colour-mapping) and lets you swap
between runs (different hypers) by pointing ``--run-dir`` at a different
folder.

Layout expected under ``--run-dir``::

    config.json
    metrics.csv
    lipid_names.json
    graph_meta.npz
    col_means.pt, col_stds.pt
    predictions/<slug>/
        test_coords_mm.npy
        test_pred_z.npy / test_pred_raw.npy
        test_std_z.npy
        test_true_z.npy
        graph_pred_z.npy / graph_pred_raw.npy
        graph_std_z.npy

Layers added to napari:

  0. reference template volume
  1. (optional) anatomical annotations
  2. test points — true z-scored values
  3. test points — GP prediction (z-scored)
  4. test points — error (true − pred), diverging
  5. test points — posterior stdev (uncertainty)
  6. whole-brain reconstruction (z-scored)
  7. whole-brain posterior stdev

Comparing two runs side-by-side: pass ``--compare <other_run_dir>``; we
build the same layer set for both runs with names prefixed by "A " and
"B ".  The lipid slider applies to both simultaneously.
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
import json
import logging
import re
from pathlib import Path

import numpy as np
import matplotlib
import napari


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize precomputed per-lipid GP predictions.",
    )
    p.add_argument("--run-dir", required=True,
                   help="Output directory of per_lipid_gp_experiment.py "
                        "(the one containing predictions/, config.json, ...).")
    p.add_argument("--compare", default=None,
                   help="Optional second run directory to overlay for "
                        "side-by-side comparison (e.g. euclidean vs manifold "
                        "with the same lipid).")
    p.add_argument("--reference-file", required=True,
                   help="Path to the CCF reference volume (.npy) used as "
                        "the anatomical background.")
    p.add_argument("--annotations-file", default=None,
                   help="Optional anatomical-labels volume.")
    p.add_argument("--initial-lipid-idx", type=int, default=0)
    p.add_argument("--initial-lipid-name", default=None,
                   help="Override --initial-lipid-idx by name.")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Display gamma for colour mapping.")
    p.add_argument("--point-size", type=float, default=2.5,
                   help="napari Points size for test scatter.")
    p.add_argument("--brain-point-size", type=float, default=1.0,
                   help="napari Points size for the dense brain reconstruction.")
    p.add_argument("--axis-order", nargs=3, type=int, default=[0, 1, 2],
                   metavar=("A0", "A1", "A2"),
                   help=("Permutation of MaLDI mm columns (xccf, yccf, zccf) "
                         "→ napari volume axes (axis 0, axis 1, axis 2). "
                         "Default '0 1 2' (identity) is right for the Allen "
                         "CCF convention where xccf=AP, yccf=DV, zccf=LR "
                         "and the template volume's axes are (AP, DV, LR). "
                         "If predictions look rotated 90° relative to the "
                         "template, try other permutations: '1 0 2', "
                         "'0 2 1', '1 2 0', '2 0 1', '2 1 0'."))
    p.add_argument("--flip-axes", nargs="*", type=int, default=[],
                   metavar="AXIS",
                   help=("Optional list of OUTPUT (post-permutation) axes "
                         "to mirror. Useful when predictions are "
                         "left-right reflected after permuting. E.g. "
                         "'--flip-axes 2' flips the last axis only."))
    p.add_argument("--template-voxel-scale", type=float, default=0.025,
                   help=("Physical voxel size (mm) of the DISPLAYED "
                         "reference volume. The default 0.025 matches "
                         "the Allen 25 µm mouse atlas. Used to convert "
                         "mm coordinates into display-volume voxel "
                         "indices. Distinct from any training-time "
                         "stride applied during graph construction."))
    p.add_argument("--brain-render-stride", type=int, default=1,
                   help=("Display every Nth brain-graph node. The full "
                         "brain reconstruction can be tens of thousands "
                         "to millions of points — napari Points layers "
                         "slow down badly above ~100K because every "
                         "lipid switch re-uploads colours to the GPU. "
                         "Use 2 (4×), 4 (16×), or 8 (64×) speedup. "
                         "Independent of the training-time graph stride."))
    p.add_argument("--color-range-mode", choices=["per_layer", "shared"],
                   default="per_layer",
                   help=("'per_layer' (default): each layer's colour "
                         "range is its own p1-p99 percentile, so each "
                         "layer shows its internal structure even if "
                         "one role has much narrower variance than "
                         "another (e.g. test scatter at z∈[-3,+3] vs "
                         "brain reconstruction at z∈[-0.2,+0.2]). "
                         "'shared': test_true / test_pred / brain_pred "
                         "share one scale — quantitative comparison "
                         "across layers, but if one role's range is "
                         "narrow it will look uniform."))
    # --- Kernel reconstruction & highlight sampling -----------------------
    p.add_argument("--eigenvector-dir", default=None,
                   help=("Path to the eigenvector cache used by the "
                         "training run (the dir containing eigvecs/ and "
                         "knn/ subdirs). When provided, the viewer "
                         "reconstructs the manifold-Matern kernel from "
                         "the run config and adds a highlight subset of "
                         "test points plus a layer rendering K(active "
                         "source, all test points). Without this, the "
                         "highlight subset still appears but the kernel "
                         "layer is skipped."))
    p.add_argument("--kernel-sample-size", type=int, default=300,
                   help=("Number of test points to include in the "
                         "highlight subset. Picked once at startup with "
                         "--kernel-seed; you cycle through them via the "
                         "prev/next source buttons."))
    p.add_argument("--kernel-seed", type=int, default=42,
                   help="RNG seed for the highlight subset sampling.")
    p.add_argument("--no-kernel", action="store_true",
                   help=("Skip kernel reconstruction even when "
                         "--eigenvector-dir is provided. Highlight subset "
                         "still appears."))
    return vars(p.parse_args())


# =============================================================================
# Helpers
# =============================================================================
def safe_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace(":", "-"))
    return s.strip("_")


def load_run(run_dir: Path) -> dict:
    """Read everything statically known about a run from disk.

    Filters lipid_names to only those that actually have a predictions
    directory on disk — i.e. the ones that were trained for THIS run.
    Without this, runs that used --lipids-file / --limit to subset would
    expose dozens of "missing" lipid indices in the slider.
    """
    log = logging.getLogger("viz")
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "lipid_names.json") as f:
        lipid_names_all = json.load(f)

    pred_root = run_dir / "predictions"
    trained_slugs = set()
    incomplete_slugs = []
    if pred_root.exists():
        # A lipid counts as "trained" only if the ESSENTIAL files exist
        # (coords + test pred). graph_pred / std are bonus — their
        # absence just disables the corresponding layers.
        # If a directory exists but lacks the essentials, the run likely
        # crashed partway through — log it and skip.
        REQUIRED = {"test_coords_mm.npy", "test_pred_z.npy",
                    "test_true_z.npy"}
        for d in pred_root.iterdir():
            if not d.is_dir():
                continue
            present = {f.name for f in d.glob("*.npy")}
            if REQUIRED.issubset(present):
                trained_slugs.add(d.name)
            elif present:
                incomplete_slugs.append(d.name)

    if incomplete_slugs:
        log.warning(
            f"  {len(incomplete_slugs)} prediction dir(s) are incomplete "
            f"and will be skipped: {incomplete_slugs[:5]}"
            + ("..." if len(incomplete_slugs) > 5 else "")
        )

    if not trained_slugs:
        # Build a helpful message: list sibling run dirs that DO have
        # something to visualise. Most of the time you cancelled this
        # run and forgot to switch RUN_DIR.
        parent = run_dir.parent
        sibling_summary = []
        if parent.exists():
            for sib in sorted(parent.iterdir()):
                if not sib.is_dir() or sib == run_dir:
                    continue
                pr = sib / "predictions"
                if not pr.exists():
                    continue
                n_complete = sum(
                    1 for d in pr.iterdir()
                    if d.is_dir() and {"test_coords_mm.npy",
                                       "test_pred_z.npy",
                                       "test_true_z.npy"}
                       .issubset({f.name for f in d.glob("*.npy")})
                )
                if n_complete:
                    sibling_summary.append((n_complete, sib.name))

        msg = (
            f"No COMPLETE lipid prediction directories under "
            f"{pred_root}.\n"
            f"This run was probably cancelled before any lipid finished.\n"
        )
        if sibling_summary:
            msg += "\nOther runs under the same parent that ARE usable:\n"
            for n, name in sibling_summary:
                msg += f"  ({n:3d} lipids)  {name}\n"
            msg += (f"\nPoint --run-dir (or $RUN_DIR) at one of those, "
                    f"or delete this empty one:\n"
                    f"  rm -r {run_dir}\n")
        else:
            msg += (f"\nNo other usable runs found under {parent}. "
                    f"Train at least one lipid before visualising.\n")
        raise RuntimeError(msg)

    # Walk lipid_names_all in ORIGINAL ORDER (so metrics.csv indices and
    # the slider stay aligned), keeping only those whose slug has a
    # predictions/<slug>/ directory on disk.
    lipid_names = []
    slugs = []
    original_indices = []   # idx into the full lipid_names_all (for metrics.csv lookups)
    for orig_i, name in enumerate(lipid_names_all):
        slug = safe_filename(name)
        if slug in trained_slugs:
            lipid_names.append(name)
            slugs.append(slug)
            original_indices.append(orig_i)
    log.info(f"  run dir       = {run_dir.name}")
    log.info(f"  kernel        = {config.get('kernel_family', 'unknown')}")
    log.info(f"  lipids on disk= {len(lipid_names)} / {len(lipid_names_all)} "
             f"(rest were not part of this run's subset)")

    graph_meta = None
    gm_path = run_dir / "graph_meta.npz"
    if gm_path.exists():
        graph_meta = np.load(gm_path)
        log.info(f"  loaded graph_meta from {gm_path.name}")

    return {
        "run_dir": run_dir,
        "config": config,
        "lipid_names": lipid_names,
        "slugs": slugs,
        "original_indices": original_indices,
        "graph_meta": graph_meta,
    }


def load_lipid_arrays(run: dict, lipid_idx: int):
    """Load all per-lipid arrays for one lipid; return None if missing."""
    slug = run["slugs"][lipid_idx]
    d = run["run_dir"] / "predictions" / slug
    if not d.exists():
        return None
    out = {}
    for name in ("test_coords_mm", "test_pred_z", "test_pred_raw",
                 "test_std_z", "test_true_z",
                 "graph_pred_z", "graph_pred_raw", "graph_std_z"):
        f = d / f"{name}.npy"
        out[name] = np.load(f) if f.exists() else None
    return out


# =============================================================================
# Colour helpers
# =============================================================================
def colors_sequential(values, cmap_name="magma", gamma=0.5,
                      vmin=None, vmax=None):
    """Map values → RGBA colours via a sequential colormap.

    If ``vmin`` / ``vmax`` are supplied, they pin the colour range — so
    multiple layers can share a scale and "the same value gets the
    same colour" across them. If left None, the range is taken from the
    data (the old per-layer-independent behaviour).
    """
    lo = float(values.min()) if vmin is None else float(vmin)
    hi = float(values.max()) if vmax is None else float(vmax)
    if hi > lo:
        norm = (values - lo) / (hi - lo)
        norm = np.clip(norm, 0, 1) ** float(gamma)
    else:
        norm = np.zeros_like(values, dtype=np.float32)
    return matplotlib.colormaps[cmap_name](norm)


def colors_diverging(values, cmap_name="RdBu_r", pct=99.0, gamma=0.5,
                     amax=None):
    """Map values → RGBA via a diverging colormap centred on 0.

    If ``amax`` is supplied, it pins the symmetric range to ±amax;
    otherwise the |.|^pct percentile of the data is used (the old
    per-layer behaviour).
    """
    if amax is None:
        amax = max(float(np.percentile(np.abs(values), pct)), 1e-12)
    else:
        amax = max(float(amax), 1e-12)
    sign = np.sign(values)
    rel = np.clip(np.abs(values) / amax, 0, 1) ** float(gamma)
    norm = 0.5 + 0.5 * sign * rel
    return matplotlib.colormaps[cmap_name](np.clip(norm, 0, 1))


def compute_color_ranges(arr: dict, mode: str = "per_layer",
                         pct: float = 99.0):
    """Compute (vmin, vmax) per layer ROLE — depending on ``mode``:

      "per_layer"  → each role gets its OWN data range (with percentile
                     clipping for robustness). Best default: lets each
                     layer show its internal structure even when one
                     has much higher variance than another (e.g. test
                     scatter at z∈[-3, +3] vs brain reconstruction at
                     z∈[-0.2, +0.2] when the GP under-fits the brain).
      "shared"     → all "pred" roles (true / pred / brain_pred) share
                     one (vmin, vmax). Same colour ↔ same z-value across
                     them, but if one role's range is much narrower it
                     will look uniform. Good for quantitative comparison,
                     bad for spotting subtle structure.

    Returns a dict keyed by ROLE (test_true, test_pred, brain_pred, ...)
    so the caller can look up the range directly.
    """
    def _range(a, lo_pct=100 - pct, hi_pct=pct, floor=None):
        if a is None or a.size == 0:
            return (0.0, 1.0)
        lo, hi = (float(np.percentile(a, lo_pct)),
                  float(np.percentile(a, hi_pct)))
        if floor is not None:
            lo = max(lo, floor)
        if hi <= lo:
            hi = lo + 1e-12
        return (lo, hi)

    if mode == "shared":
        pred_arrays = [a for a in (arr.get("test_true_z"),
                                   arr.get("test_pred_z"),
                                   arr.get("graph_pred_z")) if a is not None]
        if pred_arrays:
            cat = np.concatenate([a.ravel() for a in pred_arrays])
            pr = _range(cat)
        else:
            pr = (0.0, 1.0)
        std_arrays = [a for a in (arr.get("test_std_z"),
                                  arr.get("graph_std_z")) if a is not None]
        if std_arrays:
            cat = np.concatenate([a.ravel() for a in std_arrays])
            sr = _range(cat, floor=0.0)
        else:
            sr = (0.0, 1.0)
        ranges = {
            "test_true": pr, "test_pred": pr, "brain_pred": pr,
            "test_std": sr, "brain_std": sr,
        }
    else:  # per_layer
        ranges = {
            "test_true": _range(arr.get("test_true_z")),
            "test_pred": _range(arr.get("test_pred_z")),
            "brain_pred": _range(arr.get("graph_pred_z")),
            "test_std": _range(arr.get("test_std_z"), floor=0.0),
            "brain_std": _range(arr.get("graph_std_z"), floor=0.0),
        }

    # Diverging err range: always symmetric, always per-layer.
    if (arr.get("test_true_z") is not None
            and arr.get("test_pred_z") is not None):
        err = arr["test_true_z"] - arr["test_pred_z"]
        err_amax = max(float(np.percentile(np.abs(err), pct)), 1e-12)
    else:
        err_amax = 1.0
    ranges["err_amax"] = err_amax
    return ranges


# =============================================================================
# Coordinate conversion helpers
# =============================================================================
def mm_to_voxel_idx(coords_mm: np.ndarray,
                    voxel_scale: float = 0.025,
                    axis_order: tuple = (0, 1, 2),
                    flip_axes: tuple = ()) -> np.ndarray:
    """Convert mm coordinates → integer voxel indices, applying a
    permutation (axis_order) and optional per-axis flip.

    For the Allen CCF mouse atlas with xccf=AP, yccf=DV, zccf=LR and a
    template volume of shape (AP, DV, LR), the default identity
    permutation is right. Override with a different ``axis_order`` only
    if your dataset uses a different convention.

    axis_order = (0, 1, 2)  → identity (default; xccf→axis0, etc.)
    axis_order = (2, 1, 0)  → reverse all three axes
    axis_order = (1, 0, 2)  → swap first two only
    """
    idx = (coords_mm / voxel_scale).round().astype(np.int32)
    permuted = idx[:, list(axis_order)]
    if flip_axes:
        # Per-axis flip after permutation. Uses the data's own max as
        # the reference point — close enough for visualisation but not
        # a true atlas-aware flip.
        flipped = permuted.copy()
        for ax in flip_axes:
            flipped[:, ax] = permuted[:, ax].max() - permuted[:, ax]
        return flipped
    return permuted


def graph_pred_to_voxel_positions(graph_meta, axis_order=(0, 1, 2),
                                  flip_axes=(),
                                  display_voxel_scale: float = 0.025
                                  ) -> np.ndarray:
    """Return (N_nodes, 3) array of voxel-index positions for the
    graph nodes — in the DISPLAYED template's index space.

    Two layouts supported:

      - manifold runs: graph_meta has reference_nodes_z + coord_mean/std
        → de-standardize to mm, then convert to display-volume voxel
        indices via ``display_voxel_scale`` (default 0.025 mm/voxel for
        the Allen 25 µm atlas).
        NOTE: graph_meta also stores ``voxel_scale_mm``, but that is the
        STRIDED scale (0.025 × stride) at which the manifold graph was
        built. We must NOT use it here, because the displayed template
        is the full-resolution atlas. Using the strided scale would
        shrink the brain points by ``stride`` and pile them near the
        origin.

      - euclidean runs: graph_meta has node_voxel_idx (full-resolution
        voxel indices) directly, no conversion needed.
    """
    if "node_voxel_idx" in graph_meta:
        return graph_meta["node_voxel_idx"].astype(np.float32)
    z = graph_meta["reference_nodes_z"]
    cm_ = graph_meta["coord_mean"]
    cs_ = graph_meta["coord_std"]
    mm = z * cs_ + cm_
    return mm_to_voxel_idx(mm, voxel_scale=display_voxel_scale,
                           axis_order=axis_order,
                           flip_axes=flip_axes).astype(np.float32)


# =============================================================================
# Layer-set builders
# =============================================================================
def reconstruct_matern_kernel(run: dict, args: dict):
    """Rebuild the deployed manifold-Matern kernel from a run's config + the
    cached eigenvectors. Returns a dict with ``matern_kernel``, the coord
    normalisation (``coord_mean``, ``coord_std``), and a callback
    ``test_to_normalized(test_mm)`` that maps a (N, 3) array of test-point
    mm coordinates into the kernel's normalized space.

    The kernel is fully deterministic given:
      - the run's config.json (knn_method, knn_k, threshold, graphbandwidth,
        num_modes, nu, lengthscale, bump_scale, bump_decay, normalization,
        cross_region_inflation, stride, template_name)
      - the reference_file / annotations_file at the same path the training
        used
      - the eigenvector cache directory at the same path the training used

    No training data is loaded — the kernel only needs the graph nodes
    (above-threshold voxels) and the cached eigenpairs.

    Returns None when reconstruction can't proceed (missing config field,
    cache miss, etc.) — the caller falls back to "no kernel layer."
    """
    log = logging.getLogger("viz")
    config = run["config"]

    try:
        import torch
        from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
        from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
        from manifold_gp.utils.anatomical_knn import (
            inflate_cross_region_edges, labels_for_nodes_from_sub_atlas,
        )
        from manifold_gp.utils.compute_eigenvectors import (
            LaplacianEigensolver, make_key as make_eig_key,
        )
        from manifold_gp.utils.nearest_neighbors import (
            KnnGraphCache, make_key as make_graph_key,
        )
        from utils import crop_or_stride_volume, reference_ccf_from_subvolume
    except ImportError as e:
        log.warning(f"Kernel reconstruction unavailable — import failed: {e}")
        return None

    # Required hyperparams — fail soft if anything's missing.
    REQUIRED = ["knn_method", "knn_k", "graphbandwidth", "threshold",
                "stride", "num_modes", "nu", "lengthscale",
                "bump_scale", "bump_decay", "laplacian_norm"]
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        log.warning(f"Run config is missing {missing}; can't reconstruct kernel.")
        return None

    eigvec_dir = Path(args["eigenvector_dir"]) / "eigvecs"
    knn_dir    = Path(args["eigenvector_dir"]) / "knn"
    if not eigvec_dir.exists():
        log.warning(f"--eigenvector-dir/eigvecs not found at {eigvec_dir}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Reconstructing matern_kernel on {device} from run config + cached eigvecs …")

    template_full = np.load(args["reference_file"])
    annotations_full = (np.load(args["annotations_file"])
                        if args.get("annotations_file") else None)

    stride = int(config["stride"])
    threshold = int(config["threshold"])
    knn_method = config["knn_method"]
    knn_k = int(config["knn_k"])
    bw = float(config["graphbandwidth"])
    norm = config["laplacian_norm"]
    num_modes = int(config["num_modes"])
    nu = int(config["nu"])
    lengthscale = float(config["lengthscale"])
    bump_scale = float(config["bump_scale"])
    bump_decay = float(config["bump_decay"])
    n_list = int(config.get("n_list", 1))
    template_name = config.get("template_name", "reference")
    cross_region_inflation = float(config.get("cross_region_inflation", 10.0))

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_full, annotations_full, stride=stride, region_bbox=None,
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    reference_nodes_mm = torch.tensor(reference_ccf, dtype=torch.float32)
    coord_mean = reference_nodes_mm.mean(dim=0)
    coord_std = reference_nodes_mm.std(dim=0).clamp(min=1e-6)
    reference_nodes = ((reference_nodes_mm - coord_mean) / coord_std).to(device)

    # Same graph key composition the training run + sweep used.
    graph_key_parts = {
        "template": template_name,
        "stride":   stride,
        "thresh":   threshold,
        "method":   knn_method,
        "k":        knn_k,
        "nlist":    n_list,
        "bbox":     None,
    }
    if knn_method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"]  = 3
    graph_key = make_graph_key(graph_key_parts)
    graphs_cache = KnnGraphCache(cache_dir=knn_dir, verbose=False)

    try:
        if knn_method == "faiss":
            knn, edge_index, edge_value = graphs_cache.train_or_load(
                key=graph_key, method="faiss", coords=reference_nodes,
                k=knn_k, nlist=n_list, extra=graph_key_parts,
                device=device, force_recompute=False,
            )
        elif knn_method == "anatomical_atlas":
            if sub_atlas is None:
                log.warning("anatomical_atlas needs --annotations-file; can't rebuild.")
                return None
            knn, edge_index, edge_value = graphs_cache.train_or_load(
                key=graph_key, method="anatomical_atlas", volume=sub_volume,
                threshold=threshold, atlas_volume=sub_atlas, connectivity=3,
                coords=reference_nodes, k=knn_k, nlist=n_list,
                extra=graph_key_parts, device=device, force_recompute=False,
            )
        elif knn_method == "faiss_atlas_weighted":
            if sub_atlas is None:
                log.warning("faiss_atlas_weighted needs --annotations-file; can't rebuild.")
                return None
            base_parts = dict(graph_key_parts); base_parts["method"] = "faiss"
            base_key = make_graph_key(base_parts)
            knn, edge_index, edge_value = graphs_cache.train_or_load(
                key=base_key, method="faiss", coords=reference_nodes,
                k=knn_k, nlist=n_list, extra=base_parts,
                device=device, force_recompute=False,
            )
            node_labels = labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, threshold)
            edge_index, edge_value, _info = inflate_cross_region_edges(
                edge_index, edge_value, node_labels,
                inflation=cross_region_inflation, treat_zero_as_cross=True,
            )
            graph_key_parts["weighting"] = f"atlas_x{cross_region_inflation:g}"
            graph_key = make_graph_key(graph_key_parts)
        else:
            log.warning(f"Unknown knn_method: {knn_method}")
            return None
    except Exception as e:
        log.warning(f"KNN cache load failed: {e}")
        return None

    bw_tensor = torch.tensor(bw, device=device)
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0], bw_tensor, norm,
    )
    eigvec_key_parts = {
        "graph": graph_key, "norm": norm, "bw": bw, "modes": num_modes,
    }
    ekey = make_eig_key(eigvec_key_parts)
    solver = LaplacianEigensolver(num_modes=num_modes, verbose=False)
    try:
        eigval, eigvec = solver.compute_or_load(
            laplacian_op, cache_dir=eigvec_dir, key=ekey,
            graphbandwidth=bw, laplacian_normalization=norm,
            extra=eigvec_key_parts, force_recompute=False, device=device,
        )
    except Exception as e:
        log.warning(f"Eigvec cache load failed for key={ekey}: {e}")
        return None

    matern_kernel = RiemannMaternKernel(
        nu=nu, lengthscale=lengthscale,
        knn=knn, edge_index=edge_index, edge_value=edge_value,
        eigval=eigval, eigvec=eigvec,
        nearest_neighbors=knn_k, laplacian_normalization=norm,
        num_modes=num_modes,
        bump_scale=bump_scale, bump_decay=bump_decay,
        graphbandwidth_init=bw,
    ).to(device)

    log.info(f"  kernel built: nu={nu} lengthscale={lengthscale} "
             f"bw={bw} num_modes={num_modes}")
    return {
        "matern_kernel": matern_kernel,
        "coord_mean":    coord_mean.cpu().numpy(),
        "coord_std":     coord_std.cpu().numpy(),
        "device":        device,
        "graphbandwidth": bw,
        "bump_scale":    bump_scale,
        "bump_decay":    bump_decay,
        "lengthscale":   lengthscale,
        "nu":            nu,
    }


def add_kernel_highlight_layers(
    viewer, run, layers, args, kernel_pkg,
    test_voxel: np.ndarray, test_coords_mm: np.ndarray, prefix: str,
):
    """Add the highlight + kernel layers for one run.

    Three new layers:
      - ``highlight_subset``: scatter of N randomly chosen test points,
        ring-outlined, slightly bigger than the regular markers so they
        pop out against the prediction layers.
      - ``active_source``: a single bigger marker at the currently active
        source within the subset.
      - ``kernel_at_source``: ALL test points coloured by K(source, point).

    Returns a dict ``{indices, active_idx, refresh_kernel}`` so the prev/next
    callbacks can change which source is active and re-evaluate the kernel.
    """
    import torch

    n_test = test_voxel.shape[0]
    n_sample = min(int(args["kernel_sample_size"]), n_test)
    rng = np.random.default_rng(int(args["kernel_seed"]))
    subset_idx = rng.choice(n_test, size=n_sample, replace=False)
    subset_idx = np.sort(subset_idx)  # stable navigation order

    highlight_pos = test_voxel[subset_idx].astype(np.float32)
    state = {"indices": subset_idx, "active_within_subset": 0,
             "kernel_pkg": kernel_pkg, "test_coords_mm": test_coords_mm}

    # Subset markers — ring outline so they don't drown the underlying
    # prediction colours. White edge color makes them visible on dark
    # napari background; size slightly larger than the regular markers.
    test_size = float(args["point_size"])

    # napari ≥ 0.5 renamed `edge_color`/`edge_width` on Points layers to
    # `border_color`/`border_width`. Feature-detect via the signature so
    # we work on both old and new napari without forcing a version pin.
    import inspect
    _add_pts_sig = inspect.signature(viewer.add_points)
    _has_border  = "border_color" in _add_pts_sig.parameters
    def _outline_kwargs(color, width):
        if _has_border:
            return {"border_color": color, "border_width": width}
        return {"edge_color": color, "edge_width": width}

    highlight_layer = viewer.add_points(
        highlight_pos,
        name=f"{prefix}highlight subset ({n_sample}pts)",
        size=test_size * 1.6,
        face_color=np.zeros((n_sample, 4), dtype=np.float32),
        **_outline_kwargs("#ffffff", 0.3),
        opacity=0.8,
        blending="translucent",
        visible=True,
    )

    active_pos = highlight_pos[0:1].copy()
    active_layer = viewer.add_points(
        active_pos,
        name=f"{prefix}active source",
        size=test_size * 4.0,
        face_color="#ffff00",
        **_outline_kwargs("#ff0000", 0.6),
        opacity=1.0,
        blending="translucent",
        visible=True,
    )

    # Three kernel layers: bare manifold, Euclidean Matern, and the deployed
    # composition. The Euclidean one is independent of any cache (analytic
    # Bessel formula with the same ν/ℓ the run trained with) so it works
    # whenever kernel_pkg has been built, and provides a baseline that shows
    # what the bump-Euclidean fallback term contributes.
    # All three default invisible — toggling any of them on triggers the
    # eval lazily, and each is independently toggleable.
    kernel_layer = viewer.add_points(
        test_voxel.astype(np.float32),
        name=f"{prefix}K_m manifold (source, ·)",
        size=test_size,
        face_color=colors_sequential(
            np.zeros(n_test, dtype=np.float32), "magma",
            float(args["gamma"]), vmin=0.0, vmax=1.0,
        ),
        opacity=0.9, blending="translucent", visible=False,
    )
    kernel_eu_layer = viewer.add_points(
        test_voxel.astype(np.float32),
        name=f"{prefix}K_eu Euclidean (source, ·)",
        size=test_size,
        face_color=colors_sequential(
            np.zeros(n_test, dtype=np.float32), "magma",
            float(args["gamma"]), vmin=0.0, vmax=1.0,
        ),
        opacity=0.9, blending="translucent", visible=False,
    )
    kernel_full_layer = viewer.add_points(
        test_voxel.astype(np.float32),
        name=f"{prefix}K_full deployed (source, ·)",
        size=test_size,
        face_color=colors_sequential(
            np.zeros(n_test, dtype=np.float32), "magma",
            float(args["gamma"]), vmin=0.0, vmax=1.0,
        ),
        opacity=0.9, blending="translucent", visible=False,
    )

    def _bump(d: "torch.Tensor", scale: float, decay: float) -> "torch.Tensor":
        """Local copy of the bump function. d is distance to nearest training
        node; returns b ∈ [0, 1] with the same parametrisation visualize_
        laplacian.py and RiemannGP.modulation use."""
        out = torch.zeros_like(d)
        inside = d < scale
        if inside.any():
            u = (d[inside] / scale).clamp(0.0, 1.0 - 1e-6)
            out[inside] = torch.exp(-decay / (1.0 - u * u)) / float(np.exp(-decay))
        return out

    def _matern_euclidean_at_src(src_norm: np.ndarray, tgt_norm: np.ndarray,
                                  nu: float, lengthscale: float) -> np.ndarray:
        """Euclidean Matern kernel from one source to many targets, evaluated
        in the same NORMALIZED coordinate frame the manifold kernel uses.
        Uses scipy's Bessel form so apples-to-apples with the run's ν / ℓ."""
        from scipy.special import kv, gamma as gamma_fn
        d = np.linalg.norm(tgt_norm - src_norm[None, :], axis=-1)
        nu_f = float(nu); ls = float(lengthscale)
        out = np.ones_like(d, dtype=np.float64)
        nz = d > 0
        if nz.any():
            z = np.sqrt(2.0 * nu_f) * d[nz] / ls
            # Floor z away from 0 so 0·inf cancellation in z^nu * kv(nu, z)
            # doesn't produce NaN at near-coincident points.
            z = np.maximum(z, 1e-12)
            out[nz] = ((2.0 ** (1.0 - nu_f)) / gamma_fn(nu_f)) * (z ** nu_f) * kv(nu_f, z)
        return out.astype(np.float32)

    def _update_kernel_layer(layer, vals: np.ndarray):
        vlo = float(np.percentile(vals, 1.0))
        vhi = float(np.percentile(vals, 99.0))
        # Diverging palette when the kernel takes both signs (truncation
        # artefacts on the manifold path); sequential otherwise.
        if vlo < -1e-6 and vhi > 1e-6:
            amax = max(abs(vlo), abs(vhi), 1e-6)
            layer.face_color = colors_diverging(
                vals, "RdBu_r", gamma=float(args["gamma"]), amax=amax,
            )
        else:
            layer.face_color = colors_sequential(
                vals, "magma", float(args["gamma"]),
                vmin=max(vlo, 0.0), vmax=max(vhi, 1e-6),
            )
        layer.refresh()

    def refresh_kernel():
        """Evaluate whichever of K_m, K_eu, K_full are visible and recolour."""
        if kernel_pkg is None:
            return
        want_m  = kernel_layer.visible
        want_eu = kernel_eu_layer.visible
        want_full = kernel_full_layer.visible
        if not (want_m or want_eu or want_full):
            return  # nothing to do

        idx = subset_idx[state["active_within_subset"]]
        src_mm = state["test_coords_mm"][idx]
        tgt_mm = state["test_coords_mm"]
        mk = kernel_pkg["matern_kernel"]
        cm  = kernel_pkg["coord_mean"]
        cs  = kernel_pkg["coord_std"]
        device = kernel_pkg["device"]
        nu = kernel_pkg["nu"]
        ls = kernel_pkg["lengthscale"]
        bw_g = kernel_pkg["graphbandwidth"]
        bs = kernel_pkg["bump_scale"]
        bd = kernel_pkg["bump_decay"]

        # mm → normalized coords (same transform the training used).
        src_norm_np = ((src_mm - cm) / cs).astype(np.float32)
        tgt_norm_np = ((tgt_mm - cm) / cs).astype(np.float32)
        # K_full needs both K_m and K_eu plus bump weights, so compute
        # any component another layer depends on even if its own layer is off.
        need_m  = want_m  or want_full
        need_eu = want_eu or want_full
        need_bump = want_full

        try:
            with torch.no_grad():
                K_m_vals = None
                b_src = None
                b_tgt = None
                if need_m:
                    src_n = torch.from_numpy(src_norm_np)[None, :].to(device)
                    tgt_n = torch.from_numpy(tgt_norm_np).to(device)
                    feat_src = mk.features(src_n).squeeze(0)
                    B = 20_000
                    K_m_vals = np.empty(tgt_n.shape[0], dtype=np.float32)
                    for s_ in range(0, tgt_n.shape[0], B):
                        feat_b = mk.features(tgt_n[s_:s_+B])
                        K_m_vals[s_:s_+B] = (feat_b * feat_src).sum(dim=-1).cpu().numpy()
                    if need_bump:
                        # bump radius = bump_scale × graphbandwidth in normalized space
                        rad = float(bs) * float(bw_g)
                        ev_src, _ = mk.knn.search(src_n, 1)
                        d_src_t = ev_src.sqrt().squeeze()
                        b_src_t = _bump(d_src_t.reshape(1), rad, float(bd))
                        b_src = float(b_src_t.item())
                        ev_tgt, _ = mk.knn.search(tgt_n, 1)
                        d_tgt_t = ev_tgt.sqrt().squeeze(-1)
                        b_tgt = _bump(d_tgt_t, rad, float(bd)).cpu().numpy().astype(np.float32)

                K_eu_vals = None
                if need_eu:
                    K_eu_vals = _matern_euclidean_at_src(
                        src_norm_np, tgt_norm_np, nu, ls,
                    )

                K_full_vals = None
                if want_full:
                    K_full_vals = (
                        b_src * K_m_vals * b_tgt
                        + (1.0 - b_src) * (1.0 - b_tgt) * K_eu_vals
                    ).astype(np.float32)
        except Exception as e:
            logging.getLogger("viz").warning(f"Kernel eval failed: {e}")
            return

        if want_m   and K_m_vals    is not None: _update_kernel_layer(kernel_layer,      K_m_vals)
        if want_eu  and K_eu_vals   is not None: _update_kernel_layer(kernel_eu_layer,   K_eu_vals)
        if want_full and K_full_vals is not None: _update_kernel_layer(kernel_full_layer, K_full_vals)

    def recolor_on_toggle(event):
        # Any of the three becoming visible triggers a refresh — refresh
        # itself only computes what's actually needed.
        if kernel_layer.visible or kernel_eu_layer.visible or kernel_full_layer.visible:
            refresh_kernel()
    kernel_layer.events.visible.connect(recolor_on_toggle)
    kernel_eu_layer.events.visible.connect(recolor_on_toggle)
    kernel_full_layer.events.visible.connect(recolor_on_toggle)

    def set_active(within_subset_idx: int):
        within_subset_idx = int(within_subset_idx) % n_sample
        state["active_within_subset"] = within_subset_idx
        active_layer.data = highlight_pos[within_subset_idx:within_subset_idx + 1].copy()
        refresh_kernel()

    layers["highlight_subset"]   = highlight_layer
    layers["active_source"]      = active_layer
    layers["kernel_at_source"]   = kernel_layer
    layers["kernel_eu_source"]   = kernel_eu_layer
    layers["kernel_full_source"] = kernel_full_layer
    layers["_highlight_state"]   = state
    layers["_set_active_src"]    = set_active
    layers["_refresh_kernel"]    = refresh_kernel

    logging.getLogger("viz").info(
        f"  highlight: {n_sample} test points sampled, source 0 active"
    )


def add_run_layers(viewer, run, prefix, args, start_lipid_idx=None,
                   kernel_pkg=None):
    """Create the layer set for one run.

    ``start_lipid_idx`` overrides the args-derived "initial lipid" — used
    when SWITCHING runs so the new layers come up coloured for whatever
    lipid the slider is currently on, not for the startup default.

    ``kernel_pkg`` is the dict returned by ``reconstruct_matern_kernel``
    when --eigenvector-dir was supplied. When non-None we also add the
    highlight subset + active source + kernel layers; when None we still
    add the highlight subset (visual only) but skip the kernel layer.
    """
    gamma = args["gamma"]
    test_size = args["point_size"]
    brain_size = args["brain_point_size"]

    if start_lipid_idx is not None:
        first_idx = max(0, min(int(start_lipid_idx),
                               len(run["lipid_names"]) - 1))
    else:
        first_idx = args["initial_lipid_idx"]
        if args.get("initial_lipid_name"):
            if args["initial_lipid_name"] in run["lipid_names"]:
                first_idx = run["lipid_names"].index(args["initial_lipid_name"])
    arr = load_lipid_arrays(run, first_idx)
    if arr is None:
        for i in range(len(run["lipid_names"])):
            arr = load_lipid_arrays(run, i)
            if arr is not None:
                first_idx = i
                break
        if arr is None:
            raise RuntimeError(
                f"Run dir {run['run_dir']} has no per-lipid predictions."
            )

    axis_order = tuple(args.get("axis_order", (0, 1, 2)))
    flip_axes = tuple(args.get("flip_axes", []) or [])
    voxel_scale = float(args.get("template_voxel_scale", 0.025))
    test_voxel = mm_to_voxel_idx(arr["test_coords_mm"],
                                 voxel_scale=voxel_scale,
                                 axis_order=axis_order,
                                 flip_axes=flip_axes)
    brain_voxel = (graph_pred_to_voxel_positions(
                       run["graph_meta"],
                       axis_order=axis_order, flip_axes=flip_axes,
                       display_voxel_scale=voxel_scale)
                   if run["graph_meta"] is not None else None)

    # Colour ranges per role. With mode="per_layer" (default) each
    # layer gets its OWN range, so brain_pred's internal structure
    # shows up even when its variance is much smaller than test_pred's.
    # mode="shared" makes "true vs pred vs brain_pred" use one scale
    # (good for quantitative comparison, bad for spotting small variation).
    cr_mode = args.get("color_range_mode", "per_layer")
    cr = compute_color_ranges(arr, mode=cr_mode)
    err_amax = cr["err_amax"]

    # Diagnostic — confirms whether brain prediction is genuinely flat
    # or just looking flat because of viz compression.
    if arr.get("graph_pred_z") is not None:
        bp = arr["graph_pred_z"]
        logging.getLogger("viz").info(
            f"  brain_pred  z range: [{bp.min():+.3f}, {bp.max():+.3f}]  "
            f"std={bp.std():.3f}  range={cr['brain_pred']}")
    if arr.get("test_pred_z") is not None:
        tp = arr["test_pred_z"]
        logging.getLogger("viz").info(
            f"  test_pred   z range: [{tp.min():+.3f}, {tp.max():+.3f}]  "
            f"std={tp.std():.3f}  range={cr['test_pred']}")

    layers = {}
    layers["test_true"] = viewer.add_points(
        test_voxel.astype(np.float32),
        name=f"{prefix}test obs (z) — true",
        size=test_size, opacity=0.9, blending="translucent", visible=False,
        face_color=colors_sequential(arr["test_true_z"], "magma", gamma,
                                     vmin=cr["test_true"][0],
                                     vmax=cr["test_true"][1]),
    )
    layers["test_pred"] = viewer.add_points(
        test_voxel.astype(np.float32),
        name=f"{prefix}test pred (z)",
        size=test_size, opacity=0.9, blending="translucent", visible=True,
        face_color=colors_sequential(arr["test_pred_z"], "magma", gamma,
                                     vmin=cr["test_pred"][0],
                                     vmax=cr["test_pred"][1]),
    )
    err = arr["test_true_z"] - arr["test_pred_z"]
    layers["test_err"] = viewer.add_points(
        test_voxel.astype(np.float32),
        name=f"{prefix}test err (true−pred)",
        size=test_size, opacity=0.9, blending="translucent", visible=False,
        face_color=colors_diverging(err, "RdBu_r", gamma=gamma,
                                    amax=err_amax),
    )
    layers["test_std"] = viewer.add_points(
        test_voxel.astype(np.float32),
        name=f"{prefix}test std (uncertainty)",
        size=test_size, opacity=0.9, blending="translucent", visible=False,
        face_color=colors_sequential(arr["test_std_z"], "viridis", gamma,
                                     vmin=cr["test_std"][0],
                                     vmax=cr["test_std"][1]),
    )
    if brain_voxel is not None and arr["graph_pred_z"] is not None:
        # OPTIMIZATION: subsample brain nodes for display. The full
        # reconstruction can be 10K–1M+ points; rendering every one in
        # napari is overkill. The stride is uniform — every Nth node —
        # which still gives a dense visual when stride is small (2-4)
        # but cuts the GPU upload cost by stride² for color updates.
        bstride = max(1, int(args.get("brain_render_stride", 1)))
        if bstride > 1:
            brain_idx = np.arange(0, brain_voxel.shape[0], bstride)
            brain_voxel_disp = brain_voxel[brain_idx]
        else:
            brain_idx = None
            brain_voxel_disp = brain_voxel
        # Stash the subsample indices in the layer dict so update_run_layers
        # knows to apply them when re-colouring after a lipid change.
        layers["_brain_subsample_idx"] = brain_idx

        def _sub(arr_):
            return arr_ if brain_idx is None else arr_[brain_idx]

        layers["brain_pred"] = viewer.add_points(
            brain_voxel_disp.astype(np.float32),
            name=f"{prefix}brain pred (z)",
            size=brain_size, opacity=0.85, blending="translucent", visible=False,
            face_color=colors_sequential(_sub(arr["graph_pred_z"]),
                                         "magma", gamma,
                                         vmin=cr["brain_pred"][0],
                                         vmax=cr["brain_pred"][1]),
        )
        if arr["graph_std_z"] is not None:
            layers["brain_std"] = viewer.add_points(
                brain_voxel_disp.astype(np.float32),
                name=f"{prefix}brain std (uncertainty)",
                size=brain_size, opacity=0.85, blending="translucent",
                visible=False,
                face_color=colors_sequential(_sub(arr["graph_std_z"]),
                                              "viridis", gamma,
                                              vmin=cr["brain_std"][0],
                                              vmax=cr["brain_std"][1]),
            )
    # Highlight subset + (optional) kernel layer. Always added so the
    # subset is visible even when --eigenvector-dir wasn't supplied; the
    # kernel layer is only meaningful when kernel_pkg is non-None.
    try:
        add_kernel_highlight_layers(
            viewer, run, layers, args, kernel_pkg=kernel_pkg,
            test_voxel=test_voxel, test_coords_mm=arr["test_coords_mm"],
            prefix=prefix,
        )
    except Exception as e:
        logging.getLogger("viz").warning(
            f"Failed to add highlight/kernel layers: {e}"
        )

    layers["current_idx"] = first_idx
    return layers


def update_run_layers(layers, run, lipid_idx, args):
    arr = load_lipid_arrays(run, lipid_idx)
    if arr is None:
        return None
    gamma = args["gamma"]
    # Per-layer (or shared) ranges, re-derived from this lipid's arrays.
    cr_mode = args.get("color_range_mode", "per_layer")
    cr = compute_color_ranges(arr, mode=cr_mode)
    err_amax = cr["err_amax"]

    # OPTIMIZATION: skip recolouring invisible layers. The brain layers
    # in particular hold tens of thousands to millions of points, and
    # ``face_color = ...`` triggers a full GPU upload. We track the
    # lipid_idx that each layer was last coloured for in the ``layers``
    # dict; when a layer becomes visible later, the visibility-toggle
    # hook below recolours it on demand using these cached indices.
    pending = layers.setdefault("_pending_recolor", {})
    # Brain layers may have been built with a subsample stride — pull
    # the index array out of ``layers`` and apply it before colouring.
    brain_idx = layers.get("_brain_subsample_idx")

    def _sub_brain(arr_):
        return arr_ if brain_idx is None else arr_[brain_idx]

    def _maybe_recolor(role, array, palette, gamma_local, **kw):
        """Apply a colour update only if (a) the layer exists, (b) the
        source array is present, and (c) the layer is currently visible.
        Otherwise mark the role as 'pending' so the next visibility
        toggle recolours it from disk on demand.
        """
        if role not in layers or array is None:
            return
        if not layers[role].visible:
            pending[role] = True
            return
        if palette == "diverging":
            layers[role].face_color = colors_diverging(
                array, "RdBu_r", gamma=gamma_local, **kw)
        else:
            layers[role].face_color = colors_sequential(
                array, palette, gamma_local, **kw)
        pending.pop(role, None)

    # test_true_z + test_pred_z are guaranteed by load_run's filter;
    # test_std_z and the brain arrays are optional (a partial save or
    # interrupted run may lack them). For each, if the array is missing,
    # we keep the previous lipid's colors on that layer — a small visual
    # carry-over, but no crash.
    _maybe_recolor("test_true", arr["test_true_z"], "magma", gamma,
                   vmin=cr["test_true"][0], vmax=cr["test_true"][1])
    _maybe_recolor("test_pred", arr["test_pred_z"], "magma", gamma,
                   vmin=cr["test_pred"][0], vmax=cr["test_pred"][1])
    if arr["test_true_z"] is not None and arr["test_pred_z"] is not None:
        err = arr["test_true_z"] - arr["test_pred_z"]
        _maybe_recolor("test_err", err, "diverging", gamma, amax=err_amax)
    _maybe_recolor("test_std", arr["test_std_z"], "viridis", gamma,
                   vmin=cr["test_std"][0], vmax=cr["test_std"][1])
    # Brain layers: subsample-index applied before colouring.
    bp = arr["graph_pred_z"]
    bs_ = arr["graph_std_z"]
    _maybe_recolor("brain_pred",
                   _sub_brain(bp) if bp is not None else None,
                   "magma", gamma,
                   vmin=cr["brain_pred"][0], vmax=cr["brain_pred"][1])
    _maybe_recolor("brain_std",
                   _sub_brain(bs_) if bs_ is not None else None,
                   "viridis", gamma,
                   vmin=cr["brain_std"][0], vmax=cr["brain_std"][1])
    layers["current_idx"] = lipid_idx
    return arr


# =============================================================================
# Discover usable runs in the parent directory
# =============================================================================
REQUIRED_LIPID_FILES = {"test_coords_mm.npy", "test_pred_z.npy",
                        "test_true_z.npy"}


def _is_run_usable(run_dir: Path) -> int:
    """Return the number of COMPLETE lipid subdirectories. 0 = the run
    is empty / cancelled — same predicate the cleanup script uses."""
    pred = run_dir / "predictions"
    if not pred.is_dir():
        return 0
    n = 0
    for d in pred.iterdir():
        if d.is_dir():
            present = {f.name for f in d.glob("*.npy")}
            if REQUIRED_LIPID_FILES.issubset(present):
                n += 1
    return n


def discover_sibling_runs(seed_run_dir: Path,
                          log: logging.Logger) -> list[Path]:
    """Return all sibling run directories (incl. seed) that have at
    least one complete lipid, sorted alphabetically by name. Cancelled
    runs are skipped silently."""
    parent = seed_run_dir.parent
    if not parent.is_dir():
        log.warning(f"  parent {parent} not a directory; using only seed.")
        return [seed_run_dir]
    runs = []
    for p in sorted(parent.iterdir()):
        if not p.is_dir() or not (p / "config.json").exists():
            continue
        if _is_run_usable(p):
            runs.append(p)
    # If the seed itself isn't in the list (e.g. it has no predictions
    # but the user passed it explicitly), don't add it — we'd just hit
    # load_run's error. Instead make sure SOMETHING is returned.
    if not runs:
        runs = [seed_run_dir]
    log.info(f"  discovered {len(runs)} usable run(s) under {parent}")
    return runs


# =============================================================================
# Main
# =============================================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("viz")
    args = parse_args()

    seed_dir = Path(args["run_dir"])
    log.info("Discovering sibling runs …")
    all_runs = discover_sibling_runs(seed_dir, log)
    # Start at the user-specified run (or 0 if it's not in the list).
    # Use strict=False resolve so a non-existent seed_dir doesn't crash —
    # it just means we won't find it in all_runs and fall back to idx 0.
    try:
        seed_resolved = seed_dir.resolve(strict=False)
        current_run_idx = next(
            i for i, p in enumerate(all_runs)
            if p.resolve(strict=False) == seed_resolved
        )
    except (StopIteration, OSError):
        current_run_idx = 0
        if not seed_dir.exists():
            log.warning(f"Seed RUN_DIR {seed_dir} does not exist; "
                        f"falling back to first usable run: "
                        f"{all_runs[0].name}")

    log.info(f"Loading run A (current = idx {current_run_idx} of "
             f"{len(all_runs)}) …")
    run_a = load_run(all_runs[current_run_idx])

    run_b = None
    if args.get("compare"):
        log.info("Loading run B (--compare, anchored — not affected by "
                 "the run-navigation buttons) …")
        run_b = load_run(Path(args["compare"]))
        if run_a["lipid_names"] != run_b["lipid_names"]:
            log.warning(
                "Run A and B have different lipid_names; the slider will "
                "use Run A's ordering. Behaviour for lipids missing in "
                "either run is undefined."
            )

    log.info("Opening napari …")
    viewer = napari.Viewer()

    def _title():
        return (f"per-lipid GP — A:{all_runs[current_run_idx].name}"
                + (f"  vs  B:{run_b['run_dir'].name}" if run_b else ""))
    viewer.title = _title()

    template_full = np.load(args["reference_file"])
    viewer.add_image(
        template_full, name="0 reference (atlas)",
        rendering="attenuated_mip", colormap="gray",
        opacity=0.55, blending="additive", visible=True,
    )
    if args.get("annotations_file"):
        try:
            ann = np.load(args["annotations_file"])
            max_lbl = int(ann.max())
            if max_lbl < 256:
                ann = ann.astype(np.uint8)
            elif max_lbl < 65_536:
                ann = ann.astype(np.uint16)
            ann = np.ascontiguousarray(ann)
            viewer.add_labels(
                ann, name="1 annotations (atlas)",
                opacity=0.3, blending="translucent", visible=False,
            )
        except Exception as e:
            log.warning(f"Could not load annotations: {e}")

    # Reconstruct the manifold-Matern kernel from each run's config when
    # --eigenvector-dir is provided. Failure here is non-fatal: the viewer
    # still works, just without the kernel-overlay layer.
    kernel_pkg_a = None
    kernel_pkg_b = None
    if args.get("eigenvector_dir") and not args.get("no_kernel"):
        kernel_pkg_a = reconstruct_matern_kernel(run_a, args)
        if run_b is not None:
            kernel_pkg_b = reconstruct_matern_kernel(run_b, args)
    elif args.get("no_kernel"):
        log.info("--no-kernel set: skipping kernel reconstruction.")
    else:
        log.info("--eigenvector-dir not set: highlight subset added, but kernel layer disabled.")

    # Mutable state shared across widgets. Wrapping the per-run objects
    # in lists so the closures below can re-assign them when the user
    # navigates between runs without ``nonlocal`` everywhere.
    state = {
        "run_a": run_a,
        "layers_a": add_run_layers(viewer, run_a, prefix="A: ", args=args,
                                   kernel_pkg=kernel_pkg_a),
        "metrics_a": None,
        "run_idx": current_run_idx,
        "kernel_pkg_a": kernel_pkg_a,
        "kernel_pkg_b": kernel_pkg_b,
    }

    # Run B (the optional comparison) is NOT touched by the navigation —
    # the navigation cycles only run A. This keeps a stable baseline for
    # whatever B is set to.
    layers_b = (add_run_layers(viewer, run_b, prefix="B: ", args=args,
                               kernel_pkg=kernel_pkg_b)
                if run_b else None)

    import pandas as pd

    def _load_metrics(run):
        try:
            return pd.read_csv(run["run_dir"] / "metrics.csv")
        except Exception as e:
            log.warning(f"Could not read metrics.csv for "
                        f"{run['run_dir'].name}: {e}")
            return None

    state["metrics_a"] = _load_metrics(run_a)
    metrics_b = _load_metrics(run_b) if run_b else None

    # ---- Info panel: hypers + per-lipid metrics --------------------------
    # We use a plain Qt QLabel inside a QGroupBox so we get a multi-line
    # block of text with monospaced styling, kept in sync with whichever
    # (run, lipid) is currently selected. The label is updated by
    # _update_info_panel(), which gets called whenever the lipid slider
    # OR the run selector changes.
    from qtpy.QtWidgets import QLabel, QWidget, QVBoxLayout
    from qtpy.QtCore import Qt

    info_label = QLabel()
    info_label.setTextFormat(Qt.RichText)
    info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    info_label.setStyleSheet(
        "QLabel { font-family: monospace; font-size: 11px; padding: 6px; }"
    )
    info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    info_label.setWordWrap(True)
    info_container = QWidget()
    _vl = QVBoxLayout(info_container)
    _vl.setContentsMargins(0, 0, 0, 0)
    _vl.addWidget(info_label)

    # The keys we surface from config.json — chosen to be small enough
    # for a side panel and informative enough to identify a run. Add
    # more here if you want. Missing keys are silently skipped, so a
    # newer/older config schema won't break anything.
    INFO_KEYS = [
        "kernel_family", "kernel", "nu", "num_inducing",
        "lipid_batch_size", "epochs", "batch_size", "learning_rate",
        # Manifold-only:
        "num_modes", "stride", "knn_k", "knn_method",
        "laplacian_norm", "bump_scale", "bump_decay",
        "graphbandwidth_init",
    ]

    def _format_hypers(cfg: dict) -> str:
        lines = []
        for k in INFO_KEYS:
            if k in cfg and cfg[k] is not None:
                v = cfg[k]
                # Pretty-print floats (avoids "1.0000000004" etc.)
                if isinstance(v, float):
                    s = f"{v:.4g}"
                else:
                    s = str(v)
                lines.append(f"  {k:>20s} = {s}")
        return "\n".join(lines) if lines else "  (no hypers in config.json)"

    def _format_metrics_row(row) -> str:
        """One-line summary of a metrics.csv row."""
        if row is None:
            return "  (no metrics row for this lipid)"
        bits = []
        if "test_corr" in row and row["test_corr"] == row["test_corr"]:  # NaN check
            bits.append(f"corr = {row['test_corr']:+.4f}")
        if "test_rmse_z" in row:
            bits.append(f"RMSE(z) = {row['test_rmse_z']:.4f}")
        if "test_r2" in row:
            bits.append(f"R² = {row['test_r2']:+.4f}")
        if "mean_pred_std_z" in row:
            bits.append(f"mean σ(z) = {row['mean_pred_std_z']:.4f}")
        return "  " + "  |  ".join(bits) if bits else "  (no metric columns)"

    def _format_aggregate(mdf) -> str:
        """Aggregate stats across all trained lipids in this run."""
        if mdf is None or not len(mdf):
            return "  (no metrics)"
        n = len(mdf)
        bits = [f"  n_lipids = {n}"]
        if "test_corr" in mdf:
            bits.append(f"  mean corr    = {mdf['test_corr'].mean():+.4f}")
            bits.append(f"  median corr  = {mdf['test_corr'].median():+.4f}")
        if "test_rmse_z" in mdf:
            bits.append(f"  mean RMSE(z) = {mdf['test_rmse_z'].mean():.4f}")
        return "\n".join(bits)

    def _update_info_panel(lipid_idx: int):
        """Rebuild the info-panel HTML. Called from the lipid + run
        callbacks. Touches no napari state — just sets the label text."""
        run_a_cur = state["run_a"]
        cfg_a = run_a_cur.get("config", {})
        sections = []

        # Lipid identity
        if lipid_idx < len(run_a_cur["lipid_names"]):
            name = run_a_cur["lipid_names"][lipid_idx]
            gidx = run_a_cur["original_indices"][lipid_idx]
            sections.append(
                f"<b>Lipid</b><br>"
                f"<pre>  name        = {name}\n"
                f"  global idx  = {gidx}\n"
                f"  local idx   = {lipid_idx}</pre>"
            )

        # Run A — hypers
        sections.append(
            f"<b>Run A — {run_a_cur['run_dir'].name}</b>"
            f"<pre>{_format_hypers(cfg_a)}</pre>"
        )

        # Run A — per-lipid metrics
        mdf_a = state["metrics_a"]
        row_a = None
        if mdf_a is not None and lipid_idx < len(run_a_cur["original_indices"]):
            gidx = run_a_cur["original_indices"][lipid_idx]
            sel = mdf_a.loc[mdf_a["lipid_global_idx"] == gidx]
            if len(sel):
                row_a = sel.iloc[0]
        sections.append(
            f"<b>A — this lipid</b>"
            f"<pre>{_format_metrics_row(row_a)}</pre>"
        )
        sections.append(
            f"<b>A — aggregate across {len(mdf_a) if mdf_a is not None else 0} "
            f"trained lipids</b>"
            f"<pre>{_format_aggregate(mdf_a)}</pre>"
        )

        # Run B — same blocks if --compare was passed
        if run_b is not None:
            cfg_b = run_b.get("config", {})
            sections.append(
                f"<hr><b>Run B — {run_b['run_dir'].name}</b>"
                f"<pre>{_format_hypers(cfg_b)}</pre>"
            )
            row_b = None
            if metrics_b is not None and lipid_idx < len(run_b["original_indices"]):
                gidx_b = run_b["original_indices"][lipid_idx]
                sel = metrics_b.loc[metrics_b["lipid_global_idx"] == gidx_b]
                if len(sel):
                    row_b = sel.iloc[0]
            sections.append(
                f"<b>B — this lipid</b>"
                f"<pre>{_format_metrics_row(row_b)}</pre>"
            )
            sections.append(
                f"<b>B — aggregate</b>"
                f"<pre>{_format_aggregate(metrics_b)}</pre>"
            )

        info_label.setText("".join(sections))

    state["update_info"] = _update_info_panel

    from magicgui import magicgui

    # ---- Lipid + gamma controls (re-bound on run change) ----
    @magicgui(
        auto_call=True,
        lipid_idx={"label": "lipid",
                   "min": 0,
                   "max": max(0, len(state["run_a"]["lipid_names"]) - 1),
                   "step": 1},
        gamma={"label": "gamma", "min": 0.1, "max": 2.0, "step": 0.05},
    )
    def lipid_controls(
        lipid_idx: int = state["layers_a"]["current_idx"],
        gamma: float = args["gamma"],
    ):
        args["gamma"] = gamma
        update_run_layers(state["layers_a"], state["run_a"], lipid_idx, args)
        if run_b is not None and layers_b is not None:
            update_run_layers(layers_b, run_b, lipid_idx, args)

        run_a_cur = state["run_a"]
        if lipid_idx >= len(run_a_cur["lipid_names"]):
            return
        name = run_a_cur["lipid_names"][lipid_idx]
        global_idx_a = run_a_cur["original_indices"][lipid_idx]
        bits = [f"[{lipid_idx} → global {global_idx_a}] {name}"]
        for tag, mdf, run in (("A", state["metrics_a"], run_a_cur),
                              ("B", metrics_b, run_b)):
            if mdf is None or run is None:
                continue
            if lipid_idx < len(run["original_indices"]):
                gidx = run["original_indices"][lipid_idx]
            else:
                gidx = global_idx_a
            row = mdf.loc[mdf["lipid_global_idx"] == gidx]
            if not len(row):
                continue
            r = row.iloc[0]
            bits.append(
                f"{tag}: corr={r['test_corr']:+.3f} "
                f"RMSE(z)={r['test_rmse_z']:.3f} "
                f"R²={r['test_r2']:+.3f}"
            )
        viewer.status = "   ".join(bits)

        # Refresh the dock info panel too.
        state["update_info"](lipid_idx)

    # ---- Run-navigation controls (prev / next + jump-by-dropdown) ----
    # The dropdown is the primary control; prev/next buttons are
    # convenience. magicgui can't currently put buttons next to a
    # combobox on the same row, so they stack vertically in the dock.
    run_choices = [
        (f"[{i:2d}/{len(all_runs)}] {p.name}", i)
        for i, p in enumerate(all_runs)
    ]

    def _snapshot_layer_state(layer_dict):
        """Read the current visible / opacity / size of each layer in
        ``layer_dict`` keyed by ROLE. Used before a run switch.

        Each property has its own try/except — napari Points layers
        sometimes have ``.size`` as a per-point array which can't be
        coerced to a single float. Failing on size shouldn't lose the
        visible/opacity for that role.
        """
        snap = {}
        for role, lyr in layer_dict.items():
            if role == "current_idx":
                continue
            entry = {}
            try:
                entry["visible"] = bool(lyr.visible)
            except Exception:
                pass
            try:
                entry["opacity"] = float(lyr.opacity)
            except Exception:
                pass
            # Size may be a scalar OR a per-point array depending on
            # napari version + how the layer was constructed. We use
            # the first element as a representative scalar.
            try:
                raw = getattr(lyr, "size", None)
                if raw is None:
                    pass
                elif hasattr(raw, "shape") and getattr(raw, "size", 1) > 1:
                    entry["size"] = float(np.asarray(raw).flat[0])
                else:
                    entry["size"] = float(raw)
            except Exception:
                pass
            if entry:
                snap[role] = entry
        return snap

    def _apply_layer_state(layer_dict, snap):
        """Re-apply a snapshot. Each property is independent so a
        partial snapshot (e.g. visible without opacity) still works."""
        if not snap:
            return
        for role, lyr in layer_dict.items():
            if role == "current_idx" or role not in snap:
                continue
            s = snap[role]
            if "visible" in s:
                try:
                    lyr.visible = s["visible"]
                except Exception:
                    pass
            if "opacity" in s:
                try:
                    lyr.opacity = s["opacity"]
                except Exception:
                    pass
            if "size" in s:
                try:
                    lyr.size = s["size"]
                except Exception:
                    pass

    def _remove_layers(layer_dict):
        """Remove all napari layers we created for one run, so a switch
        can rebuild from scratch."""
        for k, lyr in list(layer_dict.items()):
            if k == "current_idx":
                continue
            try:
                viewer.layers.remove(lyr)
            except Exception:
                pass

    def _rebind_to_run(new_idx: int):
        """Tear down run-A layers, load the new run, build new layers,
        and reset the lipid slider's bounds.

        Preserves across the switch:
          - per-layer visible / opacity / size (snapshot → restore)
          - current lipid_idx (clamped to the new run's lipid count)
          - the lipid slider's value (= same lipid, if available)
        Forces a refresh of the layers (with the right lipid's colours)
        AND the info panel — both can silently go stale if the lipid
        index happens to be unchanged, because the magicgui callback
        only fires on value change.
        """
        new_idx = max(0, min(int(new_idx), len(all_runs) - 1))
        if new_idx == state["run_idx"]:
            return
        log.info(f"Switching A → run #{new_idx}: "
                 f"{all_runs[new_idx].name}")
        # Capture the slider's current value FIRST — that's the lipid we
        # want the new run's layers to start coloured for.
        current_lipid_idx = int(lipid_controls.lipid_idx.value)
        # ---- snapshot view state BEFORE removing layers ----
        snap = _snapshot_layer_state(state["layers_a"])
        _remove_layers(state["layers_a"])
        new_run = load_run(all_runs[new_idx])
        # Rebuild the kernel for the new run — its config may differ from
        # the previous one (different knn_method / num_modes / etc).
        new_kernel_pkg = None
        if args.get("eigenvector_dir") and not args.get("no_kernel"):
            new_kernel_pkg = reconstruct_matern_kernel(new_run, args)
        state["kernel_pkg_a"] = new_kernel_pkg
        # Clamp to the new run's lipid count.
        new_n = max(1, len(new_run["lipid_names"]))
        start_lipid = min(current_lipid_idx, new_n - 1)
        new_layers = add_run_layers(viewer, new_run, prefix="A: ", args=args,
                                    start_lipid_idx=start_lipid,
                                    kernel_pkg=new_kernel_pkg)
        _apply_layer_state(new_layers, snap)
        state["run_a"] = new_run
        state["layers_a"] = new_layers
        state["run_idx"] = new_idx
        state["metrics_a"] = _load_metrics(new_run)
        viewer.title = _title()

        # Re-bind the lipid slider's max to the new run's lipid count.
        try:
            lipid_controls.lipid_idx.max = new_n - 1
        except Exception:
            pass
        # Setting the value MAY fire lipid_controls (auto_call=True) if
        # the value changes. If it doesn't change (same lipid in both
        # runs), the callback won't fire — so we explicitly refresh
        # below regardless.
        lipid_controls.lipid_idx.value = start_lipid

        # FORCE refresh: layer face_colors + info panel. add_run_layers
        # already coloured for start_lipid, but update_run_layers also
        # rebuilds the shared colour ranges, which we want consistent
        # with the rest of the app. The info-panel update is the
        # critical one — without it the panel keeps showing the previous
        # run's hypers/metrics.
        update_run_layers(state["layers_a"], state["run_a"],
                          start_lipid, args)
        if run_b is not None and layers_b is not None:
            update_run_layers(layers_b, run_b, start_lipid, args)
        state["update_info"](start_lipid)

        # Re-wire visibility hooks on the new layers so the lazy-recolor
        # optimisation keeps working after a run switch.
        wire = state.get("wire_visibility_hooks")
        if wire is not None:
            wire(state["layers_a"], state["run_a"])

        # Also re-sync the run-nav widget's display if the change came
        # from a prev/next button rather than the dropdown itself.
        try:
            run_nav.run_choice.value = new_idx
        except Exception:
            pass

    @magicgui(
        auto_call=False,
        call_button=None,   # no submit button on the form itself
        run_choice={
            "label": "run",
            "choices": run_choices,
        },
    )
    def run_nav(run_choice: int = current_run_idx):
        # Triggered when the dropdown value changes (auto_call=False
        # means we wire it manually via .changed below).
        _rebind_to_run(run_choice)

    # Wire the dropdown change to trigger the rebind (auto_call=False
    # would otherwise wait for a button click).
    run_nav.run_choice.changed.connect(_rebind_to_run)

    # Convenience: dedicated prev / next buttons. They wrap around so
    # you can cycle continuously.
    @magicgui(call_button="◀ prev run")
    def prev_run():
        target = (state["run_idx"] - 1) % len(all_runs)
        _rebind_to_run(target)

    @magicgui(call_button="next run ▶")
    def next_run():
        target = (state["run_idx"] + 1) % len(all_runs)
        _rebind_to_run(target)

    # Source navigation within the highlight subset. These only do anything
    # when the highlight layers exist (which they do unless add_run_layers
    # raised during the kernel/highlight step). Cycles through the subset
    # and — when the kernel layer is visible — re-evaluates K(src, ·).
    def _step_source(delta: int):
        layers = state["layers_a"]
        set_active = layers.get("_set_active_src")
        st = layers.get("_highlight_state")
        if set_active is None or st is None:
            log.warning("Highlight subset isn't available; skip step.")
            return
        new_within = (st["active_within_subset"] + delta) % len(st["indices"])
        set_active(new_within)
        # Refresh the info panel too — it shows the active source's test idx.
        try:
            _update_info_panel(int(lipid_controls.lipid_idx.value))
        except Exception:
            pass

    @magicgui(call_button="◀ prev source")
    def prev_source():
        _step_source(-1)

    @magicgui(call_button="next source ▶")
    def next_source():
        _step_source(+1)

    # ---- Dock all widgets on the right side ----
    viewer.window.add_dock_widget(run_nav, name="run selector", area="right")
    viewer.window.add_dock_widget(prev_run, name="prev", area="right")
    viewer.window.add_dock_widget(next_run, name="next", area="right")
    viewer.window.add_dock_widget(lipid_controls, name="lipid selector",
                                  area="right")
    viewer.window.add_dock_widget(prev_source, name="prev source", area="right")
    viewer.window.add_dock_widget(next_source, name="next source", area="right")
    viewer.window.add_dock_widget(info_container, name="info", area="right")

    # ---- Visibility hooks: lazy recolor on toggle ----
    # When a layer that was skipped during a lipid update (because it
    # was invisible) gets toggled on later, recompute its face_color
    # from disk. This is what makes the "skip invisible" optimisation
    # in update_run_layers safe — turning a layer back on never shows
    # stale data.
    def _wire_visibility_hooks(layers_dict, run_dict):
        for role, lyr in layers_dict.items():
            if role.startswith("_") or role == "current_idx":
                continue

            def make_handler(role=role, run_dict=run_dict,
                             layers_dict=layers_dict):
                def handler(event):
                    layer = layers_dict.get(role)
                    if layer is None or not layer.visible:
                        return
                    pending = layers_dict.get("_pending_recolor", {})
                    if role not in pending:
                        return
                    update_run_layers(
                        layers_dict, run_dict,
                        layers_dict["current_idx"], args,
                    )
                return handler

            try:
                lyr.events.visible.connect(make_handler())
            except Exception:
                pass

    _wire_visibility_hooks(state["layers_a"], state["run_a"])
    if layers_b is not None and run_b is not None:
        _wire_visibility_hooks(layers_b, run_b)
    # Make _rebind_to_run rewire hooks on the new layers it builds.
    # We just append this to state so the run-switch code can call it.
    state["wire_visibility_hooks"] = _wire_visibility_hooks

    # Trigger the initial render — this populates layers AND the info panel
    # (the lipid_controls callback calls _update_info_panel at the end).
    lipid_controls(
        lipid_idx=state["layers_a"]["current_idx"], gamma=args["gamma"],
    )

    log.info("Viewer ready. Use the run dropdown or prev/next to step "
             "between runs.")
    napari.run()


if __name__ == "__main__":
    main()