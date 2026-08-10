"""Shared plumbing for the two ``parcelgp`` viewers.

``view_parcels`` (what the parcellation looks like on the template) and
``parcel_vs_euclidean_explorer`` (what the kernel does with it, on real MALDI
slices) need the same four things, so they live here rather than being copied:

  * a stable colour per parcel id, and the one-voxel-thick border mask that
    turns a label volume into drawable outlines;
  * the reference sub-volume that matches a field's strided grid, checked
    rather than assumed;
  * one sample's MALDI voxels, snapped to the field's nodes and laid out flat
    per section (the layout the manifold explorer uses, reproduced here so the
    two tools feel the same);
  * the kernel arithmetic -- a numpy Matern identical to the one the runner
    trains, and the parcel factor from :mod:`parcelgp.kernels` evaluated at a
    given ``B``, whether that ``B`` came from a checkpoint or from the
    no-training-needed identity fallback.

Everything here is numpy/scipy/pandas, matching the rest of the package. The
only exception is :func:`load_trained_parcel`, which imports ``torch`` inside
the function because reading a checkpoint is impossible without it -- the
viewers work fine with no torch installed as long as you do not pass a run dir.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .volume import LR_AXIS, load_reference, standardize, stride_volume

#: Index of the dorsal-ventral axis in the (AP, DV, LR) convention.
DV_AXIS = 1

log = logging.getLogger(__name__)

#: Registered CCF coordinates in the parquet, (AP, DV, LR) -- same convention
#: as :mod:`parcelgp.volume` and :mod:`parcelgp.validate`.
COORD_COLS = ("xccf", "yccf", "zccf")
SECTION_COL = "Section"
SAMPLE_COL = "Sample"


# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #
def parcel_colors(n_parcels: int, alpha: float = 0.9) -> np.ndarray:
    """(K, 4) RGBA, one stable colour per parcel id.

    Hues are spread by the golden ratio so numerically adjacent parcel ids --
    which k-means hands out in no spatial order at all -- stay visually
    distinct. Deterministic: the same parcel keeps its colour across sections,
    across the two viewers, and across re-runs.
    """
    from matplotlib.colors import hsv_to_rgb
    rgba = np.zeros((int(n_parcels), 4), np.float32)
    for i in range(int(n_parcels)):
        h = (i * 0.6180339887) % 1.0
        s = 0.55 + 0.35 * ((i * 0.7548776662) % 1.0)      # vary sat/val too, so
        v = 0.75 + 0.25 * ((i * 0.3247179572) % 1.0)      # near-hues still differ
        rgba[i, :3] = hsv_to_rgb((h, s, v))
    rgba[:, 3] = alpha
    return rgba


def heat_colors(vals, gamma=0.45, cmap="inferno", fade=True, alpha_floor=0.12,
                vlim=None) -> np.ndarray:
    """Sequential heatmap with the high end gamma-boosted.

    ``fade=True`` also drives OPACITY with the value, so low-covariance points
    recede instead of forming a solid coloured sheet -- which is what makes a
    kernel row readable as "the neighbourhood this kernel considers".

    ``vlim=None`` normalises against this frame's own 1st/99.5th percentiles, so
    the colour range is always filled and a uniform rescale of ``vals`` (the
    outputscale) is invisible. ``vlim=(lo, hi)`` fixes the scale instead, which
    is what you want when two panels must be comparable.
    """
    import matplotlib
    v = np.asarray(vals, np.float64)
    if vlim is None:
        lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99.5)
    else:
        lo, hi = float(vlim[0]), float(vlim[1])
    hi = hi if hi > lo else lo + 1e-9
    n = np.clip((v - lo) / (hi - lo), 0, 1) ** gamma
    rgba = matplotlib.colormaps[cmap](np.nan_to_num(n))
    if fade:
        rgba[:, 3] = alpha_floor + (1.0 - alpha_floor) * np.nan_to_num(n)
    return rgba


def diverging_colors(vals, cmap="coolwarm", vlim=None) -> np.ndarray:
    """Signed heatmap centred at 0; |value| drives opacity, NaN -> invisible."""
    import matplotlib
    v = np.asarray(vals, np.float64)
    m = np.isfinite(v)
    lim = vlim if vlim is not None else (
        np.nanpercentile(np.abs(v[m]), 98) if m.any() else 1.0)
    lim = float(lim) if lim and lim > 0 else 1.0
    n = np.clip(v / (2 * lim) + 0.5, 0, 1)
    rgba = matplotlib.colormaps[cmap](np.nan_to_num(n))
    rgba[:, 3] = np.where(m, np.clip(np.abs(v) / lim, 0, 1), 0.0)
    return rgba


# --------------------------------------------------------------------------- #
# template geometry
# --------------------------------------------------------------------------- #
def label_volume(field, background: int = -1) -> np.ndarray:
    """(Z, Y, X) int32 parcel-id volume on the field's strided grid."""
    return field.to_volume("labels", background=background)


def border_mask(field, both_sides: bool = False) -> np.ndarray:
    """(N,) bool: nodes that sit on a parcel border.

    Vectorised 6-neighbour label comparison on the strided grid, marking only
    the LOWER-index side of each interface by default so the outline is ONE node
    thick. Marking both sides doubles the drawn width (every border node paints
    its whole cell), which reads as a band rather than a line.

    This is deliberately not derived from ``d_border_rel``: that quantity is
    normalised per parcel, so a fixed threshold on it means different physical
    distances in a big parcel and a small one. The label comparison is exact.
    """
    shape = field.volume_shape
    if shape is None or field.node_vox is None:
        raise ValueError("field was built without node_vox/volume_shape; "
                         "rebuild it with parcelgp.build")
    lab = np.zeros(shape, np.int64)
    lab[tuple(field.node_vox.T)] = field.labels.astype(np.int64) + 1   # 0 = non-tissue
    border = np.zeros(shape, bool)
    for ax in range(3):
        sl_a = [slice(None)] * 3; sl_a[ax] = slice(0, shape[ax] - 1)
        sl_b = [slice(None)] * 3; sl_b[ax] = slice(1, shape[ax])
        a, b = lab[tuple(sl_a)], lab[tuple(sl_b)]
        diff = (a > 0) & (b > 0) & (a != b)
        border[tuple(sl_a)] |= diff
        if both_sides:
            border[tuple(sl_b)] |= diff
    return border[tuple(field.node_vox.T)]


def reference_subvolume(reference_file, field) -> np.ndarray:
    """The reference image strided to exactly the field's grid.

    Raises if the shapes disagree, because silently viewing a stride-2 label
    volume over a stride-4 template is the kind of mistake that looks plausible
    on screen.
    """
    stride = int(field.meta.get("stride", 1))
    sub, _ = stride_volume(load_reference(reference_file), stride)
    if field.volume_shape is not None and sub.shape != tuple(field.volume_shape):
        raise ValueError(
            f"reference {Path(reference_file).name} strided by {stride} gives "
            f"{sub.shape}, but the field's grid is {tuple(field.volume_shape)}. "
            f"The field was built from {field.meta.get('reference_file')!r} — "
            f"point --reference-file at that file.")
    return sub


def coord_frame(field):
    """``(coord_mean (3,), coord_std ())`` recorded in the field's metadata.

    Read from the field rather than recomputed from the reference image so a
    viewer needs only the .npz to place MALDI voxels in the right frame.
    """
    return (np.asarray(field.meta["coord_mean"], np.float32),
            np.float32(field.meta["coord_std"]))


# --------------------------------------------------------------------------- #
# MALDI sections
# --------------------------------------------------------------------------- #
def list_samples(maldi_file) -> list[str]:
    import pandas as pd
    s = pd.read_parquet(maldi_file, columns=[SAMPLE_COL])[SAMPLE_COL]
    return sorted(s.unique().tolist())


def _project_section_2d(coords_mm: np.ndarray) -> np.ndarray:
    """PCA-project a (tilted) section's mm coords onto their best-fit plane.

    Returns ``(M, 2)`` ordered ``(down, right)`` — i.e. (row, column), the
    convention napari points and ``imshow`` both use, with the first column
    increasing ventrally and the second increasing to the right.

    The plane comes from an SVD, whose basis is arbitrary up to rotation and
    sign, so the raw projection lands every section at its own random angle and
    handedness. A coronal acquisition plane is spanned by DV and LR, so
    re-ordering and re-signing the two in-plane vectors against those axes makes
    every section come out upright and the same way round — which matters here
    because two panels of the same slice are compared side by side.
    """
    c = coords_mm - coords_mm.mean(0, keepdims=True)
    if c.shape[0] < 3:
        return c[:, :2].astype(np.float32)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    e = vt[:2].copy()                                   # (2, 3) in-plane basis
    if abs(e[0, DV_AXIS]) < abs(e[1, DV_AXIS]):         # DV-aligned vector first
        e = e[::-1]
    sign = np.sign([e[0, DV_AXIS], e[1, LR_AXIS]])
    e *= np.where(sign == 0, 1.0, sign)[:, None]        # DV down, LR right
    return c @ e.T


class MaldiSections:
    """One sample's MALDI voxels, snapped to the field's nodes, grouped by section.

    Each section is laid out as a flat 2D ``(down, right)`` tile (PCA onto its
    own best-fit plane, since the acquisition planes are tilted relative to the
    CCF axes), so a viewer can show one section at a time as plain 2D points.
    Sections are NOT stacked on a third axis: napari keeps a stale
    ``_indices_view`` when 3D point data is swapped underneath a slider, which
    crashes on section change.

    Voxels further than ``max_snap_mm`` from any template node are dropped —
    those are registration outliers, and a wrong parcel label for them would
    show up exactly as spurious structure in the border overlay.
    """

    def __init__(self, field, maldi_file, sample: str, max_snap_mm: float = 0.5):
        import pandas as pd
        self.field, self.maldi_file, self.sample = field, maldi_file, str(sample)
        self.filters = [(SAMPLE_COL, "==", self.sample)]
        df = pd.read_parquet(maldi_file, columns=[*COORD_COLS, SECTION_COL],
                             filters=self.filters)
        self.n_rows = len(df)
        xyz = df[list(COORD_COLS)].to_numpy(np.float32)
        coord_mean, coord_std = coord_frame(field)
        cs = standardize(xyz, coord_mean, coord_std)

        finite = np.isfinite(cs).all(1)
        node = np.zeros(len(cs), np.int64)
        node[finite] = field.node_index(cs[finite])
        err_mm = np.full(len(cs), np.inf, np.float32)
        err_mm[finite] = (np.linalg.norm(field.node_coords[node[finite]] - cs[finite],
                                         axis=1) * coord_std)
        valid = finite & (err_mm <= float(max_snap_mm))

        sec = df[SECTION_COL].to_numpy()
        self.sections = sorted(np.unique(sec[valid]).tolist())
        self.snap_err_mm = err_mm[valid]
        #: per section: (xy (M,2), node ids (M,), parquet row ids (M,))
        self.by_section: dict = {}
        rows_all = np.arange(len(df), dtype=np.int64)
        for s in self.sections:
            m = valid & (sec == s)
            xy = _project_section_2d(xyz[m].astype(np.float64)).astype(np.float32)
            self.by_section[s] = (xy, node[m], rows_all[m])
        log.info("%s: %d/%d voxels snapped (median error %.3f mm) across %d sections",
                 self.sample, int(valid.sum()), len(df),
                 float(np.median(self.snap_err_mm)) if valid.any() else float("nan"),
                 len(self.sections))

        # point size that makes voxels TILE their section rather than scatter:
        # sqrt(bounding-box area / count), taken on a mid section for robustness.
        s0 = self.sections[len(self.sections) // 2]
        xy0 = self.by_section[s0][0]
        if len(xy0) > 3:
            span = xy0.max(0) - xy0.min(0)
            area = float(max(span[0], 1e-6) * max(span[1], 1e-6))
            self.spacing = float(np.sqrt(area / max(len(xy0), 1)))
        else:
            self.spacing = 0.1
        self._lipid_cache: dict = {}

    def layer_data(self, section):
        """``(points (M,2), node ids (M,), parquet rows (M,))`` for one section."""
        return self.by_section[section]

    def lipid(self, name: str) -> np.ndarray:
        """(n_rows,) z-scored measurements of one lipid over this sample.

        Read with the SAME filter as the coordinates, so the row order matches
        and ``rows`` from :meth:`layer_data` indexes straight into it (this is
        the alignment ``parcelgp.validate`` already relies on).
        """
        if name not in self._lipid_cache:
            import pandas as pd
            v = pd.read_parquet(self.maldi_file, columns=[name],
                                filters=self.filters)[name].to_numpy(np.float64)
            if len(v) != self.n_rows:
                raise RuntimeError(
                    f"lipid column {name!r} returned {len(v)} rows but the "
                    f"coordinates had {self.n_rows}; the parquet filter is not "
                    f"reproducing row order.")
            good = np.isfinite(v)
            z = np.full(len(v), np.nan)
            z[good] = (v[good] - v[good].mean()) / (v[good].std() + 1e-8)
            self._lipid_cache[name] = z.astype(np.float32)
        return self._lipid_cache[name]


def available_lipids(maldi_file, available_lipids_file=None) -> list[str]:
    """Lipid column names, from the shipped list when given, else the parquet."""
    if available_lipids_file:
        return [str(x) for x in np.load(available_lipids_file, allow_pickle=True)]
    import pyarrow.parquet as pq
    names = pq.ParquetFile(maldi_file).schema_arrow.names
    skip = set(COORD_COLS) | {SECTION_COL, SAMPLE_COL, "SampleSection", "x", "y",
                              "x_index", "y_index", "z_index", "__index_level_0__"}
    return [n for n in names if n not in skip]


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def matern(r: np.ndarray, nu: float) -> np.ndarray:
    """Matern correlation at scaled distance ``r``, for nu in {0.5, 1.5, 2.5}.

    Same three closed forms ``gpytorch.kernels.MaternKernel`` uses, so a kernel
    row drawn here is the one the runner trained, not a lookalike.
    """
    r = np.asarray(r, np.float64)
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5.0) * r
        return (1.0 + a + a ** 2 / 3.0) * np.exp(-a)
    raise ValueError(f"nu must be 0.5, 1.5 or 2.5 (the Euclidean MaternKernel's "
                     f"only valid values), got {nu}")


def euclidean_row(coords: np.ndarray, x0: np.ndarray, lengthscale, outputscale=1.0,
                  nu: float = 2.5) -> np.ndarray:
    """``k_base(x0, x)`` for every row of ``coords``, in standardized coordinates.

    ``lengthscale`` is a scalar (isotropic) or a (3,) ARD vector, in the same
    standardized units the model was trained in.
    """
    ls = np.broadcast_to(np.asarray(lengthscale, np.float64).ravel(), (3,))
    d = (np.asarray(coords, np.float64) - np.asarray(x0, np.float64).ravel()) / ls
    return float(outputscale) * matern(np.linalg.norm(d, axis=1), nu)


def parcel_embedding(field, nodes: np.ndarray, B: np.ndarray | None,
                     strength: float = 1.0) -> np.ndarray:
    """``z(x) = m(x)^T B`` at the given node ids, shape (M, r).

    ``B=None`` is the untrained fallback: ``B = strength * I``, i.e. the parcel
    embedding IS the membership vector, scaled. Two nodes deep inside different
    parcels then sit ``strength * sqrt(2)`` apart, so the covariance multiplier
    across a border is ``exp(-strength**2)`` and ``strength=0`` is an exact
    no-op. That makes the geometry visible without pretending a model was
    trained: nothing here is fitted to lipids.
    """
    idx = np.asarray(nodes, np.intp)
    mi, mv = field.mem_idx[idx].astype(np.intp), field.mem_val[idx].astype(np.float64)
    if B is None:
        m = np.zeros((idx.size, field.n_parcels), np.float64)
        np.put_along_axis(m, mi, mv, axis=1)
        return float(strength) * m
    B = np.asarray(B, np.float64)
    if B.shape[0] != field.n_parcels:
        raise ValueError(f"B has {B.shape[0]} parcels but the field has "
                         f"{field.n_parcels}; they must be the same build")
    return (B[mi] * mv[..., None]).sum(1)


def parcel_factor_row(Z: np.ndarray, j0: int) -> np.ndarray:
    """``exp(-||z_j0 - z||^2 / 2)`` for every row of ``Z``: the multiplier alone.

    In [0, 1] by construction, 1 at the test point. This is exactly the factor
    :class:`parcelgp.kernels.ParcelFactorKernel` multiplies onto the base kernel,
    so ``k_parcel = k_base * this``.
    """
    d2 = ((np.asarray(Z, np.float64) - np.asarray(Z, np.float64)[j0]) ** 2).sum(1)
    return np.exp(-0.5 * d2)


# --------------------------------------------------------------------------- #
# trained checkpoints
# --------------------------------------------------------------------------- #
@dataclass
class TrainedKernel:
    """Learned hyperparameters of one lipid, pulled out of a per-lipid run."""
    lipid: str
    nu: float
    lengthscale: np.ndarray        # (3,) ARD, standardized units
    outputscale: float
    B: np.ndarray | None           # (K, r) parcel embedding, None on a baseline run
    parcel_field: str | None       # the field path the run was TRAINED against
    run_dir: str
    checkpoint: str


def _constrained(raw, lower, upper):
    """Undo a gpytorch constraint from the bounds stored alongside it.

    Checkpoints record ``raw_*_constraint.lower_bound/upper_bound`` as buffers,
    which is enough to invert without reconstructing the model: an infinite
    upper bound is ``GreaterThan`` (softplus + lower), a finite one is
    ``Interval`` (lower + range * sigmoid).
    """
    raw = np.asarray(raw, np.float64)
    lower = np.asarray(lower, np.float64)
    upper = np.asarray(upper, np.float64)
    if np.all(np.isinf(upper)):
        return np.logaddexp(0.0, raw) + lower            # softplus(raw) + lower
    return lower + (upper - lower) / (1.0 + np.exp(-raw))


def load_trained_parcel(run_dir, lipid: str | None = None) -> TrainedKernel:
    """Learned ARD lengthscale / outputscale / parcel ``B`` for one lipid.

    ``run_dir`` is a ``lgp_experiment_per_lipid.py`` output directory; each
    ``checkpoints/batch_*.pt`` holds a batch of lipids with per-task parameters,
    so a lipid name has to be picked. Omit ``lipid`` once to get the list of
    names in the error message.
    """
    import torch
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("checkpoints/batch_*.pt"))
    ckpts = [p for p in ckpts if "inprogress" not in p.name]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints/batch_*.pt under {run_dir}")

    seen: list[str] = []
    for path in ckpts:
        d = torch.load(path, map_location="cpu", weights_only=False)
        names = [str(n) for n in d["lipid_names"]]
        seen.extend(names)
        if lipid is None or lipid not in names:
            continue
        k = names.index(lipid)
        s = {kk: v.numpy() if hasattr(v, "numpy") else v
             for kk, v in d["model_state"].items()}
        pre = "covar_module.base_kernel"
        has_parcel = f"{pre}.B" in s
        lpre = f"{pre}.base_kernel" if has_parcel else pre
        ls = _constrained(s[f"{lpre}.raw_lengthscale"][k],
                          s[f"{lpre}.raw_lengthscale_constraint.lower_bound"],
                          s[f"{lpre}.raw_lengthscale_constraint.upper_bound"]).ravel()
        os_ = float(_constrained(
            s["covar_module.raw_outputscale"][k],
            s["covar_module.raw_outputscale_constraint.lower_bound"],
            s["covar_module.raw_outputscale_constraint.upper_bound"]))
        args = d.get("args", {}) or {}
        return TrainedKernel(
            lipid=lipid, nu=float(args.get("nu", 2.5)),
            lengthscale=np.broadcast_to(ls, (3,)).astype(np.float64).copy(),
            outputscale=os_,
            B=(s[f"{pre}.B"][k].astype(np.float64) if has_parcel else None),
            parcel_field=args.get("parcel_field"),
            run_dir=str(run_dir), checkpoint=str(path))

    listing = "\n  ".join(sorted(set(seen)))
    raise SystemExit(
        (f"--run-lipid is required for {run_dir}\n" if lipid is None else
         f"lipid {lipid!r} is not in {run_dir}\n")
        + f"available lipids ({len(set(seen))}):\n  {listing}\n")
