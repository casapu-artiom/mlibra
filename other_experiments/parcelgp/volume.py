"""Reference-volume geometry: voxel grid <-> standardized mm coordinates.

VENDORED, deliberately. ``parcelgp`` is self-contained (see README), but the
coordinate frame it produces has to line up *exactly* with the frame the MALDI
parquet is read into, otherwise every downstream lookup is silently offset. The
three conversions below therefore reproduce the conventions used by the existing
runners (``maldi/utils.py``) rather than inventing new ones:

  * a voxel index in the (optionally strided) reference volume maps to mm via
    ``index * stride * 0.025``  (40 voxels per mm at full resolution),
  * ``coord_mean`` is per-axis (a translation) and ``coord_std`` is a SINGLE
    SCALAR, so standardization is an isotropic rescale and the metric stays a
    true Euclidean metric up to global scale,
  * axis order is (0, 1, 2) = (AP, DV, LR) throughout, matching the parquet's
    ``(xccf, yccf, zccf)`` column order. Axis 2 is the left-right axis, which is
    the one the bilateral-symmetry option folds.

``coord_norm_from_reference`` uses ``reference_image > 0`` (not the tissue
threshold) on the FULL-resolution volume, so the standardization frame is
independent of ``--stride`` and ``--threshold``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VOXELS_PER_MM = 40.0
MM_PER_VOXEL = 1.0 / VOXELS_PER_MM   # 0.025

#: Index of the left-right axis in the (AP, DV, LR) convention.
LR_AXIS = 2


def load_reference(reference_file) -> np.ndarray:
    """Load ``reference_image.npy`` (a 256-level intensity template, not a mask)."""
    return np.load(reference_file)


def stride_volume(reference_image: np.ndarray, stride: int):
    """Subsample the reference volume.

    Returns ``(sub_volume, voxel_scale_mm)`` where ``voxel_scale_mm`` converts a
    *strided* index directly to mm.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return reference_image[::stride, ::stride, ::stride], stride * MM_PER_VOXEL


def tissue_mask(sub_volume: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean tissue mask. Node order everywhere in this package is
    ``np.where(mask)`` (numpy C-order), which is what ``node_voxels`` returns."""
    return sub_volume > threshold


def node_voxels(mask: np.ndarray) -> np.ndarray:
    """(N, 3) int32 voxel indices of the tissue nodes, in canonical node order."""
    return np.stack(np.where(mask), axis=1).astype(np.int32)


def node_coords_mm(node_vox: np.ndarray, voxel_scale_mm: float) -> np.ndarray:
    """(N, 3) float32 mm coordinates for the given (strided) voxel indices."""
    return node_vox.astype(np.float32) * float(voxel_scale_mm)


def coord_norm_from_reference(reference_image, voxel_per_mm: float = VOXELS_PER_MM):
    """Global standardization frame, computed purely from the reference image.

    Returns ``(coord_mean (3,), coord_std ())`` as float32 numpy arrays.
    ``coord_std`` is a scalar so the transform is an isotropic rescale.
    """
    if isinstance(reference_image, (str, Path)):
        reference_image = np.load(reference_image)
    idx = np.stack(np.where(reference_image > 0), axis=1)
    ref_mm = idx / float(voxel_per_mm)
    coord_mean = ref_mm.mean(axis=0).astype(np.float32)
    coord_std = np.float32(ref_mm.std())
    return coord_mean, coord_std


def standardize(coords_mm: np.ndarray, coord_mean, coord_std) -> np.ndarray:
    """mm -> standardized coordinates (the frame the GPs and the parquet share)."""
    return ((np.asarray(coords_mm, dtype=np.float32) - np.asarray(coord_mean, dtype=np.float32))
            / np.float32(coord_std)).astype(np.float32)


def midline_mm(reference_image: np.ndarray, axis: int = LR_AXIS) -> float:
    """Left-right midline of the FULL-resolution reference, in mm.

    Taken as the intensity-weighted centre of mass along ``axis`` rather than the
    array centre, since the volume is not guaranteed to be centred in its bbox.
    """
    prof = reference_image.sum(axis=tuple(i for i in range(reference_image.ndim) if i != axis))
    prof = prof.astype(np.float64)
    pos = np.arange(prof.size, dtype=np.float64)
    return float((prof * pos).sum() / max(prof.sum(), 1e-12) * MM_PER_VOXEL)
