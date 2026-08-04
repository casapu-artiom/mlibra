"""Multi-scale appearance features of the reference template, sampled at nodes.

The parcellation is only ever as good as the representation it clusters. A
single-scale stack (intensity + one gradient + one 3x3x3 local mean/std) mostly
encodes "how bright and how busy is this voxel", which is why plain k-means on it
comes out speckly and only weakly aligned with anatomy. The stack here adds the
two things that actually carry anatomical structure in an intensity template:

  * **scale** -- the same descriptors at sigma = 1, 2, 4 voxels, so a boundary is
    described by how it looks across a range of neighbourhood sizes rather than
    at one arbitrary radius;
  * **shape** -- eigenvalues of the (scale-normalized) Hessian, which separate
    sheet-like from tube-like from blob-like local structure. Cortical laminae
    are sheets and fibre tracts are tubes, so these are the descriptors that let
    a clusterer put a layer boundary and a tract boundary in different places.
    Sorted by |lambda| descending, so they are invariant to orientation.

Plus ``depth`` -- Euclidean distance to the tissue surface -- which is strongly
anatomical on its own (cortical depth) and cheap.

Everything is computed on the volume and then sampled at the tissue nodes, one
derivative volume at a time, so peak memory stays at ~2 float32 copies of the
(strided) volume regardless of how many features are requested.

All features are z-scored over the nodes before being returned, so the
clusterer sees an isotropic feature space and no single descriptor dominates by
virtue of its units.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

#: Bump whenever the FEATURE VALUES change for unchanged arguments -- a new
#: descriptor, a changed scale convention, a fixed bug. It is recorded in each
#: built field's metadata and checked by ``parcelgp.build.check_cached``, so a
#: field built by older code is refused rather than silently reused. Without it,
#: every CLI argument can match while the numbers underneath have changed, which
#: is the same failure mode as the filename-based cache that made a stride-1 run
#: return stride-4 results.
#:
#: 1: initial stack.
#: 2: surface_depth fills internal background holes (was truncating depth by 31%
#:    at the maximum; see surface_depth's docstring).
#: 3: feature scales are specified in MILLIMETRES instead of voxels, so they no
#:    longer change meaning with --stride. Bit-identical at stride 4 (the mm
#:    defaults are exactly the old voxel values there); different elsewhere,
#:    which is why the version must be keyed even though stride already is.
FEATURE_VERSION = 3


@dataclass
class FeatureSpec:
    """Which descriptors to compute, at what PHYSICAL scales. Defaults are the
    full stack (16 features).

    Scales are in **millimetres**, not voxels. They used to be in voxels, which
    silently coupled them to ``--stride``: the same ``sigma=1`` meant a 0.1 mm
    blur at stride 4 and a 0.025 mm blur at stride 1, so changing the stride
    changed what the features measured as well as the resolution they were
    measured at, and the two effects could not be separated.

    The defaults are exactly the stride-4 values that were in use before
    (1, 2, 4 voxels -> 0.1, 0.2, 0.4 mm; boxes of 3, 5, 9 voxels -> 0.3, 0.5,
    0.9 mm), and the mm->voxel conversion is exact at stride 4, so a stride-4
    build is bit-identical to the previous behaviour. Only other strides change,
    and there they now describe the same physical structure.
    """
    smooth_scales_mm: tuple = (0.1, 0.2, 0.4)      # gaussian-smoothed intensity
    gradient_scales_mm: tuple = (0.1, 0.2, 0.4)    # gradient magnitude
    std_sizes_mm: tuple = (0.3, 0.5, 0.9)          # local std, uniform box width
    hessian_scales_mm: tuple = (0.1, 0.2)          # 3 eigenvalues each
    include_depth: bool = True                     # distance to tissue surface
    #: extra descriptors are appended in this fixed order; see ``names``
    _order: tuple = field(default=(), repr=False)


def _sigma_voxels(sigma_mm: float, voxel_scale_mm: float, what: str) -> float:
    """mm -> voxels for a Gaussian scale, with a check that it is resolvable."""
    s = float(sigma_mm) / float(voxel_scale_mm)
    if s < 0.5:
        log.warning(
            "%s at %.3f mm is %.2f voxels at this stride — below ~0.5 voxels a "
            "Gaussian is almost the identity, so this feature carries nothing. "
            "Use a finer stride or drop the scale.", what, sigma_mm, s)
    return s


def _box_voxels(size_mm: float, voxel_scale_mm: float) -> int:
    """mm -> an ODD voxel box width (uniform_filter is only symmetric for odd)."""
    k = int(round(float(size_mm) / float(voxel_scale_mm)))
    if k % 2 == 0:
        k += 1
    return max(3, k)


def feature_blocks(names) -> np.ndarray:
    """Group feature names into descriptor blocks: smooth / grad / std / hess / depth.

    Derived from the names rather than returned alongside them, so adding a
    descriptor needs no signature change anywhere. ``hess1_0.2mm`` -> ``hess``
    (the eigenvalue index is stripped), ``depth`` -> ``depth``.

    Used by the optional per-block normalization in ``parcellate``: k-means sums
    squared differences over columns, so without it a block's influence is simply
    how many numbers that descriptor happens to emit -- 6 for the Hessian, 1 for
    depth, for no anatomical reason.
    """
    out = []
    for n in names:
        head = n.split("_")[0] if "_" in n else n
        out.append(head.rstrip("0123456789"))
    return np.array(out, dtype=object)


def _sample(vol: np.ndarray, node_vox: np.ndarray) -> np.ndarray:
    z, y, x = node_vox.T
    return vol[z, y, x].astype(np.float32)


def _hessian_eigenvalues(vol: np.ndarray, node_vox: np.ndarray,
                         sigma: float) -> np.ndarray:
    """(N, 3) Hessian eigenvalues at ``sigma``, sorted by |lambda| descending.

    Scale-normalized (multiplied by sigma^2) so magnitudes are comparable across
    scales. The six second-derivative volumes are built and sampled one at a
    time; only the (N, 6) sampled values are retained.
    """
    orders = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    comps = np.empty((node_vox.shape[0], 6), dtype=np.float32)
    for i, order in enumerate(orders):
        d = ndi.gaussian_filter(vol, sigma=sigma, order=order, mode="nearest")
        comps[:, i] = _sample(d, node_vox) * (sigma ** 2)
        del d
    n = node_vox.shape[0]
    H = np.empty((n, 3, 3), dtype=np.float32)
    H[:, 0, 0], H[:, 1, 1], H[:, 2, 2] = comps[:, 0], comps[:, 1], comps[:, 2]
    H[:, 0, 1] = H[:, 1, 0] = comps[:, 3]
    H[:, 0, 2] = H[:, 2, 0] = comps[:, 4]
    H[:, 1, 2] = H[:, 2, 1] = comps[:, 5]
    ev = np.linalg.eigvalsh(H)                      # ascending, real (symmetric)
    order_idx = np.argsort(-np.abs(ev), axis=1)     # |lambda| descending
    return np.take_along_axis(ev, order_idx, axis=1).astype(np.float32)


def surface_depth(mask: np.ndarray, node_vox: np.ndarray,
                  voxel_scale_mm: float) -> np.ndarray:
    """(N,) distance from each node to the tissue surface, in mm.

    Internal holes are filled first. At the working tissue threshold (5) the
    sub-threshold voxels *inside* the brain are entirely noise -- 202 components
    totalling 531 voxels, the largest only 32 voxels and 140 of them single
    voxels -- but because they sit deep, leaving them in truncates depth badly:
    median 0.63 mm instead of 0.72 (-12%) and max 2.12 mm instead of 3.09 (-31%).
    The deepest structures were being told they were 2 mm in when they were 3.

    Filling is unconditional rather than size-limited, which is correct here and
    would NOT be at a higher threshold: by threshold 10 the internal background
    contains 188/178/164-voxel components that are genuine ventricles, and tissue
    bordering a ventricle really is at an anatomical boundary. Hence the warning
    -- the assumption is checked at runtime instead of being left implicit.
    """
    filled = ndi.binary_fill_holes(mask)
    added = filled & ~mask
    n_added = int(added.sum())
    if n_added:
        lab, _ = ndi.label(added)
        counts = np.bincount(lab.ravel())
        biggest = int(counts[1:].max()) if counts.size > 1 else 0
        log.info("surface depth: filled %d internal background voxels "
                 "(largest cavity %d voxels)", n_added, biggest)
        if biggest >= 50:
            log.warning(
                "surface_depth filled a %d-voxel internal cavity — at this tissue "
                "threshold that is large enough to be a ventricle rather than "
                "noise, and filling it erases a real anatomical boundary. Verify "
                "before trusting the depth feature.", biggest)
    d = ndi.distance_transform_edt(filled) * float(voxel_scale_mm)
    return _sample(d, node_vox)


def template_features(sub_volume: np.ndarray, mask: np.ndarray,
                      node_vox: np.ndarray, voxel_scale_mm: float,
                      spec: FeatureSpec | None = None):
    """Build the (N, F) z-scored feature matrix in node order.

    Args:
      sub_volume:      (Z, Y, X) reference intensity on the (strided) grid.
      mask:            boolean tissue mask, ``sub_volume > threshold``.
      node_vox:        (N, 3) node voxel indices (canonical ``np.where(mask)`` order).
      voxel_scale_mm:  mm per (strided) voxel -- only scales ``depth``.
      spec:            which descriptors to include.

    Returns:
      ``(feats (N, F) float32, names list[str])``
    """
    spec = spec or FeatureSpec()
    vol = sub_volume.astype(np.float32)
    cols, names = [], []

    for mm in spec.smooth_scales_mm:
        s = _sigma_voxels(mm, voxel_scale_mm, "smooth")
        cols.append(_sample(ndi.gaussian_filter(vol, sigma=s, mode="nearest"), node_vox))
        names.append(f"smooth_{mm:g}mm")

    for mm in spec.gradient_scales_mm:
        s = _sigma_voxels(mm, voxel_scale_mm, "gradient")
        cols.append(_sample(
            ndi.gaussian_gradient_magnitude(vol, sigma=s, mode="nearest"), node_vox))
        names.append(f"grad_{mm:g}mm")

    for mm in spec.std_sizes_mm:
        k = _box_voxels(mm, voxel_scale_mm)
        m = ndi.uniform_filter(vol, size=k, mode="nearest")
        m2 = ndi.uniform_filter(vol * vol, size=k, mode="nearest")
        sd = np.sqrt(np.maximum(m2 - m * m, 0.0))
        cols.append(_sample(sd, node_vox))
        names.append(f"std_{mm:g}mm")
        del m, m2, sd

    for mm in spec.hessian_scales_mm:
        s = _sigma_voxels(mm, voxel_scale_mm, "hessian")
        # The sigma^2 gamma-normalisation inside _hessian_eigenvalues is already
        # the PHYSICALLY correct one when sigma is in voxels: gaussian_filter
        # differentiates w.r.t. the array index, so d_vox = d_mm * voxel^2, and
        # sigma_mm^2 * d_mm == sigma_vox^2 * d_vox exactly. No extra factor.
        ev = _hessian_eigenvalues(vol, node_vox, s)
        for j in range(3):
            cols.append(ev[:, j])
            names.append(f"hess{j + 1}_{mm:g}mm")

    if spec.include_depth:
        cols.append(surface_depth(mask, node_vox, voxel_scale_mm))
        names.append("depth")

    feats = np.stack(cols, axis=1).astype(np.float32)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    log.info("template features: %d nodes x %d descriptors (%s)",
             feats.shape[0], feats.shape[1], ", ".join(names))
    return feats, names
