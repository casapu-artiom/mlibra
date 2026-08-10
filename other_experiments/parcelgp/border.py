"""Distance from each node to its parcel's border.

``d_border(i)`` is the Euclidean distance from node ``i`` to the nearest voxel
belonging to a *different* parcel. It is 0 on a border and grows into a parcel's
interior -- the parcel analogue of depth-into-tissue.

Background is not a target by default (``include_background=False``): the tissue
surface is a different kind of boundary and is already carried as the ``depth``
feature, so counting it here would make every superficial voxel look
"border-adjacent" for the wrong reason. Flip the flag if you want the outer
surface to act as a parcel border too.

Computed exactly, one EDT per parcel: for parcel ``k`` the transform runs on the
complement of "voxels not in ``k``", so its value at a voxel of ``k`` is the
straight-line distance to the nearest non-``k`` voxel. ``return_indices`` gives
the identity of that nearest voxel for free, hence ``nearest_other`` -- which
parcel you are closest to across the border. K transforms over the strided volume
is a few seconds for the usual K <= 256.

Three normalizations are returned because they answer different questions:

  * ``d_border_mm``  -- physical, comparable across parcels and strides;
  * ``d_border_z``   -- globally z-scored, the right input for a stationary
    feature or an ARD input column;
  * ``d_border_rel`` -- divided by the parcel's own 95th percentile and clipped
    to [0, 1], so "deep in my parcel" means the same thing in a big parcel and a
    small one. This is the one to drive a lengthscale with, otherwise a large
    parcel gets an unboundedly long lengthscale purely for being large.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)


@dataclass
class BorderField:
    d_border_mm: np.ndarray      # (N,) float32, mm
    d_border_z: np.ndarray       # (N,) float32, globally z-scored
    d_border_rel: np.ndarray     # (N,) float32 in [0, 1], per-parcel normalized
    nearest_other: np.ndarray    # (N,) int32, parcel id across the nearest border
    stats: dict


def border_distance(labels: np.ndarray, node_vox: np.ndarray, shape: tuple,
                    voxel_scale_mm: float, n_parcels: int | None = None,
                    include_background: bool = False,
                    rel_percentile: float = 95.0) -> BorderField:
    """Per-node distance to the nearest differently-labelled voxel.

    Args:
      labels:              (N,) parcel id per node, compact ids in [0, K).
      node_vox:            (N, 3) node voxel indices, canonical order.
      shape:               (Z, Y, X) of the strided volume.
      voxel_scale_mm:      mm per strided voxel.
      include_background:  count non-tissue voxels as a border too.
      rel_percentile:      percentile used for the per-parcel normalization.
    """
    K = int(n_parcels if n_parcels is not None else labels.max() + 1)
    z, y, x = node_vox.T
    lab_vol = np.full(shape, -1, dtype=np.int32)
    lab_vol[z, y, x] = labels
    tissue = lab_vol >= 0

    if K < 2:
        raise ValueError("border_distance needs at least 2 parcels")

    d_mm = np.zeros(labels.shape[0], dtype=np.float32)
    nearest = np.full(labels.shape[0], -1, dtype=np.int32)
    node_of = {}
    for k in range(K):
        sel = labels == k
        if not sel.any():
            continue
        other = (lab_vol != k) & (tissue | include_background)
        if not other.any():
            log.warning("parcel %d has no differently-labelled voxel; d_border=inf", k)
            d_mm[sel] = np.inf
            continue
        # EDT measures distance to the nearest ZERO of its input, so feed the
        # complement of `other`; the result at a voxel of k is the distance to
        # the nearest non-k voxel.
        dist, idx = ndi.distance_transform_edt(~other, return_indices=True)
        zk, yk, xk = z[sel], y[sel], x[sel]
        d_mm[sel] = (dist[zk, yk, xk] * float(voxel_scale_mm)).astype(np.float32)
        nearest[sel] = lab_vol[idx[0][zk, yk, xk], idx[1][zk, yk, xk],
                               idx[2][zk, yk, xk]]
        node_of[k] = int(sel.sum())
        del dist, idx

    finite = np.isfinite(d_mm)
    d_z = np.zeros_like(d_mm)
    d_z[finite] = ((d_mm[finite] - d_mm[finite].mean())
                   / (d_mm[finite].std() + 1e-8)).astype(np.float32)

    d_rel = np.zeros_like(d_mm)
    for k in range(K):
        sel = (labels == k) & finite
        if not sel.any():
            continue
        p = float(np.percentile(d_mm[sel], rel_percentile))
        d_rel[sel] = np.clip(d_mm[sel] / max(p, 1e-8), 0.0, 1.0)

    stats = {
        "d_border_mm_mean": float(d_mm[finite].mean()) if finite.any() else float("nan"),
        "d_border_mm_max": float(d_mm[finite].max()) if finite.any() else float("nan"),
        "border_adjacent_frac": float((d_mm[finite] <= voxel_scale_mm).mean())
        if finite.any() else float("nan"),
        "include_background": bool(include_background),
        "n_parcels": K,
    }
    log.info("border distance: mean %.3f mm, max %.3f mm, %.1f%% of nodes are "
             "border-adjacent (<= 1 voxel)",
             stats["d_border_mm_mean"], stats["d_border_mm_max"],
             100 * stats["border_adjacent_frac"])
    return BorderField(d_border_mm=d_mm, d_border_z=d_z, d_border_rel=d_rel,
                       nearest_other=nearest, stats=stats)
