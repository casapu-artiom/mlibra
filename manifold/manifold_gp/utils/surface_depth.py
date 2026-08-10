#!/usr/bin/env python
# encoding: utf-8
"""Distance-to-surface (depth-into-tissue) feature for the surface kernels.

``d_surface`` is the Euclidean distance from each tissue voxel to the nearest
background voxel (the tissue↔background boundary), i.e. how deep a point sits.
Computed once at training start from the reference mask, standardized, and then
sampled at arbitrary query coordinates (nearest graph node) so it can be appended
as the last input column that the ``Surface*`` kernels read.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def node_surface_depth(sub_volume: np.ndarray, threshold: float,
                       node_voxel_idx: np.ndarray, mm_per_voxel: float = 1.0
                       ) -> np.ndarray:
    """Per-node distance to the tissue surface (mm).

    Args:
      sub_volume:     (Z,Y,X) reference intensity on the (strided) grid.
      threshold:      tissue = sub_volume > threshold.
      node_voxel_idx: (N,3) int voxel indices of the graph nodes (np.argwhere
                      order — matches reference_ccf_from_subvolume).
      mm_per_voxel:   physical size of one (strided) voxel; scales the EDT to mm.
                      Correlation-invariant, so any positive value is fine.

    Returns:
      (N,) float32 depth: 0 at the surface, larger deeper in.
    """
    mask = sub_volume > threshold
    d = ndi.distance_transform_edt(mask) * float(mm_per_voxel)
    z, y, x = node_voxel_idx.T
    return d[z, y, x].astype(np.float32)


class DepthSampler:
    """Standardized ``d_surface`` sampled at arbitrary coordinates.

    Holds the graph nodes' coordinates + depths; ``sample``/``append`` map any
    query coordinate to the depth of its nearest node, z-scored with the node
    statistics (so train and test share one frame). Use ``append`` to add the
    depth column that the ``Surface*`` kernels expect as the last input dim.
    """

    def __init__(self, node_coords: np.ndarray, node_depth: np.ndarray,
                 cache_size: int = 16):
        self.tree = cKDTree(np.asarray(node_coords, dtype=np.float32))
        d = np.asarray(node_depth, dtype=np.float64)
        self.mean = float(d.mean())
        self.std = float(d.std()) or 1.0
        self.depth_z = ((d - self.mean) / self.std).astype(np.float32)
        # Memoize the nearest-node query by input content: the kernel calls
        # sample() on the SAME coords every training step (fixed inducing points
        # -> K_uu, and repeated predict coords), so this turns the recurring
        # O(M log N) cKDTree query into a dict hit. Bounded LRU so churning
        # minibatch coords can't leak; the hot inducing key is touched every step
        # (K_uu) and so never evicted.
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_size = int(cache_size)

    def sample(self, coords: np.ndarray) -> np.ndarray:
        """(M,) standardized depth for (M, D) query coords (nearest node).
        Memoized on the coords' content (bounded LRU)."""
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        key = None
        if self._cache_size > 0:
            key = (coords.shape, hash(coords.tobytes()))
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        _, idx = self.tree.query(coords, k=1)
        out = self.depth_z[idx]
        if key is not None:
            self._cache[key] = out
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return out

    def append(self, coords: torch.Tensor) -> torch.Tensor:
        """Append the standardized depth column to a (..., D) coord tensor →
        (..., D+1), so the last column is the ``Surface*`` kernels' depth feature."""
        c = coords.detach().cpu().numpy()
        flat = c.reshape(-1, c.shape[-1])
        dz = self.sample(flat).reshape(*c.shape[:-1], 1)
        out = np.concatenate([c, dz], axis=-1)
        return torch.as_tensor(out, dtype=coords.dtype, device=coords.device)
