"""``ParcelField`` -- the built parcellation as a queryable, serializable object.

This is the single artifact the rest of the pipeline consumes. It is built once
from the reference image (minutes), cached to a ``.npz``, and then answers, for
any batch of standardized coordinates:

  * which parcel am I in,
  * my sparse membership vector over parcels,
  * how far am I from my parcel's border.

Lookup is nearest-node: the parcellation is defined on the strided template grid,
queries are arbitrary MALDI voxels. That is the same approximation the rest of the
pipeline already makes when it snaps measurements onto template geometry, and at
stride 4 the node spacing is 0.1 mm.

Queries are memoized on tensor *content*, because the hot call pattern is the same
fixed coordinate set (inducing points) on every optimizer step. The cache is a
bounded LRU keyed by a hash of the bytes, so churning minibatch coordinates cannot
leak memory and the fixed set never evicts.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def publish_file(src, dest) -> Path:
    """Copy a locally-written file to ``dest``, atomically where the filesystem allows.

    Two constraints have to be satisfied at once:

    * **S3-fuse mounts cannot seek.** ``np.savez_compressed`` writes a ZIP, and
      closing a ZIP seeks back to write the central directory, so writing the
      archive *directly* to /s3/... dies with ``OSError: [Errno 95] Operation not
      supported``. The archive must therefore be built on local disk and then
      copied over sequentially, which S3 does support.
    * **Concurrent writers.** A batch sweep starts many jobs at once, none sees a
      cached field, and all build and write the same path. Staging under a
      per-process name and renaming means the losers just rewrite identical bytes
      and no reader ever sees a partial file.

    Rename is not universally supported on fuse mounts either, so the atomic path
    is attempted and a plain copy is the fallback. The fallback keeps a small
    window where a reader could see a partially written file -- unavoidable on a
    filesystem without rename, and harmless in practice because a torn read fails
    loudly at ``np.load`` rather than returning wrong data.
    """
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staged = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(src, staged)
        os.replace(staged, dest)
        return dest
    except OSError as e:
        log.debug("atomic publish unavailable on %s (%s); falling back to copy",
                  dest.parent, e)
        try:
            staged.unlink()
        except OSError:
            pass
    shutil.copyfile(src, dest)
    return dest


@dataclass
class ParcelSample:
    """Per-query-point parcel geometry."""
    label: np.ndarray        # (M,) int32
    mem_idx: np.ndarray      # (M, J) int32
    mem_val: np.ndarray      # (M, J) float32, rows sum to 1
    d_border_z: np.ndarray   # (M,) float32, globally z-scored
    d_border_rel: np.ndarray # (M,) float32 in [0, 1]


class ParcelField:
    """A reference-derived parcellation sampled at arbitrary coordinates."""

    def __init__(self, node_coords_std: np.ndarray, labels: np.ndarray,
                 mem_idx: np.ndarray, mem_val: np.ndarray,
                 d_border_z: np.ndarray, d_border_rel: np.ndarray,
                 nearest_other: np.ndarray, n_parcels: int,
                 node_vox: np.ndarray | None = None,
                 volume_shape: tuple | None = None,
                 meta: dict | None = None, cache_size: int = 16):
        self.node_coords = np.ascontiguousarray(node_coords_std, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int32)
        self.mem_idx = np.asarray(mem_idx, dtype=np.int32)
        self.mem_val = np.asarray(mem_val, dtype=np.float32)
        self.d_border_z = np.asarray(d_border_z, dtype=np.float32)
        self.d_border_rel = np.asarray(d_border_rel, dtype=np.float32)
        self.nearest_other = np.asarray(nearest_other, dtype=np.int32)
        self.n_parcels = int(n_parcels)
        self.node_vox = None if node_vox is None else np.asarray(node_vox, dtype=np.int32)
        self.volume_shape = None if volume_shape is None else tuple(int(s) for s in volume_shape)
        self.meta = dict(meta or {})
        self._tree = cKDTree(self.node_coords)
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_size = int(cache_size)

    # ---------------------------------------------------------------- lookup
    def node_index(self, coords_std: np.ndarray) -> np.ndarray:
        """(M,) index of the nearest template node for each query coordinate."""
        q = np.ascontiguousarray(np.asarray(coords_std, dtype=np.float32))
        key = None
        if self._cache_size > 0:
            key = (q.shape, hash(q.tobytes()))
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        _, idx = self._tree.query(q, k=1)
        idx = np.asarray(idx, dtype=np.int64)
        if key is not None:
            self._cache[key] = idx
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return idx

    def sample(self, coords_std: np.ndarray) -> ParcelSample:
        """Parcel geometry at arbitrary standardized coordinates."""
        i = self.node_index(coords_std)
        return ParcelSample(
            label=self.labels[i], mem_idx=self.mem_idx[i], mem_val=self.mem_val[i],
            d_border_z=self.d_border_z[i], d_border_rel=self.d_border_rel[i],
        )

    def dense_memberships(self, coords_std: np.ndarray) -> np.ndarray:
        """(M, n_parcels) dense membership matrix at the query coordinates."""
        s = self.sample(coords_std)
        m = np.zeros((s.mem_idx.shape[0], self.n_parcels), dtype=np.float32)
        np.put_along_axis(m, s.mem_idx.astype(np.intp), s.mem_val, axis=1)
        return m

    # ------------------------------------------------------------ inspection
    def to_volume(self, which: str = "labels", background=-1) -> np.ndarray:
        """Paint a per-node field back onto the strided grid (for napari/plots)."""
        if self.node_vox is None or self.volume_shape is None:
            raise ValueError("ParcelField was built without node_vox/volume_shape")
        src = {"labels": self.labels, "d_border_z": self.d_border_z,
               "d_border_rel": self.d_border_rel,
               "nearest_other": self.nearest_other}[which]
        vol = np.full(self.volume_shape, background, dtype=src.dtype)
        z, y, x = self.node_vox.T
        vol[z, y, x] = src
        return vol

    # ------------------------------------------------------------- serialize
    def save(self, path) -> Path:
        """Write the field, atomically.

        A batch sweep starts many jobs at once and they all see no cached field,
        so they all build it and all write the same path. Writing in place would
        interleave them into a corrupt archive. Writing to a per-process temp file
        and renaming means the losers of the race simply overwrite with identical
        bytes, and no reader ever sees a partial file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Build the archive on LOCAL disk: np.savez_compressed seeks, and S3-fuse
        # mounts do not support seeking. Then publish it across.
        with tempfile.TemporaryDirectory(prefix="parcelgp-") as td:
            tmp = Path(td) / "field.npz"
            np.savez_compressed(
                tmp,
                node_coords=self.node_coords, labels=self.labels,
                mem_idx=self.mem_idx, mem_val=self.mem_val,
                d_border_z=self.d_border_z, d_border_rel=self.d_border_rel,
                nearest_other=self.nearest_other,
                n_parcels=np.array(self.n_parcels),
                node_vox=(self.node_vox if self.node_vox is not None
                          else np.zeros((0, 3), np.int32)),
                volume_shape=np.array(self.volume_shape or (0, 0, 0)),
                meta=np.array(json.dumps(self.meta)),
            )
            mb = tmp.stat().st_size / 1e6
            publish_file(tmp, path)
        log.info("wrote parcel field -> %s (%d nodes, %d parcels, %.0f MB)",
                 path, self.node_coords.shape[0], self.n_parcels, mb)
        return path

    @classmethod
    def load(cls, path) -> "ParcelField":
        d = np.load(Path(path), allow_pickle=False)
        node_vox = d["node_vox"]
        shape = tuple(int(s) for s in d["volume_shape"])
        return cls(
            node_coords_std=d["node_coords"], labels=d["labels"],
            mem_idx=d["mem_idx"], mem_val=d["mem_val"],
            d_border_z=d["d_border_z"], d_border_rel=d["d_border_rel"],
            nearest_other=d["nearest_other"], n_parcels=int(d["n_parcels"]),
            node_vox=(node_vox if node_vox.size else None),
            volume_shape=(shape if any(shape) else None),
            meta=json.loads(str(d["meta"])),
        )
