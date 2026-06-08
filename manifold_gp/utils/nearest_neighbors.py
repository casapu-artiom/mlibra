#!/usr/bin/env python
# encoding: utf-8

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import faiss
from torch_sparse import coalesce
import faiss.contrib.torch_utils

class NearestNeighbors():
    def __init__(self, x=None, nlist=1) -> None:
        self.min_ivf = 5000

        if x is not None:
            self.train(x, nlist)

    def train(self, x, nlist=1):
        self.x = x
        (n, d) = self.x.shape
        if x.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2) if n >= self.min_ivf else faiss.GpuIndexFlatL2(res, d)
        else:
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist) if n >= self.min_ivf else faiss.IndexFlatL2(d)

        if n >= self.min_ivf:
            assert not self.index.is_trained
            self.index.train(self.x)
            assert self.index.is_trained
        self.index.add(self.x)

        return self

    def search(self, x, k, nprobe=1):
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        # Faiss-CPU expects numpy arrays, while GPU can take tensors
        return self.index.search(x, k)

    def graph(self, k, symmetric=True, self_loop=False, nprobe=1):
        val, idx = self.search(self.x, k, nprobe)

        if not self_loop:
            val, idx = val[:, 1:], idx[:, 1:]

        rows, cols = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1]).to(self.x.device), idx.reshape(1, -1).squeeze()
        val = val.reshape(1, -1).squeeze()

        if symmetric:
            split = cols > rows
            rows, cols = torch.cat([rows[split], cols[~split]], dim=0), torch.cat([cols[split], rows[~split]], dim=0)
            idx, val = coalesce(torch.stack([rows, cols], dim=0), torch.cat([val[split], val[~split]]), self.x.shape[0], self.x.shape[0], op='mean')
        else:
            idx = torch.stack([rows, cols], dim=0)

        return idx, val

    @property
    def min_ivf(self):
        return self._min_ivf

    @min_ivf.setter
    def min_ivf(self, value):
        self._min_ivf = value
 

# ---------------------------------------------------------------------------
# Cache key helper
# ---------------------------------------------------------------------------
def make_key(parts: Dict[str, Any]) -> str:
    """Compose a filesystem-safe cache key from a config dict.
 
    Sorted by key for stability. Floats are formatted with %.6g so trivial
    numeric noise doesn't change the key.
 
    Example:
        make_key({"template": "allen25", "stride": 4, "method": "faiss", "k": 5})
        -> "k=5_method=faiss_stride=4_template=allen25"
    """
    pieces = []
    for k in sorted(parts.keys()):
        v = parts[k]
        if isinstance(v, float):
            v = f"{v:.6g}"
        pieces.append(f"{k}={v}")
    return "_".join(pieces).replace("/", "-").replace(" ", "")
 
 
# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
@dataclass
class GraphFingerprint:
    """Structural summary stored alongside cached arrays.
 
    Strict checks: method label.
    Diagnostic-only: n_nodes/n_edges (used in log lines), coord and edge stats.
    """
    n_nodes: int
    n_edges: int
    coords_dim: int
    method: str
    coords_mean: List[float]
    coords_std: List[float]
    edge_value_mean: float
    edge_value_std: float
 
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
 
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphFingerprint":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})
 
 
# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class KnnGraphCache:
    """Disk cache and trainer for KNN graphs.
 
    The cache returns a fully-formed `NearestNeighbors` instance (so the
    kernel can plumb it straight into `self.knn`), along with the
    `edge_index` and `edge_value` that drive the Laplacian.
    """
 
    def __init__(self, cache_dir: Path,
                 strict_method: bool = False,
                 verbose: bool = True):
        self.cache_dir = Path(cache_dir)
        self.strict_method = strict_method
        self.verbose = verbose
 
    # ----------------------------------------------------------- paths/io
    @staticmethod
    def _paths(cache_dir: Path, key: str) -> Tuple[Path, Path]:
        return (
            cache_dir / f"{key}.knngraph.npz",
            cache_dir / f"{key}.knngraph.meta.json",
        )
 
    @staticmethod
    def _fingerprint(coords: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_value: torch.Tensor,
                     method: str) -> GraphFingerprint:
        c = coords.detach().float().cpu()
        ev = edge_value.detach().float().cpu()
        return GraphFingerprint(
            n_nodes=int(c.shape[0]),
            n_edges=int(edge_index.shape[-1]),
            coords_dim=int(c.shape[1]),
            method=str(method),
            coords_mean=c.mean(dim=0).tolist(),
            coords_std=c.std(dim=0).tolist(),
            edge_value_mean=float(ev.mean().item()),
            edge_value_std=float(ev.std().item()),
        )
     
    def save(self, key, knn, edge_index, edge_value, method, nlist, extra=None):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path, meta_path = self._paths(self.cache_dir, key)

        fp = self._fingerprint(knn.x, edge_index, edge_value, method)
        meta = {
            "key": key, "method": method, "nlist": int(nlist),
            "fingerprint": fp.to_dict(),
            "extra": extra or {},
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Stage to a regular filesystem, then move into place. np.savez writes a
        # zip and zipfile.close() seeks back to patch the central directory --
        # FUSE-backed S3 mounts (s3fs/goofys/mountpoint-s3) don't support
        # seeks on write-open files (Errno 95 EOPNOTSUPP), so we can't write
        # directly to the cache dir if it's on one. /tmp is local-disk scratch.
        import tempfile, shutil
        with tempfile.TemporaryDirectory(prefix="knngraph_") as tmpd:
            tmp_npz = Path(tmpd) / npz_path.name
            np.savez(
                tmp_npz,
                coords=knn.x.detach().cpu().numpy(),
                edge_index=edge_index.detach().cpu().numpy(),
                edge_value=edge_value.detach().cpu().numpy(),
            )
            shutil.copyfile(tmp_npz, npz_path)

        # JSON is fine to write directly -- it's a single sequential write,
        # no seeks involved.
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            print(f"[knn_graph] saved -> {npz_path}")
        return npz_path

 
    def load(self, key: str, device=None
             ) -> Optional[Tuple["NearestNeighbors", torch.Tensor, torch.Tensor,
                                 GraphFingerprint, Dict]]:
        """Load and reconstruct a NearestNeighbors instance from disk.
 
        Returns (knn, edge_index, edge_value, fingerprint, meta) or None
        if the cache files don't exist.
        """
        npz_path, meta_path = self._paths(self.cache_dir, key)
        if not npz_path.exists() or not meta_path.exists():
            return None
 
        with open(meta_path) as f:
            meta = json.load(f)
        fp = GraphFingerprint.from_dict(meta["fingerprint"])
        nlist = int(meta.get("nlist", 1))
 
        data = np.load(npz_path)
        coords = torch.from_numpy(data["coords"]).float()
        edge_index = torch.from_numpy(data["edge_index"]).long()
        edge_value = torch.from_numpy(data["edge_value"]).float()
 
        # Move to device BEFORE constructing NearestNeighbors so the FAISS
        # index lives on the right device (the GPU vs CPU branch in
        # NearestNeighbors.train looks at x.device.type).
        if device is not None:
            coords = coords.to(device)
            edge_index = edge_index.to(device)
            edge_value = edge_value.to(device)
        coords = coords.contiguous()
 
        knn = NearestNeighbors(coords, nlist=nlist)
        return knn, edge_index, edge_value, fp, meta
 
    # ---------------------------------------------------------- training
    def train_or_load(
        self,
        key: str,
        *,
        method: str,
        k: int,
        nlist: int = 1,
        nprobe: int = 1,
        symmetric: bool = True,
        # Coordinates -- required for both methods, in whatever space the
        # downstream kernel will be queried with at prediction time.
        coords: Optional[torch.Tensor] = None,
        # anatomical_atlas-specific (mask + atlas lookups + grid adjacency)
        volume: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        atlas_volume: Optional[np.ndarray] = None,
        connectivity: int = 1,
        # Common
        extra: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False,
        device=None,
    ) -> Tuple["NearestNeighbors", torch.Tensor, torch.Tensor]:
        """Train (or load from cache) a KNN graph.
 
        Returns
        -------
        knn:        NearestNeighbors with FAISS index built over the graph
                    nodes, ready to be plumbed into RiemannKernel(knn=...).
        edge_index: (2, E) long, PyG-style.
        edge_value: (E,) float, squared distances in `coords`'s coordinate
                    system.
 
        Coordinate system
        -----------------
        You normalize `coords` before calling. The cache does no coordinate
        math -- whatever space you pass `coords` in, that's the space the
        graph and the FAISS index live in. At prediction time
        `kernel.knn.search(test_x, k)` will compute distances in the same
        space, so make sure your test points use the same normalization.
 
        method='faiss'
            Pure Euclidean KNN over `coords`. No `volume` / atlas needed.
 
        method='anatomical_atlas'
            Per-region KNN + grid-adjacent cross-region edges. The mask,
            atlas lookup, grid-adjacency check, and per-region KNN
            topology are all selected in voxel-index space. The edge
            *weights* (squared distances) and the prediction-time FAISS
            index live in `coords`'s space. Required: `volume`,
            `threshold`, `atlas_volume`, and `coords` (one row per tissue
            voxel, in `np.where(mask)` order).
        """
        # 1. Try cache
        if not force_recompute:
            hit = self.load(key, device=device)
            if hit is not None:
                knn, ei, ev, cached_fp, _meta = hit
                if cached_fp.method != method:
                    msg = (f"[knn_graph] cache method mismatch for key={key!r}: "
                           f"cached={cached_fp.method!r}, requested={method!r}")
                    if self.strict_method:
                        raise RuntimeError(msg + " (strict_method=True)")
                    logging.warning(msg + " -- recomputing.")
                else:
                    if self.verbose:
                        print(f"[knn_graph] HIT  {self.cache_dir}/{key} "
                              f"(N={cached_fp.n_nodes:,}  E={cached_fp.n_edges:,})")
                    return knn, ei, ev
 
        # 2. Train fresh
        if self.verbose:
            print(f"[knn_graph] MISS key={key} -- training (method={method!r})")
        t0 = time.time()
 
        if method == "faiss":
            knn, ei, ev = self._train_faiss(
                coords=coords, k=k, nlist=nlist, nprobe=nprobe,
                symmetric=symmetric, device=device,
            )
        elif method == "anatomical_atlas":
            knn, ei, ev = self._train_anatomical_atlas(
                volume=volume, threshold=threshold, atlas_volume=atlas_volume,
                coords=coords,
                k=k, nlist=nlist, connectivity=connectivity,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Supported: 'faiss', 'anatomical_atlas'."
            )
 
        if self.verbose:
            print(f"[knn_graph] trained in {time.time() - t0:.1f}s "
                  f"(N={knn.x.shape[0]:,}  E={ei.shape[-1]:,})")
 
        valid_mask = (ei[0] >= 0) & (ei[1] >= 0)
        if not valid_mask.all():
            logging.warning(
                f"[KnnGraphCache] Detected {(~valid_mask).sum().item()} null neighbor padding entries (-1). "
                f"Purging them cleanly from the graph topology mapping."
            )
            ei = ei[:, valid_mask]
            ev = ev[valid_mask]
 
         # 3. Save
        self.save(key, knn, ei, ev, method=method, nlist=nlist, extra=extra)
        return knn, ei, ev
 
    # ----------------------------------------------------------- backends
    @staticmethod
    def _train_faiss(*,
                     coords: Optional[torch.Tensor],
                     k: int, nlist: int, nprobe: int,
                     symmetric: bool, device) -> Tuple["NearestNeighbors",
                                                       torch.Tensor, torch.Tensor]:
        if coords is None:
            raise ValueError("method='faiss' requires `coords`.")
        if device is not None:
            coords = coords.to(device)
        coords = coords.contiguous()
        knn = NearestNeighbors(coords, nlist=nlist)
        ei, ev = knn.graph(k, symmetric=symmetric, nprobe=nprobe)
        return knn, ei, ev
 
    @staticmethod
    def _train_anatomical_atlas(*,
                                volume: Optional[np.ndarray],
                                threshold: Optional[float],
                                atlas_volume: Optional[np.ndarray],
                                coords: Optional[torch.Tensor],
                                k: int, nlist: int, connectivity: int,
                                device) -> Tuple["NearestNeighbors",
                                                 torch.Tensor, torch.Tensor]:
        """Anatomical-atlas KNN graph.
 
        Coordinate-system note
        ----------------------
        `build_anatomical_knn` runs its per-region KNN in voxel-index space
        -- the mask, atlas lookup, grid-adjacency check, and KDTree distance
        all live there. We take its `edge_index` (the topology) and discard
        its voxel-space coords and edge values; weights and the FAISS index
        are rebuilt in `coords`'s system below.
 
        For approximately isotropic standardization (the typical brain-CCF
        case) voxel-space KNN topology and standardized-space KNN topology
        select essentially the same neighbors, so this is fine in practice.
        If you need strict consistency under anisotropic standardization,
        modify `build_anatomical_knn` to accept and use a `coords` arg.
        """
        if volume is None or threshold is None or atlas_volume is None:
            raise ValueError(
                "method='anatomical_atlas' requires `volume`, `threshold`, "
                "and `atlas_volume`."
            )
        if coords is None:
            raise ValueError(
                "method='anatomical_atlas' requires `coords` -- one row per "
                "tissue voxel (`np.where(volume > threshold)` order), in the "
                "same coordinate system the kernel will be queried with at "
                "prediction time."
            )
        from .anatomical_knn import build_anatomical_knn
 
        coords_vox, ei, _ = build_anatomical_knn(
            volume, threshold, k,
            atlas_volume=atlas_volume,
            connectivity=connectivity,
        )
        n_nodes = coords_vox.shape[0]
        if coords.shape[0] != n_nodes:
            raise ValueError(
                f"`coords` has N={coords.shape[0]} but the threshold mask "
                f"has {n_nodes} tissue voxels. The i-th row of `coords` "
                f"must correspond to the i-th tissue voxel in "
                f"`np.where(volume > threshold)` order."
            )
 
        target = device if device is not None else coords.device
        coords = coords.to(target).contiguous()
        ei = ei.to(target)
 
        # Edge weights in the user's coord system (squared Euclidean), and
        # FAISS index over the same coords.
        diffs = coords[ei[0]] - coords[ei[1]]
        ev = (diffs * diffs).sum(dim=-1)
 
        knn = NearestNeighbors(coords, nlist=nlist)
        return knn, ei, ev