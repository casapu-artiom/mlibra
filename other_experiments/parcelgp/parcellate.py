"""Reference-only parcellation: template features -> parcels + soft memberships.

Three things distinguish this from a plain k-means over voxel features:

1. **Bilateral symmetry** (``symmetric=True``, default). The left-right axis is
   folded to |lr - midline| before clustering, so a voxel and its mirror image get
   byte-identical clustering inputs and therefore the same parcel. Without this,
   k-means splits homologous structures into unrelated ids purely by seeding
   noise, which then shows up as a spurious "border" down the midline.

2. **Contiguity repair**. k-means over voxel features is speckly: a parcel comes
   out as one big blob plus a spray of stragglers, and a distance-to-border
   feature computed on that is meaningless. Every parcel is split into connected
   components; components below ``min_component_frac`` of the parcel are
   dissolved and the freed voxels are absorbed by the nearest surviving parcel
   (exact nearest-label fill via a single EDT with ``return_indices``). This
   costs some feature fidelity and buys a partition whose borders are surfaces.

3. **Soft memberships**. A hard label makes the model discontinuous at a border
   that the reference image only weakly implies. Each node also carries a sparse
   membership vector m (top ``n_membership`` parcels, softmax over distance to
   the parcel centroids, renormalized to sum to 1). Downstream, m is what the
   coregionalization factor consumes; the hard label is kept for diagnostics and
   for the border distance.

Centroids are recomputed from the FINAL (post-repair) parcels before memberships
are derived, so ``argmax(m)`` agrees with ``labels`` for the large majority of
nodes -- the reported disagreement rate is a useful sanity number, not a bug.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import MiniBatchKMeans

from .volume import LR_AXIS

log = logging.getLogger(__name__)

_STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass
class Parcellation:
    """A reference-only partition of the tissue nodes."""
    labels: np.ndarray          # (N,) int32, compact ids in [0, n_parcels)
    mem_idx: np.ndarray         # (N, J) int32 parcel ids of the top-J memberships
    mem_val: np.ndarray         # (N, J) float32, rows sum to 1
    centroids: np.ndarray       # (n_parcels, D) in the clustering feature space
    n_parcels: int

    def dense_memberships(self) -> np.ndarray:
        """(N, n_parcels) dense membership matrix. Only for small ``n_parcels``."""
        m = np.zeros((self.labels.shape[0], self.n_parcels), dtype=np.float32)
        np.put_along_axis(m, self.mem_idx.astype(np.intp), self.mem_val, axis=1)
        return m


def clustering_space(feats: np.ndarray, coords_std: np.ndarray,
                     spatial_weight: float, symmetric: bool,
                     midline_std: float | None = None,
                     feature_names=None, normalize_blocks: bool = False
                     ) -> np.ndarray:
    """Join appearance features with (optionally folded) spatial coordinates.

    ``spatial_weight`` scales the z-scored coordinate block: 0 is pure appearance
    clustering (maximally speckly), larger values buy spatial compactness at the
    cost of feature fidelity. The coordinate block is z-scored AFTER folding, so
    the fold does not shrink the left-right axis relative to the others.

    ``normalize_blocks`` divides each descriptor block by sqrt(its feature count).
    Without it, k-means weights a descriptor by how many numbers it emits -- the
    Hessian gets 37.5% of the squared distance and depth 6.3%, for no anatomical
    reason -- because squared Euclidean distance sums over columns and every
    column is unit variance. With it, each block contributes 1 and all five count
    equally (20% each). The coordinate block is normalized the same way, which
    leaves ``spatial_weight``'s meaning almost unchanged (at sw=3, coords go from
    62.8% to 64.3% of the distance).

    Equal-per-block is itself a choice, not a ground truth -- the point is that
    the weighting becomes explicit rather than an accident of arithmetic. Hence
    the flag: measure both.
    """
    c = np.array(coords_std, dtype=np.float32, copy=True)
    if symmetric:
        mid = 0.0 if midline_std is None else float(midline_std)
        c[:, LR_AXIS] = np.abs(c[:, LR_AXIS] - mid)

    f = np.asarray(feats, dtype=np.float32)
    if normalize_blocks and f.shape[1]:
        if feature_names is None:
            raise ValueError("normalize_blocks needs feature_names to group columns")
        from .features import feature_blocks
        blocks = feature_blocks(feature_names)
        if blocks.shape[0] != f.shape[1]:
            raise ValueError(
                f"feature_names ({blocks.shape[0]}) != feature columns ({f.shape[1]})")
        f = f.copy()
        for b in np.unique(blocks):
            sel = blocks == b
            f[:, sel] /= np.sqrt(float(sel.sum()))
        logging.info("block normalization ON: %s",
                     ", ".join(f"{b}x{int((blocks == b).sum())}"
                               for b in np.unique(blocks)))

    if spatial_weight <= 0:
        return np.ascontiguousarray(f, dtype=np.float32)
    c = (c - c.mean(0)) / (c.std(0) + 1e-8)
    if normalize_blocks:
        c = c / np.sqrt(float(c.shape[1]))
    return np.ascontiguousarray(
        np.hstack([float(spatial_weight) * c, f]), dtype=np.float32)


def _label_volume(labels: np.ndarray, node_vox: np.ndarray,
                  shape: tuple) -> np.ndarray:
    vol = np.full(shape, -1, dtype=np.int32)
    z, y, x = node_vox.T
    vol[z, y, x] = labels
    return vol


def enforce_contiguity(labels: np.ndarray, node_vox: np.ndarray, shape: tuple,
                       min_component_frac: float = 0.15,
                       min_component_voxels: int = 20):
    """Dissolve speckle: keep only the large connected components of each parcel.

    A component survives if it holds at least ``min_component_frac`` of its
    parcel's voxels AND at least ``min_component_voxels`` voxels. Voxels in
    dissolved components are reassigned to the nearest surviving voxel's parcel.

    Returns ``(new_labels (N,) int32, n_parcels, stats dict)``. Parcel ids are
    compacted, so a parcel that loses every component simply disappears and
    ``n_parcels`` drops.
    """
    vol = _label_volume(labels, node_vox, shape)
    keep = np.zeros(shape, dtype=bool)
    n_dissolved = 0
    for k in np.unique(labels):
        m = vol == k
        total = int(m.sum())
        cc, ncc = ndi.label(m, structure=_STRUCT26)
        if ncc == 0:
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        thresh = max(int(min_component_frac * total), int(min_component_voxels))
        good = np.flatnonzero(sizes >= thresh)
        if good.size == 0:                      # nothing big enough: keep the largest
            good = np.array([int(np.argmax(sizes))])
        keep |= np.isin(cc, good)
        n_dissolved += total - int(sizes[good].sum())

    # Nearest-surviving-voxel fill. distance_transform_edt on the COMPLEMENT of
    # `keep` returns, for every voxel, the index of the nearest voxel where
    # `keep` is True -- an exact nearest-label assignment in one pass.
    if n_dissolved:
        _, idx = ndi.distance_transform_edt(~keep, return_indices=True)
        filled = vol[tuple(idx)]
        vol = np.where(keep, vol, filled)

    z, y, x = node_vox.T
    new = vol[z, y, x]
    uniq, new_compact = np.unique(new, return_inverse=True)
    stats = {
        "dissolved_voxels": int(n_dissolved),
        "dissolved_frac": float(n_dissolved / max(labels.shape[0], 1)),
        "parcels_before": int(np.unique(labels).size),
        "parcels_after": int(uniq.size),
    }
    log.info("contiguity repair: dissolved %d/%d voxels (%.1f%%), parcels %d -> %d",
             stats["dissolved_voxels"], labels.shape[0], 100 * stats["dissolved_frac"],
             stats["parcels_before"], stats["parcels_after"])
    return new_compact.astype(np.int32), int(uniq.size), stats


def _soft_memberships(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray,
                      n_membership: int, temperature: float, chunk: int = 100_000):
    """Sparse softmax memberships over distance to the parcel centroids.

    The softmax temperature is ``temperature * median(distance to own centroid)``,
    i.e. it adapts to the scale of the clustering space instead of being an
    arbitrary constant. Returns ``(mem_idx (N, J) int32, mem_val (N, J) float32)``.
    """
    n, K = X.shape[0], centroids.shape[0]
    J = int(min(max(n_membership, 1), K))
    c_sq = (centroids ** 2).sum(1)

    own = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        b = X[s:s + chunk]
        d2 = np.maximum(
            (b ** 2).sum(1)[:, None] + c_sq[None, :] - 2.0 * (b @ centroids.T), 0.0)
        own[s:s + b.shape[0]] = np.take_along_axis(
            d2, labels[s:s + b.shape[0], None].astype(np.intp), axis=1).ravel()
    tau2 = float(temperature) ** 2 * max(float(np.median(np.sqrt(own))) ** 2, 1e-12)

    mem_idx = np.empty((n, J), dtype=np.int32)
    mem_val = np.empty((n, J), dtype=np.float32)
    for s in range(0, n, chunk):
        b = X[s:s + chunk]
        d2 = np.maximum(
            (b ** 2).sum(1)[:, None] + c_sq[None, :] - 2.0 * (b @ centroids.T), 0.0)
        top = np.argpartition(d2, J - 1, axis=1)[:, :J] if J < K else \
            np.repeat(np.arange(K)[None, :], b.shape[0], axis=0)
        td2 = np.take_along_axis(d2, top, axis=1)
        w = np.exp(-(td2 - td2.min(1, keepdims=True)) / (2.0 * tau2))
        w /= w.sum(1, keepdims=True)
        order = np.argsort(-w, axis=1)
        mem_idx[s:s + b.shape[0]] = np.take_along_axis(top, order, axis=1)
        mem_val[s:s + b.shape[0]] = np.take_along_axis(w, order, axis=1)
    return mem_idx, mem_val


def parcellate(feats: np.ndarray, coords_std: np.ndarray, node_vox: np.ndarray,
               shape: tuple, n_parcels: int = 64,
               spatial_weight: float = 1.0, symmetric: bool = True,
               midline_std: float | None = None,
               contiguity: bool = True, min_component_frac: float = 0.15,
               min_component_voxels: int = 20,
               n_membership: int = 4, temperature: float = 1.0,
               fit_subsample: int = 200_000, seed: int = 0,
               feature_names=None, normalize_blocks: bool = False,
               ) -> tuple[Parcellation, dict]:
    """Cluster the template into parcels. Returns ``(Parcellation, stats)``.

    Nothing here reads lipids or an atlas -- the only inputs are the template
    features and the node geometry.
    """
    X = clustering_space(feats, coords_std, spatial_weight, symmetric, midline_std,
                         feature_names=feature_names,
                         normalize_blocks=normalize_blocks)
    n = X.shape[0]

    rng = np.random.default_rng(seed)
    m = int(fit_subsample) if 0 < int(fit_subsample) < n else n
    train = X if m >= n else X[rng.choice(n, m, replace=False)]
    km = MiniBatchKMeans(n_clusters=int(n_parcels), random_state=int(seed),
                         n_init=10, batch_size=4096, max_iter=300)
    km.fit(train)
    labels = km.predict(X).astype(np.int32)
    log.info("k-means: %d nodes -> %d parcels (trained on %d, spatial_weight=%g, "
             "symmetric=%s, seed=%d)", n, n_parcels, m, spatial_weight, symmetric, seed)

    stats = {"n_nodes": int(n), "n_parcels_requested": int(n_parcels),
             "spatial_weight": float(spatial_weight), "symmetric": bool(symmetric),
             "normalize_blocks": bool(normalize_blocks), "seed": int(seed)}

    if contiguity:
        labels, K, cstats = enforce_contiguity(
            labels, node_vox, shape, min_component_frac, min_component_voxels)
        stats.update(cstats)
    else:
        uniq, labels = np.unique(labels, return_inverse=True)
        labels, K = labels.astype(np.int32), int(uniq.size)

    # Recompute centroids on the FINAL parcels so memberships and labels agree.
    centroids = np.zeros((K, X.shape[1]), dtype=np.float32)
    counts = np.bincount(labels, minlength=K).astype(np.float32)
    np.add.at(centroids, labels, X)
    centroids /= np.maximum(counts, 1.0)[:, None]

    mem_idx, mem_val = _soft_memberships(X, centroids, labels, n_membership, temperature)
    agree = float((mem_idx[:, 0] == labels).mean())
    stats["n_parcels"] = int(K)
    stats["membership_argmax_agreement"] = agree
    stats["parcel_size_min"] = int(counts.min())
    stats["parcel_size_median"] = float(np.median(counts))
    log.info("memberships: top-%d, argmax agrees with label on %.1f%% of nodes; "
             "parcel sizes min=%d median=%.0f",
             mem_idx.shape[1], 100 * agree, counts.min(), np.median(counts))

    return Parcellation(labels=labels, mem_idx=mem_idx, mem_val=mem_val,
                        centroids=centroids, n_parcels=K), stats
