"""
Anatomy-aware KNN graph construction.

Standard FAISS KNN connects voxels by Euclidean distance only, which produces
shortcut edges across sulcal gaps and ventricles. This module provides two
construction strategies for a soft-to-hard anatomical prior:

  1. ``build_anatomical_knn`` (HARD prior):
       - Each voxel has KNN edges only to other voxels in the same anatomical
         region (label).
       - Cross-region edges are added ONLY between voxels that are physically
         adjacent in the voxel grid AND both lie in real tissue.

  2. ``inflate_cross_region_edges`` (SOFT prior):
       - Take an existing graph (e.g. from FAISS) and INFLATE the edge values
         on any edge that connects two different atlas regions, by a
         user-chosen factor. Topology unchanged — graph stays connected,
         which avoids the disconnected-component failure mode of the hard
         prior. The Gaussian edge weight w=exp(-d²/2σ²) collapses
         super-exponentially with inflated d², so cross-region smoothness
         becomes very expensive but not impossible.

Both functions return ``(edge_index, edge_value)`` in the same shape your
RiemannKernel expects from its FAISS-based knn.graph(K) call, so they can
be plugged in directly.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

def label_atlas(atlas_volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Use an external atlas volume as labels, restricted to the mask."""
    labels = atlas_volume.astype(np.int32) * mask.astype(np.int32)
    n_unique = len(np.unique(labels)) - 1  # exclude 0 (background)
    logging.info(f"  Atlas: {n_unique} regions")
    return labels

# ---------------------------------------------------------------------------
# Cross-region adjacency in voxel grid
# ---------------------------------------------------------------------------
def find_anatomical_neighbors(label_volume: np.ndarray,
                              voxel_to_node: np.ndarray,
                              connectivity: int = 1
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Find pairs of voxels that:
       - are physically adjacent (6- or 26-connected) in the voxel grid, AND
       - lie in real tissue (label > 0), AND
       - have different region labels.

    Args:
      label_volume:   (Z, Y, X) int32 — 0 = background, > 0 = region
      voxel_to_node:  (Z, Y, X) int32 — maps voxel coords to node index in
                      the flat node array. -1 for background voxels.

    Returns:
      sources, targets: 1-D int arrays of node indices for adjacent
                        cross-region pairs (one direction; will be symmetrized
                        by the caller).
    """
    # Define neighbor offsets
    if connectivity == 1:
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 6-conn
    else:
        offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dz, dy, dx) != (0, 0, 0) and (dz, dy, dx) > (0, 0, 0)]  # 26-conn, dedup

    sources_all = []
    targets_all = []
    Z, Y, X = label_volume.shape

    for dz, dy, dx in offsets:
        # Slices for "self" and "neighbor in (dz,dy,dx) direction"
        s_self = (
            slice(max(0, -dz), Z + min(0, -dz)),
            slice(max(0, -dy), Y + min(0, -dy)),
            slice(max(0, -dx), X + min(0, -dx)),
        )
        s_neigh = (
            slice(max(0, dz), Z + min(0, dz)),
            slice(max(0, dy), Y + min(0, dy)),
            slice(max(0, dx), X + min(0, dx)),
        )

        lab_self = label_volume[s_self]
        lab_neigh = label_volume[s_neigh]

        # Both in tissue, but in different regions
        mask = (lab_self > 0) & (lab_neigh > 0) & (lab_self != lab_neigh)
        if not mask.any():
            continue

        nid_self = voxel_to_node[s_self][mask]
        nid_neigh = voxel_to_node[s_neigh][mask]
        sources_all.append(nid_self)
        targets_all.append(nid_neigh)

    if not sources_all:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    sources = np.concatenate(sources_all)
    targets = np.concatenate(targets_all)
    return sources, targets


# ---------------------------------------------------------------------------
# Per-region KNN
# ---------------------------------------------------------------------------
def per_region_knn(coords: np.ndarray,
                   labels: np.ndarray,
                   k: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each node, find K nearest neighbors restricted to nodes with the
    same label.

    Args:
      coords: (N, 3) float — voxel coordinates of each node
      labels: (N,)   int   — region label of each node (>0)
      k:      int          — desired number of neighbors

    Returns:
      sources, targets: (N*K_eff,) int — edge endpoints
      dists:           (N*K_eff,) float — Euclidean distances

    Notes:
      - Regions smaller than k+1 nodes get K_local = region_size - 1.
      - Regions of size 1 (isolated voxels) contribute no edges.
    """
    n = len(coords)
    sources_all = []
    targets_all = []
    dists_all = []

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if lab == 0:
            continue
        idx = np.flatnonzero(labels == lab)
        m = len(idx)
        if m < 2:
            continue

        k_local = min(k, m - 1)
        sub_coords = coords[idx]
        tree = cKDTree(sub_coords)
        # k_local + 1 because the first NN of each point is itself (distance 0)
        d, ni = tree.query(sub_coords, k=k_local + 1)
        # Drop the self-match column
        d = d[:, 1:]
        ni = ni[:, 1:]

        src_local = np.repeat(np.arange(m), k_local)
        tgt_local = ni.flatten()
        d_flat = d.flatten()

        # Map local indices back to global node indices
        sources_all.append(idx[src_local])
        targets_all.append(idx[tgt_local])
        dists_all.append(d_flat)

    sources = np.concatenate(sources_all) if sources_all else np.empty(0, dtype=np.int64)
    targets = np.concatenate(targets_all) if targets_all else np.empty(0, dtype=np.int64)
    dists = np.concatenate(dists_all) if dists_all else np.empty(0, dtype=np.float32)

    return sources.astype(np.int64), targets.astype(np.int64), dists.astype(np.float32)


def build_anatomical_knn(
    sub_volume: np.ndarray,
    threshold: float,
    k: int,
    atlas_volume: Optional[np.ndarray] = None,
    connectivity: int = 1,
    return_labels: bool = False,
):
    """Build an anatomy-aware KNN graph in the format RiemannKernel expects.

    Returns:
      coords:     (N, 3) float — node coordinates in voxel space
      edge_index: (2, E) int64 — edge endpoints (sources, targets)
      edge_value: (E,)  float  — *squared* Euclidean distances (matching the
                  convention of FAISS / NearestNeighbors.graph)
      labels (opt): (N,) int — region label of each node (if return_labels=True)
    """
    mask = sub_volume > threshold
    n_in = int(mask.sum())
    logging.info(f"Threshold > {threshold:.3f}: {n_in:,} tissue voxels")

    if atlas_volume is None:
        raise ValueError("method='atlas' requires atlas_volume")
    label_vol = label_atlas(atlas_volume, mask)

    # 2. Extract node coordinates and per-node labels
    z_arr, y_arr, x_arr = np.where(mask)
    coords = np.stack([z_arr, y_arr, x_arr], axis=1).astype(np.float32)
    n_nodes = len(coords)
    node_labels = label_vol[z_arr, y_arr, x_arr]

    # Voxel-to-node lookup table (-1 for background)
    voxel_to_node = np.full(sub_volume.shape, -1, dtype=np.int64)
    voxel_to_node[z_arr, y_arr, x_arr] = np.arange(n_nodes)

    # 3. Per-region KNN
    logging.info(f"Building per-region KNN with K={k}...")
    src_intra, tgt_intra, dist_intra = per_region_knn(coords, node_labels, k)
    logging.info(f"  Intra-region edges: {len(src_intra):,}")

    # 4. Cross-region adjacency (only at genuine anatomical interfaces)
    logging.info("Finding anatomical (grid-adjacent) cross-region edges...")
    src_inter, tgt_inter = find_anatomical_neighbors(
        label_vol, voxel_to_node, connectivity=connectivity
    )
    if len(src_inter) > 0:
        diffs = coords[src_inter] - coords[tgt_inter]
        dist_inter = np.linalg.norm(diffs, axis=1).astype(np.float32)
    else:
        dist_inter = np.empty(0, dtype=np.float32)
    logging.info(f"  Inter-region (anatomical) edges: {len(src_inter):,}")

    # 5. Combine, symmetrize
    sources = np.concatenate([src_intra, tgt_intra, src_inter, tgt_inter])
    targets = np.concatenate([tgt_intra, src_intra, tgt_inter, src_inter])
    dists = np.concatenate([dist_intra, dist_intra, dist_inter, dist_inter])

    # 6. Deduplicate edges (some KNN pairs may already be reciprocal,
    # and inter-region appears twice from the symmetrize step above —
    # find_anatomical_neighbors already returns each pair once, but the
    # symmetrize duplicates it; we want each undirected edge represented
    # once in each direction).
    # Use a stable hash: sort each pair, keep unique on (min, max).
    pair_min = np.minimum(sources, targets)
    pair_max = np.maximum(sources, targets)
    keys = pair_min.astype(np.int64) * np.int64(n_nodes) + pair_max.astype(np.int64)
    _, unique_idx = np.unique(keys, return_index=True)
    sources_u = sources[unique_idx]
    targets_u = targets[unique_idx]
    dists_u = dists[unique_idx]

    # Re-symmetrize so the graph has both (i, j) and (j, i)
    sources_full = np.concatenate([sources_u, targets_u])
    targets_full = np.concatenate([targets_u, sources_u])
    dists_full = np.concatenate([dists_u, dists_u])

    # 7. Pack into RiemannKernel format
    edge_index = np.stack([sources_full, targets_full], axis=0)
    edge_value = (dists_full ** 2)  # squared, to match FAISS convention

    # Diagnostics
    if len(dist_intra) > 0:
        med_intra = float(np.median(dist_intra))
        max_intra = float(np.max(dist_intra))
        logging.info(f"  Intra-region edge length: median={med_intra:.2f}, max={max_intra:.2f}")
    if len(dist_inter) > 0:
        med_inter = float(np.median(dist_inter))
        max_inter = float(np.max(dist_inter))
        logging.info(f"  Inter-region edge length: median={med_inter:.2f}, max={max_inter:.2f}")

    out = (
        coords,
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_value, dtype=torch.float32),
    )
    if return_labels:
        out = out + (node_labels,)
    return out


# ---------------------------------------------------------------------------
# Soft anatomical prior: inflate cross-region edges of an existing graph
# ---------------------------------------------------------------------------
def inflate_cross_region_edges(
    edge_index: torch.Tensor,
    edge_value: torch.Tensor,
    node_labels: np.ndarray,
    inflation: float = 10.0,
    treat_zero_as_cross: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Reweight an existing kNN graph using atlas labels — a SOFT
    anatomical prior.

    Takes any pre-built graph (typically from FAISS or another kNN
    construction) and multiplies the edge value (squared distance) by
    ``inflation`` on any edge whose two endpoints lie in different
    atlas regions. Topology is unchanged: the graph stays connected
    (assuming the input was), so the graph Laplacian has exactly one
    zero eigenvalue and the resulting spectral kernel is well-conditioned.

    The interaction with the Gaussian edge weight ``w_ij = exp(-d²_ij /
    (2 σ²))`` is the point of this transformation: inflating ``d²`` by
    a factor F shrinks ``w_ij`` from ``exp(-d²/(2σ²))`` to
    ``exp(-F·d²/(2σ²))``. For typical within-region edges where
    ``d²/(2σ²) ≈ 0.5`` (median), inflation=10 reduces cross-region
    weights from ≈ 0.6 to ≈ 0.007 — about 100× weaker, but non-zero.
    So information still flows across boundaries; just slowly.

    This is a drop-in soft alternative to ``build_anatomical_knn``:
    keep the FAISS topology (always connected → no NaN training
    failures from disconnected components) but use the atlas to
    discourage cross-region smoothness in the prior.

    Args:
      edge_index:   (2, E) long tensor of edge endpoints, as returned
                    by a faiss or knn graph builder.
      edge_value:   (E,) float tensor of squared distances (same
                    convention as ``build_anatomical_knn``).
      node_labels:  (N,) int array of atlas labels (0 = background,
                    >0 = region). MUST have length == number of nodes
                    in the graph (i.e. max(edge_index) < len(node_labels)).
      inflation:    multiplier on squared distance for cross-region
                    edges. 1.0 = no change. 10.0 = mild soft prior.
                    100.0 = strong soft prior. Effectively unbounded —
                    very large values approach the hard-prior limit
                    but still keep the graph connected.
      treat_zero_as_cross: if True (default), edges that touch an
                    atlas background voxel (label 0) are also inflated.
                    Background voxels are usually empty space adjacent
                    to the tissue and shouldn't carry "free" smoothness.

    Returns:
      new_edge_index: (2, E) — same as input, returned as a convenience
                       for chaining.
      new_edge_value: (E,)   — cloned, with cross-region entries multiplied.
      info:           dict with diagnostics ('n_cross', 'n_total',
                       'frac_cross', 'inflation') for logging.
    """
    if edge_index.numel() == 0:
        return edge_index, edge_value, {
            "n_cross": 0, "n_total": 0, "frac_cross": 0.0,
            "inflation": float(inflation),
        }

    n_nodes = int(edge_index.max().item()) + 1
    if node_labels.shape[0] < n_nodes:
        raise ValueError(
            f"node_labels has {node_labels.shape[0]} entries but graph "
            f"references node indices up to {n_nodes - 1}. Did you pass "
            f"labels for the right node set?"
        )

    labels_t = torch.as_tensor(node_labels, dtype=torch.long,
                               device=edge_index.device)
    src_labels = labels_t[edge_index[0]]
    tgt_labels = labels_t[edge_index[1]]
    cross_region = (src_labels != tgt_labels)
    if treat_zero_as_cross:
        cross_region = cross_region | (src_labels == 0) | (tgt_labels == 0)

    n_cross = int(cross_region.sum().item())
    n_total = int(cross_region.numel())

    new_edge_value = edge_value.clone()
    if inflation != 1.0 and n_cross > 0:
        new_edge_value[cross_region] = (
            new_edge_value[cross_region] * float(inflation)
        )

    info = {
        "n_cross": n_cross,
        "n_total": n_total,
        "frac_cross": n_cross / max(n_total, 1),
        "inflation": float(inflation),
    }
    logging.info(
        f"  inflate_cross_region_edges: {n_cross:,} / {n_total:,} "
        f"({100 * info['frac_cross']:.1f}%) cross-region edges, "
        f"inflated ×{inflation:g}"
    )
    return edge_index, new_edge_value, info


def labels_for_nodes_from_sub_atlas(
    sub_volume: np.ndarray,
    sub_atlas: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Helper: given a sub-volume + matching sub-atlas (as produced by
    your ``crop_or_stride_volume`` utility) plus the tissue threshold
    used to build the node set, return the (N,) array of atlas labels
    in the same order as the graph's nodes.

    Assumes the graph was built from voxels where ``sub_volume >
    threshold``, in numpy's default C-order (which matches
    ``np.argwhere`` and ``np.where``). This is how
    ``build_anatomical_knn`` and the standard faiss path both order
    their nodes.
    """
    if sub_atlas is None:
        raise ValueError(
            "sub_atlas is None — pass an annotations volume so cross-"
            "region detection has labels to work with."
        )
    mask = sub_volume > threshold
    return sub_atlas[mask].astype(np.int64)