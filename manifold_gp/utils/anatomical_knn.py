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


def dissolve_root_labels(
    node_labels: np.ndarray,
    coords: np.ndarray,
    root_label: int = 0,
) -> np.ndarray:
    """Reassign every ``root_label`` (unlabeled / catch-all) node to the region
    of its nearest real-region node, so the atlas has no catch-all left.

    The ``level_15annot`` atlas leaves a fraction of *tissue* nodes with label 0
    ("root") threaded throughout the brain. With ``treat_zero_as_cross=True`` in
    ``inflate_cross_region_edges`` every edge touching one of those nodes is
    inflated, which sprays the soft prior through region interiors (fragmenting
    the kernel) while adding almost nothing to real region↔region confinement.
    Dissolving root — snapping each label-0 node to the nearest labelled region
    by Euclidean distance — folds that tissue into the region it physically sits
    in, so it smooths normally inside a region and any true boundary running
    through former-root gets inflated properly. One-shot nearest-labelled-node
    assignment (a KD-tree query), so a root node deep in a root pocket still
    reaches the closest real region regardless of intervening root nodes.

    Args:
      node_labels: (N,) int array of atlas labels in node order (0 = root).
      coords:      (N, d) node coordinates (any consistent space — normalized
                   reference coords or voxel indices; only relative distance
                   matters).
      root_label:  the catch-all label to dissolve (default 0).

    Returns:
      (N,) int64 labels with no ``root_label`` entries (unless *every* node is
      root, in which case the input is returned unchanged).
    """
    from scipy.spatial import cKDTree

    labels = np.asarray(node_labels).astype(np.int64).copy()
    coords = np.asarray(coords, dtype=np.float32)
    root = labels == root_label
    n_root = int(root.sum())
    if n_root == 0 or root.all():
        return labels
    real = ~root
    tree = cKDTree(coords[real])
    _, nn = tree.query(coords[root], k=1)
    labels[root] = labels[real][nn]
    logging.info(
        f"  dissolve_root_labels: reassigned {n_root:,}/{labels.size:,} "
        f"({100.0 * n_root / labels.size:.1f}%) root nodes to nearest region "
        f"({np.unique(node_labels).size} -> {np.unique(labels).size} regions)"
    )
    return labels


def denoise_labels_majority_vote(node_labels: np.ndarray,
                                 edge_index,
                                 n_iters: int) -> np.ndarray:
    """Majority-vote label smoothing over the graph.

    Each node adopts the most common label among its graph neighbours (+ itself),
    iterated — so speck / 'polluted' minority nodes are absorbed into the region
    they physically sit in and the partition becomes spatially coherent. The
    self-vote breaks ties toward staying put. ``edge_index`` may be a torch tensor
    or a (2, E) numpy array; both edge directions are counted.

    NOTE: this erodes thin real boundaries (small regions shrink), so it can
    WEAKEN region confinement — prefer ``dissolve_root_labels`` for the atlas
    catch-all. Kept because it feeds the hard prune.
    """
    if hasattr(edge_index, "detach"):
        ei = edge_index.detach().cpu().numpy()
    else:
        ei = np.asarray(edge_index)
    lab = np.asarray(node_labels).astype(np.int64).copy()
    N = lab.shape[0]
    for it in range(int(n_iters)):
        uniq, comp = np.unique(lab, return_inverse=True)      # compact ids 0..K-1
        votes = np.zeros((N, uniq.size), dtype=np.int32)
        np.add.at(votes, (ei[0], comp[ei[1]]), 1)             # neighbour votes (both dirs)
        np.add.at(votes, (ei[1], comp[ei[0]]), 1)
        np.add.at(votes, (np.arange(N), comp), 1)             # self-vote (tie -> stay)
        new = uniq[votes.argmax(1)]
        changed = int((new != lab).sum())
        lab = new
        logging.info(f"  denoise-labels iter {it + 1}: {changed:,} nodes relabelled")
        if changed == 0:
            break
    return lab


def prune_cross_region_edges(edge_index, edge_value,
                             node_labels: np.ndarray, prune: float,
                             zero_is_region: bool, seed: int = 0):
    """Hard-remove a fraction of cross-region edges, connectivity-preserving.

    A HARD cut, vs ``inflate_cross_region_edges``' soft down-weight: removes
    ``prune`` of the edges whose endpoints lie in different regions. Any node the
    prune would ISOLATE keeps its single strongest incident edge (a bridge), so
    the graph stays connected (no disconnected-component NaNs, no specks) instead
    of shattering. On a clean partition this makes the boundary a real spectral
    bottleneck. ``zero_is_region`` = whether label 0 is a real region (template
    clusters) or a background/gap that is never a within/cross endpoint (atlas).

    Returns ``(edge_index, edge_value)`` as torch tensors on the input device.
    """
    device = edge_index.device
    ei_np = edge_index.detach().cpu().numpy()
    ev_np = edge_value.detach().cpu().numpy()
    Nn = int(node_labels.shape[0])
    if zero_is_region:
        cross = node_labels[ei_np[0]] != node_labels[ei_np[1]]
    else:  # label 0 = background/gap: never treat as a within/cross region pair
        cross = ((node_labels[ei_np[0]] != node_labels[ei_np[1]])
                 & (node_labels[ei_np[0]] > 0) & (node_labels[ei_np[1]] > 0))
    drop = np.zeros(cross.shape, bool)
    cidx = np.where(cross)[0]
    n_req = int(round(float(prune) * cidx.size))
    if n_req:
        drop[np.random.default_rng(seed).choice(cidx, n_req, replace=False)] = True
    # connectivity-preserving: restore one (strongest) bridge per would-be-isolated node
    deg = np.zeros(Nn)
    np.add.at(deg, ei_np[0][~drop], 1); np.add.at(deg, ei_np[1][~drop], 1)
    iso = deg == 0
    n_iso0 = int(iso.sum())
    if n_iso0:
        ic = np.where((iso[ei_np[0]] | iso[ei_np[1]]) & drop)[0]
        owner = np.where(iso[ei_np[0][ic]], ei_np[0][ic], ei_np[1][ic])
        order = np.lexsort((ev_np[ic], owner))                # per owner, smallest ev first
        firstm = np.ones(order.size, bool)
        firstm[1:] = owner[order][1:] != owner[order][:-1]
        drop[ic[order][firstm]] = False
    keep_mask = torch.as_tensor(~drop, device=device)
    n_before = edge_index.shape[1]
    edge_index = edge_index[:, keep_mask]
    edge_value = edge_value[keep_mask]
    deg2 = np.zeros(Nn)
    np.add.at(deg2, ei_np[0][~drop], 1); np.add.at(deg2, ei_np[1][~drop], 1)
    logging.info(
        f"  prune_cross_region_edges: pruned {int(drop.sum()):,}/{int(cross.sum()):,} "
        f"cross-region edges (frac={prune:g}) — {n_before:,} -> "
        f"{edge_index.shape[1]:,} edges; isolated nodes {n_iso0:,} -> "
        f"{int((deg2 == 0).sum()):,} after bridges")
    return edge_index, edge_value


def labels_for_nodes_from_template_clustering(
    sub_volume: np.ndarray,
    threshold: float,
    n_clusters: int,
    spatial_weight: float = 1.0,
    fit_subsample: int = 40000,
    seed: int = 0,
    niter: int = 25,
) -> np.ndarray:
    """Derive node labels by clustering the reference TEMPLATE with faiss k-means.

    A data-driven, whole-brain, lipid-free replacement for the anatomical atlas
    as the label source for ``inflate_cross_region_edges``. Uses only image
    structure of ``sub_volume`` (no atlas, no lipids), so it is defined everywhere
    the template is and needs no imputation.

    Node order matches the graph's: voxels where ``sub_volume > threshold`` in
    numpy C-order — the SAME ordering as ``labels_for_nodes_from_sub_atlas`` and
    the faiss node set, so the returned labels line up with ``knn.x``.

    Clustering: faiss ``Kmeans`` (GPU if available) over a joint feature space of
      [ ``spatial_weight`` * z-scored (z,y,x) coords , z-scored template features ]
    where the template features are intensity + local gradient magnitude + local
    mean + local std. The coordinate block makes clusters spatially COMPACT — the
    larger ``spatial_weight``, the more contiguous (anatomy-like) the regions; at
    0 it is pure feature k-means (speckly). k-means does not *guarantee* connected
    regions the way constrained Ward does, but faiss makes it GPU-fast and avoids
    the scipy/sklearn CPU path. Deterministic given ``seed``.

    Args:
      sub_volume:     strided reference template (grayscale intensity).
      threshold:      tissue threshold used to build the node set.
      n_clusters:     number of regions (K). For a matched comparison vs the atlas
                      pass the atlas's region count.
      spatial_weight: weight on the (z-scored) coordinate block. Higher = more
                      spatially compact/contiguous; 0 = pure feature clustering.
      fit_subsample:  random sample of nodes used to TRAIN the centroids (all
                      nodes are then assigned). <=0 or >= N trains on all nodes.
      seed:           RNG seed for k-means init + the training subsample. Make it
                      part of any graph cache key so runs stay reproducible.
      niter:          k-means iterations.

    Returns:
      labels: (N,) int64 array, one region id per node, in node order.
    """
    import faiss
    from scipy import ndimage

    mask = sub_volume > threshold
    zc, yc, xc = np.where(mask)
    node_idx = np.stack([zc, yc, xc], axis=1).astype(np.float32)   # (N,3), node order
    n_nodes = node_idx.shape[0]

    vol = sub_volume.astype(np.float32)
    gmag = ndimage.gaussian_gradient_magnitude(vol, sigma=1.0)
    lmean = ndimage.uniform_filter(vol, size=3)
    lstd = np.sqrt(np.maximum(ndimage.uniform_filter(vol ** 2, size=3) - lmean ** 2, 0.0))
    feats = np.stack(
        [vol[zc, yc, xc], gmag[zc, yc, xc], lmean[zc, yc, xc], lstd[zc, yc, xc]],
        axis=1,
    ).astype(np.float32)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)

    coords_z = (node_idx - node_idx.mean(0)) / (node_idx.std(0) + 1e-8)
    X = np.ascontiguousarray(
        np.hstack([float(spatial_weight) * coords_z, feats]), dtype=np.float32
    )
    d = X.shape[1]

    rng = np.random.default_rng(seed)
    m = int(fit_subsample) if 0 < int(fit_subsample) < n_nodes else n_nodes
    train = X if m >= n_nodes else X[rng.choice(n_nodes, m, replace=False)]

    try:
        use_gpu = faiss.get_num_gpus() > 0
    except Exception:
        use_gpu = False
    km = faiss.Kmeans(
        d, int(n_clusters), niter=int(niter), seed=int(seed),
        gpu=use_gpu, verbose=False,
    )
    km.train(np.ascontiguousarray(train, dtype=np.float32))

    # Assign every node to its nearest centroid in NUMPY (not faiss .index.search):
    # K is small so this is cheap, and it avoids the faiss-CPU search path that
    # can memory-fault alongside torch in some envs. argmin over ||x - c||^2 via
    # the (||c||^2 - 2 x·c) reduction (||x||^2 is constant per row).
    C = np.ascontiguousarray(km.centroids, dtype=np.float32).reshape(int(n_clusters), d)
    c_sq = (C ** 2).sum(1)
    labels = np.empty(n_nodes, dtype=np.int64)
    step = 200_000
    for s in range(0, n_nodes, step):
        block = X[s:s + step]
        scores = c_sq[None, :] - 2.0 * (block @ C.T)   # (b, K), monotone in distance
        labels[s:s + block.shape[0]] = np.argmin(scores, axis=1)
    logging.info(
        f"template k-means: {n_nodes} nodes -> {n_clusters} regions "
        f"(train on {m}, spatial_weight={spatial_weight}, gpu={use_gpu}, seed={seed})"
    )
    return labels