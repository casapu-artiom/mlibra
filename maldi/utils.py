"""
utils.py
Helper functions for the MALDI experiment.
"""
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


# =========================================================================
# Inducing points
# =========================================================================
def get_inducing_points(exp_path, dataset_path, num_inducing):
    """
    Get the inducing points (whole-brain, k-means with left/right symmetry).

    Args:
        exp_path (Path): Path to the experiment
        dataset_path (Path): Path to the dataset
        num_inducing (int): Number of inducing points

    Returns:
        inducing_points (torch.Tensor): Tensor of inducing points
        coord_mean (torch.Tensor): Mean of the coordinates
        coord_std (torch.Tensor): Standard deviation of the coordinates
    """
    if (num_inducing % 2) != 0:
        raise ValueError("num_inducing must be even")

    inducing_points_file = exp_path / f"inducing_points_{num_inducing}.pth"
    labels_file = exp_path / f"labels_{num_inducing}.pth"
    coord_mean, coord_std = _load_or_compute_coord_norm(exp_path, dataset_path)
    reference_image = None

    # create the inducing points as random samples of the 3d coordinates
    if not (inducing_points_file).exists():
        if reference_image is None:
            logging.info("Loading reference_image")
            reference_image = np.load(dataset_path / "reference_image.npy")
            reference_image_index = np.array(np.where(reference_image > 0)).T
            # convert to ccf
            reference_image = reference_image_index / 40
        logging.info("normalizing reference_image coordinates")
        reference_image = torch.tensor(reference_image, dtype=torch.float32)
        reference_image = (reference_image - coord_mean) / coord_std
        logging.info("reference_image normalization successful")
        # we do a k-means clustering of the reference image to find N inducing points
        x_median = np.median(reference_image[:, 0])
        logging.info("Clustering inducing points")
        # image is symetric from x_median along x axis, so we just need to fit the half
        logging.info("Using KMeans on symmetric points")
        inducing_points = get_symmetric_points(reference_image, exp_path, num_inducing, x_median, labels_file)
    else:
        inducing_points = torch.load(exp_path / f"inducing_points_{num_inducing}.pth")

    return inducing_points, coord_mean, coord_std

def coord_norm_from_reference(reference_image, voxel_per_mm: float = 40.0):
    """Single source of truth for the standardized coordinate space.
 
    Computes the global whole-brain normalization PURELY from the reference
    image — no exp_path, no caching, no dependency on a training run having
    happened anywhere. Training, visualization, and diagnostics should all call
    this so they share an identical metric regardless of where they run.
 
    Convention (do not change without re-solving every cached graph):
      * coordinates are tissue voxels (reference_image > 0) converted to mm
        via voxel / voxel_per_mm (40 vox/mm = 25 um),
      * coord_mean is PER-AXIS (3,)  — a translation, geometrically irrelevant,
      * coord_std is a SINGLE SCALAR  — isotropic scaling, so the metric stays
        a true (un-warped) Euclidean metric up to a global scale.
 
    Accepts either a loaded array or a path to reference_image.npy.
 
    Returns
    -------
    coord_mean : torch.Tensor (3,)
    coord_std  : torch.Tensor (scalar)
    """
    if isinstance(reference_image, (str, Path)):
        reference_image = np.load(reference_image)
    idx = np.array(np.where(reference_image > 0)).T
    ref_mm = idx / float(voxel_per_mm)
    coord_mean = torch.tensor(ref_mm.mean(axis=0), dtype=torch.float32)
    coord_std = torch.tensor(ref_mm.std(), dtype=torch.float32)   # scalar -> isotropic
    return coord_mean, coord_std


def _inducing_from_coords(coords, num_inducing, method="kmeans_snap", seed=0):
    """Select `num_inducing` inducing points that are ACTUAL data voxels.

    ``coords``: (N, D) array of (already standardized) data coordinates.
    ``method``:
      'kmeans_snap' (default): k-means centroids snapped to the nearest real
          voxel — density-aware placement, guaranteed on-data.
      'fps': farthest-point / k-center sampling — maximal coverage of sparse
          regions, on-data. O(N*M); fine for M up to a few thousand.
      'random': uniform random subset (fast; leaves coverage gaps).

    Returns (M', D) array of real voxel coordinates (M' <= num_inducing after
    de-duplicating snap collisions for 'kmeans_snap')."""
    from scipy.spatial import cKDTree
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    N = coords.shape[0]
    M = int(min(num_inducing, N))
    rng = np.random.default_rng(seed)
    if method == "random":
        idx = rng.choice(N, M, replace=False)
    elif method == "fps":
        idx = np.empty(M, dtype=np.int64)
        idx[0] = int(rng.integers(N))
        d2 = ((coords - coords[idx[0]]) ** 2).sum(1)
        for i in range(1, M):
            idx[i] = int(d2.argmax())
            d2 = np.minimum(d2, ((coords - coords[idx[i]]) ** 2).sum(1))
    elif method == "kmeans_snap":
        km = MiniBatchKMeans(n_clusters=M, random_state=seed, n_init=3).fit(coords)
        _, idx = cKDTree(coords).query(km.cluster_centers_, k=1)
        idx = np.unique(idx.astype(np.int64))      # dedupe collisions in sparse areas
    else:
        raise ValueError(f"unknown inducing method: {method!r}")
    return coords[idx]


def get_data_inducing_points(maldi_file, section_filter, num_inducing,
                             reference_image, method="kmeans_snap",
                             exp_path=None, force_recompute=False,
                             coord_cols=("xccf", "yccf", "zccf"),
                             seed=0, voxel_per_mm: float = 40.0):
    """Inducing points drawn from the ACTUAL measured MALDI voxels (sparse-data
    aware), standardized into the SAME space as ``get_inducing_points``.

    Drop-in replacement: returns ``(inducing_points, coord_mean, coord_std)``
    with the identical signature, so callers can swap it in.

    Unlike ``get_inducing_points`` (k-means over the reference *tissue* image,
    which can place centroids where no MALDI was measured), this reads the
    *training* coordinates selected by ``section_filter``, standardizes them
    with the global ``coord_norm_from_reference`` normalization, and picks
    ``num_inducing`` of them via ``method`` so every inducing point sits on a
    real measurement. ``section_filter`` is the train filter.

    Cached to ``exp_path/inducing_points_data_{method}_n{num_inducing}.pth``
    (mirrors get_inducing_points; the per-experiment exp_path scopes it to the
    fold). Pass ``exp_path=None`` to disable caching."""
    import pandas as pd
    coord_mean, coord_std = coord_norm_from_reference(reference_image, voxel_per_mm)

    cache_file = None
    if exp_path is not None:
        cache_file = Path(exp_path) / f"inducing_points_data_{method}_n{num_inducing}.pth"
        if cache_file.exists() and not force_recompute:
            logging.info(f"Loading cached data inducing points from {cache_file.name}")
            return torch.load(cache_file), coord_mean, coord_std

    df = pd.read_parquet(maldi_file, columns=list(coord_cols), filters=section_filter)
    coords_mm = torch.tensor(df.values, dtype=torch.float32)
    coords_z = ((coords_mm - coord_mean) / coord_std).numpy()
    ind = _inducing_from_coords(coords_z, num_inducing, method=method, seed=seed)
    inducing_points = torch.tensor(ind, dtype=torch.float32)
    logging.info(f"[data inducing] method={method}: {inducing_points.shape[0]} points "
                 f"from {coords_z.shape[0]:,} measured voxels")
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inducing_points, cache_file)
        logging.info(f"[data inducing] cached -> {cache_file.name}")
    return inducing_points, coord_mean, coord_std


def get_symmetric_points(reference_image, exp_path, num_inducing, x_median, labels_file):
    """
    Get symmetric inducing points from the reference image.

    Perform k-means clustering on the left half of the reference image and then
    returns the centroids (plus their mirrored copies) as inducing points.
    """
    half_reference_image = reference_image[reference_image[:, 0] <= x_median]
    complementary_half_reference_image = half_reference_image.clone()
    complementary_half_reference_image[:, 0] = x_median + (x_median - complementary_half_reference_image[:, 0])
    new_reference_image = np.concatenate([half_reference_image, complementary_half_reference_image], axis=0)
    shape_difference = reference_image.shape[0] - new_reference_image.shape[0]
    logging.info(f"Shape difference: {shape_difference}")
    kmeans = MiniBatchKMeans(n_clusters=(num_inducing // 2)).fit(half_reference_image)
    inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    assert inducing_points.shape == (num_inducing // 2, 3)

    labels = kmeans.predict(half_reference_image)
    complementary_points = inducing_points.clone()
    complementary_points[:, 0] = x_median + (x_median - complementary_points[:, 0])

    complementary_labels = labels + (num_inducing // 2)
    inducing_points = torch.cat([inducing_points, complementary_points], dim=0)
    labels = np.concatenate([labels, complementary_labels])
    torch.save(inducing_points, exp_path / f"inducing_points_{num_inducing}.pth")
    torch.save(labels, exp_path / f"labels_{num_inducing}.pth")
    torch.save(new_reference_image, exp_path / f"reference_image_{num_inducing}.pth")
    return inducing_points


def _load_or_compute_coord_norm(exp_path: Path, dataset_path: Path):
    """
    Load global (whole-brain) coord_mean / coord_std, or compute and cache
    them from the reference image. Shared across whole-brain and bbox
    inducing-point routines so the standardized coordinate space matches.
 
    The computation itself lives in `coord_norm_from_reference` (the single
    source of truth); this wrapper only adds the per-experiment cache.
    """
    colmean_file = exp_path / "colmean.pth"
    colstd_file = exp_path / "colstd.pth"
    if colmean_file.exists() and colstd_file.exists():
        return torch.load(colmean_file), torch.load(colstd_file)
 
    logging.info("Computing global coord_mean / coord_std from reference_image.npy")
    coord_mean, coord_std = coord_norm_from_reference(dataset_path / "reference_image.npy")
    torch.save(coord_mean, colmean_file)
    torch.save(coord_std, colstd_file)
    return coord_mean, coord_std


# =========================================================================
# Volume striding helpers (shared by the manifold experiments)
# =========================================================================
def crop_or_stride_volume(reference_image, annotation_volume, stride):
    """
    Subsample both volumes with the given stride.

    Returns
    -------
    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm
        sub_volume / sub_atlas : the strided (z, y, x) slabs.
        voxel_offset : always (0, 0, 0); kept for call-site compatibility.
        voxel_scale_mm : `stride * 0.025`, applied to local indices directly.
    """
    sub_volume = reference_image[::stride, ::stride, ::stride]
    sub_atlas = (annotation_volume[::stride, ::stride, ::stride]
                 if annotation_volume is not None else None)
    voxel_offset = (0, 0, 0)
    voxel_scale_mm = stride * 0.025
    logging.info(
        f"Stride subsample: stride={stride}, sub_volume.shape={sub_volume.shape}"
    )
    return sub_volume, sub_atlas, voxel_offset, voxel_scale_mm


def reference_ccf_from_subvolume(sub_volume, voxel_offset, voxel_scale_mm, threshold):
    """
    Build a (N, 3) array of mm reference coordinates for tissue voxels in
    `sub_volume`, in (z, y, x) order matching the atlas indexing.
    """
    z, y, x = np.where(sub_volume > threshold)
    oz, oy, ox = voxel_offset
    if voxel_scale_mm == 0.025:
        # Cropped path: add offset to recover full-res voxel indices
        idx = np.stack([z + oz, y + oy, x + ox], axis=1)
    else:
        # Strided path: indices are already in strided units; scale handles it
        idx = np.stack([z, y, x], axis=1)
    return idx.astype(np.float32) * voxel_scale_mm


