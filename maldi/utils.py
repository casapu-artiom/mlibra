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


def get_bbox_inducing_points(
    exp_path: Path,
    dataset_path: Path,
    num_inducing: int,
    region_bbox,
    force_recompute: bool = False,
):
    """
    Get inducing points restricted to a voxel-space bbox at full 25um resolution.

    Coord normalization (coord_mean / coord_std) is *identical* to
    `get_inducing_points`, so the standardized coordinate space is shared
    across whole-brain and region runs. Only the spatial support of the
    inducing points changes.

    No left/right symmetry is enforced (the bbox is arbitrary and may sit
    fully in one hemisphere). Plain MiniBatchKMeans inside the bbox.

    Parameters
    ----------
    exp_path : Path
        Experiment dir for caching.
    dataset_path : Path
        Must contain reference_image.npy (full-res 25um atlas template).
    num_inducing : int
        Target number of inducing points. Clamped to the number of tissue
        voxels inside the bbox if there are fewer.
    region_bbox : sequence of 6 ints
        (zmin, zmax, ymin, ymax, xmin, xmax) in voxel coords of the full-res
        25um atlas (atlas indexing).
    force_recompute : bool
        Ignore any existing cache and re-run k-means.

    Returns
    -------
    inducing_points : torch.Tensor (n, 3) standardized
    coord_mean : torch.Tensor (3,)
    coord_std  : torch.Tensor (scalar)
    """
    region_bbox = tuple(int(b) for b in region_bbox)
    bbox_str = "_".join(str(b) for b in region_bbox)
    inducing_file = exp_path / f"inducing_points_bbox_{bbox_str}_n{num_inducing}.pth"

    coord_mean, coord_std = _load_or_compute_coord_norm(exp_path, dataset_path)

    if inducing_file.exists() and not force_recompute:
        logging.info(f"Loading cached bbox inducing points from {inducing_file.name}")
        return torch.load(inducing_file), coord_mean, coord_std

    logging.info(
        f"Computing bbox inducing points (bbox={region_bbox}, num_inducing={num_inducing})"
    )
    reference_image = np.load(dataset_path / "reference_image.npy")
    zmin, zmax, ymin, ymax, xmin, xmax = region_bbox
    sub = reference_image[zmin:zmax, ymin:ymax, xmin:xmax]
    z, y, x = np.where(sub > 0)
    if z.shape[0] == 0:
        raise ValueError(
            f"Region bbox {region_bbox} contains no tissue voxels in the "
            f"reference image. Pick a different bbox."
        )

    # Reconstruct full-res voxel indices, then convert to mm (voxel / 40).
    pts_mm = np.stack([z + zmin, y + ymin, x + xmin], axis=1).astype(np.float32) / 40.0
    pts_mm = torch.tensor(pts_mm, dtype=torch.float32)
    pts_std = (pts_mm - coord_mean) / coord_std

    n_target = min(int(num_inducing), int(pts_std.shape[0]))
    if n_target < num_inducing:
        logging.warning(
            f"Bbox contains {pts_std.shape[0]} tissue voxels at full resolution, "
            f"fewer than requested num_inducing={num_inducing}. Using {n_target}."
        )

    kmeans = MiniBatchKMeans(n_clusters=n_target, n_init="auto").fit(pts_std.numpy())
    inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    torch.save(inducing_points, inducing_file)
    return inducing_points, coord_mean, coord_std


def _load_or_compute_coord_norm(exp_path: Path, dataset_path: Path):
    """
    Load global (whole-brain) coord_mean / coord_std, or compute and cache
    them from the reference image. Shared across whole-brain and bbox
    inducing-point routines so the standardized coordinate space matches.
    """
    colmean_file = exp_path / "colmean.pth"
    colstd_file = exp_path / "colstd.pth"
    if colmean_file.exists() and colstd_file.exists():
        return torch.load(colmean_file), torch.load(colstd_file)

    logging.info("Computing global coord_mean / coord_std from reference_image.npy")
    reference_image = np.load(dataset_path / "reference_image.npy")
    reference_image_index = np.array(np.where(reference_image > 0)).T
    reference_mm = reference_image_index / 40
    coord_mean = torch.tensor(reference_mm.mean(axis=0), dtype=torch.float32)
    coord_std = torch.tensor(reference_mm.std(), dtype=torch.float32)
    torch.save(coord_mean, colmean_file)
    torch.save(coord_std, colstd_file)
    return coord_mean, coord_std


# =========================================================================
# Region / bbox helpers (shared by lgp_experiment.py and lgp_manifold_experiment.py)
# =========================================================================
def crop_or_stride_volume(reference_image, annotation_volume, stride, region_bbox):
    """
    Either crop both volumes to a (z, y, x) bbox at full resolution, or
    subsample with the given stride.

    Returns
    -------
    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm
        sub_volume / sub_atlas : the processed (z, y, x) slabs.
        voxel_offset : (oz, oy, ox) ints; offset to add to local indices to
                       recover full-resolution voxel indices.
        voxel_scale_mm : float; multiply (local + offset) by this to get mm
                         in the cropped path. In the strided path,
                         voxel_offset is (0, 0, 0) and voxel_scale_mm is
                         `stride * 0.025`, applied to local indices directly.
    """
    if region_bbox is not None:
        zmin, zmax, ymin, ymax, xmin, xmax = region_bbox
        sub_volume = reference_image[zmin:zmax, ymin:ymax, xmin:xmax]
        if annotation_volume is not None:
            sub_atlas = annotation_volume[zmin:zmax, ymin:ymax, xmin:xmax]
        else:
            sub_atlas = None
        voxel_offset = (zmin, ymin, xmin)
        voxel_scale_mm = 0.025
        logging.info(
            f"Region crop: bbox={tuple(region_bbox)}, "
            f"sub_volume.shape={sub_volume.shape} (full resolution, stride ignored)"
        )
    else:
        sub_volume = reference_image[::stride, ::stride, ::stride]
        if annotation_volume is not None:
            sub_atlas = annotation_volume[::stride, ::stride, ::stride]
        else:
            sub_atlas = None
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


def bbox_to_mm_bounds(region_bbox):
    """
    Convert a (zmin, zmax, ymin, ymax, xmin, xmax) voxel bbox at 25um into mm
    bounds in the (xccf, yccf, zccf) convention used by the MALDI parquet.
    """
    zmin, zmax, ymin, ymax, xmin, xmax = region_bbox
    return {
        "x_min_mm": xmin * 0.025, "x_max_mm": xmax * 0.025,
        "y_min_mm": ymin * 0.025, "y_max_mm": ymax * 0.025,
        "z_min_mm": zmin * 0.025, "z_max_mm": zmax * 0.025,
    }


def extend_filter_with_bbox(parquet_filter, mm_bounds):
    """
    Append (xccf, yccf, zccf) bbox predicates to a pyarrow-style filter.

    Handles both forms accepted by pd.read_parquet:
      - Flat list (conjunction):    [(c, op, v), (c, op, v), ...]
      - DNF (disjunction of ANDs):  [[(c, op, v), ...], [(c, op, v), ...]]
    """
    bbox_preds = [
        ("xccf", ">=", mm_bounds["x_min_mm"]), ("xccf", "<=", mm_bounds["x_max_mm"]),
        ("yccf", ">=", mm_bounds["y_min_mm"]), ("yccf", "<=", mm_bounds["y_max_mm"]),
        ("zccf", ">=", mm_bounds["z_min_mm"]), ("zccf", "<=", mm_bounds["z_max_mm"]),
    ]
    if parquet_filter is None or len(parquet_filter) == 0:
        return bbox_preds
    if isinstance(parquet_filter[0], list):
        # DNF form: distribute the bbox over each conjunction
        return [list(conj) + bbox_preds for conj in parquet_filter]
    # Flat conjunction form
    return list(parquet_filter) + bbox_preds


def apply_region_to_config(config, region_bbox):
    """
    Patch `config.section_filter` and `config.test_filter` so the MALDI
    parquet reads only return points inside the bbox. Mutates `config`
    in place and returns it for convenience.
    """
    if region_bbox is None:
        return config
    mm_bounds = bbox_to_mm_bounds(region_bbox)
    logging.info(
        f"Restricting MALDI parquet filters to mm bbox: "
        f"x in [{mm_bounds['x_min_mm']:.3f}, {mm_bounds['x_max_mm']:.3f}], "
        f"y in [{mm_bounds['y_min_mm']:.3f}, {mm_bounds['y_max_mm']:.3f}], "
        f"z in [{mm_bounds['z_min_mm']:.3f}, {mm_bounds['z_max_mm']:.3f}]"
    )
    config.section_filter = extend_filter_with_bbox(config.section_filter, mm_bounds)
    config.test_filter    = extend_filter_with_bbox(config.test_filter,    mm_bounds)
    return config