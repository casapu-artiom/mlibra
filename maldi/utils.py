"""
utils.py
Helper functions for the MALDI experiment.
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import logging
import torch

def get_inducing_points(exp_path, dataset_path, num_inducing):
    """
    Get the inducing points.

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
    colmean_file = exp_path / "colmean.pth"
    colstd_file = exp_path / "colstd.pth"
    reference_image = None
    # compute the mean and std of the coordinates of the reference image
    if not colmean_file.exists() or not colstd_file.exists():
        logging.info("Loading reference_image")
        reference_image = np.load(dataset_path / "reference_image.npy")
        # then we convert it to a matrix with three columns x,y,z with the coordinates of these non zero values
        reference_image_index = np.array(np.where(reference_image>0)).T
        # convert to ccf
        reference_image = reference_image_index / 40
        logging.info("reference_image loaded")
        logging.info("normalizing reference_image coordinates")
        # normalize the coordinates
        coord_mean = reference_image.mean(axis=0)
        coord_std = reference_image.std()
        coord_mean = torch.tensor(coord_mean, dtype=torch.float32)
        coord_std = torch.tensor(coord_std, dtype=torch.float32)
        torch.save(coord_mean, colmean_file)
        torch.save(coord_std, colstd_file)
    else:
        coord_mean = torch.load(colmean_file)
        coord_std = torch.load(colstd_file)

    # create the inducing points as random samples of the 3d coordinates
    if not (inducing_points_file).exists():
        if reference_image is None:
            logging.info("Loading reference_image")
            reference_image = np.load(dataset_path / "reference_image.npy")
            reference_image_index = np.array(np.where(reference_image>0)).T
            # convert to ccf
            reference_image = reference_image_index / 40
        logging.info("normalizing reference_image coordinates")
        reference_image = torch.tensor(reference_image, dtype=torch.float32)
        coord_mean = torch.tensor(coord_mean, dtype=torch.float32)
        coord_std = torch.tensor(coord_std, dtype=torch.float32)
        reference_image = (reference_image - coord_mean) / coord_std
        logging.info("reference_image normalization successful")
        # we do a k-means clustering of the reference image to find N inducing points
        x_median = np.median(reference_image[:, 0])
        logging.info("Clustering inducing points")
        # image is symetric from x_median along x axis, so we just need to fitthe half
        logging.info("Using KMeans on symmetric points")
        inducing_points = get_symmetric_points(reference_image, exp_path, num_inducing, x_median, labels_file)
    else:
        inducing_points = torch.load(exp_path / f"inducing_points_{num_inducing}.pth")

    return inducing_points, coord_mean, coord_std

def get_symmetric_points(reference_image, exp_path, num_inducing, x_median, labels_file):
    """
    Get symmetric inducing points from the reference image.

    Perform k-means clustering on the left half of the reference image and then returns the centroids as inducing points.

    Args:
        reference_image (np.ndarray): The reference image with coordinates.
        exp_path (Path): Path to the experiment.
        num_inducing (int): Number of inducing points.
        x_median (float): Median x-coordinate for symmetry.
        labels_file (Path): Path to save labels.

    Returns:
        inducing_points (torch.Tensor): Tensor of inducing points.
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
    # we can live with the fact that some points are outside the image, empirically seems like only 3 to 5 points are.

    labels = kmeans.predict(half_reference_image)
    # add the symetric points
    complementary_points = inducing_points.clone()
    complementary_points[:, 0] = x_median + (x_median - complementary_points[:, 0])

    complementary_labels = labels + (num_inducing // 2)
    inducing_points = torch.cat([inducing_points, complementary_points], dim=0)
    labels = np.concatenate([labels, complementary_labels])
    # we normalize to the coordinates range
    torch.save(inducing_points, exp_path / f"inducing_points_{num_inducing}.pth")
    torch.save(labels, exp_path / f"labels_{num_inducing}.pth")
    torch.save(new_reference_image, exp_path / f"reference_image_{num_inducing}.pth")
    return inducing_points
