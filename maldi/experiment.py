"""
experiment.py
Main experiment logic and classes for the MALDI experiment.
"""
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gpytorch.settings
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from config import MaldiConfig
from l3di.lgp import LGP
from manifold_gp.utils.nearest_neighbors import faiss_cpu_recon

def density_scatter(ax, x, y, x_min, x_max, y_min, y_max, bins=50, **kwargs):
    """
    Creates a scatter plot on the provided axis using a density-based color
    from a 2D histogram with square bins.
    """
    # Calculate limits and ensure square bins
    # valid_mask = ~np.isnan(y) & ~np.isneginf(y)
    # y = y[valid_mask]
    # x = x[valid_mask]
    # x_min, x_max = np.nanmin(x), np.nanmax(x)
    # y_min, y_max = np.nanmin(y), np.nanmax(y)
    data_range = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_range = data_range / 2
    # x_lim = (x_center - half_range, x_center + half_range)
    # y_lim = (y_center - half_range, y_center + half_range)
    x_lim = (x_min, x_max)
    y_lim = (y_min, y_max)

    # Create bin edges
    xedges = np.linspace(x_lim[0], x_lim[1], bins + 1)
    yedges = np.linspace(y_lim[0], y_lim[1], bins + 1)

    # Determine the bin index for each point
    x_bin = np.digitize(x, xedges) - 1
    y_bin = np.digitize(y, yedges) - 1

    # Build a 2D histogram for the points
    counts = np.zeros((bins, bins))
    for xb, yb in zip(x_bin, y_bin):
        if xb == bins:
            xb = bins - 1
        if yb == bins:
            yb = bins - 1
        if 0 <= xb < bins and 0 <= yb < bins:
            counts[yb, xb] += 1

    # Get density for each point from its bin
    density = np.array(
        [
            counts[yb if yb < bins else bins - 1, xb if xb < bins else bins - 1]
            for xb, yb in zip(x_bin, y_bin)
        ]
    )

    # Create the scatter plot with density coloring using the inferno colormap
    sc = ax.scatter(x, y, c=density, cmap="inferno", **kwargs)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    return sc

def scatter_comparison(true_values, predicted_values, lipid ):
    fig, ax = plt.subplots(figsize=(10, 10))
    true_min = np.nanmin(true_values)
    true_max = np.nanmax(true_values)
    predicted_min = np.nanmin(predicted_values)
    predicted_max = np.nanmax(predicted_values)
    
    correlation = np.corrcoef(true_values.flatten(), predicted_values.flatten())[0, 1]

    ax.plot([true_min, true_max],[true_min, true_max], "k--", lw=2)
    density_scatter(ax, true_values, predicted_values,
                    x_min=true_min, x_max=true_max,
                    y_min=predicted_min, y_max=predicted_max,
                    s=0.1, alpha=0.1)
    ax.set_title(f"Train set: {lipid} vs {lipid}_predicted\nCorrelation: {correlation:.2f}")
    ax.set_xlabel(f"{lipid} (true)")
    ax.set_ylabel(f"{lipid} (predicted)")
    plt.show()

class MaldiExperiment:
    def __init__(self, config:MaldiConfig, lgp_model:LGP, coord_mean:torch.Tensor, coord_std:torch.Tensor):
        self.config = config
        self.coord_mean = coord_mean
        self.coord_std = coord_std

        self.train_filter = config.section_filter
        self.test_filter = config.test_filter
        self.lgp_model = lgp_model

        self.coordinates_train = None
        self.coordinates_test = None
        self.train_data_original = None
        self.test_data_original = None
        self.train_data = None
        self.test_data = None
        self.pixel_coordinates_train = None
        self.pixel_coordinates_test = None
        self.col_means = None
        self.col_stds = None

    # ------------------------------------------------------------------
    # Internal MALDI parquet reader: cuts down boilerplate and makes it
    # trivial to swap the filter (e.g. when restricting to a region bbox).
    # ------------------------------------------------------------------
    def _read_maldi(self, columns, *, train=True, as_tensor=False, dtype=torch.float32):
        flt = self.train_filter if train else self.test_filter
        df = pd.read_parquet(self.config.maldi_file, columns=columns, filters=flt)
        if as_tensor:
            return torch.tensor(df.values, dtype=dtype)
        return df

    def load_train_sections(self):
        self.train_sections = pd.read_parquet(self.config.maldi_file,
                                                  columns=["Section"],
                                                  filters=self.train_filter)

    def load_train_pixel_coordinates(self):
        logging.info("Loading training pixel coordinates")
        self.pixel_coordinates_train = self._read_maldi(["x", "y"], train=True).values

    def load_train_pixel_index_coordinates(self):
        logging.info("Loading training pixel index coordinates")
        self.pixel_index_coordinates_train = self._read_maldi(
            ["x_index", "y_index", "z_index"], train=True, as_tensor=True,
        )

    def load_test_pixel_index_coordinates(self):
        logging.info("Loading test pixel index coordinates")
        self.pixel_index_coordinates_test = self._read_maldi(
            ["x_index", "y_index", "z_index"], train=False, as_tensor=True,
        )

    def load_train_samples(self):
        logging.info("Loading training samples")
        self.train_samples = self._read_maldi(["Sample"], train=True)

    def load_test_samples(self):
        logging.info("Loading test samples")
        self.test_samples = self._read_maldi(["Sample"], train=False)


    def load_train_data(self):
        self.train_data = pd.read_parquet(self.config.maldi_file,
                                          columns=[str(i) for i in self.config.selected_lipids_names],
                                          filters=self.train_filter).values
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32)
        assert self.train_data.shape[0] > 0, "No training data found with the given filter"
        n_nans = torch.isnan(self.train_data).sum()
        assert not torch.isnan(self.train_data).any(), f"Training data contains NaNs: {n_nans} NaNs found"
        n_zeros = (self.train_data == 0).sum()
        logging.info(f"Training data contains {n_zeros} zeros")
        n_negatives = (self.train_data < 0).sum()
        logging.info(f"Training data contains {n_negatives} negative values")
        logging.info("imputing negative values to zero")
        self.train_data[self.train_data < 0] = 0

        log_transform = self.config.log_transform
        if log_transform:
            logging.info("Applying log transformation to training data")
            self.train_data = np.log(self.train_data + 1e-10)
            n_nans = torch.isnan(self.train_data).sum()
            assert not torch.isnan(self.train_data).any(), f"Training data contains NaNs after log transformation: {n_nans} NaNs found after log transformation"
        else:
            logging.info("Skipping log transformation")
        logging.info(f"Training data shape: {self.train_data.shape}")

        logging.info("normalizing training data")
        if (self.config.exp_path / "lipid_means.pth").exists() and (self.config.exp_path / "lipid_stds.pth").exists():
            logging.info("Loading column mean and std from files")
            col_means = torch.load(self.config.exp_path / "lipid_means.pth")
            col_stds = torch.load(self.config.exp_path / "lipid_stds.pth")
        else:
            logging.info("computing channels means and stds")
            col_means = self.train_data.mean(dim=0)
            col_stds = self.train_data.std(dim=0)
            torch.save(col_means, self.config.exp_path / "lipid_means.pth")
            torch.save(col_stds, self.config.exp_path / "lipid_stds.pth")
        assert not torch.isnan(col_means).any(), "there are nans in col_means"
        assert not torch.isnan(col_stds).any(), "there are nans in col_stds"
        self.col_means = col_means
        self.col_stds = col_stds
        self.train_data = (self.train_data - col_means) / col_stds
        n_nans = torch.isnan(self.train_data).sum()
        assert not torch.isnan(self.train_data).any(), f"Training data contains NaNs after normalization: {n_nans} NaNs found after normalization"
        logging.info("Training data normalization successful")

        logging.info("Loading coordinates")
        coordinates_names =["xccf","yccf","zccf"]
        self.coordinates_train = pd.read_parquet(self.config.maldi_file,
                                                 columns=coordinates_names,
                                                 filters=self.train_filter).values
        self.coordinates_train = torch.tensor(self.coordinates_train, dtype=torch.float32)
        logging.info(f"Coordinates shape: {self.coordinates_train.shape}")
        logging.info("Normalizing coordinates")
        self.coordinates_train = (self.coordinates_train - self.coord_mean) / self.coord_std
        n_nans = torch.isnan(self.coordinates_train).sum()
        assert not torch.isnan(self.coordinates_train).any(), f"Coordinates contain NaNs: {n_nans} NaNs found in coordinates"
        logging.info("Coordinates normalization successful")

        self.dataset_train = torch.utils.data.TensorDataset(self.train_data, self.coordinates_train)

    def load_test_data(self):
        self.test_data = pd.read_parquet(self.config.maldi_file,
                                         columns=[str(i) for i in self.config.selected_lipids_names],
                                         filters=self.test_filter).values
        self.test_data = torch.tensor(self.test_data, dtype=torch.float32)
        assert self.test_data.shape[0] > 0, "No test data found with the given filter"
        n_nans = torch.isnan(self.test_data).sum()
        assert not torch.isnan(self.test_data).any(), f"Test data contains NaNs: {n_nans} NaNs found"
        n_zeros = (self.test_data == 0).sum()
        logging.info(f"Test data contains {n_zeros} zeros")
        n_negatives = (self.test_data < 0).sum()
        logging.info(f"Test data contains {n_negatives} negative values")
        logging.info("imputing negative values to zero")
        self.test_data[self.test_data < 0] = 0

        log_transform = self.config.log_transform
        if log_transform:
            logging.info("Applying log transformation to test data")
            self.test_data = np.log(self.test_data + 1e-10)
            n_nans = torch.isnan(self.test_data).sum()
            assert not torch.isnan(self.test_data).any(), f"Test data contains NaNs after log transformation: {n_nans} NaNs found after log transformation"
        else:
            logging.info("Skipping log transformation")
        logging.info(f"Test data shape: {self.test_data.shape}")

        logging.info("normalizing test data")
        if (self.config.exp_path / "lipid_means.pth").exists() and (self.config.exp_path / "lipid_stds.pth").exists():
            logging.info("Loading column mean and std from files")
            col_means = torch.load(self.config.exp_path / "lipid_means.pth")
            col_stds = torch.load(self.config.exp_path / "lipid_stds.pth")
        else:
            logging.info("computing channels means and stds")
            col_means = self.train_data.mean(dim=0)
            col_stds = self.train_data.std(dim=0)
            torch.save(col_means, self.config.exp_path / "lipid_means.pth")
            torch.save(col_stds, self.config.exp_path / "lipid_stds.pth")
        assert not torch.isnan(col_means).any(), "there are nans in col_means"
        assert not torch.isnan(col_stds).any(), "there are nans in col_stds"
        # Use the SAME (training) stats the model was trained with, and actually
        # apply them. Mirrors load_train_data: predict_original_scale undoes this
        # exact z-score (* col_stds + col_means) to recover the original scale, so
        # without it the saved test true_values are doubly-transformed garbage.
        self.col_means = col_means
        self.col_stds = col_stds
        self.test_data = (self.test_data - col_means) / col_stds
        n_nans = torch.isnan(self.test_data).sum()
        assert not torch.isnan(self.test_data).any(), f"Test data contains NaNs after normalization: {n_nans} NaNs found after normalization"

    def _load_coords(self, train=True):
        """Shared loader for {train,test}_coordinates: read xccf/yccf/zccf,
        normalize with global coord_mean/coord_std, and cache on self."""
        attr = "coordinates_train" if train else "coordinates_test"
        if getattr(self, attr) is not None:
            return
        logging.info(f"Loading coordinates for {'training' if train else 'test'} data")
        coords = self._read_maldi(
            ["xccf", "yccf", "zccf"], train=train, as_tensor=True,
        )
        logging.info(f"Coordinates shape: {coords.shape}")
        coords = (coords - self.coord_mean) / self.coord_std
        n_nans = torch.isnan(coords).sum()
        assert not torch.isnan(coords).any(), \
            f"Coordinates contain NaNs: {n_nans} NaNs found in coordinates"
        setattr(self, attr, coords)
        logging.info("Coordinates normalization successful")

    def load_coord_train_data(self):
        self._load_coords(train=True)

    def load_coord_test_data(self):
        self._load_coords(train=False)

    def load_checkpoint(self):
        if len(list(self.config.checkpoint_path.glob("*.pth"))) > 0:
            last_checkpoint = max(self.config.checkpoint_path.glob("*.pth"), key=lambda x: x.stat().st_mtime)
            self.lgp_model.load_state_dict(torch.load(last_checkpoint, map_location=torch.device(self.config.device)))
            self.current_epoch = len(list(self.config.checkpoint_path.glob("*.pth")))
            logging.info(f"loaded checkpoint {last_checkpoint}")

    @property
    def true_values_train(self):
        """
        Returns the true values for the training set as a numpy array.
        """
        if self.train_data_original is None:
            self.train_data_original = np.load(self.config.exp_path / "train" / "true_values.npy")

        return self.train_data_original

    @property
    def true_values_test(self):
        """
        Returns the true values for the test set as a numpy array.
        """
        if self.test_data_original is None:
            self.test_data_original = np.load(self.config.exp_path / "test" / "true_values.npy")

        return self.test_data_original

    @property
    def predictions_train(self):
        """
        Returns the predictions for the training set as a numpy array.
        """
        train_path = self.config.exp_path / "train"
        predictions_file = train_path / "predictions.npy"
        if not predictions_file.exists():
            logging.error("Predictions for the training set do not exist. Please run the experiment first.")
            return None
        return np.load(predictions_file)

    @property
    def predictions_test(self):
        """
        Returns the predictions for the test set as a numpy array.
        """
        test_path = self.config.exp_path / "test"
        predictions_file = test_path / "predictions.npy"
        if not predictions_file.exists():
            logging.error("Predictions for the test set do not exist. Please run the experiment first.")
            return None
        return np.load(predictions_file)

    @property
    def ccf_train(self):
        """
        Returns the CCF coordinates for the training set as a numpy array.
        """
        if self.coordinates_train is None:
            self.load_coord_train_data()
        return self.coordinates_train.numpy()

    @property
    def ccf_test(self):
        """
        Returns the CCF coordinates for the test set as a numpy array.
        """
        if self.coordinates_test is None:
            self.load_coord_test_data()
        return self.coordinates_test.numpy()

    @property
    def sectionid_train(self):
        """
        Returns the section numbers for the training set as a pandas DataFrame.
        """
        self.train_sections = pd.read_parquet(self.config.maldi_file,
                                              columns=["Section", "Sample"],
                                              filters=self.train_filter)
        # use sklearn to create unique ids from the concatenation of Section and Sample
        self.train_sections["Section"] = self.train_sections["Section"].astype(str)
        self.train_sections["Sample"] = self.train_sections["Sample"].astype(str)
        # convert SectionID to integer
        return pd.factorize(self.train_sections["Section"] + "_" + self.train_sections["Sample"])[0]

    @property
    def sectionid_test(self):
        """
        Returns the section numbers for the test set as a pandas DataFrame.
        """
        if self.test_sections is None:
            self.test_sections = pd.read_parquet(self.config.maldi_file,
                                                 columns=["Section", "Sample"],
                                                 filters=self.test_filter)
            # use sklearn to create unique ids from the concatenation of Section and Sample
            self.test_sections["Section"] = self.test_sections["Section"].astype(str)
            self.test_sections["Sample"] = self.test_sections["Sample"].astype(str)
            self.test_sections["SectionID"] = self.test_sections["Section"] + "_" + self.test_sections["Sample"]
            # convert SectionID to integer
            self.test_sections["SectionID"] = pd.factorize(self.test_sections["SectionID"])[0]
        return self.test_sections["SectionID"].values

    @property
    def pixel_train(self):
        """
        Returns the pixel coordinates for the training set as a numpy array.
        """
        if self.pixel_coordinates_train is None:
            self.load_train_pixel_coordinates()
        return self.pixel_coordinates_train

    @property
    def pixel_test(self):
        """
        Returns the pixel coordinates for the test set as a numpy array.
        """
        if self.pixel_coordinates_test is None:
            self.load_test_pixel_coordinates()
        return self.pixle_coordinates_test

    @property
    def train_mean(self):
        """
        Returns the mean of the training data.
        """
        if self.col_means is None:
            self.col_means = torch.load(self.config.exp_path / "lipid_means.pth")
        return self.col_means.to(self.config.device)

    @property
    def train_std(self):
        """
        Returns the standard deviation of the training data.
        """
        if self.col_stds is None:
            self.col_stds = torch.load(self.config.exp_path / "lipid_stds.pth")
        return self.col_stds.to(self.config.device)

    def plot_lipid_distribution(self, sections: list[int], selected_lipid_indexes: list[int], dataset="train", add_scatter=False):
        """
        Generates a multi-column, two-row figure displaying the spatial distribution of
        true and predicted lipid values for multiple sections and lipids.

        Each column represents a (section, lipid_index) pair.
        The top row shows true values, and the bottom row shows predicted values.
        The coordinates (zccf, -yccf) are plotted, colored by lipid concentration.
        Each column has its own thin, horizontal color bar positioned above its top subplot,
        displaying only the two extreme values relevant to that column's lipid data.

        Args:
            sections (list[int]): A list of section numbers to filter the data by.
            selected_lipid_indexes (list[int]): A list of indices of the lipids to plot,
                                                corresponding to experiment.config.selected_lipids_names.
        """
        exp_path = self.config.exp_path
        section_id = self.sectionid_train if dataset == "train" else self.sectionid_test
        if len(sections) != len(selected_lipid_indexes):
            print("Error: The 'sections' list and 'selected_lipid_indexes' list must have the same length.")
            return

        num_columns = len(sections)
        if num_columns == 0:
            print("No sections and lipid indexes provided for plotting.")
            return

        predictions_path = exp_path / "train"

        # Load raw data from .npy files
        true_values_raw = self.true_values_train if dataset == "train" else self.true_values_test
        predictions_raw = self.predictions_train if dataset == "train" else self.predictions_test

        # Convert numpy arrays to pandas DataFrames with appropriate column names
        true_values_df = pd.DataFrame(data=true_values_raw, columns=self.config.selected_lipids_names)
        predictions_df = pd.DataFrame(data=predictions_raw, columns=self.config.selected_lipids_names)
        coordinates_df = pd.DataFrame(data=self.coordinates_train, columns=["xccf","yccf","zccf"])
        sections_df = pd.DataFrame(data=section_id, columns=["Section"])

        # Reset indices to ensure proper concatenation without misalignment
        sections_df = sections_df.reset_index(drop=True)
        true_values_df = true_values_df.reset_index(drop=True)
        predictions_df = predictions_df.reset_index(drop=True)
        coordinates_df = coordinates_df.reset_index(drop=True)

        # Concatenate all relevant data into single DataFrames for easier filtering
        true_values_full = pd.concat([true_values_df, sections_df, coordinates_df], axis=1)
        predictions_full = pd.concat([predictions_df, sections_df, coordinates_df], axis=1)

        # Create a figure with two rows and `num_columns` columns
        # Increased figsize height slightly to accommodate individual colorbars better
        if add_scatter:
        # Ensure axes is always a 2D array, even for a single column
            fig, axes = plt.subplots(3, num_columns, figsize=(5 * num_columns, 16))
        else:
            fig, axes = plt.subplots(2, num_columns, figsize=(5 * num_columns, 12))
        if num_columns == 1:
            axes = np.array([axes]).reshape(2, 1)

        # Iterate through each column (each section-lipid pair)
        for col_idx in range(num_columns):
            section = sections[col_idx]
            selected_lipid_index = selected_lipid_indexes[col_idx]

            try:
                current_lipid = self.config.selected_lipids_names[selected_lipid_index]
            except IndexError:
                print(f"Invalid lipid index: {selected_lipid_index} for column {col_idx+1}. Please choose an index within the range [0, {len(self.config.selected_lipids_names) - 1}]. Skipping this column.")
                axes[0, col_idx].set_title(f'Skipped: Invalid Lipid Index', fontsize=12, color='red')
                axes[1, col_idx].set_title(f'Skipped: Invalid Lipid Index', fontsize=12, color='red')
                # Hide axes ticks/labels for skipped columns for cleaner look
                axes[0, col_idx].set_xticks([])
                axes[0, col_idx].set_yticks([])
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_yticks([])
                continue

            true_values_section = true_values_full[true_values_full.Section == section]
            predictions_section = predictions_full[predictions_full.Section == section]

            if true_values_section.empty or predictions_section.empty:
                print(f"No data found for Section {section} for column {col_idx+1}. Skipping this column.")
                axes[0, col_idx].set_title(f'Skipped: No Data for Sec {section}', fontsize=12, color='red')
                axes[1, col_idx].set_title(f'Skipped: No Data for Sec {section}', fontsize=12, color='red')
                # Hide axes ticks/labels for skipped columns for cleaner look
                axes[0, col_idx].set_xticks([])
                axes[0, col_idx].set_yticks([])
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_yticks([])
                continue

            # Determine min and max values for the current column's lipid data
            column_lipid_values = pd.concat([
                true_values_section[current_lipid],
                predictions_section[current_lipid]
            ])
            column_min_val = column_lipid_values.min()
            column_max_val = column_lipid_values.max()

            # --- Plot for True Values (Top Row) ---
            scatter_true = axes[0, col_idx].scatter(
                true_values_section.zccf,
                -true_values_section.yccf,
                c=true_values_section[current_lipid].values,
                s=0.5,
                cmap='viridis',
                alpha=0.8,
                vmin=column_min_val, # Use column-specific min/max
                vmax=column_max_val  # Use column-specific min/max
            )
            axes[0, col_idx].set_title(f'True: {current_lipid}\n(Sec {section})', fontsize=12)
            axes[0, col_idx].set_xlabel('Z-coordinate (zccf)', fontsize=10)
            axes[0, col_idx].set_ylabel('-Y-coordinate (-yccf)', fontsize=10)
            axes[0, col_idx].set_aspect('equal', adjustable='box')
            axes[0, col_idx].grid(True, linestyle='--', alpha=0.6)

            # --- Plot for Predicted Values (Bottom Row) ---
            scatter_pred = axes[1, col_idx].scatter(
                predictions_section.zccf,
                -predictions_section.yccf,
                c=predictions_section[current_lipid].values,
                s=0.5,
                cmap='viridis',
                alpha=0.8,
                vmin=column_min_val, # Use column-specific min/max
                vmax=column_max_val  # Use column-specific min/max
            )
            axes[1, col_idx].set_title(f'Pred: {current_lipid}\n(Sec {section})', fontsize=12)
            axes[1, col_idx].set_xlabel('Z-coordinate (zccf)', fontsize=10)
            axes[1, col_idx].set_ylabel('-Y-coordinate (-yccf)', fontsize=10)
            axes[1, col_idx].set_aspect('equal', adjustable='box')
            axes[1, col_idx].grid(True, linestyle='--', alpha=0.6)

            # --- Individual Horizontal Colorbar for the current column ---
            pos_top_subplot = axes[0, col_idx].get_position()
            # Define position for the colorbar above the current column's top subplot
            # Adjusted 'bottom' slightly up (from 0.05 to 0.06) for more space
            colorbar_ax_position = [pos_top_subplot.x0, pos_top_subplot.y1 + 0.06, pos_top_subplot.width, 0.01]

            cbar_ax = fig.add_axes(colorbar_ax_position)
            fig.colorbar(scatter_true, cax=cbar_ax, orientation='horizontal',
                         label=f'{current_lipid} Concentration', # Label specific to this lipid
                         ticks=[column_min_val, column_max_val])
            cbar_ax.xaxis.set_ticks_position('top') # Place ticks on top of the colorbar
            cbar_ax.xaxis.set_label_position('top') # Place label on top of the colorbar
            if add_scatter:
                # --- Scatter Plot for the current section and lipid ---
                scatter_ax = axes[2, col_idx]
                true_values_mask = true_values_section[current_lipid].values > 1e-5
                predictions_mask = predictions_section[current_lipid].values > 1e-5
                final_mask = true_values_mask & predictions_mask
                true_value = true_values_section[current_lipid].values[final_mask]
                predicted_value = predictions_section[current_lipid].values[final_mask]
                true_min = np.nanmin(true_value)
                true_max = np.nanmax(true_value)
                predicted_min = np.nanmin(predicted_value)
                predicted_max = np.nanmax(predicted_value)
                correlation = true_values_section[current_lipid].corr(predictions_section[current_lipid],method='spearman')
                scatter_ax.plot([true_min, true_max],[true_min, true_max], "k--", lw=2)
                density_scatter(scatter_ax, true_value, predicted_value,
                                x_min=true_min, x_max=true_max,
                                y_min=predicted_min, y_max=predicted_max,
                                s=0.1, alpha=0.1)
                scatter_ax.set_title(f"True: {current_lipid} vs {current_lipid} Predicted\nCorrelation: {correlation:.2f}")
                scatter_ax.set_xlabel(f"{current_lipid} (true)")
                scatter_ax.set_ylabel(f"{current_lipid} (predicted)")
                scatter_ax.set_yscale('log')
                scatter_ax.set_xscale('log')
        # Adjust layout to make space for the top colorbars and prevent overlap
        # Increased rect top margin to 0.9 for more overall space
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.suptitle('Lipid Distribution: True vs. Predicted Values', y=0.99, fontsize=16) # Overall title

    def plot_lipid_scatter(self, lipid, dataset="train"):
        """
        Generates a scatter plot for the lipid distribution in the specified sections and lipids.

        Args:
            sections (list[int]): A list of section numbers to filter the data by.
            selected_lipid_indexes (list[int]): A list of indices of the lipids to plot,
                                                corresponding to experiment.config.selected_lipids_names.
        """
        true_values = self.true_values_train if dataset == "train" else self.true_values_test
        predictions = self.predictions_train if dataset == "train" else self.predictions_test
        fig, ax = plt.subplots(figsize=(10, 10))
        if lipid not in self.config.selected_lipids_names:
            print(f"Lipid {lipid} not found in the selected lipids.")
            return
        lipid_index = self.config.selected_lipids_names.index(lipid)
        true_values_lipid = true_values[:, lipid_index]
        predictions_lipid = predictions[:, lipid_index]
        true_min = np.nanmin(true_values_lipid)
        true_max = np.nanmax(true_values_lipid)
        predicted_min = np.nanmin(predictions_lipid)
        predicted_max = np.nanmax(predictions_lipid)
        correlation = np.corrcoef(true_values_lipid, predictions_lipid)[0, 1]
        ax.plot([true_min, true_max],[true_min, true_max], "k--", lw=2)
        density_scatter(ax, true_values_lipid, predictions_lipid,
                        x_min=true_min, x_max=true_max,
                        y_min=predicted_min, y_max=predicted_max,
                        s=0.1, alpha=0.1)
        ax.set_title(f"True: {lipid} vs {lipid} Predicted\nCorrelation: {correlation:.2f}")
        ax.set_xlabel(f"{lipid} (true)")
        ax.set_ylabel(f"{lipid} (predicted)")
        plt.show()

    def whole_brain_reconstruction(self, lipid_indices=None, lipid_names=None):
        """One-pass reconstruction. By default reconstructs all lipids; pass
        lipid_indices (list of int) or lipid_names (list of str) to restrict.

        Filtering happens AFTER the GPU forward pass — the model still predicts
        all 172 lipids per voxel; we just persist a subset. This is intentional:
        re-running the model is far more expensive than discarding columns, and
        the consolidated `all_predictions.npy` then contains the full prediction
        cube, so adding a lipid later only re-writes per-lipid volumes (no GPU
        needed).

        If you want to ACTUALLY save GPU memory or disk by not even predicting
        the unused lipids, see _make_lipid_filter — but for a standard LGP this
        isn't a win because the decoder produces all 172 lipids in one matmul.
        """
        if not self._load_model_for_reconstruction():
            return
        suffix = "_diffusion" if self.config.use_diffusion else ""
        volume_path = self.config.exp_path / ("volume_diffusion"
                                            if self.config.use_diffusion else "volume")
        volume_path.mkdir(parents=True, exist_ok=True)
        self.config.reference_file
        template_volume = np.load(self.config.reference_file)
        non_zero_indices = np.argwhere(template_volume > 5).astype(np.int32)
        logging.info(f"Whole-brain reconstruction: {non_zero_indices.shape[0]:,} voxels")

        lipid_filter = self._resolve_lipid_filter(lipid_indices, lipid_names)
        # faiss_cpu_recon() pins KNN searches to CPU iff --faiss-cpu-recon is set.
        with faiss_cpu_recon():
            self._reconstruct_voxels(
                non_zero_indices, volume_path, template_volume.shape, suffix=suffix,
                lipid_filter=lipid_filter,
            )
        self.render_reconstruction(template_volume, volume_path, lipid_filter, suffix)

    def region_reconstruction(self, region_bbox, threshold=5.0,
                            lipid_indices=None, lipid_names=None):
        """Same filtering for the regional path."""
        if not self._load_model_for_reconstruction():
            return
        bbox_str = "_".join(str(int(b)) for b in region_bbox)
        volume_path = self.config.exp_path / f"volume_region_{bbox_str}"
        volume_path.mkdir(parents=True, exist_ok=True)
        template_volume = np.load(self.config.reference_file)
        zmin, zmax, ymin, ymax, xmin, xmax = (int(b) for b in region_bbox)
        sub = template_volume[zmin:zmax, ymin:ymax, xmin:xmax]
        z, y, x = np.where(sub > threshold)
        if z.shape[0] == 0:
            logging.warning(f"No tissue voxels (>{threshold}) inside bbox {region_bbox}.")
            return
        non_zero_indices = np.stack([z + zmin, y + ymin, x + xmin], axis=1).astype(np.int32)
        logging.info(f"Region reconstruction: {non_zero_indices.shape[0]:,} voxels in {region_bbox}")
        lipid_filter = self._resolve_lipid_filter(lipid_indices, lipid_names)
        with faiss_cpu_recon():
            self._reconstruct_voxels(
                non_zero_indices, volume_path, template_volume.shape,
                suffix=f"_region_{bbox_str}", lipid_filter=lipid_filter,
            )


    def _resolve_lipid_filter(self, lipid_indices=None, lipid_names=None):
        """Resolve lipid filter precedence: explicit arg > config > all.

        Returns a numpy int array of indices, or None to mean 'all lipids'.
        Errors loudly on miss / ambiguity (no silent drops).
        """
        # Explicit args take precedence over config
        if lipid_indices is None and lipid_names is None:
            cfg_val = getattr(self.config, "reconstruction_lipids", None)
            if cfg_val is None or len(cfg_val) == 0:
                return None
            # Allow either int or str entries in the config list
            if all(isinstance(v, int) for v in cfg_val):
                lipid_indices = list(cfg_val)
            elif all(isinstance(v, str) for v in cfg_val):
                lipid_names = list(cfg_val)
            else:
                raise ValueError("config.reconstruction_lipids must be all int or all str")

        all_names = [str(n) for n in self.config.selected_lipids_names]

        if lipid_names is not None:
            resolved = []
            for name in lipid_names:
                target = name.strip()
                exact = [i for i, n in enumerate(all_names) if n == target]
                if len(exact) == 1:
                    resolved.append(exact[0]); continue
                if len(exact) > 1:
                    raise ValueError(f"{name!r} matches multiple lipids exactly: {exact}")
                sub = [i for i, n in enumerate(all_names) if target.lower() in n.lower()]
                if len(sub) == 0:
                    raise ValueError(f"No lipid matches {name!r}. First 10 names: "
                                    f"{all_names[:10]}")
                if len(sub) > 1:
                    raise ValueError(
                        f"Ambiguous lipid name {name!r}; matches: "
                        + ", ".join(f"[{i}]{all_names[i]}" for i in sub)
                    )
                resolved.append(sub[0])
            idx = np.asarray(resolved, dtype=np.int64)
        else:
            idx = np.asarray(lipid_indices, dtype=np.int64)
            if idx.min() < 0 or idx.max() >= len(all_names):
                raise ValueError(f"lipid index out of range [0, {len(all_names)})")

        # Dedupe but preserve order
        _, keep = np.unique(idx, return_index=True)
        idx = idx[np.sort(keep)]

        selected = [all_names[i] for i in idx]
        logging.info(f"Reconstruction restricted to {len(idx)} lipid(s): {selected}")
        return idx

    def _load_model_for_reconstruction(self):
        model_path = self.config.exp_path / "model.pth"
        if not model_path.exists():
            logging.error("Model not found. Please run the experiment first.")
            return False
        logging.info(f"Loading model from {model_path}")
        self.lgp_model.load_state_dict(
            torch.load(model_path, map_location=self.config.device)
        )
        return True

    def _reconstruct_voxels(self, non_zero_indices, volume_path, template_shape,
                            suffix="", lipid_filter=None):
        """Core reconstruction loop. One forward pass over all voxels, then
        write everything.

        Memory model: when lipid_filter is set, we only allocate / save / load
        those columns. The forward pass still runs the full decoder (one matmul
        per batch produces all lipids — slicing them out doesn't speed it up),
        but everything downstream operates on the subset.

        Resume: the cached predictions array is shape (N, len(filter)) when
        filtered, (N, n_lipids) when not. The two are not interchangeable —
        if you re-run with a different filter and the cache shape mismatches,
        we redo the GPU loop.
        """
        n_voxels = non_zero_indices.shape[0]
        all_names = list(self.config.selected_lipids_names)
        n_lipids_total = len(all_names)

        if lipid_filter is None:
            col_indices = np.arange(n_lipids_total, dtype=np.int64)
            filter_tag = ""
        else:
            col_indices = np.asarray(lipid_filter, dtype=np.int64)
            # Build a tag so cached files for different filters don't collide
            if len(col_indices) <= 4:
                filter_tag = "_lipids_" + "_".join(str(i) for i in col_indices)
            else:
                import hashlib
                h = hashlib.sha1(col_indices.tobytes()).hexdigest()[:8]
                filter_tag = f"_lipids_{len(col_indices)}_{h}"
        n_lipids_out = len(col_indices)

        preds_file       = volume_path / f"predictions{filter_tag}.npy"
        indices_file     = volume_path / "predictions_indices.npy"
        filter_meta_file = volume_path / f"predictions{filter_tag}.lipidcols.npy"

        # --- Resume: same voxel set AND same lipid filter ---
        if preds_file.exists() and indices_file.exists():
            saved_indices = np.load(indices_file)
            same_voxels = (saved_indices.shape == non_zero_indices.shape and
                        np.array_equal(saved_indices, non_zero_indices))
            same_filter = True
            if filter_meta_file.exists():
                saved_cols = np.load(filter_meta_file)
                same_filter = (saved_cols.shape == col_indices.shape and
                            np.array_equal(saved_cols, col_indices))
            if same_voxels and same_filter:
                logging.info(f"Reusing cached predictions at {preds_file}")
                all_preds = np.load(preds_file, mmap_mode="r")
                self._write_per_lipid_volumes(
                    volume_path, all_preds, non_zero_indices, template_shape, suffix,
                    col_indices=col_indices,
                )
                return
            else:
                logging.warning(
                    f"Cache mismatch (same_voxels={same_voxels}, same_filter={same_filter}); "
                    "recomputing."
                )

        # --- Diffusion setup ---
        ddpm = unet = true_values_min = true_values_max = padding = None
        pad_len = 0
        if self.config.use_diffusion:
            from l3di.simple_ddpm1d import SimpleDDPM1D, SimpleUNet1D
            unet = SimpleUNet1D(in_channels=1, model_channels=64, out_channels=1,
                                z_dim=self.config.latent_dim)
            ddpm = SimpleDDPM1D(unet, T=1000, var_type="fixedsmall")
            voxel_model_file = self.config.exp_path / "voxel_diffusion.pt"
            if not voxel_model_file.exists():
                logging.error("Voxel diffusion model not found.")
                return
            ddpm.load_model(voxel_model_file, self.config.device)
            true_values = torch.tensor(self.true_values_train)
            true_values_min = true_values.min()
            true_values_max = true_values.max()
            n_channels = true_values.shape[1]
            target_len = ((n_channels - 1) // 16 + 1) * 16
            pad_len = target_len - n_channels
            padding = (0, pad_len)

        # --- Coordinates + dataloader ---
        non_zero_ccf = torch.tensor(non_zero_indices.astype(np.float32) * 0.025,
                                    dtype=torch.float32)
        non_zero_ccf = (non_zero_ccf - self.coord_mean) / self.coord_std
        ccf_dataset = torch.utils.data.TensorDataset(
            non_zero_ccf, torch.tensor(non_zero_indices, dtype=torch.int32),
        )
        ccf_loader = torch.utils.data.DataLoader(
            ccf_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0,
        )

        # --- Pre-allocate accumulator (filtered width only) ---
        pred_bytes = n_voxels * n_lipids_out * 4
        logging.info(
            f"Allocating {pred_bytes / 1e9:.2f} GB for predictions "
            f"({n_voxels:,} voxels × {n_lipids_out} lipid{'s' if n_lipids_out != 1 else ''})"
        )
        all_preds = np.empty((n_voxels, n_lipids_out), dtype=np.float32)

        # Slice de-standardization params to the requested columns
        train_mean_np = self.train_mean.cpu().numpy()[col_indices]
        train_std_np  = self.train_std.cpu().numpy()[col_indices]

        cursor = 0
        self.lgp_model.eval()
        with torch.no_grad():
            for batch in tqdm(ccf_loader, desc="reconstructing voxels"):
                coords = batch[0].to(self.config.device)
                preds, gp_posterior = self.lgp_model.predict(coords)

                if self.config.use_diffusion and ddpm is not None:
                    cond = preds
                    x_t = torch.randn(cond.shape[0], 1, cond.shape[1] + pad_len,
                                    device=self.config.device)
                    cond_padded = F.pad(cond, padding, "constant", 0).unsqueeze(1)
                    sample_dict = ddpm.sample(
                        x_t, cond_padded, gp_posterior.mean,
                        n_steps=500, clip_denoised=True,
                    )
                    ddpm_pred = sample_dict["500"].squeeze().detach().cpu()
                    if pad_len > 0:
                        ddpm_pred = ddpm_pred[:, :-pad_len]
                    ddpm_pred = ddpm_pred * (true_values_max - true_values_min) + true_values_min
                    full_preds_np = (cond.detach().cpu() - ddpm_pred).numpy()
                else:
                    full_preds_np = preds.detach().cpu().numpy()

                # Slice to the requested lipids BEFORE de-standardization (so we
                # never materialize the full (B, 172) post-processed cube)
                sliced = full_preds_np[:, col_indices]
                sliced = sliced * train_std_np + train_mean_np
                if self.config.log_transform:
                    sliced = np.exp(sliced) - 1e-10

                n_b = sliced.shape[0]
                all_preds[cursor:cursor + n_b] = sliced.astype(np.float32, copy=False)
                cursor += n_b

        # --- Save consolidated ---
        np.save(preds_file, all_preds)
        np.save(indices_file, non_zero_indices)
        np.save(filter_meta_file, col_indices)
        logging.info(
            f"Saved predictions: {all_preds.shape} → {preds_file}  "
            f"(lipids: {[all_names[i] for i in col_indices]})"
        )

        # --- Per-lipid volumes ---
        self._write_per_lipid_volumes(
            volume_path, all_preds, non_zero_indices, template_shape, suffix,
            col_indices=col_indices,
        )

    def _write_per_lipid_volumes(self, volume_path, predictions, indices, template_shape,
                                suffix="", col_indices=None):
        """predictions: (N, K) where K = len(col_indices). col_indices maps each
        column back to its lipid index in selected_lipids_names."""
        z, y, x = indices[:, 0], indices[:, 1], indices[:, 2]
        if col_indices is None:
            col_indices = np.arange(predictions.shape[1], dtype=np.int64)

        for col_pos, lipid_global_idx in enumerate(tqdm(col_indices,
                                                        desc="per-lipid volumes")):
            lipid_name = self.config.selected_lipids_names[int(lipid_global_idx)]
            out_path = volume_path / f"{lipid_name}_volume{suffix}.npy"
            out_255  = volume_path / f"{lipid_name}_volume255{suffix}.npy"
            if out_path.exists() and out_255.exists():
                continue
            vol = np.full(template_shape, np.nan, dtype=np.float32)
            vol[z, y, x] = predictions[:, col_pos]
            np.save(out_path, vol)
            vmin = np.nanmin(vol); vmax = np.nanmax(vol)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                np.save(out_255, 255.0 * (vol - vmin) / (vmax - vmin))
            else:
                np.save(out_255, vol)

    def render_reconstruction(self, template_volume, volume_path, lipid_filter,
                              region_suffix: str = ""):
        """Render per-lipid composite PNGs after reconstruction. Honors the
        same lipid filter resolution as whole_brain_reconstruction."""
        from render_lipid_volumes import (
            render_selected_lipids, render_lipid_diagnostics, render_error_slice,
        )
        suffix = "_diffusion" if self.config.use_diffusion else region_suffix
        renders_dir = self.config.exp_path / "renders"

        render_selected_lipids(
            template_volume=template_volume,
            volume_dir=volume_path,
            output_dir=renders_dir,
            selected_lipids_names=self.config.selected_lipids_names,
            lipid_indices=list(lipid_filter),
            suffix=suffix,
            n_rotation_frames=10,
        )

        # Per-lipid diagnostics (value distribution + true-vs-pred scatter,
        # linear & log) for the reconstructed lipids. Uses the held-out TEST
        # points, which carry both a measurement and a prediction in the raw
        # (de-standardized) intensity scale. Non-fatal: a plotting hiccup must
        # never invalidate a completed reconstruction.
        try:
            names = list(self.config.selected_lipids_names)
            filt = (list(lipid_filter) if lipid_filter is not None
                    else list(range(len(names))))
            true_te = self.true_values_test
            pred_te = self.predictions_test
            if true_te is not None and pred_te is not None:
                for gi in filt:
                    render_lipid_diagnostics(
                        true_te[:, gi], pred_te[:, gi], str(names[gi]),
                        renders_dir / f"{names[gi]}_diagnostics.png",
                    )
                # One MALDI section as a reconstruction-error heatmap, from the
                # same held-out TEST points (row-aligned ccf mm coordinates).
                coords_te = self.ccf_test
                for gi in filt:
                    render_error_slice(
                        coords_te, true_te[:, gi], pred_te[:, gi], str(names[gi]),
                        renders_dir / f"{names[gi]}_error_slice.png",
                        axis_labels=("xccf", "yccf", "zccf"),
                    )
            else:
                logging.warning("Skipping diagnostics: test predictions/true "
                                "values not found (run the experiment first).")
        except Exception as e:
            logging.warning(f"Reconstruction diagnostics plotting failed: {e}")

    def load_whole_brain_reconstruction(self, lipid):
        """Return per-lipid volume; builds on demand from consolidated array."""
        suffix = "_diffusion" if self.config.use_diffusion else ""
        lipid_name = self.config.selected_lipids_names[lipid]
        lipid_volume_name = self.config.exp_path / f"{lipid_name}_volume{suffix}.npy"
        if lipid_volume_name.exists():
            return np.load(lipid_volume_name)

        volume_path = self.config.exp_path / ("volume_diffusion"
                                            if self.config.use_diffusion else "volume")
        preds_file = volume_path / "all_predictions.npy"
        idx_file   = volume_path / "all_indices.npy"
        template_file = self.config.reference_file
        if not (preds_file.exists() and idx_file.exists() and template_file.exists()):
            logging.error("No consolidated predictions found. Run whole_brain_reconstruction().")
            return None
        template_shape = np.load(template_file).shape
        indices = np.load(idx_file)
        predictions = np.load(preds_file, mmap_mode="r")
        vol = np.full(template_shape, np.nan, dtype=np.float32)
        vol[indices[:, 0], indices[:, 1], indices[:, 2]] = predictions[:, lipid]
        np.save(lipid_volume_name, vol)
        return vol

    def load_posterior_uncertainty(self, latent_idx=None):
        """Return a 3D volume of GP posterior uncertainty.

        Args:
            latent_idx: int or None. None → returns the total variance summed
                        over all latents (the 'overall uncertainty' map).
                        int → returns the variance for that specific latent dim.
        """
        suffix = "_diffusion" if self.config.use_diffusion else ""
        if latent_idx is None:
            path = self.config.exp_path / f"posterior_total_variance{suffix}.npy"
        else:
            path = self.config.exp_path / f"latent_{latent_idx:02d}_var{suffix}.npy"
        if path.exists():
            return np.load(path)

        # Build on demand from consolidated array
        volume_path = self.config.exp_path / ("volume_diffusion"
                                            if self.config.use_diffusion else "volume")
        pv_file = volume_path / "predictions_posterior_var.npy"
        idx_file = volume_path / "predictions_indices.npy"
        template_file = self.config.reference_file
        if not (pv_file.exists() and idx_file.exists() and template_file.exists()):
            return None
        template_shape = np.load(template_file).shape
        indices = np.load(idx_file)
        post_var = np.load(pv_file, mmap_mode="r")
        vol = np.full(template_shape, np.nan, dtype=np.float32)
        if latent_idx is None:
            vol[indices[:, 0], indices[:, 1], indices[:, 2]] = np.asarray(post_var).sum(axis=1)
        else:
            vol[indices[:, 0], indices[:, 1], indices[:, 2]] = post_var[:, latent_idx]
        return vol

    def train_fit(self):
            with torch.no_grad():
                # Pull a small sample of training coords. Reuse the loader the experiment
                # already built to avoid re-reading the parquet.
                #coords_train = experiment.coordinates_train.to(experiment.config.device)
                x1 = self.coordinates_train[:50].to(self.config.device).contiguous()
                x2 = self.coordinates_train[50:100].to(self.config.device).contiguous()
                # Inducing GPs expose the prior kernel as covar_module (a
                # ScaleKernel); SpectralLatentGP has no covar_module and holds the
                # RiemannMaternKernel directly as .kernel. Either way this probes
                # the PRIOR kernel spread.
                kernel = getattr(self.lgp_model.gp_model, "covar_module", None)
                if kernel is None:
                    kernel = self.lgp_model.gp_model.kernel
                K = kernel(x1, x2).evaluate()
                print(f"[init kernel spread] K shape:  {tuple(K.shape)}")
                print(f"[init kernel spread] K mean:   {K.mean().item():.4g}")
                print(f"[init kernel spread] K std:    {K.std().item():.4g}")
                print(f"[init kernel spread] K min/max:{K.min().item():.4g} / {K.max().item():.4g}")
                print(f"[init kernel spread] std/|mean|: {(K.std() / K.abs().mean()).item():.4g}")

            # let's use ADAMW instead of ADAM
            optimizer = torch.optim.AdamW(self.lgp_model.parameters(), lr=self.config.learning_rate, weight_decay=1e-3)
            logging.info("ready to roll")


            dataloader_train= torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0
            )

            logging.info("Model file not found, starting training from scratch")
            if self.current_epoch < self.config.epochs:
                logging.info(f"Starting training from epoch {self.current_epoch}")
                with gpytorch.settings.cholesky_jitter(1e-3, 1e-4):
                    self.lgp_model.train_model(self.config.exp_path,
                                        dataloader_train,
                                        optimizer,
                                        epochs=self.config.epochs,
                                        current_epoch=self.current_epoch,
                                        print_every=1000)
                    
            with torch.no_grad():
                x1 = self.coordinates_train[:50].to(self.config.device).contiguous()
                x2 = self.coordinates_train[50:100].to(self.config.device).contiguous()
                # Inducing GPs expose the prior kernel as covar_module (a
                # ScaleKernel); SpectralLatentGP has no covar_module and holds the
                # RiemannMaternKernel directly as .kernel. Either way this probes
                # the PRIOR kernel spread.
                kernel = getattr(self.lgp_model.gp_model, "covar_module", None)
                if kernel is None:
                    kernel = self.lgp_model.gp_model.kernel
                K = kernel(x1, x2).evaluate()
                print(f"[trained kernel spread] K shape:  {tuple(K.shape)}")
                print(f"[trained kernel spread] K mean:   {K.mean().item():.4g}")
                print(f"[trained kernel spread] K std:    {K.std().item():.4g}")
                print(f"[trained kernel spread] std/|mean|: {(K.std() / K.abs().mean()).item():.4g}")


    def predict_original_scale(self, train_predictions, test_predictions):
        if self.train_data is None:
            self.load_train_data()
        if self.test_data is None:
            self.load_test_data()
        # evaluate predictions:
        logging.info("Evaluating predictions")
        train_predictions = train_predictions * self.col_stds + self.col_means
        test_predictions = test_predictions * self.col_stds + self.col_means
        if self.config.log_transform:
            train_predictions = np.exp(train_predictions) - 1e-10
            test_predictions = np.exp(test_predictions) - 1e-10
        train_data = self.train_data * self.col_stds + self.col_means
        test_data = self.test_data * self.col_stds + self.col_means
        if self.config.log_transform:
            train_data = np.exp(train_data) - 1e-10
            test_data = np.exp(test_data) - 1e-10
        # save data and predictions in the original scale
        self.train_data_original = train_data.numpy()
        self.test_data_original = test_data.numpy()
        np.save(self.config.exp_path / "train" / "predictions.npy", train_predictions.numpy())
        np.save(self.config.exp_path / "test" / "predictions.npy", test_predictions.numpy())
        np.save(self.config.exp_path / "train" / "true_values.npy", train_data.numpy())
        np.save(self.config.exp_path / "test" / "true_values.npy", test_data.numpy())

    def predict_original_scale_(self, train_predictions, test_predictions):
        if self.train_data is None:
            self.load_train_data()
        if self.test_data is None:
            self.load_test_data()
            
        logging.info("Evaluating predictions (Memory Safe Mode)")
        
        # Ensure std and mean are on CPU
        stds = self.col_stds.cpu()
        means = self.col_means.cpu()

        # 1. IN-PLACE math on predictions (use .mul_() and .add_())
        train_predictions.mul_(stds).add_(means)
        test_predictions.mul_(stds).add_(means)

        if self.config.log_transform:
            # IN-PLACE NumPy exponentiation
            np.exp(train_predictions.numpy(), out=train_predictions.numpy())
            train_predictions.sub_(1e-10)
            
            np.exp(test_predictions.numpy(), out=test_predictions.numpy())
            test_predictions.sub_(1e-10)

        # 2. IN-PLACE math on original data
        self.train_data.mul_(stds).add_(means)
        self.test_data.mul_(stds).add_(means)

        if self.config.log_transform:
            np.exp(self.train_data.numpy(), out=self.train_data.numpy())
            self.train_data.sub_(1e-10)
            
            np.exp(self.test_data.numpy(), out=self.test_data.numpy())
            self.test_data.sub_(1e-10)

        # 3. Save to disk (Releasing memory as it goes)
        logging.info("Saving numpy files to disk...")
        self.train_data_original = self.train_data.numpy()
        self.test_data_original = self.test_data.numpy()
        
        np.save(self.config.exp_path / "train" / "predictions.npy", train_predictions.numpy())
        np.save(self.config.exp_path / "test" / "predictions.npy", test_predictions.numpy())
        np.save(self.config.exp_path / "train" / "true_values.npy", self.train_data_original)
        np.save(self.config.exp_path / "test" / "true_values.npy", self.test_data_original)
        logging.info("Predictions saved successfully.")

    def run(self):
        """Run the experiment."""
        
        # Fit he model using the training data,
        logging.info("Starting training loop VAE")
        if(self.config.exp_path / "model.pth").exists():
            logging.info("Loading model from file")
            self.lgp_model.load_state_dict(torch.load(self.config.exp_path / "model.pth", map_location=self.config.device))
            logging.info("Model loaded successfully")
        else:
            self.load_train_data()
            self.current_epoch = 0
            self.load_checkpoint()
            wandb.init(name=self.config.exp_name,
                       project="l3di_maldi",
                       config=self.config.to_dict()
                       )
            self.train_fit()
            wandb.finish()
            logging.info("Training completed, saving model")

        # Predict in the train set
        logging.info("Predicting in the train set")
        train_path = self.config.exp_path / "train"
        train_path.mkdir(parents=True, exist_ok=True)
        train_predictions_file = train_path / "predictions.pth"
        if not train_predictions_file.exists():
            self.load_coord_train_data()
            self.load_train_pixel_index_coordinates()
            logging.info("Predicting in the train set")
            self.lgp_model.eval()
            train_pred_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.coordinates_train, self.pixel_index_coordinates_train),
                                                                batch_size=self.config.batch_size,
                                                                shuffle=False,
                                                                num_workers=0)
            logging.info("Predicting in the train set using the model")
            predictions_list = []
            posterior_list = []
            coordinates_list = []
            coordinates_pixel_index_list = []
            for batch in tqdm(train_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                coordinates_pixel_index = batch[1].to(self.config.device)
                predictions, gp_posterior = self.lgp_model.predict(coordinates_batch)
                coordinates_list.append(coordinates_batch.detach().cpu())
                coordinates_pixel_index_list.append(coordinates_pixel_index.detach().cpu())
                predictions_list.append(predictions.detach().cpu())
                posterior_list.append(gp_posterior.mean.detach().cpu())
            train_predictions = torch.cat(predictions_list, dim=0)
            posterior = torch.cat(posterior_list, dim=0)
            coordinates = torch.cat(coordinates_list, dim=0)
            coordinates_pixel_index = torch.cat(coordinates_pixel_index_list, dim=0)
            torch.save(train_predictions, train_predictions_file)
            torch.save(posterior, train_predictions_file.with_name("posterior.pth"))
            torch.save(coordinates, train_predictions_file.with_name("coordinates.pth"))
            torch.save(coordinates_pixel_index, train_predictions_file.with_name("coordinates_pixel_index.pth"))
        else:
            logging.info("Train predictions already exist, loading from file")
            train_predictions = torch.load(train_predictions_file)

        # Predict in the test set in the training data scale (not original scale)
        logging.info("Predicting in the test set")
        test_path = self.config.exp_path / "test"
        test_path.mkdir(parents=True, exist_ok=True)
        test_predictions_file = test_path / "predictions.pth"
        if not test_predictions_file.exists():
            self.load_coord_test_data()
            logging.info("Predicting in the test set")
            self.lgp_model.eval()
            self.load_test_pixel_index_coordinates()
            test_pred_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.coordinates_test, self.pixel_index_coordinates_test),
                                                               batch_size=self.config.batch_size,
                                                               shuffle=False,
                                                               num_workers=0)
            predictions_list = []
            posterior_list = []
            coordinates_list = []
            coordinates_pixel_index_list = []
            for batch in tqdm(test_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                coordinates_pixel_index = batch[1].to(self.config.device)
                predictions,posterior = self.lgp_model.predict(coordinates_batch)
                coordinates_list.append(coordinates_batch.detach().cpu())
                coordinates_pixel_index_list.append(coordinates_pixel_index.detach().cpu())
                predictions_list.append(predictions.detach().cpu())
                posterior_list.append(posterior.mean.detach().cpu())
            test_predictions = torch.cat(predictions_list, dim=0)
            posterior = torch.cat(posterior_list, dim=0)
            coordinates = torch.cat(coordinates_list, dim=0)
            coordinates_pixel_index = torch.cat(coordinates_pixel_index_list, dim=0)
            torch.save(test_predictions, test_predictions_file)
            torch.save(posterior, test_predictions_file.with_name("posterior.pth"))
            torch.save(coordinates, test_predictions_file.with_name("coordinates.pth"))
            torch.save(coordinates_pixel_index, test_predictions_file.with_name("coordinates_pixel_index.pth"))
        else:
            logging.info("Test predictions already exist, loading from file")
            test_predictions = torch.load(test_predictions_file)
        train_predictions_file = train_path / "predictions.npy"
        test_predictions_file = test_path / "predictions.npy"
        if not train_predictions_file.exists() or not test_predictions_file.exists():
            logging.info("Train and test numpy predictions do not exist, saving predictions to file")
            predict_original_scale = self.predict_original_scale_(train_predictions, test_predictions)

        # Per-lipid held-out metrics table (read by lgp_report.py). Best-effort:
        # never let metrics caching abort an otherwise-finished run.
        try:
            from lgp_metrics import write_metrics
            for _split in ("test", "train"):
                write_metrics(self.config.exp_path, _split,
                              lipid_names=self.config.selected_lipids_names)
            logging.info("Wrote per-lipid metrics.csv")
        except Exception as _e:  # noqa: BLE001
            logging.warning(f"metrics.csv generation failed: {_e}")

        if self.config.use_diffusion:
            logging.info("Using diffusion model for the experiment")
            from l3di.simple_ddpm1d import SimpleDDPM1D, SimpleUNet1D
            unet = SimpleUNet1D(in_channels=1, model_channels=64, out_channels=1, z_dim=self.config.latent_dim)
            ddpm = SimpleDDPM1D(unet,T=1000,var_type="fixedsmall")
            vae_model_weights = torch.load(self.config.exp_path / "model.pth")
            self.lgp_model.load_state_dict(vae_model_weights)
            vae=self.lgp_model
            voxel_model_file = self.config.exp_path / "voxel_diffusion.pt"
            
            # Load true_values in original scale for diffusion training
            # This follows voxel_diffusion.py which uses true_values.npy
            true_values = torch.tensor(np.load(self.config.exp_path / "train" / "true_values.npy"))
            true_values_min = true_values.min()
            true_values_max = true_values.max()
            
            # Determine padding based on channel count to match UNet requirements (often multiple of 16/32)
            # In voxel_diffusion.py, hardcoded padding=(0,3) for 173 channels => 176
            # We will use dynamic padding to nearest multiple of 16
            n_channels = true_values.shape[1]
            target_len = ((n_channels - 1) // 16 + 1) * 16
            pad_len = target_len - n_channels
            padding = (0, pad_len)
            logging.info(f"Using padding {padding} for channels {n_channels} -> {target_len}")

            if not voxel_model_file.exists():
                # Normalize [0, 1] using global min/max
                norm_values = (true_values - true_values_min) / (true_values_max - true_values_min)
                
                # Make sure coordinates are loaded
                if self.coordinates_train is None:
                    self.load_coord_train_data()
                    
                diffusion_dataset = torch.utils.data.TensorDataset(norm_values, self.coordinates_train)
                data_loader_train = torch.utils.data.DataLoader(
                    diffusion_dataset,
                    batch_size=100,
                    shuffle=True,
                    num_workers=0)
                
                # Train with schedule similar to voxel_diffusion.py?
                # Using simple training call as in existing code, but ensure params match
                ddpm.train(data_loader_train, epochs=10, lr=2e-4, vae=vae)
                ddpm.save_model(voxel_model_file)
            else:
                logging.info("Loading voxel diffusion model from file")
                ddpm.load_model(voxel_model_file, self.config.device)
                logging.info("Voxel diffusion model loaded successfully")

            # predict training set
            logging.info("Predicting in the training set using the diffusion model")
            train_predictions_diffusion_file = train_path / "predictions_diffusion.npy"
            
            if not train_predictions_diffusion_file.exists():
                if self.coordinates_train is None:
                    self.load_coord_train_data()
                    
                ccf_dataset = torch.utils.data.TensorDataset(self.coordinates_train)
                ccf_dataloader = torch.utils.data.DataLoader(ccf_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
                predictions_list = []
                
                with torch.no_grad():
                    for batch in tqdm(ccf_dataloader):
                        coordinates_batch = batch[0].to(self.config.device)
                        # LGP prediction (cond)
                        cond, gp_posterior = self.lgp_model.predict(coordinates_batch)
                        
                        # Prepare for DDPM
                        x_t = torch.randn(cond.shape[0], 1, cond.shape[1] + pad_len, device=self.config.device)
                        
                        # Apply padding to condition
                        cond_padded = F.pad(cond, padding, "constant", 0)
                        cond_padded = cond_padded.unsqueeze(1).to(self.config.device)
                        
                        # Sample
                        sample_dict = ddpm.sample(x_t, cond_padded, gp_posterior.mean, n_steps=500, clip_denoised=True)
                        
                        # Process output as per voxel_diffusion.py logic
                        cond_orig = cond.detach().cpu() # Original LGP output (standardized space)
                        
                        # Get DDPM prediction and unpad
                        predictions = sample_dict["500"].squeeze().detach().cpu()
                        if pad_len > 0:
                            predictions = predictions[:, :-pad_len]
                        
                        # Rescale to original domain [min, max]
                        predictions = predictions * (true_values_max - true_values_min) + true_values_min
                        
                        # Apply the subtraction logic from voxel_diffusion.py:
                        # predictions = cond - predictions
                        # Note: cond is in standardized space (mean=0, std=1), predictions is in Original space.
                        # This matches the notebook exactly, regardless of physical interpretation.
                        predictions = cond_orig - predictions
                        
                        # Un-standardize result
                        predictions = predictions * self.train_std.cpu() + self.train_mean.cpu()
                        
                        predictions = predictions.numpy()
                        if self.config.log_transform:
                            predictions = np.exp(predictions) - 1e-10
                            
                        predictions_list.append(predictions)
                        
                train_predictions_diffusion = np.concatenate(predictions_list, axis=0)
                np.save(train_predictions_diffusion_file, train_predictions_diffusion)

            # predict test set
            test_predictions_diffusion_file = test_path / "predictions_diffusion.npy"
            if not test_predictions_diffusion_file.exists():
                logging.info("Predicting in the test set using the diffusion model")
                self.load_coord_test_data()

                ccf_dataset = torch.utils.data.TensorDataset(self.coordinates_test)
                ccf_dataloader = torch.utils.data.DataLoader(ccf_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
                predictions_list = []
                
                with torch.no_grad():
                    for batch in tqdm(ccf_dataloader):
                        coordinates_batch = batch[0].to(self.config.device)
                        cond, gp_posterior = self.lgp_model.predict(coordinates_batch)
                        
                        x_t = torch.randn(cond.shape[0], 1, cond.shape[1] + pad_len, device=self.config.device)
                        
                        cond_padded = F.pad(cond, padding, "constant", 0)
                        cond_padded = cond_padded.unsqueeze(1).to(self.config.device)
                        
                        sample_dict = ddpm.sample(x_t, cond_padded, gp_posterior.mean, n_steps=500, clip_denoised=True)
                        
                        cond_orig = cond.detach().cpu()
                        predictions = sample_dict["500"].squeeze().detach().cpu()
                        if pad_len > 0:
                            predictions = predictions[:, :-pad_len]
                            
                        predictions = predictions * (true_values_max - true_values_min) + true_values_min
                        
                        # Apply same logic
                        predictions = cond_orig - predictions
                        predictions = predictions * self.train_std.cpu() + self.train_mean.cpu()
                        
                        predictions = predictions.numpy()
                        if self.config.log_transform:
                            predictions = np.exp(predictions) - 1e-10
                            
                        predictions_list.append(predictions)
                        
                test_predictions_diffusion = np.concatenate(predictions_list, axis=0)
                np.save(test_predictions_diffusion_file, test_predictions_diffusion)