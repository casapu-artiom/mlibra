"""
experiment.py
Main experiment logic and classes for the MALDI experiment.
"""
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from config import MaldiConfig
from l3di.lgp import LGP

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
    ax.plot([true_min, true_max],[true_min, true_max], "k--", lw=2)
    density_scatter(ax, true_value, predicted_value,
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

    def load_train_sections(self):
        self.train_sections = pd.read_parquet(self.config.maldi_file,
                                                  columns=["Section"],
                                                  filters=self.train_filter)

    def load_train_pixel_coordinates(self):
        logging.info("Loading training pixel coordinates")
        coordinates_names = ["x", "y"]
        self.pixel_coordinates_train = pd.read_parquet(self.config.maldi_file,
                                                    columns=coordinates_names,
                                                    filters=self.train_filter).values

    def load_train_samples(self):
        logging.info("Loading training samples")
        self.train_samples = pd.read_parquet(self.config.maldi_file,
                                             columns=["Sample"],
                                             filters=self.train_filter)

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
        if (self.config.exp_path / "lipid_means.pth").exists() and (self.config.exp_path / "lipid_std.pth").exists():
            logging.info("Loading column mean and std from files")
            col_means = torch.load(self.config.exp_path / "lipid_means.pth")
            col_std = torch.load(self.config.exp_path / "lipid_std.pth")
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
        if (self.config.exp_path / "lipid_means.pth").exists() and (self.config.exp_path / "lipid_std.pth").exists():
            logging.info("Loading column mean and std from files")
            col_means = torch.load(self.config.exp_path / "lipid_means.pth")
            col_stds = torch.load(self.config.exp_path / "lipid_std.pth")
        else:
            logging.info("computing channels means and stds")
            col_means = self.train_data.mean(dim=0)
            col_stds = self.train_data.std(dim=0)
            torch.save(col_means, self.config.exp_path / "lipid_means.pth")
            torch.save(col_stds, self.config.exp_path / "lipid_stds.pth")

    def load_coord_train_data(self):
        if self.coordinates_train is None:
            logging.info("Loading coordinates for training data")
            coordinates_names = ["xccf", "yccf", "zccf"]
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

    def load_coord_test_data(self):
        if self.coordinates_test is None:
            logging.info("Loading coordinates for test data")
            coordinates_names = ["xccf", "yccf", "zccf"]
            self.coordinates_test = pd.read_parquet(self.config.maldi_file,
                                                    columns=coordinates_names,
                                                    filters=self.test_filter).values
            self.coordinates_test = torch.tensor(self.coordinates_test, dtype=torch.float32)
            logging.info(f"Coordinates shape: {self.coordinates_test.shape}")
            logging.info("Normalizing coordinates")
            self.coordinates_test = (self.coordinates_test - self.coord_mean) / self.coord_std
            n_nans = torch.isnan(self.coordinates_test).sum()
            assert not torch.isnan(self.coordinates_test).any(), f"Coordinates contain NaNs: {n_nans} NaNs found in coordinates"
            logging.info("Coordinates normalization successful")

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

    def whole_brain_reconstruction(self):
        """
        Reconstructs the whole brain volume for all lipids using the trained model.

        This function loads the model from the specified path, prepares the template volume,
        and iterates through the non-zero indices of the template volume to predict lipid concentrations
        using the trained model. The predictions are saved in batches to avoid memory issues.
        The template volume is downloaded if it does not exist, and the predictions are saved in a specified directory.
        The function assumes that the model has been trained and saved in the specified path.
        """
        if(self.config.exp_path / "model.pth").exists():
            logging.info("Loading model from file")
            self.lgp_model.load_state_dict(torch.load(self.config.exp_path / "model.pth", map_location=self.config.device))
            logging.info("Model loaded successfully")
        else:
            logging.error("Model not found. Please run the experiment first.")
            return

        volume_path = self.config.exp_path / "volume"
        volume_path.mkdir(parents=True, exist_ok=True)
        template_file = volume_path / "template_volume.npy"
        if not template_file.exists():
            logging.info("Downloading template volume...")
            from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
            # Specify the resolution you want for the template volume (in microns)
            resolution_um = 25
            mcc = MouseConnectivityCache(resolution=resolution_um)
            logging.info(f"Downloading/loading reference TEMPLATE volume at {resolution_um} um resolution...")
            template_volume, _ = mcc.get_template_volume()
            logging.info(f"Template volume shape: {template_volume.shape}")
            logging.info(f"Template volume data type: {template_volume.dtype}")
            np.save(template_file, template_volume)
        else:
            logging.info("Template volume already exists, loading from file")
            template_volume = np.load(template_file)
        non_zero_indices = np.argwhere(template_volume > 5)
        # concert from 25 um to 1 mm
        non_zero_ccf = non_zero_indices * 0.025
        non_zero_ccf = torch.tensor(non_zero_ccf, dtype=torch.float32)
        non_zero_ccf = (non_zero_ccf - self.coord_mean) / self.coord_std
        ccf_dataset = torch.utils.data.TensorDataset(torch.tensor(non_zero_ccf, dtype=torch.float32), torch.tensor(non_zero_indices, dtype=torch.float32))
        ccf_dataloader = torch.utils.data.DataLoader(ccf_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        logging.info("Reconstructing whole brain volume...")

        self.lgp_model.eval()
        with torch.no_grad():
            for i,batch in enumerate(tqdm(ccf_dataloader)):
                batch_file = volume_path / f"batch_{i}.pth"
                if batch_file.exists():
                    logging.info(f"Batch {i} already exists, skipping...")
                    continue
                coordinates_batch = batch[0].to(self.config.device)
                indices_batch = batch[1]
                predictions, gp_posterior = self.lgp_model.predict(coordinates_batch)
                # we save the coordinates the indices and the predictions in a single file
                predictions = predictions * self.train_std + self.train_mean
                predictions = predictions.detach().cpu().numpy()
                if self.config.log_transform:
                    predictions = np.exp(predictions) - 1e-10
                torch.save({
                    "coordinates": coordinates_batch.cpu(),
                    "indices": indices_batch.cpu(),
                    "predictions": predictions,
                    "posterior": gp_posterior.mean.detach().cpu()
                }, batch_file)

    def load_whole_brain_reconstruction(self, lipid):
        """
        Loads and saves the whole brain reconstruction for a specific lipid.

        This does not need gpu to be run, as all the data was already batched and saved.

        Args:
            lipid (int): The index of the lipid to reconstruct.
        """
        volume_path = self.config.exp_path / "volume"
        template_file = volume_path / "template_volume.npy"
        if not template_file.exists():
            logging.error("Template volume does not exist. Please run whole_brain_reconstruction() first.")
            return None
        template_volume = np.load(template_file)
        non_zero_indices = np.argwhere(template_volume > 5)
        # fill template volume with zeros
        template_volume = np.zeros_like(template_volume, dtype=np.float32)

        lipid_name = self.config.selected_lipids_names[lipid]
        lipid_volume_name = self.config.exp_path / f"{lipid_name}_volume.npy"
        if lipid_volume_name.exists():
            logging.info(f"Lipid volume for {lipid_name} already exists, loading from file")
            lipid_volume = np.load(lipid_volume_name)
            return lipid_volume
        else:
            for i in tqdm(range(len(non_zero_indices) // self.config.batch_size + 1)):
                batch_file = volume_path / f"batch_{i}.pth"
                if not batch_file.exists():
                    continue
                batch_data = torch.load(batch_file)
                template_volume[batch_data["indices"][:, 0].long(),
                                batch_data["indices"][:, 1].long(),
                                batch_data["indices"][:, 2].long()] = batch_data["predictions"][:, lipid]
            # Save the reconstructed lipid volume
            # We need to ensure the volume is saved in the same shape as the template
            # set 0 to nan
            template_volume[template_volume == 0] = np.nan

            np.save(lipid_volume_name, template_volume)
            logging.info(f"Lipid volume for {lipid_name} saved to {lipid_volume_name}")
            template_volume = 255*( template_volume - np.nanmin(template_volume)) / (np.nanmax(template_volume)- np.nanmin(template_volume))
            np.save(self.config.exp_path / f"{lipid_name}_volume255.npy", template_volume)
            return template_volume

    def train_fit(self):
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
                self.lgp_model.train_model(self.config.exp_path,
                                      dataloader_train,
                                      optimizer,
                                      epochs=self.config.epochs,
                                      current_epoch=self.current_epoch,
                                      print_every=1000)

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
            logging.info("Predicting in the train set")
            self.lgp_model.eval()
            train_pred_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.coordinates_train),
                                                                batch_size=self.config.batch_size,
                                                                shuffle=False,
                                                                num_workers=0)
            logging.info("Predicting in the train set using the model")
            predictions_list = []
            posterior_list = []
            for batch in tqdm(train_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                predictions, gp_posterior = self.lgp_model.predict(coordinates_batch)
                predictions_list.append(predictions.detach().cpu())
                posterior_list.append(gp_posterior.mean.detach().cpu())
            train_predictions = torch.cat(predictions_list, dim=0)
            posterior = torch.cat(posterior_list, dim=0)
            torch.save(train_predictions, train_predictions_file)
            torch.save(posterior, train_predictions_file.with_name("posterior.pth"))
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
            self.load_coord_test_data()
            test_pred_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.coordinates_test),
                                                               batch_size=self.config.batch_size,
                                                               shuffle=False,
                                                               num_workers=0)
            predictions_list = []
            posterior_list = []
            for batch in tqdm(test_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                predictions,posterior = self.lgp_model.predict(coordinates_batch)
                predictions_list.append(predictions.detach().cpu())
                posterior_list.append(posterior.mean.detach().cpu())
            test_predictions = torch.cat(predictions_list, dim=0)
            posterior = torch.cat(posterior_list, dim=0)
            torch.save(test_predictions, test_predictions_file)
            torch.save(posterior, test_predictions_file.with_name("posterior.pth"))
        else:
            logging.info("Test predictions already exist, loading from file")
            test_predictions = torch.load(test_predictions_file)
        train_predictions_file = train_path / "predictions.npy"
        test_predictions_file = test_path / "predictions.npy"
        if not train_predictions_file.exists() or not test_predictions_file.exists():
            logging.info("Train and test numpy predictions do not exist, saving predictions to file")
            predict_original_scale = self.predict_original_scale(train_predictions, test_predictions)

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
