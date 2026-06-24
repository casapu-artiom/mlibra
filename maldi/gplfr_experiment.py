"""Run a MALDI experiment with GPLFR (collapsed linear decoder).

GPyTorch port of Gaussian Process Latent Factor Regression
(https://github.com/edstevenson/GPLFR, arXiv:2606.06576). This is the same
Euclidean inducing-point latent GP as ``lgp_experiment.py``, but the nonlinear
MLP decoder is replaced by GPLFR's *analytically marginalized linear decoder*.
Everything else (data loading, CV splits, whole-brain reconstruction) reuses the
existing ``MaldiExperiment`` harness unchanged.

Note: observation noise is a single learnable scalar (homoscedastic), matching
the upstream collapsed-likelihood derivation. Per-lipid noise is a possible
extension but breaks the shared closed-form marginalization, so it is out of
scope here.
"""
from argparse import ArgumentParser
import logging

from l3di.lgp import IndependentMultitaskGPModel
from gplfr import GPLFR
from experiment import MaldiExperiment
from config import MaldiConfig
from utils import (
    get_inducing_points,
    get_data_inducing_points,
)


def parse_args():
    """Parse command line arguments (mirrors lgp_experiment.py)."""
    parser = ArgumentParser(description="Run MALDI experiment with GPLFR.")
    parser.add_argument("--mode", type=str, required=True, help="Experiment mode (e.g., 'train', 'test').")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi-file", dest="maldi_file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--exp-name", dest="exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available-lipids-file", dest="available_lipids_file", type=str, required=True, help="File with available lipids.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--slices-dataset-file", dest="slices_dataset_file", type=str, required=True, help="File for slices dataset.")
    parser.add_argument("--template-name", dest="template_name", type=str, required=True, help="The template name.")
    parser.add_argument("--reference-file", dest="reference_file", type=str, required=True, help="The reference image npy.")
    parser.add_argument("--annotations-file", dest="annotations_file", type=str, help="The annotations if needed.")
    parser.add_argument("--num-inducing", dest="num_inducing", type=int, default=200, help="Number of inducing points.")
    parser.add_argument("--inducing-source", dest="inducing_source", default="reference",
                        choices=["reference", "data"],
                        help="'reference' (default): k-means over the reference tissue "
                             "image. 'data': draw inducing points from ACTUAL measured "
                             "MALDI voxels (sparse-data aware).")
    parser.add_argument("--inducing-method", dest="inducing_method", default="kmeans_snap",
                        choices=["kmeans_snap", "fps", "random"],
                        help="(--inducing-source data) how to pick on-data inducing points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of (full-batch) training epochs.")
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=8, help="Latent dimension q.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on.")
    parser.add_argument("--kernel", type=str, default="matern", help="Kernel type for the latent GP.")
    parser.add_argument("--log-transform", dest="log_transform", action='store_true', help="Apply log transformation to the data.")
    parser.add_argument("--nu", type=float, default=1.5, help="Matern smoothness parameter.")
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10, help="Pixels used for the minimal length scale.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2000, help="Loader batch size (training is full-batch).")
    parser.add_argument("--load-args", dest="load_args", action='store_true', help="Load arguments from a saved file.")
    parser.add_argument("--inverse-temperature", dest="inverse_temperature", type=float, default=0.1,
                        help="GPLFR inverse temperature scaling the collapsed log-likelihood.")
    parser.add_argument("--no-rsample", dest="no_rsample", action='store_true',
                        help="Use the latent posterior mean instead of a marginal sample (closer to upstream MAP).")
    parser.add_argument("--use-diffusion", dest="use_diffusion", action='store_true', help="Use diffusion model in the experiment.")
    parser.add_argument("--do-brain-reconstruction", dest="do_brain_reconstruction", action='store_true', help="Perform whole brain prediction.")
    parser.add_argument(
        "--reconstruction-lipids", dest="reconstruction_lipids",
        nargs="+", default=None,
        help="Restrict reconstruction to these lipids (indices or names). Default: all.",
    )
    return vars(parser.parse_args())


def setup_experiment(args):
    config = MaldiConfig.from_args(args)
    logging.info("Configuration created successfully")

    # Inducing points (reused verbatim from the LGP baseline).
    logging.info("Getting inducing points")
    if args.get("inducing_source", "reference") == "data":
        inducing_points, coord_mean, coord_std = get_data_inducing_points(
            config.maldi_file, config.section_filter, config.num_inducing,
            config.reference_file, method=args.get("inducing_method", "kmeans_snap"),
            exp_path=config.exp_path, seed=args["seed"],
        )
        config.num_inducing = inducing_points.shape[0]
    else:
        inducing_points, coord_mean, coord_std = get_inducing_points(
            config.exp_path, config.dataset_path, config.num_inducing,
        )
    logging.info(f"Got {inducing_points.shape[0]} inducing points")

    logging.info("Creating latent GP model")
    voxel_size = 0.025
    n_pixel = args["n_pixels"]
    minimal_length_scale = args["n_pixels"] * voxel_size / (coord_std.sum() / 3)
    logging.info(f"minimal length scale in um: {n_pixel * voxel_size}")
    gp_model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        kernel_type=config.kernel,
        nu=config.nu,
        minimal_length_scale=minimal_length_scale,
        input_dim=3,
    )
    logging.info("Latent GP model created successfully")

    logging.info("Creating GPLFR instance")
    model = GPLFR(
        gp_model=gp_model,
        p=len(config.selected_lipids_names),
        d=config.latent_dim,
        device=config.device,
        inverse_temperature=args.get("inverse_temperature", 0.1),
        use_rsample=not args.get("no_rsample", False),
    )
    return MaldiExperiment(config, model, coord_mean, coord_std)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting MALDI experiment (GPLFR collapsed linear decoder)")
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")
    experiment = setup_experiment(args)
    experiment.run()
    if experiment.config.do_brain_reconstruction:
        if experiment.config.reconstruction_lipids_by_index:
            lipid_names = None
            lipid_indices = experiment.config.reconstruction_lipids
        else:
            lipid_names = experiment.config.reconstruction_lipids
            lipid_indices = None
        experiment.whole_brain_reconstruction(lipid_indices=lipid_indices, lipid_names=lipid_names)
