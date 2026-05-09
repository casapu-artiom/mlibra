"""This script sets up and runs a MALDI experiment using the l3di library with a Riemann Manifold."""
import logging
import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from experiment import MaldiExperiment
from config import MaldiConfig
from utils import get_inducing_points

# Import the new Manifold classes we added to lgp.py
from l3di.lgp_manifold import LatentRiemannGP, ManifoldLGP, SpectralLatentGP, SpectralManifoldLGP
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run MALDI experiment with l3di.")
    parser.add_argument("--mode", type=str, required=True, help="Experiment mode (e.g., 'train', 'test').")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi-file", dest="maldi_file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--exp-name", dest="exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available-lipids-file", dest="available_lipids_file", type=str, required=True, help="File with available lipids.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--slices-dataset-file", dest="slices_dataset_file", type=str, required=True, help="File for slices dataset.")
    parser.add_argument("--num-inducing", dest="num_inducing", type=int, default=500, help="Number of inducing points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=10, help="Dimensionality of the latent space.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on (e.g., 'cpu', 'cuda').")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for the GP model.")
    parser.add_argument("--log-transform", dest="log_transform", action='store_true', help="Apply log transformation to the data.")
    parser.add_argument("--nu", type=float, default=1.0, help="Smoothness parameter for the Riemann GP model.")
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10, help="Number of pixels to consider in the experiment.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2000, help="Batch size for training")
    parser.add_argument("--load-args", dest="load_args", action='store_true', help="Load arguments from a file instead of command line.")
    parser.add_argument("--use-diffusion", dest="use_diffusion", action='store_true', help="Use diffusion model in the experiment.")

    return vars(parser.parse_args())

def setup_experiment(args):
    config = MaldiConfig.from_args(args)
    logging.info("Configuration created successfully")
    
    # # 1. We still use get_inducing_points to get the scaling factors
    # # (coord_mean and coord_std) to ensure normalization is consistent.
    logging.info("Calculating coordinate normalization factors...")
    _, coord_mean, coord_std = get_inducing_points(
        config.exp_path, config.dataset_path, config.num_inducing
    )

    # 2. Load the full CCF coordinates for the Riemann Graph
    logging.info("Loading full training CCF coordinates to build the Manifold...")    
    volume_path = config.exp_path / "volume"
    template_file = volume_path / "template_volume.npy"
    
    if not template_file.exists():
        logging.info("Downloading template volume via BrainGlobe Atlas API...")
        
        # Import BrainGlobe instead of AllenSDK
        from bg_atlasapi.bg_atlas import BrainGlobeAtlas
        
        # This automatically downloads, unpacks, and caches the 25um Allen CCF
        atlas = BrainGlobeAtlas("allen_mouse_25um")
        template_volume = atlas.reference
        
        logging.info(f"Template volume shape: {template_volume.shape}")
        logging.info(f"Template volume data type: {template_volume.dtype}")
        
        volume_path.mkdir(parents=True, exist_ok=True)
        np.save(template_file, template_volume)
    else:
        logging.info("Template volume already exists, loading from file")
        template_volume = np.load(template_file)    
        
    # Subsample to avoid cuSPARSE memory crash (Stride 4 = ~100um resolution)
    stride = 4
    z, y, x = np.where(template_volume[::stride, ::stride, ::stride] > 5)
    
    # Convert to 1mm CCF space (matching experiment.py logic)
    reference_ccf = np.stack([z, y, x], axis=1) * stride * 0.025
    reference_nodes = torch.tensor(reference_ccf, dtype=torch.float32)
    reference_nodes = (reference_nodes - coord_mean) / coord_std
    reference_nodes = reference_nodes.to(config.device).contiguous()

    # 4. Initialize the Manifold Kernel and compute geometry
    logging.info("Building Riemann Kernel and solving Laplacian Eigenvectors...")
    manifold_kernel = RiemannMaternKernel(
        nu=config.nu, 
        x=reference_nodes, 
        nearest_neighbors=5,
        num_modes=1100,      # Adjust based on tissue complexity (200-500 is typical)
        bump_scale=3.0,     # Wide enough to cover voxel gaps during out-of-sample prediction
        bump_decay=0.05
    ).to(config.device)
    
    # This triggers the eigensolver (Torch LOBPCG / CuPy depending on your riemann_kernel.py)
    manifold_kernel.eval()  

    # 5. Create the Latent Riemann GP using the fixed anchors
    logging.info("Creating LatentRiemannGP model...")
    gp_model = SpectralLatentGP(
        num_tasks=config.latent_dim,
        manifold_kernel=manifold_kernel
    ).to(config.device)

    # 6. Create the end-to-end Manifold LGP
    logging.info("Creating ManifoldLGP instance...")
    lgp_model = SpectralManifoldLGP(
        p=len(config.selected_lipids_names),
        d=config.latent_dim,
        n_neurons=[512, 512],
        dropout=[0.1, 0.1],
        activation='silu',  # Switched to SiLU as it generally decodes spatial fields smoother than ReLU
        device=config.device,
        gp_model=gp_model
    )

    return MaldiExperiment(config, lgp_model, coord_mean, coord_std)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting MALDI experiment with Riemann Manifold")
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")
    
    experiment = setup_experiment(args)
    
    # Standard execution pipeline
    experiment.run()
    experiment.whole_brain_reconstruction()
    # selected_reconstructions = [0, 3, 5, 10, 131, 72, 16, 89, 4, 74]
    # for i in selected_reconstructions:
    #    experiment.load_whole_brain_reconstruction(i)