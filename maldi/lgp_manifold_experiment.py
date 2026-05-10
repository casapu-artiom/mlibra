"""This script sets up and runs a MALDI experiment using the l3di library with a Riemann Manifold."""
import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from experiment import MaldiExperiment
from config import MaldiConfig
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver
from manifold_gp.utils.nearest_neighbors import KnnGraphCache, make_key as make_graph_key
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from utils import get_inducing_points

# Import the new Manifold classes we added to lgp.py
from l3di.lgp_manifold import LatentRiemannGP, ManifoldLGP
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
    parser.add_argument("--eigenvector-dir", dest="eigenvector_dir", type=str, required=True, help="Directory for eigenvector files.")
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
    parser.add_argument("--knn-method", dest="knn_method", type=str, default="faiss",
                        choices=["faiss", "anatomical_atlas"])
    parser.add_argument("--laplacian-norm", dest="laplacian_norm", type=str, default="symmetric", help="Normalization for the graph laplacian",
                        choices=["symmetric", "randomwalk"])
    parser.add_argument("--stride", dest="stride", type=int, default=4, help="Stride to downsample the template.")
    parser.add_argument("--knn-k", dest="knn_k", type=int, default=15, help="Number of knn neighbours for the Graph Laplacian.")
    parser.add_argument("--bump-scale", dest="bump_scale", type=float, default=3.0, help="Bump function param.")
    parser.add_argument("--bump-decay", dest="bump_decay", type=float, default=0.05, help="Bump function param.")
    parser.add_argument("--num-modes", dest="num_modes", type=int, default=200, help="Number of eigenvectors to use.")
    
    return vars(parser.parse_args())

def coarsen_annotation(annotation, atlas, max_depth=4):
    """Collapse leaf labels to a chosen ancestor depth in the structure tree."""
    structures = atlas.structures
    id_remap = {0: 0}
    for sid, info in structures.items():
        if sid == 0:
            continue
        path = info.get("structure_id_path", [sid])
        if len(path) <= max_depth + 1:
            id_remap[sid] = sid
        else:
            id_remap[sid] = path[max_depth]
    max_id = int(annotation.max()) + 1
    lut = np.zeros(max_id + 1, dtype=np.int32)
    for src, dst in id_remap.items():
        if src < lut.shape[0]:
            lut[src] = dst
    return lut[annotation]

def _load_or_download_template(volume_path: Path):
    from bg_atlasapi.bg_atlas import BrainGlobeAtlas
    atlas = BrainGlobeAtlas("allen_mouse_25um")
 
    template_file = volume_path / "template_volume.npy"
    annotation_file = volume_path / "annotations.npy"
 
    if not template_file.exists() or not annotation_file.exists():
        logging.info("Downloading template volume via BrainGlobe Atlas API...")
        template_volume = atlas.reference
        annotation_volume = atlas.annotation
        logging.info(f"Template volume shape: {template_volume.shape}")
        volume_path.mkdir(parents=True, exist_ok=True)
        np.save(template_file, template_volume)
        np.save(annotation_file, annotation_volume)
    else:
        logging.info("Template volume already exists, loading from file")
        template_volume = np.load(template_file)
        annotation_volume = np.load(annotation_file)
 
    return template_volume, annotation_volume, atlas


def setup_experiment(args):
    config = MaldiConfig.from_args(args)
    logging.info("Configuration created successfully")
    
    # 1. We still use get_inducing_points to get the scaling factors
    # (coord_mean and coord_std) to ensure normalization is consistent.
    logging.info("Calculating coordinate normalization factors...")
    inducing_points, coord_mean, coord_std = get_inducing_points(
        config.exp_path, config.dataset_path, config.num_inducing
    )

    volume_path = config.exp_path / "volume"
    template_volume, annotation_volume, atlas = _load_or_download_template(volume_path)
    annotation_coarse = coarsen_annotation(annotation_volume, atlas, max_depth=4)

    template_name        = "allen_mouse_25um"
    threshold            = 5
    stride               = args.get("stride", 4)
    knn_k                = args.get("knn_k", 4)
    nlist                = 1
    num_modes            = args.get("num_modes", 200)
    bump_scale           = args.get("bump_scale", 3.0)
    bump_decay           = args.get("bump_scale", 0.05)
    laplacian_norm       = "symmetric"
    graphbandwidth_init  = 1.0   # pinned: used for the eigensolve AND for kernel init
    knn_method           = args.get("knn_method", "faiss")

    sub_volume = template_volume[::stride, ::stride, ::stride]
    z, y, x = np.where(sub_volume > threshold)
    reference_ccf = np.stack([z, y, x], axis=1) * stride * 0.025
    reference_nodes = torch.tensor(reference_ccf, dtype=torch.float32)
    reference_nodes = (reference_nodes - coord_mean) / coord_std
    reference_nodes = reference_nodes.to(config.device).contiguous()


    eigenvector_dir = Path(args.get("eigenvector_dir"))
    eigenvector_dir.mkdir(parents=True, exist_ok=True)

    graph_cache_dir = eigenvector_dir / "knn"
    graphs = KnnGraphCache(cache_dir=graph_cache_dir, verbose=True)
 
    graph_key_parts = {
        "template": template_name,
        "stride": stride,
        "thresh": threshold,
        "method": knn_method,
        "k": knn_k,
        "nlist": nlist,
    }
    if knn_method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
 
    graph_key = make_graph_key(graph_key_parts)
    logging.info(f"Graph cache key: {graph_key}")
 
    if knn_method == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key,
            method="faiss",
            coords=reference_nodes,
            k=knn_k,
            nlist=nlist,
            extra=graph_key_parts,
            force_recompute=bool(args.get("force_recompute_graph", False)),
            device=config.device,
        )
    elif knn_method == "anatomical_atlas":
        # Same standardized coords as the FAISS path. The anatomical builder
        # uses these for KNN distance and inter-region edge values; voxel
        # indices for atlas lookup and grid-adjacency come from `volume` +
        # `threshold` (np.where(volume > threshold) inside the builder).
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key,
            method="anatomical_atlas",
            volume=sub_volume,
            threshold=threshold,
            atlas_volume=annotation_coarse,
            connectivity=3,
            coords=reference_nodes,
            k=knn_k,
            nlist=nlist,
            extra=graph_key_parts,
            force_recompute=bool(args.get("force_recompute_graph", False)),
            device=config.device,
        )
    else:
        raise ValueError(f"Unknown knn_method: {knn_method!r}")
 
    # ----------------------------------------------------------------------
    # 5. Eigenpairs: compute (or load from cache) BEFORE the kernel exists.
    #
    #    The kernel now requires eigvecs at construction, with no fallback,
    #    so we have to do this step first. We build a transient
    #    GraphLaplacianOperator from (edges, n_nodes, graphbandwidth_init,
    #    laplacian_norm) and hand it to the solver. The same
    #    `graphbandwidth_init` is also passed to the kernel below so the
    #    eigvecs match the kernel's actual initial bandwidth.
    # ----------------------------------------------------------------------
    eigvec_cache_dir = eigenvector_dir / "eigvecs"
 
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0],
        torch.tensor(graphbandwidth_init, device=config.device),
        laplacian_norm,
    )
 
    eigvec_key_parts = {
        "graph": graph_key,
        "norm": laplacian_norm,
        "bw": graphbandwidth_init,
    }
    eigvec_key = make_eig_key(eigvec_key_parts)
    logging.info(f"Eigenvector cache key: {eigvec_key}")
 
    solver = LaplacianEigensolver(
        num_modes=num_modes, backend="cupy", tol=1e-4, ncv_min=1500, verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op,
        cache_dir=eigvec_cache_dir,
        key=eigvec_key,
        graphbandwidth=graphbandwidth_init,
        laplacian_normalization=laplacian_norm,
        extra=eigvec_key_parts,
        force_recompute=bool(args.get("force_recompute_eigvecs", False)),
        device=config.device,
    )
 
    # ----------------------------------------------------------------------
    # 6. Kernel: fully formed at construction. No attach, no eval-time setup.
    # ----------------------------------------------------------------------
    logging.info("Building Riemann Kernel (knn + edges + eigenpairs at construction)...")
    manifold_kernel = RiemannMaternKernel(
        nu=config.nu,
        knn=knn,
        edge_index=edge_index,
        edge_value=edge_value,
        eigval=eigval,
        eigvec=eigvec,
        nearest_neighbors=knn_k,
        num_modes=num_modes,
        bump_scale=bump_scale,
        bump_decay=bump_decay,
        laplacian_normalization=laplacian_norm,
        graphbandwidth_init=graphbandwidth_init,
    ).to(config.device)
    manifold_kernel.eval()
 
    # ----------------------------------------------------------------------
    # 7. Latent Riemann GP, then end-to-end Manifold LGP
    # ----------------------------------------------------------------------
    logging.info(f"Locking {config.num_inducing} anchor nodes on the graph...")
    gp_model = LatentRiemannGP(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        manifold_kernel=manifold_kernel,
    ).to(config.device)
 
    lgp_model = ManifoldLGP(
        p=len(config.selected_lipids_names),
        d=config.latent_dim,
        n_neurons=[100, 100],
        dropout=[0.1, 0.1],
        activation="relu",
        device=config.device,
        gp_model=gp_model,
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
    #selected_reconstructions = [0, 3, 5, 10, 131, 72, 16, 89, 4, 74]
    #for i in selected_reconstructions:
    #   experiment.load_whole_brain_reconstruction(i)

    #experiment.run(num_patches=50, model_type="riemann")
    
    # 2. Stitch the specific lipids back together!
    #selected_reconstructions = [0, 3, 5, 10, 131, 72, 16, 89, 4, 74]
    
    # for lipid_idx in selected_reconstructions:
    #    volume = experiment.load_whole_brain_reconstruction_patches(
    #        lipid=lipid_idx, 
    #        model_type="riemann"
    #    )