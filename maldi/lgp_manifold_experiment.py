"""Riemann manifold MALDI experiment.

Pair this with `lgp_experiment.py` (the Euclidean Matern baseline) for the
kernel comparison. Both scripts share `--region-bbox` semantics via utils.py
so a region run is configured the same way on either side.
"""
import logging
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np

import wandb

from experiment import MaldiExperiment
from config import MaldiConfig
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import KnnGraphCache, make_key as make_graph_key
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from l3di.lgp_manifold import LatentRiemannGP, ManifoldLGP

from utils import (
    get_inducing_points,
    get_bbox_inducing_points,
    apply_region_to_config,
    crop_or_stride_volume,
    reference_ccf_from_subvolume,
)

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run MALDI experiment with l3di (Riemann).")
    parser.add_argument("--mode", type=str, required=True, help="Experiment mode (e.g., 'train', 'test').")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi-file", dest="maldi_file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--exp-name", dest="exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available-lipids-file", dest="available_lipids_file", type=str, required=True, help="File with available lipids.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--eigenvector-dir", dest="eigenvector_dir", type=str, required=True, help="Directory for eigenvector files.")
    parser.add_argument("--slices-dataset-file", dest="slices_dataset_file", type=str, required=True, help="File for slices dataset.")
    parser.add_argument("--template-name", dest="template_name", type=str, required=True, help="The template name.")
    parser.add_argument("--reference-file", dest="reference_file", type=str, required=True, help="The reference image npy.")
    parser.add_argument("--annotations-file", dest="annotations_file", type=str, help="The annotations if needed.")
    parser.add_argument("--num-inducing", dest="num_inducing", type=int, default=500, help="Number of inducing points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=10, help="Dimensionality of the latent space.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on.")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for the GP model (legacy, ignored by Riemann).")
    parser.add_argument("--log-transform", dest="log_transform", action='store_true', help="Apply log transformation to the data.")
    parser.add_argument("--nu", type=float, default=1.0, help="Smoothness parameter for the Riemann GP model.")
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10, help="Number of pixels to consider in the experiment.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2000, help="Batch size for training")
    parser.add_argument("--load-args", dest="load_args", action='store_true', help="Load arguments from a file instead of command line.")
    parser.add_argument("--use-diffusion", dest="use_diffusion", action='store_true', help="Use diffusion model in the experiment.")
    parser.add_argument("--knn-method", dest="knn_method", type=str, default="faiss",
                        choices=["faiss", "anatomical_atlas"])
    parser.add_argument("--laplacian-norm", dest="laplacian_norm", type=str, default="symmetric",
                        choices=["symmetric", "randomwalk"])
    parser.add_argument("--stride", dest="stride", type=int, default=4, help="Stride to downsample the template.")
    parser.add_argument("--knn-k", dest="knn_k", type=int, default=15, help="Number of knn neighbours for the Graph Laplacian.")
    parser.add_argument("--n-list", type=int, default=1, help="FAISS nlist parameter")
    parser.add_argument("--graphbandwidth-init", dest="graphbandwidth_init", type=float, default=1.0, help="Initial graph bandwidth.")
    parser.add_argument("--bump-scale", dest="bump_scale", type=float, default=3.0, help="Bump function param.")
    parser.add_argument("--bump-decay", dest="bump_decay", type=float, default=0.05, help="Bump function param.")
    parser.add_argument("--num-modes", dest="num_modes", type=int, default=200, help="Number of eigenvectors to use.")
    parser.add_argument("--use-rsample", dest="use_rsample", action='store_false', help="Use rsample instead of mean.")
    parser.add_argument("--do-brain-reconstruction", dest="do_brain_reconstruction", action='store_true', help="Perform whole brain prediction")
    parser.add_argument(
        "--reconstruction-lipids", dest="reconstruction_lipids",
        nargs="+", default=None,
        help="Restrict reconstruction to these lipids. Accepts indices (0 5 10) "
            "or names ('PA 36:4' 'PE 40:7'). Default: all lipids.",
    )

    # ---- region restriction ----
    parser.add_argument(
        "--region-bbox", dest="region_bbox", type=int, nargs=6, default=None,
        metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"),
        help=("Optional bbox in voxel coords of the full-res 25um atlas. "
              "If set: the atlas is cropped at full resolution (stride is "
              "ignored), inducing points are placed inside the bbox via "
              "k-means, and MALDI train/test points are filtered to the "
              "same bbox in mm."),
    )

    return vars(parser.parse_args())

def setup_experiment(args):
    config = MaldiConfig.from_args(args)
    logging.info("Configuration created successfully")

    region_bbox = args.get("region_bbox", None)

    # 1. Patch the config's MALDI parquet filters so train/test only sees
    #    points inside the bbox. No-op when region_bbox is None.
    apply_region_to_config(config, region_bbox)

    # 2. Inducing points: bbox-restricted k-means when bbox is set, else
    #    the original whole-brain symmetric k-means. Both routines use the
    #    *global* coord_mean / coord_std so the standardized space matches.
    logging.info("Calculating coordinate normalization factors and inducing points...")
    if region_bbox is not None:
        inducing_points, coord_mean, coord_std = get_bbox_inducing_points(
            config.exp_path, config.dataset_path, config.num_inducing, region_bbox,
        )
        config.num_inducing = inducing_points.shape[0]
    else:
        inducing_points, coord_mean, coord_std = get_inducing_points(
            config.exp_path, config.dataset_path, config.num_inducing,
        )
    logging.info(f"Got {inducing_points.shape[0]} inducing points")

    # 3. Atlas: download (if needed), coarsen annotations, then crop or stride.
    template_name = config.template_name
    template_volume = np.load(config.reference_file)
    annotations_volume = None
    if config.annotations_file is not None:
        annotations_volume = np.load(config.annotations_file)

    threshold            = 5
    stride               = args.get("stride", 4)
    knn_k                = args.get("knn_k", 15)
    nlist                = args.get("n_list", 1)
    num_modes            = args.get("num_modes", 200)
    bump_scale           = args.get("bump_scale", 20.0)
    bump_decay           = args.get("bump_scale", 0.01)
    laplacian_norm       = args.get("laplacian_norm", "symmetric")
    graphbandwidth_init  = args.get("graphbandwidth_init", 1.0)   # used for both eigensolve and kernel init
    knn_method           = args.get("knn_method", "faiss")

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_volume, annotations_volume, stride, region_bbox,
    )
    
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    reference_nodes = torch.tensor(reference_ccf, dtype=torch.float32)
    reference_nodes = (reference_nodes - coord_mean) / coord_std
    reference_nodes = reference_nodes.to(config.device).contiguous()

    # 4. Build (or load) the KNN graph.
    eigenvector_dir = Path(args.get("eigenvector_dir"))
    eigenvector_dir.mkdir(parents=True, exist_ok=True)

    graph_cache_dir = eigenvector_dir / "knn"
    graphs = KnnGraphCache(cache_dir=graph_cache_dir, verbose=True)

    graph_key_parts = {
        "template": template_name,
        "stride": stride if region_bbox is None else 1,
        "thresh": threshold,
        "method": knn_method,
        "k": knn_k,
        "nlist": nlist,
        "bbox": tuple(region_bbox) if region_bbox is not None else None,
    }
    if knn_method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3

    graph_key = make_graph_key(graph_key_parts)
    logging.info(f"Graph cache key: {graph_key}")

    wandb.init(project=config.exp_name + "_knn_eig", config=args)

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
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key,
            method="anatomical_atlas",
            volume=sub_volume,
            threshold=threshold,
            atlas_volume=sub_atlas,
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

    # 5. Eigenpairs: compute (or load from cache) before kernel construction.
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
        "modes": num_modes,
    }
    eigvec_key = make_eig_key(eigvec_key_parts)
    logging.info(f"Eigenvector cache key: {eigvec_key}")

    # Lanczos basis must comfortably exceed num_modes; bump as needed.
    ncv_min = max(1500, 3 * num_modes + 20)
    solver = LaplacianEigensolver(
        num_modes=num_modes, backend="cupy", tol=1e-4, ncv_min=ncv_min, verbose=True,
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

    # 6. Riemann Matern kernel, fully formed at construction.
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

    # 7. Latent Riemann GP, then end-to-end Manifold LGP.
    logging.info(f"Locking {inducing_points.shape[0]} anchor nodes on the graph...")
    gp_model = LatentRiemannGP(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        manifold_kernel=manifold_kernel,
    ).to(config.device)

    lgp_model = ManifoldLGP(
        p=len(config.selected_lipids_names),
        d=config.latent_dim,
        n_neurons=[256, 256, 128],
        dropout=[0.1, 0.1, 0.1],
        activation="silu",
        device=config.device,
        gp_model=gp_model,
        use_rsample=args.get("use_rsample", True),
    )

    wandb.finish()

    return MaldiExperiment(config, lgp_model, coord_mean, coord_std), region_bbox

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting MALDI experiment with Riemann Manifold")
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")

    experiment, region_bbox = setup_experiment(args)
    experiment.run()

    if experiment.config.do_brain_reconstruction:
        if experiment.config.reconstruction_lipids_by_index:
            lipid_names = None
            lipid_indices = experiment.config.reconstruction_lipids
        else:
            lipid_names = experiment.config.reconstruction_lipids
            lipid_indices = None
        if region_bbox is not None:
            # Skip whole-brain reconstruction in region mode -- a GP trained
            # only on points inside the bbox will extrapolate poorly outside it.
            experiment.region_reconstruction(region_bbox, lipid_indices=lipid_indices, lipid_names=lipid_names)
        else:
            experiment.whole_brain_reconstruction(lipid_indices=lipid_indices, lipid_names=lipid_names)