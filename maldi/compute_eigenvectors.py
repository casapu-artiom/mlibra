#!/usr/bin/env python
# encoding: utf-8

import argparse
import logging
from pathlib import Path
import time

import numpy as np
import torch
import wandb

# Import your custom modules
from manifold_gp.utils.nearest_neighbors import KnnGraphCache, make_key as knn_make_key
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver, make_key as eigen_make_key
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator

def main():
    parser = argparse.ArgumentParser(description="Pre-compute Graph & Eigenvectors for MALDI")
    parser.add_argument("--reference-volume", type=str, required=True, help="Base reference volume for calculations")
    parser.add_argument("--annotations-volume", type=str, required=True, help="Base annotations")
    parser.add_argument("--output-path", type=str, required=True, help="Base experiment path")
    
    parser.add_argument("--stride", type=int, default=4, help="Subsampling stride for the 25um atlas")
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors")
    parser.add_argument("--modes", type=int, default=200, help="Number of eigenmodes to compute")
    parser.add_argument("--bandwidth", type=float, default=1.0, help="Graph bandwidth for Laplacian")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project", type=str, default="riemann-eigensolver", help="W&B Project name")
    args = parser.parse_args()

    # 1. Initialize W&B
    wandb.init(project=args.project, config=vars(args))
    device = torch.device(args.device)
    cache_path = Path(args.output_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # --- Load + downsample (do this once; everything else derives from it) ---
    reference_image   = np.load(args.reference_volume)
    annotation_volume = np.load(args.annotations_volume)

    sub_volume = reference_image[::args.stride, ::args.stride, ::args.stride]
    sub_atlas  = annotation_volume[::args.stride, ::args.stride, ::args.stride]
    threshold  = 5.0

    mask = sub_volume > threshold
    z, y, x = np.where(mask)

    voxel_idx = np.stack([z, y, x], axis=1).astype(np.float32)
    mm_coords = voxel_idx * (args.stride * 0.025)

    coord_mean = mm_coords.mean(axis=0)
    coord_std  = mm_coords.std(axis=0)

    reference_coords = torch.from_numpy(
        (mm_coords - coord_mean) / coord_std
    ).float().to(device).contiguous()

    # ==========================================
    knn_config = {
        "template": "allen_25um",
        "stride": args.stride,
        "method": "anatomical_atlas",
        "k": args.k,
        "thresh": 5.0
    }
    knn_key = knn_make_key(knn_config)
    
    knn_cache = KnnGraphCache(cache_dir=cache_path / "knn", verbose=True)
    
    t0 = time.time()
    logging.info(f"Building Anatomical KNN Graph (stride={args.stride})...")
    knn_model, edge_index, edge_value = knn_cache.train_or_load(
        key=knn_key,
        method="faiss",
        k=args.k,
        coords=reference_coords,
        volume=sub_volume,
        threshold=5.0,
        atlas_volume=sub_atlas,
        connectivity=1,
        device=device
    )
    knn_time = time.time() - t0
    wandb.log({"timing/knn_seconds": knn_time})

    # ==========================================
    # Phase 2: Laplacian Eigensolver
    # ==========================================
    logging.info("Constructing Graph Laplacian Operator...")
    bw_tensor = torch.tensor(args.bandwidth, dtype=torch.float32, device=device)
    
    laplacian_op = GraphLaplacianOperator(
        x=edge_value,
        idx=edge_index,
        operator_dimension=knn_model.x.shape[0],
        graphbandwidth=bw_tensor,
        normalization="symmetric", 
        self_loops=True 
    )

    eigen_config = {
        "knn_key": knn_key, 
        "modes": args.modes, 
        "norm": "symmetric",
        "bw": args.bandwidth
    }
    eigen_key = eigen_make_key(eigen_config)
    
    solver = LaplacianEigensolver(
        num_modes=args.modes, 
        backend="cupy" if args.device == "cuda" else "scipy",
        verbose=True
    )
    
    t1 = time.time()
    logging.info(f"Computing bottom {args.modes} eigenvectors...")
    eigval, eigvec = solver.compute_or_load(
        laplacian_op=laplacian_op,
        cache_dir=cache_path / "eigvecs",
        key=eigen_key,
        graphbandwidth=args.bandwidth,
        laplacian_normalization="symmetric",
        device=device
    )
    eigen_time = time.time() - t1
    
    # ==========================================
    # Phase 3: Final Logging
    # ==========================================
    wandb.log({
        "timing/eigen_seconds": eigen_time,
        "timing/total_seconds": knn_time + eigen_time,
        "results/min_eigenvalue": eigval[0].item(),
        "results/max_eigenvalue": eigval[-1].item(),
        "results/nonzero_min_eigenvalue": eigval[1].item() if len(eigval) > 1 else 0,
    })
    
    logging.info(f"Pipeline complete! Cache saved. Found {len(eigval)} eigenmodes.")
    wandb.finish()

if __name__ == "__main__":
    main()