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

from utils import crop_or_stride_volume, reference_ccf_from_subvolume

def main():
    parser = argparse.ArgumentParser(description="Pre-compute Graph & Eigenvectors for MALDI")
    parser.add_argument("--reference-volume", type=str, required=True, help="Base reference volume for calculations")
    parser.add_argument("--annotations-volume", type=str, required=True, help="Base annotations")
    parser.add_argument("--output-path", type=str, required=True, help="Base experiment path")
    
    parser.add_argument("--stride", type=int, default=4, help="Subsampling stride for the 25um atlas")
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors")
    parser.add_argument("--modes", type=int, default=200, help="Number of eigenmodes to compute")
    parser.add_argument("--bandwidth", type=float, default=1.0, help="Graph bandwidth for Laplacian")
    parser.add_argument("--n-list", type=int, default=1, help="FAISS nlist parameter")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project", type=str, default="riemann-eigensolver", help="W&B Project name")

    parser.add_argument(
        "--region-bbox", type=int, nargs=6, default=None,
        metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"),
        help="Optional voxel bbox in the full-res 25um atlas. If set, "
             "stride is ignored and the volume is cropped at full resolution.",
    )
    args = parser.parse_args()

    # 1. Initialize W&B
    wandb.init(project=args.project, config=vars(args))
    device = torch.device(args.device)
    cache_path = Path(args.output_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # --- Load + downsample (do this once; everything else derives from it) ---
    reference_image   = np.load(args.reference_volume)
    annotation_volume = np.load(args.annotations_volume)
    threshold  = 5.0

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        reference_image, annotation_volume, args.stride, args.region_bbox,
    )
    mm_coords = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    n_nodes = mm_coords.shape[0]
    logging.info(f"Tissue nodes after threshold={threshold}: {n_nodes}")
    wandb.log({"graph/num_nodes": n_nodes})
 
    coord_mean = mm_coords.mean(axis=0)
    coord_std  = mm_coords.std(axis=0)
 
    reference_coords = torch.from_numpy(
        (mm_coords - coord_mean) / coord_std
    ).float().to(device).contiguous()
    # ==========================================
    knn_config = {
        "template": "allen_25um",
        "stride": args.stride,
        "method": "faiss",
        "k": args.k,
        "thresh": 5.0,
        "bbox": tuple(args.region_bbox) if args.region_bbox is not None else None,
    }
    knn_key = knn_make_key(knn_config)
    
    knn_cache = KnnGraphCache(cache_dir=cache_path / "knn", verbose=True)
    
    t0 = time.time()
    logging.info(f"Building Anatomical KNN Graph (stride={args.stride})...")
    knn_model, edge_index, edge_value = knn_cache.train_or_load(
        key=knn_key,
        method=knn_config["method"],
        k=args.k,
        nlist=args.n_list,
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