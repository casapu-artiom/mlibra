#!/usr/bin/env python
# encoding: utf-8
"""Pre-compute the graph-Laplacian eigenvectors for the MALDI manifold GP, on GPU.

CUDA path (cupy/LOBPCG) counterpart to manifold/slepc/slepc_eigensolve.py -- it produces
the SAME cache (KnnGraphCache + LaplacianEigensolver layout) under the SAME keys
as the training pipeline (lgp_experiment_per_lipid.py) and the slepc solver, so
its output is a drop-in cache hit. Use this when you have a GPU and want to skip
the SLEPc/MPI/MUMPS (and conda-MKL) stack.

Parity with the training key scheme:
  * template "reference", threshold + normalization are ARGS (not hardcoded);
  * coordinate frame from utils.coord_norm_from_reference (per-axis mean + scalar
    isotropic std over ALL tissue voxels) -- the single source of truth, so the
    kNN graph matches bit-for-bit;
  * knn methods faiss / anatomical_atlas / faiss_atlas_weighted /
    faiss_cluster_weighted, with cross-region inflation, root-handling
    (dissolve/ignore/cross), majority-vote denoise, and hard cross-region prune,
    all folded into the eigvec cache key exactly as the pipeline does.

Idempotent: builds the KNN graph only if absent, and skips the eigensolve if the
eigvec cache already exists (unless --force-recompute[-graph]).

Example (exact/flat nlist=1 -> connected graph; atlas-weighted, cleaned):
  python manifold/compute_eigenvectors.py \
    --reference-volume  /path/reference_image.npy \
    --annotations-volume /path/level_15annot.npy \
    --output-path /path/eigenvectors \
    --stride 4 --threshold 5 --knn-k 15 --modes 2300 \
    --nlist 1 --bandwidth 0.1 --normalization randomwalk \
    --knn-method faiss_atlas_weighted --cross-region-inflation 50 \
    --root-handling dissolve --denoise-labels 3 --prune-cross-region 0.95
"""

# --- repo path bootstrap (this file moved out of maldi/) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_), str(_REPO_ / "maldi"),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as knn_make_key, resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as eigen_make_key,
)
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
    dissolve_root_labels, denoise_labels_majority_vote, prune_cross_region_edges,
    labels_for_nodes_from_template_clustering,
)
from utils import (
    crop_or_stride_volume, reference_ccf_from_subvolume, coord_norm_from_reference,
)

log = logging.getLogger("compute_eigenvectors")


def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute Graph & Eigenvectors (CUDA)")
    p.add_argument("--reference-volume", required=True)
    p.add_argument("--annotations-volume", default=None,
                   help="Atlas .npy. Required for anatomical_atlas / faiss_atlas_weighted.")
    p.add_argument("--output-path", required=True,
                   help="Cache root; graph -> <path>/knn, eigvecs -> <path>/eigvecs.")
    p.add_argument("--template", default="reference")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--knn-k", dest="knn_k", type=int, default=15)
    p.add_argument("--modes", type=int, default=2300)
    p.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1)
    p.add_argument("--bandwidth", type=float, default=0.1)
    p.add_argument("--normalization", choices=["symmetric", "randomwalk"], default="randomwalk")
    p.add_argument("--tol", type=float, default=1e-4)
    # IVF
    p.add_argument("--nlist", default="1",
                   help="FAISS IVF nlist: int or 'sqrt'. 1 = exact flat index "
                        "(connected graph); keyed only when != 1.")
    p.add_argument("--nprobe", default="8",
                   help="FAISS IVF nprobe: int or 'sqrt'. Not keyed; drives build recall. "
                        "Default 8 (recall ~1.0 in 3D) so nlist>1 graphs come out CONNECTED "
                        "-- nprobe=1 fragments the graph into ~nlist components.")
    # method + anatomy prior
    p.add_argument("--knn-method", dest="knn_method", default="faiss",
                   choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted",
                            "faiss_cluster_weighted"])
    p.add_argument("--cross-region-inflation", dest="cross_region_inflation",
                   type=float, default=10.0)
    p.add_argument("--root-handling", dest="root_handling",
                   choices=["dissolve", "ignore", "cross"], default="dissolve",
                   help="faiss_atlas_weighted: atlas label-0 'root' handling "
                        "(matches the training default 'dissolve').")
    p.add_argument("--denoise-labels", dest="denoise_labels", type=int, default=0)
    p.add_argument("--prune-cross-region", dest="prune_cross_region", type=float, default=0.0)
    # cluster-weighted params
    p.add_argument("--cluster-k", dest="cluster_k", type=int, default=64)
    p.add_argument("--cluster-spatial-weight", dest="cluster_spatial_weight", type=float, default=1.0)
    p.add_argument("--cluster-seed", dest="cluster_seed", type=int, default=0)
    p.add_argument("--cluster-fit-subsample", dest="cluster_fit_subsample", type=int, default=40000)
    # misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute", action="store_true",
                   help="Recompute eigvecs even if cached.")
    p.add_argument("--wandb", action="store_true", help="Log to W&B (off by default).")
    p.add_argument("--project", default="riemann-eigensolver")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    cache_path = Path(args.output_path)
    (cache_path / "knn").mkdir(parents=True, exist_ok=True)
    (cache_path / "eigvecs").mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, config=vars(args))

    # ---- nodes + standardized coordinate frame (single source of truth) ----
    reference_image = np.load(args.reference_volume)
    annotation_volume = np.load(args.annotations_volume) if args.annotations_volume else None
    if args.knn_method in ("anatomical_atlas", "faiss_atlas_weighted") and annotation_volume is None:
        raise SystemExit(f"--knn-method {args.knn_method} requires --annotations-volume")

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        reference_image, annotation_volume, args.stride)
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, args.threshold)
    coord_mean, coord_std = coord_norm_from_reference(reference_image)
    reference_nodes = ((torch.tensor(reference_ccf, dtype=torch.float32) - coord_mean)
                       / coord_std).to(device).contiguous()
    node_coords_np = reference_nodes.detach().cpu().numpy()
    N = reference_nodes.shape[0]
    nlist = resolve_nlist(args.nlist, N)
    nprobe = resolve_nprobe(args.nprobe, nlist)
    log.info(f"N={N:,} nodes  nlist={nlist} nprobe={nprobe}  method={args.knn_method}")

    # ---- graph key scheme (identical to lgp_experiment_per_lipid / slepc) ----
    atlas_stem = Path(args.annotations_volume).stem if args.annotations_volume else "noatlas"
    _legacy_atlas = (atlas_stem == "level_15annot")
    graph_key_parts = {
        "template": args.template, "stride": args.stride, "thresh": args.threshold,
        "method": args.knn_method, "k": args.knn_k, "bbox": None,
    }
    if nlist != 1:
        graph_key_parts["nlist"] = nlist
    if args.knn_method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4" if _legacy_atlas else atlas_stem
        graph_key_parts["conn"] = 3
    graph_key = knn_make_key(graph_key_parts)

    graphs = KnnGraphCache(cache_dir=cache_path / "knn", verbose=True)
    frg = args.force_recompute_graph
    t0 = time.time()

    if args.knn_method == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss", coords=reference_nodes,
            k=args.knn_k, nlist=nlist, nprobe=nprobe, extra=graph_key_parts,
            device=args.device, force_recompute=frg)
    elif args.knn_method == "anatomical_atlas":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas", volume=sub_volume,
            threshold=args.threshold, atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=args.knn_k, nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts, device=args.device, force_recompute=frg)
    else:  # faiss_atlas_weighted | faiss_cluster_weighted -- base graph is plain faiss
        base_parts = dict(graph_key_parts, method="faiss")
        base_parts.pop("atlas", None); base_parts.pop("conn", None)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=knn_make_key(base_parts), method="faiss", coords=reference_nodes,
            k=args.knn_k, nlist=nlist, nprobe=nprobe, extra=base_parts,
            device=args.device, force_recompute=frg)

        if args.knn_method == "faiss_atlas_weighted":
            node_labels = labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, args.threshold)
            root_mode = args.root_handling
            if root_mode == "dissolve":
                node_labels = dissolve_root_labels(node_labels, node_coords_np)
            infl = args.cross_region_inflation
            edge_index, edge_value, _ = inflate_cross_region_edges(
                edge_index, edge_value, node_labels, inflation=infl,
                treat_zero_as_cross=(root_mode == "cross"))
            _base_wt = (f"atlas_x{infl:g}" if _legacy_atlas else f"{atlas_stem}_x{infl:g}")
            graph_key_parts["weighting"] = (_base_wt if root_mode == "cross"
                                            else f"{_base_wt}_root{root_mode}")
            zero_is_region = False
        else:  # faiss_cluster_weighted
            node_labels = labels_for_nodes_from_template_clustering(
                sub_volume, args.threshold, n_clusters=args.cluster_k,
                spatial_weight=args.cluster_spatial_weight,
                fit_subsample=args.cluster_fit_subsample, seed=args.cluster_seed)
            infl = args.cross_region_inflation
            edge_index, edge_value, _ = inflate_cross_region_edges(
                edge_index, edge_value, node_labels, inflation=infl,
                treat_zero_as_cross=False)
            graph_key_parts["weighting"] = (
                f"tmplclust_k{args.cluster_k}_sw{args.cluster_spatial_weight:g}"
                f"_s{args.cluster_seed}_x{infl:g}")
            zero_is_region = True
        graph_key = knn_make_key(graph_key_parts)

        # denoise + hard prune (after inflation), keyed like the pipeline:
        # prune changes edges -> keyed; the denoise that shaped it rides along.
        n_denoise = int(args.denoise_labels or 0)
        prune = float(args.prune_cross_region or 0.0)
        if n_denoise > 0:
            node_labels = denoise_labels_majority_vote(node_labels, edge_index, n_denoise)
        if prune > 0.0:
            edge_index, edge_value = prune_cross_region_edges(
                edge_index, edge_value, node_labels, prune, zero_is_region=zero_is_region)
            graph_key_parts["prune"] = f"{prune:g}"
            if n_denoise > 0:
                graph_key_parts["denoise"] = n_denoise
            graph_key = knn_make_key(graph_key_parts)

    log.info(f"graph ready in {time.time()-t0:.1f}s  ({edge_index.shape[1]:,} edges)  "
             f"key={graph_key}")

    # ---- Laplacian + eigensolve (CUDA cupy backend) ----
    bw_tensor = torch.tensor(args.bandwidth, dtype=torch.float32, device=device)
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0], bw_tensor,
        args.normalization, self_loops=True)

    eigvec_key_parts = {"graph": graph_key, "norm": args.normalization,
                        "bw": args.bandwidth, "modes": args.modes}
    eigvec_key = eigen_make_key(eigvec_key_parts)
    log.info(f"eigvec_key={eigvec_key}")

    solver = LaplacianEigensolver(
        num_modes=args.modes,
        backend="cupy" if device.type == "cuda" else "scipy",
        tol=args.tol, ncv_min=resolve_ncv_min(args.modes, args.ncv_min), verbose=True)
    t1 = time.time()
    eigval, eigvec = solver.compute_or_load(
        laplacian_op=laplacian_op, cache_dir=cache_path / "eigvecs", key=eigvec_key,
        graphbandwidth=args.bandwidth, laplacian_normalization=args.normalization,
        extra=eigvec_key_parts, force_recompute=args.force_recompute,
        device=device, allow_larger_modes=True)
    log.info(f"{eigvec.shape[1]} eigenmodes ready in {time.time()-t1:.1f}s  "
             f"lambda in [{eigval[0].item():.4g}, {eigval[-1].item():.4g}]")

    if run is not None:
        run.log({"graph/num_nodes": N, "graph/num_edges": int(edge_index.shape[1]),
                 "results/num_modes": int(eigvec.shape[1]),
                 "results/min_eigenvalue": float(eigval[0].item()),
                 "results/max_eigenvalue": float(eigval[-1].item())})
        run.finish()
    log.info("done.")


if __name__ == "__main__":
    main()
