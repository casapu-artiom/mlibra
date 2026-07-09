"""Riemann manifold MALDI experiment.

Pair this with `lgp_experiment.py` (the Euclidean Matern baseline) for the
kernel comparison.
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
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
    add_faiss_cpu_args, apply_faiss_cpu_args,
    resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
    labels_for_nodes_from_template_clustering, dissolve_root_labels,
    denoise_labels_majority_vote, prune_cross_region_edges,
)
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from l3di.lgp_manifold import LatentRiemannGP, ManifoldLGP

from utils import (
    get_inducing_points,
    get_data_inducing_points,
    crop_or_stride_volume,
    reference_ccf_from_subvolume,
    augment_nodes_with_maldi,
    inducing_blend_density_maldi,
    maldi_voxels_standardized,
    coord_norm_from_reference,
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
    parser.add_argument("--inducing-source", dest="inducing_source", default="reference",
                        choices=["reference", "data"],
                        help="'reference' (default): k-means over the reference tissue "
                             "image. 'data': draw inducing points from ACTUAL measured "
                             "MALDI voxels (sparse-data aware). Snapped to graph nodes.")
    parser.add_argument("--inducing-method", dest="inducing_method", default="kmeans_snap",
                        choices=["kmeans_snap", "fps", "random"],
                        help="(--inducing-source data) on-data selection: 'kmeans_snap' "
                             "(default), 'fps' (max coverage), 'random'.")
    parser.add_argument("--inducing-from-maldi-nodes", dest="inducing_from_maldi_nodes",
                        action="store_true",
                        help="Blend the inducing points over the UNALTERED strided "
                             "graph: ~--inducing-density-frac from the densest graph "
                             "nodes + the rest from the measured MALDI voxels that snap "
                             "onto the graph most cheaply. Every inducing point is an "
                             "exact graph node. Independent of --augment-maldi-nodes (the "
                             "KNN graph is NOT modified). Overrides --inducing-source.")
    parser.add_argument("--inducing-density-frac", dest="inducing_density_frac",
                        type=float, default=0.8,
                        help="(--inducing-from-maldi-nodes) fraction of inducing points "
                             "drawn from the densest graph nodes; the remainder come from "
                             "cheapest-to-snap MALDI nodes. Default 0.8 (80/20 blend).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=10, help="Dimensionality of the latent space.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on.")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for the GP model (legacy, ignored by Riemann).")
    parser.add_argument("--log-transform", dest="log_transform", action='store_true', help="Apply log transformation to the data.")
    parser.add_argument("--nu", type=float, default=1.0, help="Smoothness parameter for the Riemann GP model.")
    parser.add_argument("--threshold", type=int, default=5, help="Threshold for the reference")
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10, help="Number of pixels to consider in the experiment.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2000, help="Batch size for training")
    parser.add_argument("--load-args", dest="load_args", action='store_true', help="Load arguments from a file instead of command line.")
    parser.add_argument("--use-diffusion", dest="use_diffusion", action='store_true', help="Use diffusion model in the experiment.")
    parser.add_argument("--knn-method", dest="knn_method", type=str, default="faiss",
                        choices=["faiss", "anatomical_atlas", "faiss_atlas_weighted",
                                 "faiss_cluster_weighted"],
                        help=("'faiss': Euclidean kNN, no anatomical prior. "
                              "'anatomical_atlas': edges restricted to same atlas region + "
                              "voxel-adjacent cross-region links (strong but can produce "
                              "disconnected components → NaN). "
                              "'faiss_atlas_weighted': use the faiss kNN graph but INFLATE "
                              "squared distances on edges that cross atlas regions by "
                              "--cross-region-inflation. Soft prior — graph stays connected. "
                              "'faiss_cluster_weighted': same soft inflation, but the region "
                              "labels are CLUSTERED from the reference template (no atlas "
                              "file needed) — data-driven, whole-brain, lipid-free."))
    parser.add_argument("--cross-region-inflation", dest="cross_region_inflation", type=float,
                        default=10.0,
                        help=("For --knn-method=faiss_atlas_weighted|faiss_cluster_weighted. "
                              "Multiplier applied to squared distance on cross-region edges. "
                              "Default 10 = mild prior; 100 = strong prior."))
    parser.add_argument("--cluster-k", dest="cluster_k", type=int, default=64,
                        help="(faiss_cluster_weighted) number of template-clustering regions.")
    parser.add_argument("--cluster-spatial-weight", dest="cluster_spatial_weight", type=float, default=1.0,
                        help="(faiss_cluster_weighted) weight on (z-scored) coords in the k-means feature "
                             "space; higher = more spatially compact/contiguous regions, 0 = pure feature k-means.")
    parser.add_argument("--cluster-fit-subsample", dest="cluster_fit_subsample", type=int, default=40000,
                        help="(faiss_cluster_weighted) nodes agglomerated before nearest-node propagation.")
    parser.add_argument("--cluster-seed", dest="cluster_seed", type=int, default=0,
                        help="(faiss_cluster_weighted) RNG seed for the template clustering.")
    parser.add_argument("--root-handling", dest="root_handling", type=str, default="dissolve",
                        choices=["dissolve", "ignore", "cross"],
                        help="(faiss_atlas_weighted) how to treat the atlas label-0 'root' "
                             "catch-all tissue. 'dissolve' (default): reassign each root node to "
                             "its nearest real region before inflating (frees interiors, keeps "
                             "region↔region confinement). 'ignore': keep root, don't inflate "
                             "root-touching edges. 'cross': legacy (inflate all root edges; "
                             "reuses old eigvec caches).")
    parser.add_argument("--denoise-labels", dest="denoise_labels", type=int, default=0,
                        help="Majority-vote label smoothing passes over the graph before the "
                             "prune (absorbs speck nodes). 0 = off. NOTE: erodes thin real "
                             "boundaries — prefer --root-handling dissolve for the catch-all.")
    parser.add_argument("--prune-cross-region", dest="prune_cross_region", type=float, default=0.0,
                        help="Fraction of cross-region edges to HARD-remove (vs the soft "
                             "--cross-region-inflation), connectivity-preserving. 0 = off. "
                             "Changes edges → distinct eigvec cache key.")
    parser.add_argument("--laplacian-norm", dest="laplacian_norm", type=str, default="symmetric",
                        choices=["symmetric", "randomwalk"])
    parser.add_argument("--stride", dest="stride", type=int, default=4, help="Stride to downsample the template.")
    parser.add_argument("--augment-maldi-nodes", dest="augment_maldi_nodes", action='store_true',
                        help="Add the measured MALDI voxels to the graph node set so every "
                             "measured point is an exact graph node (independent of stride).")
    parser.add_argument("--knn-k", dest="knn_k", type=int, default=15, help="Number of knn neighbours for the Graph Laplacian.")
    parser.add_argument("--n-list", default="1",
                        help="FAISS IVF nlist: an int, or 'sqrt' for round(sqrt(N)). "
                             "sqrt makes the CPU path near-linear (see faiss_bench_report).")
    parser.add_argument("--n-probe", default="1",
                        help="FAISS IVF nprobe: an int, or 'sqrt' for round(sqrt(nlist)). "
                             "For 3D coords a small FIXED value (~8) already gives recall "
                             "~1.0 at every N; 'sqrt' over-scans (grows with N for no gain).")
    parser.add_argument("--graphbandwidth-init", dest="graphbandwidth_init", type=float, default=1.0, help="Initial graph bandwidth.")
    parser.add_argument("--diffusion-scale-init", dest="diffusion_scale_init", type=float, default=1.0,
                        help="Initial multiplicative scale on the Laplacian spectrum "
                             "(lambda_k -> diffusion_scale*lambda_k in the Matern spectral density). 1.0 = identity.")
    parser.add_argument("--learn-diffusion-scale", dest="learn_diffusion_scale", action='store_true',
                        help="Make the diffusion scale learnable. Reshapes the spectral density WITHOUT "
                             "recomputing eigenpairs (scaling an operator leaves its eigenvectors unchanged); "
                             "frozen at --diffusion-scale-init otherwise.")
    parser.add_argument("--bump-scale", dest="bump_scale", type=float, default=3.0, help="Bump function param.")
    parser.add_argument("--bump-decay", dest="bump_decay", type=float, default=0.05, help="Bump function param.")
    parser.add_argument("--num-modes", dest="num_modes", type=int, default=200, help="Number of eigenvectors to use.")
    parser.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1,
                        help="Lanczos Krylov subspace floor. <=0 (default) auto-picks "
                             "max(1500, 3*num_modes+20); set a small explicit value "
                             "(e.g. 100) to fit large-N (stride=1) eigensolves in GPU memory.")
    parser.add_argument("--per-task-lengthscale", dest="per_task_lengthscale", action='store_true',
                        help="Give each latent dimension its OWN learnable lengthscale "
                             "(one RiemannMaternKernel per task, sharing the eigenpairs/graph) "
                             "instead of a single lengthscale shared across all latents.")
    parser.add_argument("--product-ard-matern", dest="product_ard_matern", action='store_true',
                        help="Multiply the Riemann kernel by an ARD Euclidean Matern (per-axis "
                             "lengthscales) to regain ambient anisotropy while keeping geodesic "
                             "routing: k = k_geo * k_eucl-ARD.")
    parser.add_argument("--product-ard-nu", dest="product_ard_nu", type=float, default=2.5,
                        help="Matern smoothness (nu) for the ARD Euclidean factor of "
                             "--product-ard-matern.")
    parser.add_argument("--lengthscale-init", dest="lengthscale_init", type=float, default=None,
                        help="Initial kernel lengthscale (z-units). Default: gpytorch default (~0.69). "
                             "Smaller favours more local covariance.")
    parser.add_argument("--no-rsample", dest="no_rsample", action='store_false', help="Use rsample instead of mean.")
    parser.add_argument("--do-brain-reconstruction", dest="do_brain_reconstruction", action='store_true', help="Perform whole brain prediction")
    parser.add_argument(
        "--reconstruction-lipids", dest="reconstruction_lipids",
        nargs="+", default=None,
        help="Restrict reconstruction to these lipids. Accepts indices (0 5 10) "
            "or names ('PA 36:4' 'PE 40:7'). Default: all lipids.",
    )

    parser.add_argument("--force-recompute-graph", dest="force_recompute_graph",
                        action="store_true",
                        help="Ignore the cached KNN graph and rebuild it from "
                             "scratch (needed to actually time graph construction, "
                             "e.g. under --faiss-cpu-graph).")

    add_faiss_cpu_args(parser)

    return vars(parser.parse_args())

def setup_experiment(args):
    config = MaldiConfig.from_args(args)
    logging.info("Configuration created successfully")

    # Inducing points (later snapped to graph nodes). 'data' draws from the
    # measured training voxels (sparse-data aware); 'reference' is k-means over
    # the tissue image. Both use the *global* coord_mean / coord_std.
    #
    # --inducing-from-maldi-nodes overrides --inducing-source: the points are
    # (re)selected by the density/MALDI blend AFTER the graph is built, so we
    # skip the discarded initial selection here and only fetch the (identical)
    # coord normalization directly -- no wasted k-means, no even-count
    # requirement, and config.num_inducing stays at the requested value.
    logging.info("Calculating coordinate normalization factors and inducing points...")
    if bool(args.get("inducing_from_maldi_nodes", False)):
        coord_mean, coord_std = coord_norm_from_reference(config.reference_file)
        inducing_points = None   # replaced by the blend once the graph exists
    elif args.get("inducing_source", "reference") == "data":
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
    if inducing_points is not None:
        logging.info(f"Got {inducing_points.shape[0]} inducing points")

    # 3. Atlas: download (if needed), coarsen annotations, then crop or stride.
    template_name = config.template_name
    template_volume = np.load(config.reference_file)
    annotations_volume = None
    if config.annotations_file is not None:
        annotations_volume = np.load(config.annotations_file)

    threshold            = args.get("threshold", 5)
    stride               = args.get("stride", 4)
    knn_k                = args.get("knn_k", 15)
    nlist_spec           = args.get("n_list", "1")    # int or 'sqrt'; resolved below
    nprobe_spec          = args.get("n_probe", "1")   # int or 'sqrt'; resolved below
    num_modes            = args.get("num_modes", 200)
    bump_scale           = args.get("bump_scale", 20.0)
    bump_decay           = args.get("bump_decay", 0.01)
    laplacian_norm       = args.get("laplacian_norm", "symmetric")
    graphbandwidth_init  = args.get("graphbandwidth_init", 1.0)   # used for both eigensolve and kernel init
    knn_method           = args.get("knn_method", "faiss")

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_volume, annotations_volume, stride,
    )
    
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    reference_nodes = torch.tensor(reference_ccf, dtype=torch.float32)
    reference_nodes = (reference_nodes - coord_mean) / coord_std
    reference_nodes = reference_nodes.to(config.device).contiguous()

    # Optionally add the measured MALDI voxels to the node set BEFORE the KNN
    # graph is built, so every measurement is an exact graph node (no
    # Nyström/bump zeroing in grid gaps), independent of stride.
    augment_maldi = bool(args.get("augment_maldi_nodes", False))
    if augment_maldi:
        reference_nodes = augment_nodes_with_maldi(
            reference_nodes, config.maldi_file, config.section_filter,
            coord_mean, coord_std,
        )

    # Resolve IVF nlist/nprobe now that the node count N is fixed (after any
    # MALDI augmentation). 'sqrt' -> nlist=round(sqrt(N)), nprobe=round(sqrt(nlist)).
    nlist = resolve_nlist(nlist_spec, reference_nodes.shape[0])
    nprobe = resolve_nprobe(nprobe_spec, nlist)
    logging.info(f"FAISS IVF: N={reference_nodes.shape[0]:,} -> nlist={nlist} nprobe={nprobe}")

    # 4. Build (or load) the KNN graph.
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
        "bbox": None,   # kept (always None) so existing graph/eig cache keys match
    }
    # nlist/nprobe are IVF index-construction knobs, not graph *content*: with an
    # adequate nprobe the build has recall ~1.0, so it produces the exact KNN
    # graph regardless of the specific nprobe (and nlist) value. So we don't key
    # on nprobe at all, and key on nlist only when it's non-default -- keeping the
    # default (nlist=1) compatible with caches built before nlist was ever keyed,
    # while still preventing an approximate IVF build from colliding with the
    # exact nlist=1 graph.
    if nlist != 1:
        graph_key_parts["nlist"] = nlist
    # Level-aware atlas id so different annotation volumes (e.g. level_5 vs
    # level_15) never share a graph/eigenvector cache. The historical default
    # (level_15annot) keeps its original un-suffixed keys so existing caches stay
    # valid; any other atlas is tagged by its filename stem.
    atlas_stem = Path(config.annotations_file).stem if config.annotations_file else "noatlas"
    _legacy_atlas = (atlas_stem == "level_15annot")
    if knn_method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4" if _legacy_atlas else atlas_stem
        graph_key_parts["conn"] = 3
    elif knn_method == "faiss_atlas_weighted":
        # The base graph is pure faiss — reuse that cache. Atlas weighting is
        # applied after loading, so we encode it in the eigenvector key only.
        pass

    if augment_maldi:
        # Augmented graphs have a different node set (and N) than strided-only
        # ones — key them separately so the KNN/eigenvector caches don't collide.
        graph_key_parts["augmaldi"] = int(reference_nodes.shape[0])
        if knn_method in ("anatomical_atlas", "faiss_atlas_weighted", "faiss_cluster_weighted"):
            # These map nodes 1:1 back to strided template voxels for region
            # labels; appended MALDI nodes break that mapping.
            raise ValueError(
                f"--augment-maldi-nodes is only supported with --knn-method=faiss "
                f"(got {knn_method!r}); atlas/cluster-label methods assume nodes == strided "
                f"template voxels."
            )

    graph_key = make_graph_key(graph_key_parts)
    logging.info(f"Graph cache key: {graph_key}")

    wandb.init(name=config.exp_name, project="l3di_maldi_knn_eig", config=config.to_dict())

    if knn_method == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key,
            method="faiss",
            coords=reference_nodes,
            k=knn_k,
            nlist=nlist,
            nprobe=nprobe,
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
            nprobe=nprobe,
            extra=graph_key_parts,
            force_recompute=bool(args.get("force_recompute_graph", False)),
            device=config.device,
        )
    elif knn_method == "faiss_atlas_weighted":
        # ---- 1. Get (or load) the base faiss graph ------------------
        # Cache key uses "faiss" so it's shared with vanilla faiss runs
        # of the same k / stride / threshold.
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key,
            method="faiss",
            coords=reference_nodes,
            k=knn_k,
            nlist=nlist,
            nprobe=nprobe,
            extra=base_key_parts,
            force_recompute=bool(args.get("force_recompute_graph", False)),
            device=config.device,
        )
        # ---- 2. Map nodes to atlas regions --------------------------
        if sub_atlas is None:
            raise ValueError(
                "--knn-method=faiss_atlas_weighted requires --annotations-file; "
                "got none."
            )
        node_labels = labels_for_nodes_from_sub_atlas(
            sub_volume, sub_atlas, threshold,
        )
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Mismatch between labelled voxels ({node_labels.shape[0]}) "
                f"and graph nodes ({knn.x.shape[0]}). Check that atlas and "
                f"reference template were cropped/strided identically."
            )
        # ---- 3. Root handling + inflate cross-region edges ----------
        # The atlas label-0 'root' is real tissue, not background; dissolving it
        # into the nearest region (default) stops the soft prior from spraying
        # through interiors. 'cross' = legacy treat_zero_as_cross=True.
        root_mode = str(args.get("root_handling", "dissolve"))
        if root_mode == "dissolve":
            node_labels = dissolve_root_labels(
                node_labels, reference_nodes.detach().cpu().numpy())
        inflation = float(args.get("cross_region_inflation", 10.0))
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=(root_mode == "cross"),
        )
        # ---- 4. Re-key so the eigvec cache doesn't collide with
        #         vanilla faiss runs (same topology, different weights).
        _base_wt = (f"atlas_x{inflation:g}" if _legacy_atlas
                    else f"{atlas_stem}_x{inflation:g}")
        graph_key_parts["weighting"] = (
            _base_wt if root_mode == "cross" else f"{_base_wt}_root{root_mode}")
        graph_key = make_graph_key(graph_key_parts)
    elif knn_method == "faiss_cluster_weighted":
        # Same soft-inflation as faiss_atlas_weighted, but the node labels are
        # CLUSTERED from the reference template itself (no annotations-file).
        # Data-driven, whole-brain, lipid-free. Base topology = plain faiss graph.
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key,
            method="faiss",
            coords=reference_nodes,
            k=knn_k,
            nlist=nlist,
            nprobe=nprobe,
            extra=base_key_parts,
            force_recompute=bool(args.get("force_recompute_graph", False)),
            device=config.device,
        )
        cluster_k = int(args.get("cluster_k", 64))
        sw = float(args.get("cluster_spatial_weight", 1.0))
        cseed = int(args.get("cluster_seed", 0))
        node_labels = labels_for_nodes_from_template_clustering(
            sub_volume, threshold, n_clusters=cluster_k,
            spatial_weight=sw,
            fit_subsample=int(args.get("cluster_fit_subsample", 40000)),
            seed=cseed,
        )
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Mismatch between clustered voxels ({node_labels.shape[0]}) "
                f"and graph nodes ({knn.x.shape[0]}). Check the reference "
                f"template stride/threshold."
            )
        inflation = float(args.get("cross_region_inflation", 10.0))
        # treat_zero_as_cross=False: every node is tissue and cluster id 0 is a
        # real region (not background), so it must NOT be inflated as a gap.
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=False,
        )
        graph_key_parts["weighting"] = (
            f"tmplclust_k{cluster_k}_sw{sw:g}_s{cseed}_x{inflation:g}"
        )
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"Unknown knn_method: {knn_method!r}")

    # ---- Label denoise + hard prune (weighted methods, after inflation) ----
    # Refine the HARD topology on the region labels the graph was weighted on.
    # Prune changes edges → the eigvecs change, so prune (+ the denoise that
    # shaped it) go into the eigvec cache key; denoise alone leaves edges intact.
    if knn_method in ("faiss_atlas_weighted", "faiss_cluster_weighted"):
        n_denoise = int(args.get("denoise_labels", 0) or 0)
        prune = float(args.get("prune_cross_region", 0.0) or 0.0)
        if n_denoise > 0:
            node_labels = denoise_labels_majority_vote(
                node_labels, edge_index, n_denoise)
        if prune > 0.0:
            edge_index, edge_value = prune_cross_region_edges(
                edge_index, edge_value, node_labels, prune,
                zero_is_region=(knn_method == "faiss_cluster_weighted"))
            graph_key_parts["prune"] = f"{prune:g}"
            if n_denoise > 0:
                graph_key_parts["denoise"] = n_denoise
            graph_key = make_graph_key(graph_key_parts)

    # Optionally REPLACE the inducing points with a density/MALDI blend over the
    # UNALTERED (strided) graph: ~density_frac from the densest graph nodes + the
    # rest from the measured MALDI voxels that snap onto the graph most cheaply.
    # The KNN graph is NOT modified (independent of --augment-maldi-nodes); both
    # sources resolve to exact graph nodes. Overrides --inducing-source. The snap
    # step below is then a no-op (the blend already returns exact graph nodes).
    if bool(args.get("inducing_from_maldi_nodes", False)):
        density_frac = float(args.get("inducing_density_frac", 0.8))
        graph_coords = knn.x.detach().cpu().numpy()
        maldi_coords = maldi_voxels_standardized(
            config.maldi_file, config.section_filter, coord_mean, coord_std,
        ).numpy()
        sel = inducing_blend_density_maldi(
            graph_coords, maldi_coords, config.num_inducing,
            density_frac=density_frac,
            density_k=knn_k,
            method=args.get("inducing_method", "kmeans_snap"),
            seed=args["seed"],
        )
        inducing_points = torch.tensor(sel, dtype=torch.float32)
        n_dense = int(round(density_frac * config.num_inducing))
        logging.info(
            f"Inducing points blended: ~{n_dense} from densest graph nodes + "
            f"~{config.num_inducing - n_dense} from cheapest-to-snap MALDI voxels "
            f"(graph N={graph_coords.shape[0]:,}, MALDI={maldi_coords.shape[0]:,})"
        )

    # Snap the inducing points to the nearest graph nodes. The unique-dedup
    # count depends on the (non bit-reproducible) KNN graph, so the resulting
    # num_inducing varies run-to-run. If a trained checkpoint already exists in
    # this folder, the model will be rebuilt only to load that checkpoint -- so
    # take the inducing geometry straight from it, otherwise the freshly-snapped
    # (slightly different) count makes load_state_dict fail. The stored param is
    # (num_tasks, M, 3); any task slice gives the [M, 3] the constructor wants,
    # and load_state_dict restores the exact per-task values afterwards.
    ckpt_path = config.exp_path / "model.pth"
    ind_key = "gp_model.variational_strategy.base_variational_strategy.inducing_points"
    with torch.no_grad():
        if ckpt_path.exists():
            inducing_points = torch.load(ckpt_path, map_location="cpu")[ind_key][0].contiguous()
            logging.info(
                f"Reusing {inducing_points.shape[0]} inducing points from checkpoint "
                f"{ckpt_path}"
            )
        else:
            ind_gpu = inducing_points.to(config.device)
            _, nn_idx = knn.search(ind_gpu, 1)          # (M, 1) — nearest graph node
            nn_idx = nn_idx.squeeze(1).cpu()            # (M,)
            nn_idx_unique = torch.unique(nn_idx)
            inducing_points = knn.x[nn_idx_unique].cpu()
            logging.info(
                f"Inducing points snapped to graph nodes: {inducing_points.shape[0]} unique"
            )
        config.num_inducing = inducing_points.shape[0]

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
    ncv_min = resolve_ncv_min(num_modes, args.get("ncv_min", -1))
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
        # Reuse a cache with more modes if one exists (eigenpairs are nested;
        # the extra modes are simply trimmed off on load).
        allow_larger_modes=True,
    )
    print(f"eigvec shape:          {tuple(eigvec.shape)}")  # (N_nodes, num_modes)
    print(f"eigval shape:          {tuple(eigval.shape)}")
    print(f"NaN in eigvec:         {torch.isnan(eigvec).any().item()}")
    print(f"NaN in eigval:         {torch.isnan(eigval).any().item()}")
    print(f"eigval range:          [{eigval.min().item():.4g}, {eigval.max().item():.4g}]")
    print(f"first 10 eigvals:      {eigval[:10].cpu().numpy()}")

    # Orthonormality check — eigvec.T @ eigvec should be ~I for a well-conditioned eigensolve
    inner = eigvec.T @ eigvec
    off_diag = inner - torch.eye(inner.shape[0], device=inner.device)
    print(f"|eigvec.T @ eigvec - I|_max (off-diag): {off_diag.abs().max().item():.4g}")

    # Eigval ordering — should be non-decreasing
    diffs = eigval[1:] - eigval[:-1]
    print(f"min eigval gap (should be >= 0): {diffs.min().item():.4g}")

    # 6. Riemann Matern kernel(s), fully formed at construction.
    logging.info("Building Riemann Kernel (knn + edges + eigenpairs at construction)...")
    def _build_kernel():
        k = RiemannMaternKernel(
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
            diffusion_scale_init=args.get("diffusion_scale_init", 1.0),
            learn_diffusion_scale=args.get("learn_diffusion_scale", False),
        ).to(config.device)
        ls_init = args.get("lengthscale_init", None)
        if ls_init is not None:
            k.lengthscale = torch.tensor(float(ls_init))
        k.eval()
        return k

    if args.get("per_task_lengthscale", False):
        # one kernel per latent dimension, sharing eigenpairs/graph tensors but
        # with an independent lengthscale (multi-scale latent basis).
        manifold_kernel = [_build_kernel() for _ in range(config.latent_dim)]
        logging.info(f"Per-task lengthscale: {config.latent_dim} kernels (shared eigenpairs)")
    else:
        manifold_kernel = _build_kernel()

    # 7. Latent Riemann GP, then end-to-end Manifold LGP.
    logging.info(f"Locking {inducing_points.shape[0]} anchor nodes on the graph...")
    gp_model = LatentRiemannGP(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        manifold_kernel=manifold_kernel,
        product_ard_matern=args.get("product_ard_matern", False),
        product_ard_nu=float(args.get("product_ard_nu", 2.5)),
    ).to(config.device)

    use_rsample = not args.get("no_rsample", False)
    lgp_model = ManifoldLGP(
        p=len(config.selected_lipids_names),
        d=config.latent_dim,
        n_neurons=[256, 256, 128],
        dropout=[0.1, 0.1, 0.1],
        activation="silu",
        device=config.device,
        gp_model=gp_model,
        use_rsample=use_rsample,
    )

    wandb.finish()

    return MaldiExperiment(config, lgp_model, coord_mean, coord_std)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting MALDI experiment with Riemann Manifold")
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")

    # Pin FAISS phases to CPU if requested (benchmarking knob). Graph build
    # happens inside setup_experiment, so apply the flags before it.
    apply_faiss_cpu_args(args)

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