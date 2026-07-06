"""Shared builder for the Riemann manifold kernel + eigenbasis.

Both manifold latent-GP bases used by the GPLFR runner — the inducing-point
``LatentRiemannGP`` and the weight-space ``SpectralLatentGP`` — need the same
front-end pipeline: build (or load) the FAISS KNN graph over the reference
tissue nodes, form the Graph Laplacian, solve for its leading eigenpairs, and
wrap them in a ``RiemannMaternKernel``. That pipeline is lifted verbatim from
``spectral_lgp_manifold_experiment.py`` so the GPLFR path shares the exact same
graph / eigenvector caches (same keys) as the standalone manifold runners.

``add_manifold_args`` registers only the manifold-specific CLI flags — the ones
NOT already defined by ``gplfr_experiment.py`` (which owns template/reference/
num-inducing/nu/etc.), so it composes onto that parser without option clashes.
"""
import logging
from pathlib import Path

import torch
import numpy as np

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
    add_faiss_cpu_args, resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
    labels_for_nodes_from_template_clustering,
)
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

from utils import (
    crop_or_stride_volume,
    reference_ccf_from_subvolume,
    augment_nodes_with_maldi,
)


def add_manifold_args(parser):
    """Add the graph/Laplacian/eigensolve CLI flags to ``parser``.

    Only adds flags NOT already registered by the GPLFR runner, so it can be
    called after that runner's own ``add_argument`` calls without conflicts.
    """
    parser.add_argument("--eigenvector-dir", dest="eigenvector_dir", type=str, default=None,
                        help="Directory for the KNN graph / eigenvector caches "
                             "(required for --base-gp riemann|spectral).")
    parser.add_argument("--threshold", type=int, default=5, help="Reference-image tissue threshold.")
    parser.add_argument("--stride", dest="stride", type=int, default=4, help="Stride to downsample the template.")
    parser.add_argument("--knn-k", dest="knn_k", type=int, default=15, help="Number of KNN neighbours for the Graph Laplacian.")
    parser.add_argument("--n-list", default="1",
                        help="FAISS IVF nlist: an int, or 'sqrt' for round(sqrt(N)).")
    parser.add_argument("--n-probe", default="1",
                        help="FAISS IVF nprobe: an int, or 'sqrt' for round(sqrt(nlist)).")
    parser.add_argument("--num-modes", dest="num_modes", type=int, default=200, help="Number of eigenvectors (spectral modes) to use.")
    parser.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1,
                        help="Lanczos Krylov subspace floor. <=0 auto-picks max(1500, 3*num_modes+20).")
    parser.add_argument("--graphbandwidth-init", dest="graphbandwidth_init", type=float, default=1.0, help="Initial graph bandwidth.")
    parser.add_argument("--diffusion-scale-init", dest="diffusion_scale_init", type=float, default=1.0,
                        help="Initial multiplicative scale on the Laplacian spectrum (1.0 = identity).")
    parser.add_argument("--learn-diffusion-scale", dest="learn_diffusion_scale", action="store_true",
                        help="Make the diffusion scale learnable (no eigenpair recompute).")
    parser.add_argument("--bump-scale", dest="bump_scale", type=float, default=3.0, help="Bump function param.")
    parser.add_argument("--bump-decay", dest="bump_decay", type=float, default=0.05, help="Bump function param.")
    parser.add_argument("--lengthscale-init", dest="lengthscale_init", type=float, default=None,
                        help="Initial kernel lengthscale (z-units). Default: gpytorch default (~0.69).")
    parser.add_argument("--laplacian-norm", dest="laplacian_norm", type=str, default="symmetric",
                        help="Graph Laplacian normalization ('symmetric' | 'randomwalk' | 'none').")
    parser.add_argument("--knn-method", dest="knn_method", type=str, default="faiss",
                        help="'faiss' | 'anatomical_atlas' | 'faiss_atlas_weighted' | "
                             "'faiss_cluster_weighted'.")
    parser.add_argument("--cross-region-inflation", dest="cross_region_inflation", type=float, default=10.0,
                        help="(--knn-method faiss_atlas_weighted|faiss_cluster_weighted) "
                             "cross-region edge-weight inflation.")
    parser.add_argument("--cluster-k", dest="cluster_k", type=int, default=64,
                        help="(--knn-method faiss_cluster_weighted) number of template-clustering "
                             "regions used as node labels (no annotations-file needed).")
    parser.add_argument("--cluster-spatial-weight", dest="cluster_spatial_weight", type=float, default=1.0,
                        help="(faiss_cluster_weighted) weight on (z-scored) coords in the k-means feature "
                             "space; higher = more spatially compact/contiguous regions, 0 = pure feature k-means.")
    parser.add_argument("--cluster-fit-subsample", dest="cluster_fit_subsample", type=int, default=40000,
                        help="(faiss_cluster_weighted) nodes agglomerated before nearest-node propagation.")
    parser.add_argument("--cluster-seed", dest="cluster_seed", type=int, default=0,
                        help="(faiss_cluster_weighted) RNG seed for the template clustering.")
    parser.add_argument("--augment-maldi-nodes", dest="augment_maldi_nodes", action="store_true",
                        help="Add measured MALDI voxels to the node set before graph construction.")
    parser.add_argument("--force-recompute-graph", dest="force_recompute_graph", action="store_true",
                        help="Ignore the cached KNN graph and rebuild it.")
    parser.add_argument("--force-recompute-eigvecs", dest="force_recompute_eigvecs", action="store_true",
                        help="Ignore the cached eigenpairs and recompute them.")
    add_faiss_cpu_args(parser)
    return parser


def build_manifold_graph(args, config, coord_mean, coord_std):
    """Build (or load) the FAISS KNN graph over the reference tissue nodes.

    The graph-only front half of ``build_manifold_kernel`` (no Laplacian
    eigensolve), so consumers that need only the graph topology — e.g. a GCN over
    the manifold graph — don't pay for an eigensolve. Reference nodes are scaled
    by ``coord_mean`` / ``coord_std`` (use ``coord_norm_from_reference`` so the
    frame matches the cached graphs).

    Returns ``(knn, edge_index, edge_value, graph_key)``. ``knn`` exposes ``.x``
    (node coords) and ``.search(x, k)`` (nearest-node lookup); ``edge_index`` is
    PyG-style ``(2, E)``; ``graph_key`` keys the (downstream) eigenvector cache.
    """
    if args.get("eigenvector_dir") is None:
        raise ValueError(
            "--eigenvector-dir is required for the manifold graph "
            "(--base-gp riemann|spectral, or --model mlp_eigen|gcn_faiss)."
        )

    threshold           = args.get("threshold", 5)
    stride              = args.get("stride", 4)
    knn_k               = args.get("knn_k", 15)
    nlist_spec          = args.get("n_list", "1")
    nprobe_spec         = args.get("n_probe", "1")
    knn_method          = args.get("knn_method", "faiss")

    template_name = config.template_name
    template_volume = np.load(config.reference_file)
    annotations_volume = None
    if config.annotations_file is not None:
        annotations_volume = np.load(config.annotations_file)

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_volume, annotations_volume, stride,
    )

    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    reference_nodes = torch.tensor(reference_ccf, dtype=torch.float32)
    reference_nodes = (reference_nodes - coord_mean) / coord_std
    reference_nodes = reference_nodes.to(config.device).contiguous()

    augment_maldi = bool(args.get("augment_maldi_nodes", False))
    if augment_maldi:
        reference_nodes = augment_nodes_with_maldi(
            reference_nodes, config.maldi_file, config.section_filter,
            coord_mean, coord_std,
        )

    nlist = resolve_nlist(nlist_spec, reference_nodes.shape[0])
    nprobe = resolve_nprobe(nprobe_spec, nlist)
    logging.info(f"FAISS IVF: N={reference_nodes.shape[0]:,} -> nlist={nlist} nprobe={nprobe}")

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
    if nlist != 1:
        graph_key_parts["nlist"] = nlist
    if knn_method == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
    elif knn_method == "faiss_atlas_weighted":
        pass

    if augment_maldi:
        graph_key_parts["augmaldi"] = int(reference_nodes.shape[0])
        if knn_method in ("anatomical_atlas", "faiss_atlas_weighted", "faiss_cluster_weighted"):
            raise ValueError(
                f"--augment-maldi-nodes is only supported with --knn-method=faiss "
                f"(got {knn_method!r}); atlas/cluster-label methods assume nodes == "
                f"strided template voxels."
            )

    graph_key = make_graph_key(graph_key_parts)
    logging.info(f"Graph cache key: {graph_key}")

    force_graph = bool(args.get("force_recompute_graph", False))
    if knn_method == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss", coords=reference_nodes, k=knn_k,
            nlist=nlist, nprobe=nprobe, extra=graph_key_parts,
            force_recompute=force_graph, device=config.device,
        )
    elif knn_method == "anatomical_atlas":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas", volume=sub_volume,
            threshold=threshold, atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=knn_k, nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts, force_recompute=force_graph, device=config.device,
        )
    elif knn_method == "faiss_atlas_weighted":
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss", coords=reference_nodes, k=knn_k,
            nlist=nlist, nprobe=nprobe, extra=base_key_parts,
            force_recompute=force_graph, device=config.device,
        )
        if sub_atlas is None:
            raise ValueError(
                "--knn-method=faiss_atlas_weighted requires --annotations-file; got none."
            )
        node_labels = labels_for_nodes_from_sub_atlas(sub_volume, sub_atlas, threshold)
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Mismatch between labelled voxels ({node_labels.shape[0]}) "
                f"and graph nodes ({knn.x.shape[0]}). Check that atlas and "
                f"reference template were cropped/strided identically."
            )
        inflation = float(args.get("cross_region_inflation", 10.0))
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=True,
        )
        graph_key_parts["weighting"] = f"atlas_x{inflation:g}"
        graph_key = make_graph_key(graph_key_parts)
    elif knn_method == "faiss_cluster_weighted":
        # Data-driven, whole-brain, lipid-free labels: cluster the reference
        # TEMPLATE itself (no annotations-file). Base topology is the plain
        # faiss graph (cache shared with --knn-method=faiss); only the edge
        # weights differ, via the same soft cross-region inflation.
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss", coords=reference_nodes, k=knn_k,
            nlist=nlist, nprobe=nprobe, extra=base_key_parts,
            force_recompute=force_graph, device=config.device,
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

    return knn, edge_index, edge_value, graph_key


def build_manifold_kernel(args, config, coord_mean, coord_std):
    """Build a fully-formed ``RiemannMaternKernel`` from the reference geometry.

    Reproduces the graph/Laplacian/eigensolve pipeline of the standalone
    manifold runners (same cache keys), scaling the reference nodes by the given
    ``coord_mean`` / ``coord_std`` (use ``coord_norm_from_reference`` so the frame
    matches the cached graphs).

    Returns ``(manifold_kernel, knn)`` — ``knn`` is returned so the Riemann
    inducing base can pick inducing points directly from the graph nodes.
    """
    num_modes           = args.get("num_modes", 200)
    bump_scale          = args.get("bump_scale", 3.0)
    bump_decay          = args.get("bump_decay", 0.05)
    laplacian_norm      = args.get("laplacian_norm", "symmetric")
    graphbandwidth_init = args.get("graphbandwidth_init", 1.0)
    knn_k               = args.get("knn_k", 15)

    knn, edge_index, edge_value, graph_key = build_manifold_graph(
        args, config, coord_mean, coord_std,
    )
    eigenvector_dir = Path(args.get("eigenvector_dir"))

    # Eigenpairs: compute (or load from cache) before kernel construction.
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

    ncv_min = resolve_ncv_min(num_modes, args.get("ncv_min", -1))
    solver = LaplacianEigensolver(
        num_modes=num_modes, backend="cupy", tol=1e-4, ncv_min=ncv_min, verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op, cache_dir=eigvec_cache_dir, key=eigvec_key,
        graphbandwidth=graphbandwidth_init, laplacian_normalization=laplacian_norm,
        extra=eigvec_key_parts,
        force_recompute=bool(args.get("force_recompute_eigvecs", False)),
        device=config.device, allow_larger_modes=True,
    )
    logging.info(f"eigvec shape: {tuple(eigvec.shape)}  eigval range: "
                 f"[{eigval.min().item():.4g}, {eigval.max().item():.4g}]")

    logging.info("Building RiemannMaternKernel (knn + edges + eigenpairs)...")
    manifold_kernel = RiemannMaternKernel(
        nu=config.nu, knn=knn, edge_index=edge_index, edge_value=edge_value,
        eigval=eigval, eigvec=eigvec, nearest_neighbors=knn_k, num_modes=num_modes,
        bump_scale=bump_scale, bump_decay=bump_decay,
        laplacian_normalization=laplacian_norm,
        graphbandwidth_init=graphbandwidth_init,
        diffusion_scale_init=args.get("diffusion_scale_init", 1.0),
        learn_diffusion_scale=args.get("learn_diffusion_scale", False),
    ).to(config.device)
    ls_init = args.get("lengthscale_init", None)
    if ls_init is not None:
        manifold_kernel.lengthscale = torch.tensor(float(ls_init))
    manifold_kernel.eval()

    return manifold_kernel, knn
