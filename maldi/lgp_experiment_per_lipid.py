#!/usr/bin/env python
"""Per-lipid GP training pipeline.

Trains one independent GP per lipid for either the Euclidean Matern kernel
or the Riemann Manifold kernel. Reuses:

  - ``IndependentMultitaskGPModel``  (from l3di.lgp)            for Euclidean
  - ``LatentRiemannGP``              (from l3di.lgp_manifold)   for Manifold
  - All MaLDI / atlas / graph / inducing-point machinery from
    ``lgp_experiment.py`` and ``lgp_manifold_experiment.py``.

Why "per lipid" if we already have ``IndependentMultitaskGPModel`` with
``num_tasks``?  Because a single (num_tasks=173, num_inducing=500) variational
distribution is 173 × 500² floats = ~170M parameters in the Cholesky factor
alone, plus the matching kernel hyperparameter and inducing-point batches
— it OOMs on most GPUs and is wasteful.  We instead loop in batches of
``--lipid-batch-size`` lipids (default 10), reusing the exact same class
with ``num_tasks=batch_size``. Each batch gets its own optimiser and its
own variational fit, so lipids in different batches are fully independent
and we get the same statistical model as 173 separate fits.

Output layout (under ``<output_dir>/<exp_name>/``)::

    config.json                # snapshot of every CLI flag
    metrics.csv                # one row per lipid (test RMSE, R², etc.)
    lipid_names.json           # int idx -> name mapping
    graph_meta.npz             # voxel indices / atlas shape for the brain
    predictions/<lipid_slug>/
        test_coords_mm.npy     # (N_test, 3)   physical mm coordinates
        test_pred_z.npy        # (N_test,)     z-scored mean
        test_pred_raw.npy      # (N_test,)     raw (de-standardised) mean
        test_std_z.npy         # (N_test,)     posterior stdev
        test_true_z.npy        # (N_test,)     ground truth
        graph_pred_z.npy       # (N_nodes,)    whole-brain reconstruction
        graph_pred_raw.npy     # (N_nodes,)
        graph_std_z.npy        # (N_nodes,)
    checkpoints/batch_<bb>.pt  # one file per lipid batch (state_dict + meta)

The on-disk layout is what ``visualize_lipid_gp.py`` (the slim viewer) reads.
No model checkpoints need to be reloaded for visualisation — predictions
are precomputed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm

# Re-use the EXISTING project pieces — no copy-paste.
from config import MaldiConfig
from l3di.lgp import IndependentMultitaskGPModel
from l3di.lgp_manifold import LatentRiemannGP, SpectralLatentGP
from utils import (
    get_inducing_points,
    get_data_inducing_points,
    crop_or_stride_volume,
    reference_ccf_from_subvolume,
    augment_nodes_with_maldi,
)

# Manifold kernel + graph stack. Imports kept identical to
# ``lgp_manifold_experiment.py`` for compatibility.
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
    add_faiss_cpu_args, apply_faiss_cpu_args, faiss_cpu_recon,
    resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
    labels_for_nodes_from_template_clustering, dissolve_root_labels,
    denoise_labels_majority_vote, prune_cross_region_edges,
)
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.kernels.surface_kernels import SurfaceRiemannMaternKernel

# Reference-only parcel geometry (--parcel-field). Self-contained package at the
# repo root; the path shim covers environments whose editable install predates it
# (a fresh `pip install -e .` picks it up automatically via `packages = find:`).
try:
    from parcelgp.field import ParcelField
    from parcelgp.kernels import wrap_scale_kernel
except ModuleNotFoundError:                                          # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from parcelgp.field import ParcelField
    from parcelgp.kernels import wrap_scale_kernel


# =============================================================================
# CLI parsing — strict superset of lgp_*_experiment.py flags
# =============================================================================
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=(
            "Train one independent GP per lipid (Euclidean or Manifold), "
            "batched in groups of --lipid-batch-size lipids to control "
            "GPU memory. Outputs per-lipid prediction arrays ready for "
            "off-line analysis or for the slim napari viewer."
        ),
    )

    # ---- which kernel ----
    p.add_argument("--kernel-family",
                   choices=["euclidean", "manifold", "eigenmap", "spectral"],
                   required=True,
                   help="'euclidean' → IndependentMultitaskGPModel "
                        "(reuse from l3di.lgp), 'manifold' → "
                        "LatentRiemannGP (reuse from l3di.lgp_manifold), "
                        "'eigenmap' → project coords into the leading "
                        "--embed-dim Laplacian eigenfunctions, then a Euclidean "
                        "ARD Matern GP over that embedding (reuses the euclidean "
                        "path; needs --eigenvector-dir), 'spectral' → weight-space "
                        "SpectralLatentGP over the manifold spectrum, one per lipid "
                        "(needs --eigenvector-dir).")
    p.add_argument("--embed-dim", dest="embed_dim", type=int, default=10,
                   help="(--kernel-family eigenmap) number of leading Laplacian "
                        "eigenfunctions to use as the GP input coordinates. Keep "
                        "small (diffusion-map regime); ARD lets the kernel weight "
                        "modes. Default 10.")

    # ---- everything common to both experiment scripts ----
    p.add_argument("--exp-name", required=True,
                   help="Sub-directory under --output-dir for this run.")
    p.add_argument("--output-dir", required=True,
                   help="Top-level results directory.")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--maldi-file", required=True)
    p.add_argument("--available-lipids-file", required=True)
    p.add_argument("--slices-dataset-file", required=True)
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None)
    p.add_argument("--mode", default="per_lipid",
                   help="Passed to MaldiConfig for compatibility.")

    p.add_argument("--num-inducing", type=int, default=500,
                   help="Inducing points per GP (per lipid). "
                        "For the Euclidean kernel these are learned; "
                        "for the manifold kernel they are anchored.")
    p.add_argument("--inducing-source", default="reference",
                   choices=["reference", "data"],
                   help="'reference' (default): k-means over the reference "
                        "tissue image. 'data': draw inducing points from "
                        "ACTUAL measured MALDI voxels (sparse-data aware). "
                        "For the manifold kernel they are then snapped to "
                        "graph nodes as usual.")
    p.add_argument("--inducing-method", default="kmeans_snap",
                   choices=["kmeans_snap", "fps", "random"],
                   help="(--inducing-source data) on-data selection: "
                        "'kmeans_snap' (default), 'fps' (max coverage), "
                        "'random'.")
    p.add_argument("--inducing-from-maldi-nodes", dest="inducing_from_maldi_nodes",
                   action="store_true",
                   help="(manifold) blend the inducing points over the UNALTERED "
                        "strided graph: ~--inducing-density-frac from the densest "
                        "graph nodes + the rest from the measured MALDI voxels "
                        "that snap onto the graph most cheaply (smallest snap "
                        "distance). Every inducing point is an exact graph node "
                        "(subsampling via --inducing-method). Independent of "
                        "--augment-maldi-nodes (the KNN graph is NOT modified). "
                        "Overrides --inducing-source for the manifold kernel.")
    p.add_argument("--inducing-density-frac", dest="inducing_density_frac",
                   type=float, default=0.8,
                   help="(--inducing-from-maldi-nodes) fraction of inducing "
                        "points drawn from the densest graph nodes; the "
                        "remainder come from cheapest-to-snap MALDI nodes. "
                        "Default 0.8 (80/20 blend).")
    p.add_argument("--learn-inducing", dest="learn_inducing",
                   action="store_true", default=False,
                   help="Optimize the inducing-point LOCATIONS jointly with the "
                        "variational/kernel parameters (learn_inducing_locations="
                        "True). Default off: inducing points stay anchored where "
                        "they were initialized. For the manifold kernel anchored "
                        "points sit exactly on graph nodes (exact-eigenvector "
                        "branch); learning lets them drift off-graph onto the "
                        "kernel's Nyström path, so enable deliberately.")
    p.add_argument("--lipid-batch-size", type=int, default=10,
                   help="Number of lipids to fit simultaneously (= "
                        "num_tasks of the multitask GP). Increase this "
                        "to amortise per-step overhead; decrease if you "
                        "OOM. With 173 lipids and default 10, the run "
                        "does 18 batches.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20,
                   help="Adam epochs per lipid-batch fit.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-transform", action="store_true")
    p.add_argument("--nu", type=float, default=2.0)
    p.add_argument("--per-task-lengthscale", dest="per_task_lengthscale",
                   action="store_true",
                   help="(manifold kernel) give each lipid in the batch its OWN "
                        "learnable lengthscale instead of one shared across the "
                        "batch. Each task reuses the same eigenpairs/graph.")
    p.add_argument("--product-ard-matern", dest="product_ard_matern",
                   action="store_true",
                   help="(manifold kernel) multiply the Riemann kernel by an ARD "
                        "Euclidean Matern (per-axis lengthscales) so the model "
                        "regains ambient anisotropy while keeping geodesic routing: "
                        "k = k_geo * k_eucl-ARD.")
    p.add_argument("--product-ard-nu", dest="product_ard_nu", type=float,
                   default=2.5,
                   help="Matern smoothness (nu) for the ARD Euclidean factor of "
                        "--product-ard-matern.")
    p.add_argument("--lengthscale-init", dest="lengthscale_init", type=float,
                   default=None,
                   help="Initial kernel lengthscale (z-units). Default: gpytorch "
                        "default (~0.69). Smaller favours more local covariance.")

    # ---- reference-only parcel factor (parcelgp) ----
    p.add_argument("--parcel-field", dest="parcel_field", default=None,
                   help="Path to a parcelgp .npz parcel field. When set, the "
                        "spatial kernel is multiplied by a learned parcel-similarity "
                        "factor exp(-||z(x)-z(x')||^2/2), z(x)=m(x)^T B — letting "
                        "covariance break across a template-derived boundary and "
                        "reach across a gap within a region, which no stationary "
                        "kernel can express. The partition is reference-only; only "
                        "B is learned. Build one with `python -m parcelgp.build`.")
    p.add_argument("--parcel-rank", dest="parcel_rank", type=int, default=8,
                   help="(--parcel-field) width r of the learned parcel embedding.")
    p.add_argument("--parcel-init-scale", dest="parcel_init_scale", type=float,
                   default=0.05,
                   help="(--parcel-field) std of the B init. Small keeps training "
                        "starting at the unmodified base kernel; 0.0 makes the "
                        "factor an exact no-op at init.")
    p.add_argument("--parcel-shared-B", dest="parcel_per_task",
                   action="store_false", default=True,
                   help="(--parcel-field) share one B across the lipid batch "
                        "instead of one per lipid.")
    p.add_argument("--lengthscale-no-decay", dest="lengthscale_no_decay",
                   action="store_true",
                   help="Exclude raw_lengthscale / raw_graphbandwidth from AdamW "
                        "weight_decay (decay otherwise nudges the lengthscale toward "
                        "its init regardless of the data).")
    p.add_argument("--n-pixels", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=4096,
                   help="Mini-batch size for variational SGD over MaLDI "
                        "points.")

    # ---- variational family ----
    p.add_argument("--variational", choices=["analytic", "nngp"],
                   default="analytic",
                   help=("'analytic' (default): the batched multitask "
                         "SVGP with the analytic expected-log-likelihood "
                         "ELBO (IndependentMultitaskGPModel / "
                         "LatentRiemannGP). 'nngp': Variational Nearest "
                         "Neighbor GP (Wu et al. 2022) — inducing points "
                         "= a dense subset of the TRAINING voxels, and "
                         "each point only couples to its --nn-k nearest "
                         "inducing neighbours, so there is no O(M^3) "
                         "Cholesky and M can be huge (kills the "
                         "'inducing points too scattered' problem). "
                         "EUCLIDEAN ONLY — the manifold spectral kernel "
                         "does not compose with the NN factorisation. "
                         "Fits one lipid at a time (VNNGP is O(k^3) per "
                         "step, so this is cheap)."))
    p.add_argument("--nn-k", type=int, default=256,
                   help=("(--variational nngp) Number of nearest "
                         "inducing neighbours each point conditions on. "
                         "Typical 64-256. Larger = closer to exact, more "
                         "compute per step (O(k^3))."))
    p.add_argument("--nngp-num-inducing", type=int, default=0,
                   help=("(--variational nngp) Number of training voxels "
                         "to use as inducing points. 0 = use ALL training "
                         "voxels (the VNNGP default; recommended). Set a "
                         "positive number to cap via a random subsample "
                         "if memory is tight (the mean-field q has one "
                         "mean+var per inducing point)."))
    p.add_argument("--nn-metric", choices=["euclidean", "geodesic"],
                   default="euclidean",
                   help=("(--variational nngp) Metric for choosing each "
                         "point's VNNGP conditioning neighbours. 'euclidean' "
                         "(default): gpytorch's built-in faiss L2 kNN. "
                         "'geodesic': build a faiss kNN graph over the "
                         "inducing voxels and rank neighbours by SHORTEST-"
                         "PATH distance on it (so neighbours respect tissue "
                         "shape — they route around gaps/ventricles instead "
                         "of jumping across). Kernel stays Euclidean Matern "
                         "(PSD-safe). The shortest-path kNN is precomputed "
                         "once and cached to the output dir."))
    p.add_argument("--geodesic-graph-k", type=int, default=16,
                   help=("(--nn-metric geodesic) Degree of the faiss kNN "
                         "graph whose shortest paths define the geodesic "
                         "metric. The graph only needs to be locally "
                         "connected (small k); multi-hop Dijkstra reaches "
                         "the --nn-k farther neighbours. 16 is a good "
                         "default."))

    # ---- Euclidean-only knob ----
    p.add_argument("--kernel", default="matern",
                   choices=["rbf", "matern", "symmetric"],
                   help="Sub-type for IndependentMultitaskGPModel.")
    p.add_argument("--no-ard", dest="no_ard", action="store_true",
                   help="EUCLIDEAN ONLY — use an isotropic (single shared) "
                        "lengthscale instead of per-axis ARD. Without this "
                        "flag the Euclidean RBF/Matern kernel learns one "
                        "lengthscale per spatial axis (ard_num_dims=3).")

    # ---- Manifold-only knobs (ignored when --kernel-family=euclidean) ----
    p.add_argument("--eigenvector-dir", default=None,
                   help="Required when --kernel-family=manifold.")
    p.add_argument("--knn-method", default="faiss",
                   choices=["faiss", "anatomical_atlas",
                            "faiss_atlas_weighted", "faiss_cluster_weighted"],
                   help=("'faiss': Euclidean kNN, no anatomical prior. "
                         "'anatomical_atlas': edges restricted to same "
                         "atlas region + voxel-adjacent cross-region "
                         "links (strong but can produce disconnected "
                         "components → NaN). "
                         "'faiss_atlas_weighted': use the faiss kNN "
                         "graph as-is but INFLATE squared distances "
                         "on edges that cross atlas regions, by a "
                         "factor of --cross-region-inflation. Soft "
                         "anatomical prior — graph stays connected "
                         "(no NaN) and the spectral kernel still "
                         "prefers within-region smoothness because "
                         "the Gaussian edge weight w=exp(-d²/2σ²) "
                         "collapses for inflated d². "
                         "'faiss_cluster_weighted': same soft inflation, "
                         "but region labels are CLUSTERED from the "
                         "reference template (no atlas file needed) — "
                         "data-driven, whole-brain, lipid-free."))
    p.add_argument("--cross-region-inflation", type=float, default=10.0,
                   help=("For --knn-method=faiss_atlas_weighted|faiss_cluster_weighted. "
                         "Multiplier applied to squared Euclidean "
                         "distance on edges that connect two regions. "
                         "Default 10 = mild prior; try 100 "
                         "for a strong prior, 1 for none. The "
                         "underlying graph topology is identical to "
                         "pure faiss; only the edge weights change."))
    p.add_argument("--cluster-k", dest="cluster_k", type=int, default=64,
                   help="(faiss_cluster_weighted) number of template-clustering regions.")
    p.add_argument("--cluster-spatial-weight", dest="cluster_spatial_weight", type=float, default=1.0,
                   help="(faiss_cluster_weighted) weight on (z-scored) coords in the k-means "
                        "feature space; higher = more spatially compact regions, 0 = pure feature k-means.")
    p.add_argument("--cluster-fit-subsample", dest="cluster_fit_subsample", type=int, default=40000,
                   help="(faiss_cluster_weighted) nodes used to train the k-means centroids.")
    p.add_argument("--cluster-seed", dest="cluster_seed", type=int, default=0,
                   help="(faiss_cluster_weighted) RNG seed for the template clustering.")
    p.add_argument("--root-handling", dest="root_handling", type=str, default="dissolve",
                   choices=["dissolve", "ignore", "cross"],
                   help="(faiss_atlas_weighted) how to treat the atlas label-0 'root' catch-all "
                        "tissue. 'dissolve' (default): reassign each root node to its nearest real "
                        "region before inflating (frees interiors, keeps region↔region "
                        "confinement). 'ignore': keep root, don't inflate root-touching edges. "
                        "'cross': legacy (inflate all root edges; reuses old eigvec caches).")
    p.add_argument("--denoise-labels", dest="denoise_labels", type=int, default=0,
                   help="Majority-vote label smoothing passes over the graph before the prune "
                        "(absorbs speck nodes). 0 = off. NOTE: erodes thin real boundaries — "
                        "prefer --root-handling dissolve for the catch-all.")
    p.add_argument("--prune-cross-region", dest="prune_cross_region", type=float, default=0.0,
                   help="Fraction of cross-region edges to HARD-remove (vs the soft "
                        "--cross-region-inflation), connectivity-preserving. 0 = off. Changes "
                        "edges → distinct eigvec cache key.")
    p.add_argument("--laplacian-norm", default="randomwalk",
                   choices=["symmetric", "randomwalk"])
    p.add_argument("--normalize-features", dest="normalize_features", action="store_true",
                   help="Use the cosine/correlation Riemann kernel: L2-normalize the feature "
                        "rows so the prior variance is constant (diagonal=1) and the "
                        "sqrt(degree) sampling artifact is quotiented out; magnitude is then "
                        "carried by the ScaleKernel outputscale.")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--augment-maldi-nodes", dest="augment_maldi_nodes", action="store_true",
                   help="Add the measured MALDI voxels to the graph node set so every "
                        "measured point is an exact graph node (independent of stride). "
                        "Supported with --knn-method faiss / faiss_atlas_weighted "
                        "(not anatomical_atlas).")
    p.add_argument("--max-maldi-nodes", dest="max_maldi_nodes", type=int, default=0,
                   help="Cap on MALDI voxels added by --augment-maldi-nodes (0 = all). "
                        "The eigensolve workspace is ~ncv*N, so use this to bound N "
                        "(e.g. 200000) when the full measured set blows up GPU memory.")
    p.add_argument("--maldi-subsample-method", dest="maldi_subsample_method",
                   default="random", choices=["random", "fps", "kmeans_snap"],
                   help="How to subsample MALDI voxels down to --max-maldi-nodes.")
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--n-list", default="1",
                   help="FAISS IVF nlist: an int, or 'sqrt' for round(sqrt(N)).")
    p.add_argument("--n-probe", default="1",
                   help="FAISS IVF nprobe: an int, or 'sqrt' for round(sqrt(nlist)). "
                        "For 3D coords a small FIXED value (~8) gives recall ~1.0 at "
                        "every N; 'sqrt' over-scans (grows with N for no gain).")
    p.add_argument("--graphbandwidth-init", type=float, default=0.1)
    p.add_argument("--diffusion-scale-init", dest="diffusion_scale_init",
                   type=float, default=1.0,
                   help="Initial multiplicative scale on the Laplacian spectrum "
                        "(lambda_k -> diffusion_scale*lambda_k in the Matern spectral "
                        "density). 1.0 = identity.")
    p.add_argument("--learn-diffusion-scale", dest="learn_diffusion_scale",
                   action="store_true",
                   help="Make the diffusion scale learnable. Reshapes the spectral "
                        "density WITHOUT recomputing eigenpairs (scaling an operator "
                        "leaves its eigenvectors unchanged), so it is a cheap, "
                        "eigenpairs-free companion to the lengthscale. Frozen at "
                        "--diffusion-scale-init otherwise.")
    p.add_argument("--learn-spectral-weights", dest="learn_spectral_weights",
                   action="store_true",
                   help="Learn ONE free positive weight per Laplacian mode instead of "
                        "tying them to the monotone Matern density "
                        "S_k=(2*nu/l^2+diffusion_scale*lambda_k)^-nu. Makes the prior "
                        "spectral density anisotropic in lambda (can up-weight the "
                        "boundary-discriminating modes a monotone S_k cannot). "
                        "Warm-started at the Matern density. Applies to the manifold and "
                        "spectral families; with one knob per mode, use more epochs / a "
                        "smoothness prior to avoid overfitting.")
    p.add_argument("--surface-kernel", dest="surface_kernel", action="store_true",
                   help="Multiply the manifold kernel by a depth (distance-to-surface) "
                        "factor: k = k_manifold(x) * k_depth(d_surface(x)), i.e. use "
                        "SurfaceRiemannMaternKernel. d_surface = EDT distance-to-tissue-"
                        "surface, computed once at startup; the kernel samples it from "
                        "the (unchanged 3-D) coords internally (nearest node), so the "
                        "coords/inducing/predict paths are untouched. Manifold family only.")
    p.add_argument("--surface-depth-lengthscale", dest="surface_depth_lengthscale",
                   type=float, default=1.0,
                   help="(--surface-kernel) init lengthscale of the depth factor "
                        "(standardized-depth units). Large -> depth factor ~1 (opt out); "
                        "learnable, one per task under --per-task-lengthscale.")
    p.add_argument("--bump-scale", type=float, default=20.0)
    p.add_argument("--bump-decay", type=float, default=0.01)
    p.add_argument("--num-modes", type=int, default=1300)
    p.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1,
                   help="Lanczos Krylov subspace floor. <=0 (default) auto-picks "
                        "max(1500, 3*num_modes+20); set a small explicit value "
                        "(e.g. 100) to fit large-N (stride=1) eigensolves in GPU "
                        "memory.")
    p.add_argument("--threshold", type=int, default=5)

    # ---- lipid restriction / debugging ----
    p.add_argument("--lipids", nargs="+", default=None,
                   help="Restrict to a subset of lipids (names or "
                        "indices). Default: all of them. NOTE: lipid "
                        "names with spaces won't survive shell word-"
                        "splitting; use --lipids-file for those.")
    p.add_argument("--lipids-file", default=None,
                   help="Path to a text file with one lipid name (or "
                        "integer index) per line. Lines beginning with "
                        "'#' and blank lines are ignored. This is the "
                        "right way to pass lipid names that contain "
                        "spaces (e.g. 'PC 35:1 PE 38:1'). Combined "
                        "with --lipids if both given.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of lipids processed (applied "
                        "after --lipids/--lipids-file filtering).")
    p.add_argument("--resume", choices=["auto", "force"], default="auto",
                   help=("Checkpoint behaviour. 'auto' (default): if the "
                         "output dir already has complete .npy "
                         "predictions for any lipid in a batch, skip "
                         "that entire batch and continue with the next. "
                         "Restores metrics.csv too. Designed for cloud "
                         "jobs that crash + auto-restart with the same "
                         "--exp-name. 'force': re-run every batch from "
                         "scratch (overwrites existing predictions)."))
    p.add_argument("--checkpoint-every-epochs", type=int, default=5,
                   help=("Save the in-progress per-batch model state "
                         "every N training epochs, overwriting the "
                         "previous in-progress checkpoint. On resume "
                         "(after a crash), training continues from this "
                         "checkpoint rather than starting the batch "
                         "over from scratch. 0 = disable."))
    p.add_argument("--dataloader-workers", type=int, default=2,
                   help=("Number of background worker processes the "
                         "training DataLoader uses to prepare "
                         "minibatches. Workers prefetch the next "
                         "minibatch (shuffle index + tensor slice + "
                         "pin to page-locked memory) while the GPU "
                         "is busy with the current step, hiding the "
                         "data-prep latency entirely. 0 = synchronous "
                         "(useful for debugging tracebacks); 2-4 is "
                         "a good default; >4 hits diminishing returns "
                         "and uses more RAM since each worker keeps a "
                         "copy of the dataset tensors."))
    p.add_argument("--wandb", dest="wandb", action="store_true", default=False,
                   help=("Log training metrics (loss / recon / KL / kernel "
                         "hypers / observation noise / per-group gradient "
                         "norms, including the inducing points when "
                         "--learn-inducing is set) to Weights & Biases. One "
                         "run for the whole job; steps are global across "
                         "lipid batches."))
    p.add_argument("--wandb-project", dest="wandb_project",
                   default="l3di_maldi_per_lipid",
                   help="W&B project name (used only when --wandb is set).")
    p.add_argument("--grad-clip", type=float, default=10.0,
                   help=("Gradient L2-norm clip. The analytic ELBO can "
                         "produce very large gradients at initialization "
                         "(the posterior variance term is huge under "
                         "the GP prior, before training contracts it), "
                         "and the manifold kernel's K_uu Cholesky can "
                         "be ill-conditioned. Clipping the gradient "
                         "norm to this value keeps the first few "
                         "epochs numerically stable. Default 10.0 = "
                         "permissive (rarely binds late in training); "
                         "try 1.0 if you still see non-finite gradient "
                         "warnings."))
    p.add_argument("--bad-grad-skip-frac", type=float, default=0.01,
                   help=("Fraction of gradient entries that must be "
                         "NaN/Inf before skipping the whole step. Below "
                         "this threshold we zero the bad entries and "
                         "step normally (well-defined directions still "
                         "informative). At or above the threshold, the "
                         "step is skipped entirely — too much zeroing "
                         "means moving in an essentially random "
                         "direction. Default 0.01 (1%% of params). Set "
                         "higher (e.g. 0.1) to tolerate more zeroing; "
                         "set lower (e.g. 0.001) to skip more "
                         "aggressively."))
    p.add_argument("--max-consecutive-bad-grads", type=int, default=1000,
                   help=("Bail-out threshold: if this many consecutive "
                         "training steps had ANY NaN/Inf gradient "
                         "entries (regardless of whether we zeroed or "
                         "skipped), raise RuntimeError and abort the "
                         "whole pipeline. Sustained NaN noise — even "
                         "at low per-step fractions — signals a kernel "
                         "configuration that isn't healing itself; "
                         "better to stop and let the operator pick a "
                         "different config than to grind through a "
                         "long run that may never converge. The "
                         "counter resets on the first fully-clean "
                         "step, so a single recovery clears it."))
    p.add_argument("--hyperpriors", dest="hyperpriors", action="store_true",
                   default=True,
                   help=("Register weakly-informative priors on the GP "
                         "hyperparameters (signal variance / outputscale and "
                         "observation noise) so the analytic multitask path "
                         "does MAP-II rather than ML-II. This matches the "
                         "regularization the exact manifold_gp model applies "
                         "via manifold_informed_train, making exact-vs-"
                         "variational comparisons apples-to-apples. The "
                         "VariationalELBO folds the prior log-probs in at the "
                         "correct 1/num_data scale automatically. Default on."))
    p.add_argument("--no-hyperpriors", dest="hyperpriors", action="store_false",
                   help=("Disable hyperparameter priors (pure ML-II). Use for "
                         "A/B comparisons against the --hyperpriors default."))
    p.add_argument("--lengthscale-prior-mean", dest="lengthscale_prior_mean",
                   type=float, default=None,
                   help=("If set (and --hyperpriors), also register a "
                         "Gamma(3, 3/mean) prior on the base-kernel "
                         "lengthscale centered at this value. Off by default "
                         "because the lengthscale's natural scale is data-"
                         "dependent (it carries a GreaterThan(minimal_length_"
                         "scale) constraint), so a bad center fights the "
                         "constraint. Set it explicitly when you know the "
                         "expected lengthscale for your z-scored coordinates."))

    # ---- compatibility scarecrows: accepted but unused ----
    # The l3di MaldiConfig may demand these. We provide harmless defaults
    # so that the same shell wrapper can drive both this script and the
    # existing experiment scripts.
    p.add_argument("--latent-dim", type=int, default=1,
                   help="Not used in per-lipid mode (kept for "
                        "MaldiConfig compatibility).")
    p.add_argument("--no-rsample", action="store_false")
    p.add_argument("--use-diffusion", action="store_true")
    p.add_argument("--do-brain-reconstruction", action="store_true",
                   default=True,
                   help="In per-lipid mode this is always on — the "
                        "whole-brain prediction is what the viewer needs.")

    # ---- whole-brain renders (same output as lgp_manifold_experiment) ----
    p.add_argument("--render", dest="render", action="store_true", default=True,
                   help="After training, render composite PNGs of the whole-brain "
                        "reconstruction for the trained lipids into "
                        "<exp>/renders/, mirroring lgp_manifold_experiment's "
                        "whole_brain_reconstruction render. On by default; only "
                        "the lipids trained in THIS run are rendered.")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="Disable the post-training lipid renders.")
    p.add_argument("--render-max-lipids", dest="render_max_lipids", type=int,
                   default=8,
                   help="Cap on how many trained lipids get a PNG render — a cloud "
                        "batch job just needs a few sanity-check images, not one "
                        "per lipid. 0 = render every trained lipid.")

    p.add_argument("--force-recompute-graph", dest="force_recompute_graph",
                   action="store_true",
                   help="Ignore the cached KNN graph and rebuild it from scratch "
                        "(needed to time graph construction, e.g. under "
                        "--faiss-cpu-graph).")

    add_faiss_cpu_args(p)

    return vars(p.parse_args())


# =============================================================================
# Helpers
# =============================================================================
def safe_filename(name: str) -> str:
    """Turn an arbitrary lipid name into a filesystem-safe slug.
    'PA 36:1 PA 38:4' → 'PA_36-1_PA_38-4'
    """
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace(":", "-"))
    return s.strip("_")


# Files we expect every COMPLETED lipid prediction dir to contain.
# Used by the --resume auto path: if all these exist, the lipid is done.
# (test_*_raw.npy and graph_*_raw.npy are derivable from _z + col_means/std,
# so we don't require them — but training will produce them.)
_LIPID_REQUIRED_FILES = (
    "test_coords_mm.npy",
    "test_pred_z.npy",
    "test_true_z.npy",
    "test_std_z.npy",
    "graph_pred_z.npy",
    "graph_std_z.npy",
)


def lipid_is_complete(predictions_root: Path, slug: str) -> bool:
    """True iff predictions/<slug>/ exists and contains all the
    essential .npy files for a finished lipid."""
    d = predictions_root / slug
    if not d.is_dir():
        return False
    return all((d / f).is_file() for f in _LIPID_REQUIRED_FILES)


def savez_safe(path, **arrays):
    """
    Always writes .npz to a local temporary file first, then uses a 
    purely sequential copy to the final destination. This completely 
    avoids the seek() errors on S3 FUSE mounts without risking broken
    file states from failed direct write attempts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the temp file on the local filesystem
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Write the ZIP archive locally (where seek() is fully supported)
        np.savez(tmp_path, **arrays)
        
        # Use shutil.copyfile (not move) — move tries rename() first,
        # which fails across filesystems; copyfile is strictly sequential.
        shutil.copyfile(tmp_path, str(path))
    finally:
        # Clean up the local temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

def resolve_lipids(spec, lipid_names: list[str],
                   log: logging.Logger) -> list[int]:
    """Convert --lipids tokens (names OR integer indices) → list of ints."""
    if spec is None:
        return list(range(len(lipid_names)))
    out = []
    for tok in spec:
        # Try integer first
        try:
            i = int(tok)
            if 0 <= i < len(lipid_names):
                out.append(i)
                continue
        except ValueError:
            pass
        if tok in lipid_names:
            out.append(lipid_names.index(tok))
        else:
            log.warning(f"Lipid spec '{tok}' not found; skipped.")
    # de-dup, preserve order
    seen = set()
    return [i for i in out if not (i in seen or seen.add(i))]


def load_maldi_columns(maldi_file, filter_expr, lipid_indices,
                       lipid_names_all, coord_cols=("xccf", "yccf", "zccf")):
    """Read coords + a SUBSET of lipid columns from the parquet.

    Only loads the lipid columns we care about — much cheaper than
    reading all 173 every time. ``lipid_indices`` indexes into
    ``lipid_names_all``.
    """
    df_coords = pd.read_parquet(
        maldi_file, columns=list(coord_cols), filters=filter_expr,
    )
    cols = [lipid_names_all[i] for i in lipid_indices]
    df_lip = pd.read_parquet(
        maldi_file, columns=cols, filters=filter_expr,
    )
    coords = torch.from_numpy(df_coords.values.astype(np.float32))
    values = torch.from_numpy(df_lip.values.astype(np.float32))
    return coords, values


def embed_eigenmap(kernel, x, r, device, chunk=50000):
    """Project coordinates into the leading ``r`` UNSCALED Laplacian eigenfunctions.

    ``e(x) = kernel.raw_eigenvectors(x)[:, :r]`` (smoothest r modes, since the
    eigenpairs are sorted by ascending eigenvalue). Chunked so the
    ``(chunk, num_modes)`` intermediate stays bounded for million-point inputs.
    Returns an ``(N, r)`` tensor on CPU.
    """
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            # faiss GPU search (inside raw_eigenvectors) requires a contiguous
            # input, like the reference_nodes the manifold path already snaps.
            xb = x[i:i + chunk].to(device).contiguous()
            outs.append(kernel.raw_eigenvectors(xb)[:, :r].cpu())
    return torch.cat(outs, dim=0)


# =============================================================================
# Manifold-only setup (kernel + graph) — extracted unchanged from
# lgp_manifold_experiment.setup_experiment(), minus the LGP / decoder bits.
# =============================================================================
def setup_manifold_kernel(args, config, coord_mean, coord_std, log):
    """Build the RiemannMaternKernel exactly as ``lgp_manifold_experiment``
    does. Returns the kernel ready to plug into LatentRiemannGP.
    """
    template_volume = np.load(args["reference_file"])
    annotations_volume = (np.load(args["annotations_file"])
                          if args.get("annotations_file") else None)

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_volume, annotations_volume, args["stride"],
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, args["threshold"],
    )
    reference_nodes = torch.tensor(reference_ccf, dtype=torch.float32)
    reference_nodes = (reference_nodes - coord_mean) / coord_std
    reference_nodes = reference_nodes.to(args["device"]).contiguous()

    # Optionally add the measured MALDI voxels to the node set BEFORE the KNN
    # graph is built, so every measurement is an exact graph node (no
    # Nyström/bump zeroing in grid gaps), independent of stride.
    augment_maldi = bool(args.get("augment_maldi_nodes", False))
    n_atlas_nodes = reference_nodes.shape[0]   # before any MALDI nodes are added
    if augment_maldi:
        reference_nodes = augment_nodes_with_maldi(
            reference_nodes, config.maldi_file, config.section_filter,
            coord_mean, coord_std,
            max_nodes=args.get("max_maldi_nodes", 0),
            subsample_method=args.get("maldi_subsample_method", "random"),
            seed=args.get("seed", 0),
        )

    # Resolve IVF nlist/nprobe now that N is fixed (after any MALDI augmentation).
    # 'sqrt' -> nlist=round(sqrt(N)), nprobe=round(sqrt(nlist)).
    nlist = resolve_nlist(args.get("n_list", "1"), reference_nodes.shape[0])
    nprobe = resolve_nprobe(args.get("n_probe", "1"), nlist)
    log.info(f"FAISS IVF: N={reference_nodes.shape[0]:,} -> nlist={nlist} nprobe={nprobe}")

    eig_dir = Path(args["eigenvector_dir"])
    eig_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_dir = eig_dir / "knn"
    eigvec_cache_dir = eig_dir / "eigvecs"

    graphs = KnnGraphCache(cache_dir=graph_cache_dir, verbose=True)
    graph_key_parts = {
        "template": args["template_name"],
        "stride": args["stride"],
        "thresh": args["threshold"],
        "method": args["knn_method"],
        "k": args["knn_k"],
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
    # (level_15annot) keeps its original un-suffixed keys.
    atlas_stem = (Path(args["annotations_file"]).stem
                  if args.get("annotations_file") else "noatlas")
    _legacy_atlas = (atlas_stem == "level_15annot")
    if args["knn_method"] == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4" if _legacy_atlas else atlas_stem
        graph_key_parts["conn"] = 3
    elif args["knn_method"] == "faiss_atlas_weighted":
        # The base graph is pure faiss — we reuse that cache. The atlas
        # weighting is applied AFTER loading, so we only need to encode
        # the inflation factor in the eigenvector key (the kNN graph
        # cache key uses "faiss" so it's shareable with vanilla faiss
        # runs of the same k / stride / threshold).
        pass
    if augment_maldi:
        # Augmented graphs have a different node set (and N) than strided-only
        # ones — key them separately so the KNN/eigenvector caches don't collide.
        graph_key_parts["augmaldi"] = int(reference_nodes.shape[0])
        if args["knn_method"] == "anatomical_atlas":
            # anatomical_atlas builds edges from voxel ADJACENCY over the
            # strided grid, so appended MALDI coords would get no edges at all.
            raise ValueError(
                f"--augment-maldi-nodes is not supported with "
                f"--knn-method=anatomical_atlas (edges come from strided-voxel "
                f"adjacency, so MALDI nodes would be disconnected). Use "
                f"--knn-method=faiss or faiss_atlas_weighted."
            )
    graph_key = make_graph_key(graph_key_parts)

    # Build / load KNN graph
    if args["knn_method"] == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss",
            coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts, device=args["device"],
            force_recompute=bool(args.get("force_recompute_graph", False)),
        )
    elif args["knn_method"] == "anatomical_atlas":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas",
            volume=sub_volume, threshold=args["threshold"],
            atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts, device=args["device"],
            force_recompute=bool(args.get("force_recompute_graph", False)),
        )
    elif args["knn_method"] == "faiss_atlas_weighted":
        # ---- 1. Get the base faiss graph ---------------------------
        # Build (or load from cache) the *exact same* faiss kNN graph
        # that --knn-method=faiss would produce, so caches are shared
        # across the two methods.
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"  # cache under the faiss key
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss",
            coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=base_key_parts, device=args["device"],
            force_recompute=bool(args.get("force_recompute_graph", False)),
        )
        # ---- 2. Look up each node's atlas region -------------------
        # Use the helper in anatomical_knn so other models can do the
        # same conversion (sub_volume + sub_atlas + threshold → labels
        # in node order).
        if sub_atlas is None:
            raise ValueError(
                "--knn-method=faiss_atlas_weighted requires "
                "--annotations-file; got none."
            )
        node_labels = labels_for_nodes_from_sub_atlas(
            sub_volume, sub_atlas, args["threshold"],
        )
        if augment_maldi and reference_nodes.shape[0] > n_atlas_nodes:
            # The strided grid only labels the first n_atlas_nodes; give each
            # appended MALDI node the region of its NEAREST strided atlas node.
            from scipy.spatial import cKDTree
            atlas_xyz = reference_nodes[:n_atlas_nodes].detach().cpu().numpy()
            maldi_xyz = reference_nodes[n_atlas_nodes:].detach().cpu().numpy()
            _, nn = cKDTree(atlas_xyz).query(maldi_xyz, k=1)
            node_labels = np.concatenate([node_labels, node_labels[nn]])
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Mismatch between labelled voxels ({node_labels.shape[0]}) "
                f"and graph nodes ({knn.x.shape[0]}). The atlas and "
                f"reference template were probably cropped/strided "
                f"differently — check that they have the same shape."
            )

        # ---- 3. Root handling + inflate cross-region edges ----------
        # The atlas label-0 'root' is real tissue, not background; dissolving it
        # into the nearest region (default) stops the soft prior from spraying
        # through interiors. Applied AFTER the MALDI-augment relabel above so any
        # appended root nodes are dissolved too. 'cross' = legacy behaviour.
        root_mode = str(args.get("root_handling", "dissolve"))
        if root_mode == "dissolve":
            node_labels = dissolve_root_labels(
                node_labels, reference_nodes.detach().cpu().numpy())
        inflation = float(args.get("cross_region_inflation", 10.0))
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=(root_mode == "cross"),
        )

        # ---- 4. Re-key the eigenvector cache -----------------------
        # Same graph topology but different edge weights → different
        # Laplacian → different eigenmodes. Encode the inflation (+ root mode)
        # in graph_key so the eigvec cache doesn't collide with vanilla faiss
        # runs. Legacy 'cross' stays un-suffixed so old caches still load.
        _base_wt = (f"atlas_x{inflation:g}" if _legacy_atlas
                    else f"{atlas_stem}_x{inflation:g}")
        graph_key_parts["weighting"] = (
            _base_wt if root_mode == "cross" else f"{_base_wt}_root{root_mode}")
        graph_key = make_graph_key(graph_key_parts)
    elif args["knn_method"] == "faiss_cluster_weighted":
        # ---- 1. Base faiss graph (cache shared with --knn-method=faiss) ----
        base_key_parts = dict(graph_key_parts)
        base_key_parts["method"] = "faiss"
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss",
            coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=base_key_parts, device=args["device"],
            force_recompute=bool(args.get("force_recompute_graph", False)),
        )
        # ---- 2. CLUSTER the reference template into node labels ----
        # Data-driven, whole-brain, lipid-free (no annotations-file).
        cluster_k = int(args.get("cluster_k", 64))
        sw = float(args.get("cluster_spatial_weight", 1.0))
        cseed = int(args.get("cluster_seed", 0))
        node_labels = labels_for_nodes_from_template_clustering(
            sub_volume, args["threshold"], n_clusters=cluster_k,
            spatial_weight=sw,
            fit_subsample=int(args.get("cluster_fit_subsample", 40000)),
            seed=cseed,
        )
        if augment_maldi and reference_nodes.shape[0] > n_atlas_nodes:
            # give each appended MALDI node the cluster of its nearest strided node
            from scipy.spatial import cKDTree
            strided_xyz = reference_nodes[:n_atlas_nodes].detach().cpu().numpy()
            maldi_xyz = reference_nodes[n_atlas_nodes:].detach().cpu().numpy()
            _, nn = cKDTree(strided_xyz).query(maldi_xyz, k=1)
            node_labels = np.concatenate([node_labels, node_labels[nn]])
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Mismatch between clustered voxels ({node_labels.shape[0]}) "
                f"and graph nodes ({knn.x.shape[0]}). Check the reference "
                f"template stride/threshold."
            )
        # ---- 3. Inflate cross-cluster edges. treat_zero_as_cross=False:
        #         cluster id 0 is a real region, not background. ----
        inflation = float(args.get("cross_region_inflation", 10.0))
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=False,
        )
        graph_key_parts["weighting"] = (
            f"tmplclust_k{cluster_k}_sw{sw:g}_s{cseed}_x{inflation:g}"
        )
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"unknown knn_method: {args['knn_method']}")

    # ---- Label denoise + hard prune (weighted methods, after inflation) ----
    # Refine the HARD topology on the region labels the graph was weighted on.
    # Prune changes edges → the eigvecs change, so prune (+ the denoise that
    # shaped it) go into the eigvec cache key; denoise alone leaves edges intact.
    if args["knn_method"] in ("faiss_atlas_weighted", "faiss_cluster_weighted"):
        n_denoise = int(args.get("denoise_labels", 0) or 0)
        prune = float(args.get("prune_cross_region", 0.0) or 0.0)
        if n_denoise > 0:
            node_labels = denoise_labels_majority_vote(
                node_labels, edge_index, n_denoise)
        if prune > 0.0:
            edge_index, edge_value = prune_cross_region_edges(
                edge_index, edge_value, node_labels, prune,
                zero_is_region=(args["knn_method"] == "faiss_cluster_weighted"))
            graph_key_parts["prune"] = f"{prune:g}"
            if n_denoise > 0:
                graph_key_parts["denoise"] = n_denoise
            graph_key = make_graph_key(graph_key_parts)

    # Eigenpairs
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0],
        torch.tensor(args["graphbandwidth_init"], device=args["device"]),
        args["laplacian_norm"],
    )
    eigvec_key_parts = {
        "graph": graph_key,
        "norm": args["laplacian_norm"],
        "bw": args["graphbandwidth_init"],
        "modes": args["num_modes"],
    }
    eigvec_key = make_eig_key(eigvec_key_parts)
    ncv_min = resolve_ncv_min(args["num_modes"], args.get("ncv_min", -1))
    solver = LaplacianEigensolver(
        num_modes=args["num_modes"], backend="cupy",
        tol=1e-4, ncv_min=ncv_min, verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op, cache_dir=eigvec_cache_dir, key=eigvec_key,
        graphbandwidth=args["graphbandwidth_init"],
        laplacian_normalization=args["laplacian_norm"],
        extra=eigvec_key_parts, device=args["device"],
        # Reuse a cache with more modes if one exists (eigenpairs are nested;
        # the extra modes are simply trimmed off on load).
        allow_larger_modes=True,
    )

    # --- optional surface (distance-to-surface) depth factor ---
    # d_surface = EDT depth per strided node; SurfaceRiemannMaternKernel samples it
    # from the coords INTERNALLY (nearest node), so inducing snap / learn-inducing /
    # predict all stay unchanged (coords remain 3-D). DepthSampler z-scores depth,
    # so the mm_per_voxel scale is irrelevant. Built on the strided grid nodes
    # (reference_nodes[:n_atlas_nodes]) so it's independent of any MALDI augmentation.
    surface_sampler = None
    if args.get("surface_kernel", False):
        from manifold_gp.utils.surface_depth import node_surface_depth, DepthSampler
        node_vox = np.argwhere(sub_volume > args["threshold"])
        node_depth = node_surface_depth(
            sub_volume, args["threshold"], node_vox, mm_per_voxel=float(voxel_scale_mm))
        surface_sampler = DepthSampler(
            reference_nodes[:n_atlas_nodes].detach().cpu().numpy(), node_depth)
        log.info(f"surface-kernel: d_surface for {node_vox.shape[0]:,} nodes "
                 f"(mean={surface_sampler.mean:.3f} std={surface_sampler.std:.3f}); "
                 f"depth_lengthscale_init={args.get('surface_depth_lengthscale', 1.0)}")

    # Kernel: SurfaceRiemannMaternKernel when --surface-kernel, else RiemannMaternKernel.
    _KernelCls = SurfaceRiemannMaternKernel if surface_sampler is not None else RiemannMaternKernel
    _surf_kw = (dict(depth_sampler=surface_sampler,
                     depth_lengthscale=float(args.get("surface_depth_lengthscale", 1.0)))
                if surface_sampler is not None else {})
    manifold_kernel = _KernelCls(
        nu=args["nu"], knn=knn, edge_index=edge_index, edge_value=edge_value,
        eigval=eigval, eigvec=eigvec,
        nearest_neighbors=args["knn_k"], num_modes=args["num_modes"],
        bump_scale=args["bump_scale"], bump_decay=args["bump_decay"],
        laplacian_normalization=args["laplacian_norm"],
        normalize_features=args.get("normalize_features", False),
        graphbandwidth_init=args["graphbandwidth_init"],
        diffusion_scale_init=args.get("diffusion_scale_init", 1.0),
        learn_diffusion_scale=args.get("learn_diffusion_scale", False),
        learn_spectral_weights=args.get("learn_spectral_weights", False),
        **_surf_kw,
    ).to(args["device"])
    manifold_kernel.eval()

    # CRITICAL: snap inducing points to nearest graph nodes.
    # The RiemannKernel.features() method has two branches:
    #   - is_on_graph (dist² < 1e-8): exact eigenvector lookup — always stable
    #   - out-of-sample: Nyström extension with a different spectral density
    #     formula that can diverge for small graphbandwidth + large eigvals.
    # k-means inducing points from MaLDI coordinates are almost never exact
    # graph nodes, so without snapping every K_uu evaluation goes through the
    # OOS path, producing near-singular K_uu → NotPSDError / NaN loss from
    # iter 0. lgp_manifold_experiment.setup_experiment() always snaps;
    # this function must do the same.
    log.info("Snapping inducing points to nearest graph nodes …")
    return {
        "kernel": manifold_kernel,
        "knn": knn,                      # needed by caller for snapping
        "reference_nodes": reference_nodes,
        "sub_volume": sub_volume,
        "voxel_offset": voxel_offset,
        "voxel_scale_mm": voxel_scale_mm,
        "template_shape": template_volume.shape,
        # Node bookkeeping for MALDI-node inducing selection. When augment is
        # on, nodes [n_atlas_nodes:] are the appended measured MALDI voxels.
        "augment_maldi": augment_maldi,
        "n_atlas_nodes": n_atlas_nodes,
    }


# =============================================================================
# Per-lipid-batch trainer (the actual GP fit)
# =============================================================================
def _record_error(config, kind, it, detail, streak, log):
    """Append an error event to ERRORS.txt and throttle console output.

    Per-iter logging would flood the cluster log when failures happen
    in bursts (e.g. a thousand "grad_zero" warnings in a row). We
    instead:
      - ALWAYS append every event to ``ERRORS.txt`` (one line per event,
        with timestamp + iter + streak + detail). The file is the
        complete record for forensic analysis after the run.
      - ECHO to console only periodically: the first 3 of any streak,
        then one per 100. Enough to see "something is going wrong" in
        real-time without obscuring the actual training progress.

    The ERRORS.txt path is derived from args["output_dir"]/exp_name —
    the same out_root used for everything else. Failures during the
    append (e.g. S3 hiccup) are non-fatal; we silently keep training.
    """
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}  it={it:>6d}  "
        f"kind={kind:<10s}  streak={streak:>4d}  {detail}\n"
    )
    out_root = config.exp_path
    errors_path = out_root / "ERRORS.txt"
    try:
        with open(errors_path, "a") as f:
            f.write(line)
    except OSError:
        pass  # don't kill training over a log-write failure
    if streak <= 3 or streak % 100 == 0:
        log.warning(f"[it {it}] {kind} streak={streak}: {detail}")


def _maybe_bail_on_nan_streak(streak, it, args, skip_frac, log, pbar):
    """Raise RuntimeError if NaN gradients have persisted too long.

    Called from both branches of the NaN handler (skip-step and
    zero-and-step) so a sustained streak triggers a bail-out regardless
    of which branch is firing. The threshold is the same in both cases
    because the underlying signal — "kernel keeps producing NaN
    gradients" — is the same; whether we skip or zero only affects per-
    step recovery, not whether we should keep trying long-term.

    The streak is reset to 0 on the first clean step (no NaN at all),
    so a single recovery clears the alarm.
    """
    cap = int(args.get("max_consecutive_bad_grads", 1000))
    if streak < cap:
        return
    pbar.close()
    raise RuntimeError(
        f"{streak} consecutive steps with NaN/Inf gradients at "
        f"iter {it}. The model has diverged or is stuck in a "
        f"numerically singular region. Common fixes:\n"
        f"  • smaller --learning-rate (currently "
        f"{args['learning_rate']})\n"
        f"  • smaller --grad-clip (currently "
        f"{args.get('grad_clip', 10.0)}; try 1.0)\n"
        f"  • smaller --bump-scale on the manifold kernel\n"
        f"  • larger gpytorch cholesky_jitter\n"
        f"  • change --knn-method (try faiss_atlas_weighted with "
        f"smaller --cross-region-inflation)\n"
        f"  • raise --max-consecutive-bad-grads if you want the "
        f"alarm later (currently {cap})"
    )


# Bounds on the log observation-noise variance for z-scored outputs (unit
# variance by construction). These are the SAME bounds the old manual
# log_var_n clamp used: -5.0 → σ²≈0.0067 (tight but reachable), +1.5 → σ²≈4.5
# (genuinely noisy lipids). With the MultitaskGaussianLikelihood they become a
# hard Interval constraint on task_noises rather than a post-step clamp, so the
# noise can't leave the safe range regardless of the optimizer or prior.
NOISE_LOG_MIN, NOISE_LOG_MAX = -5.0, 1.5


def _build_task_likelihood(n_tasks: int, device: str):
    """Per-task Gaussian noise model — the GaussianLikelihood analogue of the
    old ``log_var_n`` (shape (n_tasks,), independent per task, no cross-task
    coupling and no shared global noise). ``task_noises`` is the noise VARIANCE
    per task; initialised to 1.0 to match the old ``log_var_n = 0`` init. The
    Interval constraint reproduces the old clamp range exactly."""
    constraint = gpytorch.constraints.Interval(
        math.exp(NOISE_LOG_MIN), math.exp(NOISE_LOG_MAX)
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=n_tasks, rank=0,
        has_global_noise=False, has_task_noise=True,
        noise_constraint=constraint,
    ).to(device)
    with torch.no_grad():
        likelihood.task_noises = torch.ones(n_tasks, device=device)
    return likelihood


def _register_hyperpriors(model, likelihood, args, log):
    """Register weakly-informative priors on the GP hyperparameters so the
    VariationalELBO does MAP-II. The mll picks these up via ``named_priors()``
    on its (model, likelihood) submodules and adds their log-probs at the
    correct ``1/num_data`` scale. Priors are placed only on parameters whose
    natural scale is known a priori for z-scored data:

      • outputscale (signal variance) → ~1   : Gamma(2, 2), mean 1
      • observation noise (task_noises) → ~1  : SmoothedBox in [e^-3, e^0.5],
        a soft version of the noise bounds that discourages collapse/explosion
        without the hard wall doing all the work.

    The lengthscale prior is opt-in (``--lengthscale-prior-mean``) because its
    scale is data/constraint-dependent."""
    from gpytorch.priors import GammaPrior, SmoothedBoxPrior
    registered = []

    covar = getattr(model, "covar_module", None)
    if isinstance(covar, gpytorch.kernels.ScaleKernel):
        covar.register_prior("outputscale_prior", GammaPrior(2.0, 2.0),
                             "outputscale")
        registered.append("outputscale~Gamma(2,2)")
        ls_mean = args.get("lengthscale_prior_mean", None)
        base = getattr(covar, "base_kernel", None)
        if ls_mean is not None and base is not None and hasattr(base, "raw_lengthscale"):
            conc = 3.0
            base.register_prior("lengthscale_prior",
                                GammaPrior(conc, conc / float(ls_mean)),
                                "lengthscale")
            registered.append(f"lengthscale~Gamma(mean={ls_mean})")

    likelihood.register_prior(
        "task_noise_prior",
        SmoothedBoxPrior(math.exp(-3.0), math.exp(0.5), sigma=0.1),
        "task_noises",
    )
    registered.append("task_noises~SmoothedBox[e^-3,e^0.5]")

    device = next(likelihood.parameters()).device
    model.to(device)
    likelihood.to(device)

    log.info("  hyperpriors registered: " + "; ".join(registered))


def _restore_noise(likelihood, ckpt, device):
    """Load the per-task noise from a checkpoint into ``likelihood``. Handles
    both the new format (``likelihood_state``) and pre-VariationalELBO
    checkpoints that stored a bare ``log_var_n`` tensor (noise = exp(log_var_n),
    clamped into the constraint's range)."""
    if "likelihood_state" in ckpt:
        likelihood.load_state_dict(ckpt["likelihood_state"])
    elif "log_var_n" in ckpt:
        with torch.no_grad():
            likelihood.task_noises = torch.exp(
                ckpt["log_var_n"].to(device)
            ).clamp(math.exp(NOISE_LOG_MIN), math.exp(NOISE_LOG_MAX))


def _grad_norms_by_group(model, likelihood):
    """L2 gradient norm, total and per parameter-group, for W&B logging.

    Buckets parameters by name so each W&B line isolates a distinct part of
    the model:
      * inducing    — inducing-point locations (only nonzero when
                      learn_inducing_locations=True; zero/absent otherwise)
      * variational — variational distribution (mean + Cholesky factor)
      * kernel      — covar_module (lengthscale / outputscale / graphbandwidth)
      * mean        — mean_module
      * likelihood  — observation-noise params
    'grad_norm/total' is the global L2 norm (== what clip_grad_norm_ measures).
    Returns a flat {key: float} dict ready to hand to ``wandb.log``.
    """
    groups = {"inducing": 0.0, "variational": 0.0, "kernel": 0.0,
              "mean": 0.0, "likelihood": 0.0, "other": 0.0}

    def bucket(name):
        if "inducing_points" in name:        # check first: name also has "variational"
            return "inducing"
        if "variational" in name:            # variational_mean / chol_variational_covar
            return "variational"
        if "mean_module" in name:
            return "mean"
        if ("covar_module" in name or "raw_lengthscale" in name
                or "raw_graphbandwidth" in name):
            return "kernel"
        return "other"

    total_sq = 0.0
    named = (list(model.named_parameters())
             + [(f"likelihood.{n}", p) for n, p in likelihood.named_parameters()])
    for name, p in named:
        if p.grad is None:
            continue
        g2 = float(p.grad.detach().pow(2).sum().item())
        total_sq += g2
        grp = "likelihood" if name.startswith("likelihood.") else bucket(name)
        groups[grp] += g2
    out = {f"grad_norm/{k}": math.sqrt(v) for k, v in groups.items()}
    out["grad_norm/total"] = math.sqrt(total_sq)
    return out


def _kernel_hypers_for_log(model):
    """Best-effort scalar kernel hyperparameters for W&B (means across tasks /
    per-task kernels). Wrapped in try/except so a kernel layout we don't expect
    never breaks training — logging is non-essential."""
    out = {}
    try:
        oscale = [float(m.outputscale.detach().mean().item())
                  for m in model.modules()
                  if isinstance(m, gpytorch.kernels.ScaleKernel)]
        if oscale:
            out["kernel/outputscale_mean"] = float(np.mean(oscale))
        ls, gb = [], []
        for m in model.modules():
            if getattr(m, "lengthscale", None) is not None:
                try:
                    ls.append(float(m.lengthscale.detach().mean().item()))
                except Exception:
                    pass
            if getattr(m, "graphbandwidth", None) is not None:
                try:
                    gb.append(float(torch.as_tensor(m.graphbandwidth).detach()
                                    .mean().item()))
                except Exception:
                    pass
        if ls:
            out["kernel/lengthscale_mean"] = float(np.mean(ls))
        if gb:
            out["kernel/graphbandwidth_mean"] = float(np.mean(gb))
        # Parcel factor: offdiag_mean is the mean covariance multiplier between
        # DIFFERENT parcels. It starts at ~1 (no-op) and falls as the model learns
        # to stop smoothing across borders — the single number that says whether
        # the factor is doing anything at all.
        for m in model.modules():
            if hasattr(m, "factor_stats"):
                out.update(m.factor_stats())
    except Exception:
        pass
    return out


def train_lipid_batch(
    *,
    coords_train: torch.Tensor,   # (N, 3) on device, already z-scored
    y_train: torch.Tensor,        # (N, B)   B = lipid batch size
    inducing_points: torch.Tensor,
    config: MaldiConfig,
    args: dict,
    manifold_kernel=None,         # if None → Euclidean path
    parcel_field=None,            # optional parcelgp.ParcelField
    device: str,
    log: logging.Logger,
    pbar_desc: str,
):
    """Build one multitask model with num_tasks=B, run variational SGD,
    return the trained model + per-task Gaussian likelihood.

    The objective is the sparse-variational ELBO computed by
    ``gpytorch.mlls.VariationalELBO`` against a per-task
    ``MultitaskGaussianLikelihood``. The likelihood's ``task_noises`` (one
    noise variance per task) is the GaussianLikelihood analogue of LGP's
    ``log_var_n`` — see :func:`_build_task_likelihood`.

    Loss scale.  ``VariationalELBO`` returns the *per-datum* ELBO
    (full-data ELBO / num_data). We multiply by ``n_train`` so the loss is
    on the SAME full-data scale as the old inlined ``(n_train/bs) * recon +
    kl`` formulation — recon coefficient ``n_train/bs`` and KL coefficient
    ``1`` are reproduced exactly (verified: the two losses differ only by the
    gradient-free constant ``0.5·log(2π)·n_tasks·n_train``). Keeping the
    scale identical means the tuned ``--grad-clip``, learning rate, and
    bail-out thresholds remain valid.

    Why VariationalELBO instead of the previous inlined recon+KL?  It does
    the same expected-log-likelihood math (``MultitaskGaussianLikelihood.
    expected_log_prob`` is the analytic ``0.5·sum[((y-mean)²+var)/σ² +
    logσ²]`` we used to write by hand), but additionally folds in
    hyperparameter prior log-probs at the correct ``1/num_data`` scale (see
    :func:`_register_hyperpriors`), turning ML-II into MAP-II. That matches
    the regularization the exact ``manifold_informed_train`` applies, making
    exact-vs-variational comparisons apples-to-apples.
    """
    n_tasks = y_train.shape[1]
    n_train = y_train.shape[0]

    wandb_on = bool(args.get("wandb", False))
    learn_inducing = bool(args.get("learn_inducing", False))
    if learn_inducing and manifold_kernel is not None:
        log.info(
            "  --learn-inducing (manifold): inducing points use the "
            "differentiable Nyström feature map (features_differentiable), so "
            "their coordinates receive gradients and move ALONG the manifold "
            "within bump support. Watch grad_norm/inducing in W&B — it is "
            "structurally zero with the default exact-lookup feature map."
        )

    if manifold_kernel is None:
        # Euclidean — same as run_final.sh's setup. For the eigenmap family the
        # inputs are the r-dim Laplacian-eigenfunction embedding (args["_embed_dim"]),
        # so the GP runs over r features with per-mode ARD and no physical
        # length-scale floor (the voxel-size floor is meaningless in eigenspace).
        embed_dim = int(args.get("_embed_dim", 0))
        if embed_dim:
            input_dim = embed_dim
            minimal_length_scale = 0.0
            ard_num_dims = None if args.get("no_ard", False) else embed_dim
        else:
            voxel_size = 0.025
            # Equivalent to lgp_experiment.minimal_length_scale
            minimal_length_scale = config.n_pixels * voxel_size / 3.0
            input_dim = 3
            # ard_num_dims: None → isotropic (--no-ard); 3 → per-axis ARD (default).
            ard_num_dims = None if args.get("no_ard", False) else 3
        model = IndependentMultitaskGPModel(
            inducing_points=inducing_points,
            num_tasks=n_tasks,
            kernel_type=config.kernel,
            nu=config.nu,
            minimal_length_scale=minimal_length_scale,
            input_dim=input_dim,
            ard_num_dims=ard_num_dims,
            learn_inducing_locations=learn_inducing,
        ).to(device)
    else:
        if args.get("per_task_lengthscale", False):
            # One RiemannMaternKernel per task, *sharing* the eigenpairs/graph
            # tensors (passed by reference) but with an independent lengthscale.
            mk = manifold_kernel
            # Match the base kernel's class; carry the surface depth factor (shared
            # sampler, per-task learnable depth_lengthscale) when it's a surface kernel.
            _KernelCls = type(mk)
            _surf_kw = (dict(depth_sampler=mk._depth_sampler,
                             depth_lengthscale=float(mk.depth_lengthscale))
                        if isinstance(mk, SurfaceRiemannMaternKernel) else {})
            manifold_arg = [
                _KernelCls(
                    nu=mk.nu, knn=mk.knn, edge_index=mk.edge_index,
                    edge_value=mk.edge_value, eigval=mk.eigval, eigvec=mk.eigvec,
                    nearest_neighbors=mk.nearest_neighbors, num_modes=mk.num_modes,
                    bump_scale=mk.bump_scale, bump_decay=mk.bump_decay,
                    laplacian_normalization=mk.laplacian_normalization,
                    normalize_features=mk.normalize_features,
                    graphbandwidth_init=float(mk.graphbandwidth),
                    diffusion_scale_init=float(mk.diffusion_scale),
                    learn_diffusion_scale=args.get("learn_diffusion_scale", False),
                    learn_spectral_weights=args.get("learn_spectral_weights", False),
                    **_surf_kw,
                ).to(device)
                for _ in range(n_tasks)
            ]
            log.info(f"  per-task lengthscale: {n_tasks} kernels (shared eigenpairs)")
        else:
            manifold_arg = manifold_kernel
        model = LatentRiemannGP(
            inducing_points=inducing_points,
            num_tasks=n_tasks,
            manifold_kernel=manifold_arg,
            product_ard_matern=args.get("product_ard_matern", False),
            product_ard_nu=float(args.get("product_ard_nu", 2.5)),
            learn_inducing_locations=learn_inducing,
        ).to(device)

    # Optional lengthscale initialisation (manifold path).
    ls_init = args.get("lengthscale_init", None)
    if manifold_kernel is not None and ls_init is not None:
        for m in model.modules():
            if isinstance(m, RiemannMaternKernel):
                m.lengthscale = torch.tensor(float(ls_init))
        log.info(f"  lengthscale init = {ls_init}")

    # Parcel factor. Wrapped INSIDE the ScaleKernel (see wrap_scale_kernel) so
    # covar_module stays a ScaleKernel and _register_hyperpriors below still
    # finds outputscale/lengthscale where it expects them. Applied before the
    # optimizer is built so B is picked up as a trainable parameter.
    if parcel_field is not None:
        covar = getattr(model, "covar_module", None)
        if not isinstance(covar, gpytorch.kernels.ScaleKernel):
            raise ValueError(
                f"--parcel-field needs a ScaleKernel covar_module to wrap, got "
                f"{type(covar).__name__}. The euclidean and manifold families "
                f"both provide one; check --kernel-family.")
        wrap_scale_kernel(
            covar, parcel_field, num_tasks=n_tasks,
            rank=int(args.get("parcel_rank", 8)),
            init_scale=float(args.get("parcel_init_scale", 0.05)),
            per_task=bool(args.get("parcel_per_task", True)),
        )
        model.to(device)

    # Per-task observation noise model — the GaussianLikelihood analogue of
    # LGP's log_var_n (task_noises = noise variance per task, init 1.0).
    likelihood = _build_task_likelihood(n_tasks, device)
    if args.get("hyperpriors", True):
        _register_hyperpriors(model, likelihood, args, log)

    # VariationalELBO over the full dataset. num_data=n_train makes the KL and
    # prior terms full-data-scaled; we multiply the returned per-datum ELBO by
    # n_train in the loop to recover the old full-data loss scale exactly.
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train)

    model.train()
    likelihood.train()
    if args.get("lengthscale_no_decay", False):
        # keep raw_lengthscale / raw_graphbandwidth out of weight_decay so AdamW
        # doesn't pull the (log-)scale toward its init independent of the data.
        no_decay, decay = [], []
        for pname, p_ in model.named_parameters():
            (no_decay if ("raw_lengthscale" in pname or "raw_graphbandwidth" in pname
                          or "raw_diffusion_scale" in pname)
             else decay).append(p_)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-3},
             {"params": no_decay, "weight_decay": 0.0},
             {"params": list(likelihood.parameters()), "weight_decay": 0.0}],
            lr=config.learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=config.learning_rate, weight_decay=1e-3,
        )

    bs = min(int(config.batch_size), n_train)
    # iters_per_epoch matches DataLoader(drop_last=True) — partial final
    # minibatch is skipped so each epoch has a stable iter count.
    iters_per_epoch = max(1, n_train // bs)
    n_iters = int(config.epochs) * iters_per_epoch

    # Log frequency for cluster runs — write a line every ~5% of training
    # so cluster job logs show progress even when the tqdm bar is hidden.
    log_every = max(1, n_iters // 20)

    # W&B global step: each lipid batch occupies a contiguous [b*n_iters,
    # (b+1)*n_iters) block, so steps stay monotonic across the whole job
    # (n_iters is identical for every batch — same n_train / bs / epochs).
    wandb_step0 = int(args.get("wandb_batch_index", 0)) * n_iters

    # ---- Resume from in-progress checkpoint if one exists ---------------
    # ckpt_path is set by the caller (per-batch). If a previous crash
    # left a checkpoint file, restore model + log_var_n + iter counter
    # so training picks up where it left off rather than restarting
    # from scratch.
    ckpt_path = args.get("checkpoint_path", None)
    iter_start = 0
    if ckpt_path is not None and Path(ckpt_path).exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if ckpt.get("n_tasks") == n_tasks:
                model.load_state_dict(ckpt["model_state"])
                _restore_noise(likelihood, ckpt, device)
                iter_start = int(ckpt.get("iter", 0))
                log.info(
                    f"  [{pbar_desc}] resumed from "
                    f"{Path(ckpt_path).name} @ iter {iter_start}/"
                    f"{n_iters} (epoch {ckpt.get('epoch', 0)})"
                )
            else:
                log.warning(
                    f"  [{pbar_desc}] checkpoint n_tasks mismatch "
                    f"({ckpt.get('n_tasks')} vs {n_tasks}); ignoring"
                )
        except Exception as ex:
            log.warning(
                f"  [{pbar_desc}] could not load checkpoint "
                f"{ckpt_path}: {ex}; starting from scratch"
            )

    log.info(
        f"  [{pbar_desc}] starting fit: "
        f"n_train={n_train:,} bs={bs} epochs={args['epochs']} "
        f"iters_per_epoch={iters_per_epoch} total_iters={n_iters} "
        f"n_tasks={n_tasks}; logging every {log_every} iters; "
        f"iter_start={iter_start}."
    )

    # ---- DataLoader: epoch-shuffled minibatches with async prefetching ---
    # Matches the framework's approach in ManifoldLGP.train_model — TensorDataset
    # + DataLoader(shuffle=True). num_workers>0 gives prefetching: while the GPU
    # is busy with iter N, worker processes prepare iter N+1's minibatch. The
    # CPU-side tensors are pinned (pin_memory) so the H→D copy is async.
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(coords_train, y_train),
        batch_size=bs,
        shuffle=True,
        num_workers=int(args.get("dataloader_workers", 2)),
        pin_memory=device.startswith("cuda"),
        persistent_workers=int(args.get("dataloader_workers", 2)) > 0,
        drop_last=True,  # consistent iters_per_epoch
    )

    pbar = tqdm(total=n_iters, initial=iter_start,
                desc=pbar_desc, leave=False, dynamic_ncols=True)
    last_recon = float("nan")
    last_kl = float("nan")
    # Observation-noise bounds are enforced by the likelihood's Interval
    # constraint (see _build_task_likelihood / NOISE_LOG_MIN, NOISE_LOG_MAX),
    # so there is no manual log_var_n clamp here anymore.

    fit_t0 = time.time()
    epoch_t0 = fit_t0
    current_epoch = iter_start // iters_per_epoch
    # Counter for consecutive non-finite gradients. Resets on every
    # successful step. We use it to throttle the warning log (otherwise
    # one bad iter fills 100+ lines) and to bail out after sustained
    # divergence rather than spinning forever.
    bad_grad_count = 0
    # CUDA generator for fast on-device random index sampling. Using
    # torch.randperm(n_train) on CPU every iter wastes 50-100ms per
    # step on a 5M-point training set (allocates an int64 tensor of
    # length n_train just to slice the first `bs`). torch.randint on
    # GPU draws only `bs` integers and stays on-device — ~3 orders of
    # magnitude faster. Sampling WITH replacement (vs randperm's
    # without-replacement) is fine for SVI: at bs=4096 and n=5M, the
    # probability of any collision within a minibatch is ~bs²/(2n) =
    # 0.17%, statistically indistinguishable from sampling without
    # replacement at the gradient level.
    # ---- Periodic checkpointing ----------------------------------------
    # Cloud jobs that auto-restart on OOM lose all in-progress training
    # if we only save at end-of-batch. Save the model state every
    # ``checkpoint_every_epochs`` epochs, overwriting a single file per
    # batch so we never bloat the S3 output dir.
    ckpt_every = int(args.get("checkpoint_every_epochs", 5))
    ckpt_path = args.get("checkpoint_path", None)  # caller passes per-batch path

    def _save_ckpt(iter_idx):
        """Save the latest in-progress model state to ckpt_path. The
        whole save is a torch.save of a dict including model state,
        likelihood state (per-task noise), current iter, epoch, and last
        losses. Atomic via write-to-temp-then-rename so a SIGKILL mid-save
        doesn't leave a corrupt checkpoint."""
        if ckpt_path is None:
            return
        try:
            tmp = Path(str(ckpt_path) + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "likelihood_state": likelihood.state_dict(),
                "iter": iter_idx,
                "epoch": current_epoch,
                "recon": last_recon,
                "kl": last_kl,
                "n_tasks": n_tasks,
            }, tmp)
            os.replace(tmp, ckpt_path)  # atomic
            log.info(
                f"  [{pbar_desc}] ckpt @ epoch {current_epoch} "
                f"(it {iter_idx}) → {Path(ckpt_path).name}"
            )
        except Exception as ex:
            # Checkpoint failures shouldn't kill training. Log and continue.
            log.warning(f"  [{pbar_desc}] ckpt save failed: {ex}")
    with gpytorch.settings.cholesky_jitter(1e-3, 1e-4):
        # ``it`` is the global iter counter (so the periodic-log + ckpt
        # logic in the rest of the loop body is unchanged). It starts at
        # iter_start when resuming from a checkpoint.
        it = iter_start
        # Figure out which epoch we're resuming in (if any), and skip
        # earlier epochs cleanly. Within the resumed epoch we restart
        # from its first minibatch — that's a minor inaccuracy vs the
        # exact crash point, but acceptable for a 5-epoch save cadence.
        start_epoch = iter_start // iters_per_epoch
        try:
            for epoch_idx in range(start_epoch, int(args["epochs"])):
                for x_b_cpu, y_b_cpu in train_loader:
                    if it >= n_iters:
                        break
                    # Async H→D copy (works because the loader pinned the
                    # tensors). The GPU op queue serialises this against the
                    # subsequent model(x_b) so we don't need explicit sync.
                    x_b = x_b_cpu.to(device, non_blocking=True)
                    y_b = y_b_cpu.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    # ---- forward + post-hoc NaN check -------------------------
                    # Two failure modes during the GP forward:
                    #   (a) gpytorch raises NotPSDError when K_uu Cholesky
                    #       gives up after the jitter ladder is exhausted.
                    #   (b) Cholesky numerically "succeeds" but produces NaN
                    #       in the output because the matrix was so
                    #       ill-conditioned that floating-point ops broke.
                    # Both are handled identically: skip the iter, log to
                    # ERRORS.txt, count toward the bail-out streak. No
                    # in-iter recovery (extra-jitter retry) — keeps the
                    # logic and gradient semantics simple.
                    fwd_failure = None
                    try:
                        gp_posterior = model(x_b)
                        if (not torch.isfinite(gp_posterior.mean).all() or
                                not torch.isfinite(gp_posterior.variance).all()):
                            fwd_failure = "nan_posterior"
                    except gpytorch.utils.errors.NotPSDError as ex:
                        fwd_failure = f"NotPSDError"

                    if fwd_failure is not None:
                        bad_grad_count += 1
                        _record_error(config, "forward", it, fwd_failure,
                                      bad_grad_count, log)
                        optimizer.zero_grad()
                        it += 1
                        pbar.update(1)
                        _maybe_bail_on_nan_streak(
                            bad_grad_count, it, args, skip_frac, log, pbar
                        )
                        if it >= n_iters:
                            break
                        continue
                        # NOTE: forward failures always increment bad_grad_count
                        # and never reach the else-branch reset below. That is
                        # intentional — a forward failure IS a genuine problem
                        # (unlike a single NaN gradient entry), and sustained
                        # forward failures should trigger the bail-out.

                    # ---- Variational ELBO loss -------------------------------
                    # gp_posterior is the latent MultitaskMVN q(f|x), shape
                    # (bs, n_tasks). VariationalELBO computes the analytic
                    # expected log-likelihood (MultitaskGaussianLikelihood.
                    # expected_log_prob = the SAME closed form
                    #   0.5·sum[((y-mean)² + var)/σ² + logσ²]
                    # we used to inline — exact in the expectation, no MC
                    # sampling), minus the KL, minus hyperparameter prior
                    # log-probs, all on a PER-DATUM scale (full ELBO / num_data).
                    #
                    # Multiplying by n_train recovers the old full-data loss
                    # scale EXACTLY: recon coefficient n_train/bs and KL
                    # coefficient 1 (the two losses differ only by the
                    # gradient-free constant 0.5·log(2π)·n_tasks·n_train). Same
                    # scale ⇒ the tuned --grad-clip / lr / bail thresholds stay
                    # valid. The prior log-probs are folded in by the mll at the
                    # correct 1/num_data scale, so this is MAP-II.
                    loss = -n_train * mll(gp_posterior, y_b)
                    # KL alone (cheap) for the progress logs; the data-fit +
                    # prior remainder is reported as `recon` so the existing
                    # log/postfix lines stay meaningful.
                    kl = model.variational_strategy.kl_divergence().sum()
                    recon = loss.detach() - kl.detach()

                    # ---- Loss-NaN: treat same as any forward failure ----
                    # The forward + posterior-NaN check upstream already
                    # catches most kernel ill-conditioning. A NaN here is
                    # rarer — usually noise drift or KL overflow. Route
                    # it through the same bad-grad streak counter so it
                    # contributes to the bail-out logic uniformly.
                    if not torch.isfinite(loss):
                        bad_grad_count += 1
                        lvn = likelihood.task_noises.detach().log()
                        _record_error(
                            config, "loss_nan", it,
                            f"loss={loss.item()}, "
                            f"recon={recon.item():.3g}, "
                            f"kl={kl.item():.3g}, "
                            f"lvn=[{float(lvn.min().item()):.2f},"
                            f"{float(lvn.max().item()):.2f}]",
                            bad_grad_count, log,
                        )
                        optimizer.zero_grad()
                        it += 1
                        pbar.update(1)
                        _maybe_bail_on_nan_streak(
                            bad_grad_count, it, args, skip_frac, log, pbar
                        )
                        if it >= n_iters:
                            break
                        continue
                        # NOTE: same as forward failures — NaN loss is always
                        # a real problem, always increments, never resets.

                    # Capture loss components NOW (before backward) so the
                    # diagnostic shown in NaN warnings reflects the CURRENT
                    # iter, not the last clean-grad iter. Without this,
                    # warnings keep showing stale `recon=6.64e+04` from
                    # iter 1 even after recon has dropped 20×, making the
                    # logs look like nothing is happening.
                    last_recon = float(recon.detach().item())
                    last_kl = float(kl.detach().item())

                    loss.backward()

                    # ---- NaN-to-zero gradient sanitization with skip safety -
                    # Backward through the manifold kernel's Cholesky can
                    # produce NaN/Inf in a SUBSET of parameter gradients —
                    # typically variational covariance entries near a
                    # singular eigenmode of K_uu. We have two modes:
                    #
                    #   (a) Small fraction bad (<bad_grad_skip_frac): zero
                    #       the offending entries and step normally. The
                    #       remaining finite gradient direction is still
                    #       informative, just missing a few components.
                    #
                    #   (b) Large fraction bad (>=bad_grad_skip_frac): skip
                    #       the step entirely. Stepping with mostly-zeroed
                    #       gradients means moving in an essentially random
                    #       direction, which can push the model further into
                    #       the singular regime.
                    #
                    # Default threshold 1% — small enough that occasional
                    # numerical hiccups don't block training, big enough
                    # that a genuinely diverged step gets caught.
                    n_bad_grad = 0
                    n_total_grad = 0
                    # Collect (param_name, n_bad) so error messages identify
                    # which parameter had the bad gradient entries.
                    bad_params: list[tuple[str, int]] = []
                    for pname, p in (
                        list(model.named_parameters())
                        + [(f"likelihood.{n}", p_) for n, p_ in likelihood.named_parameters()]
                    ):
                        if p.grad is None:
                            continue
                        n_total_grad += p.grad.numel()
                        bad = ~torch.isfinite(p.grad)
                        if bad.any():
                            nb = int(bad.sum().item())
                            n_bad_grad += nb
                            bad_params.append((pname, nb))

                    bad_frac = n_bad_grad / max(n_total_grad, 1)
                    skip_frac = float(args.get("bad_grad_skip_frac", 0.01))
                    # e.g. "variational_strategy.chol_variational_covar:1"
                    bad_params_str = ", ".join(f"{n}:{k}" for n, k in bad_params)

                    # Minimum number of bad gradient entries before we treat
                    # the step as genuinely problematic. A single NaN in 5M
                    # parameters (0.00002%) is a harmless numerical artifact
                    # from one eigenvalue boundary or one Laplacian pivot —
                    # it carries no information about kernel health and must
                    # not accumulate toward the bail-out counter. Only counts
                    # at or above this threshold are suspicious.
                    MIN_BAD_GRAD_ENTRIES = 100

                    if n_bad_grad > 0:# and bad_frac >= skip_frac:
                        # Large fraction bad — skip the step entirely rather
                        # than moving in an essentially random direction.
                        bad_grad_count += 1
                        _record_error(
                            config, "grad_skip", it,
                            f"{n_bad_grad}/{n_total_grad} ({100*bad_frac:.2f}%) "
                            f"grad entries NaN/Inf [{bad_params_str}], "
                            f"loss={loss.item():.3g}",
                            bad_grad_count, log,
                        )
                        optimizer.zero_grad()
                        it += 1
                        pbar.update(1)
                        _maybe_bail_on_nan_streak(
                            bad_grad_count, it, args, skip_frac, log, pbar
                        )
                        if it >= n_iters:
                            break
                        continue

                    # Always zero whatever bad entries exist so downstream
                    # grad-clip and optimizer.step() see finite values.
                    if n_bad_grad > 0:
                        for p in list(model.parameters()) + list(likelihood.parameters()):
                            if p.grad is None:
                                continue
                            bad = ~torch.isfinite(p.grad)
                            if bad.any():
                                p.grad.masked_fill_(bad, 0.0)
                    else:
                        bad_grad_count = 0

                    # if n_bad_grad >= MIN_BAD_GRAD_ENTRIES:
                    #     # Enough bad entries to be suspicious — count toward
                    #     # the bail-out streak and log to ERRORS.txt.
                    #     bad_grad_count += 1
                    #     _record_error(
                    #         config, "grad_zero", it,
                    #         f"{n_bad_grad}/{n_total_grad} ({100*bad_frac:.3f}%) "
                    #         f"grad entries zeroed [{bad_params_str}], "
                    #         f"recon={last_recon:.3g}, "
                    #         f"kl={last_kl:.3g}, loss={loss.item():.3g}",
                    #         bad_grad_count, log,
                    #     )
                    #     _maybe_bail_on_nan_streak(
                    #         bad_grad_count, it, args, skip_frac, log, pbar
                    #     )
                    # elif n_bad_grad > 0:
                    #     # 1–(MIN_BAD_GRAD_ENTRIES-1) bad entries out of
                    #     # millions: a harmless numerical glitch (one
                    #     # eigenvalue boundary, one Laplacian pivot). Silently
                    #     # zeroed above. Do NOT increment bad_grad_count and
                    #     # do NOT reset it — this step is neither clean nor
                    #     # genuinely problematic, so it's neutral w.r.t. the
                    #     # streak. The streak only resets on a fully clean step
                    #     # (n_bad_grad == 0), ensuring that a sequence of
                    #     # single-entry glitches interleaved with clean steps
                    #     # never accumulates toward the bail-out threshold.
                    #     pass
                    # else:
                    #     # Fully clean step — reset the streak counter.
                    #     bad_grad_count = 0

                    # ---- per-group gradient norms (pre-clip) for W&B ---------
                    # Captured BEFORE clipping so they reflect the true gradient
                    # magnitude. Only computed at the log cadence (the per-param
                    # reduction is wasteful every iter). Includes the inducing-
                    # point grad norm, which is nonzero only with --learn-inducing.
                    wandb_metrics = None
                    if wandb_on and (it % log_every == 0 or it == n_iters - 1):
                        wandb_metrics = _grad_norms_by_group(model, likelihood)

                    # ---- gradient clipping ---------------------------------
                    # After NaN sanitization, gradients are finite — clipping
                    # caps any remaining large-but-finite values. Helps with
                    # the analytic ELBO's large gradients at init.
                    grad_clip = float(args.get("grad_clip", 10.0))
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(likelihood.parameters()),
                        max_norm=grad_clip,
                    )

                    optimizer.step()

                    # ---- W&B per-iter metrics (log cadence) ------------------
                    if wandb_metrics is not None:
                        import wandb
                        lvn = likelihood.task_noises.detach().log()
                        wandb_metrics.update({
                            "loss": float(loss.detach().item()),
                            "recon": last_recon,
                            "kl": last_kl,
                            "noise/log_min": float(lvn.min().item()),
                            "noise/log_max": float(lvn.max().item()),
                            "noise/log_mean": float(lvn.mean().item()),
                            "epoch": current_epoch,
                            "batch": int(args.get("wandb_batch_index", 0)),
                        })
                        wandb_metrics.update(_kernel_hypers_for_log(model))
                        wandb.log(wandb_metrics, step=wandb_step0 + it)
                    # No manual noise clamp: the likelihood's Interval
                    # constraint maps raw_task_noises through a sigmoid into
                    # [e^NOISE_LOG_MIN, e^NOISE_LOG_MAX], so task_noises is
                    # ALWAYS in range regardless of the optimizer step — this
                    # replaces the old post-step log_var_n.clamp_.

                    # ---- periodic log line (cluster-friendly) -----------------
                    # Visible even when tqdm output is suppressed (non-tty, log
                    # capture, runai, etc). Includes recon / KL / log-noise
                    # range / elapsed / ETA so a cluster job log gives a
                    # full picture.
                    if it % log_every == 0 or it == n_iters - 1:
                        elapsed = time.time() - fit_t0
                        done = it + 1
                        rate = done / max(elapsed, 1e-9)
                        eta = (n_iters - done) / max(rate, 1e-9)
                        lvn = likelihood.task_noises.detach().log()
                        lvn_min = float(lvn.min().item())
                        lvn_max = float(lvn.max().item())
                        log.info(
                            f"  [{pbar_desc}] it {done:>6d}/{n_iters} "
                            f"({100 * done / n_iters:5.1f}%) "
                            f"recon={last_recon:.4g} kl={last_kl:.4g} "
                            f"lvn=[{lvn_min:+.2f},{lvn_max:+.2f}] "
                            f"{rate:.1f} it/s  elapsed={elapsed:.0f}s "
                            f"eta={eta:.0f}s"
                        )

                    # ---- epoch boundary log + checkpoint ---------------------
                    new_epoch = (it + 1) // iters_per_epoch
                    if new_epoch > current_epoch and new_epoch <= int(args["epochs"]):
                        epoch_elapsed = time.time() - epoch_t0
                        log.info(
                            f"  [{pbar_desc}] === epoch {new_epoch}/{args['epochs']} "
                            f"done in {epoch_elapsed:.1f}s "
                            f"(recon={last_recon:.4g}, kl={last_kl:.4g}) ==="
                        )
                        current_epoch = new_epoch
                        epoch_t0 = time.time()
                        # Periodic checkpoint — every N epochs (configurable).
                        # Saves the SAME path each time so disk usage is bounded.
                        if ckpt_every > 0 and current_epoch % ckpt_every == 0:
                            _save_ckpt(it + 1)

                    if (it % max(1, n_iters // 30)) == 0:
                        lvn = likelihood.task_noises.detach().log()
                        pbar.set_postfix(
                            recon=f"{last_recon:.3g}",
                            kl=f"{last_kl:.3g}",
                            lvn=f"[{lvn.min().item():.2f},"
                                f"{lvn.max().item():.2f}]",
                        )

                    # Advance the global iter counter + progress bar. The
                    # `for x_b, y_b in train_loader:` loop doesn't auto-
                    # increment `it` like the old `for it in pbar:` did.
                    it += 1
                    pbar.update(1)
                    if it >= n_iters:
                        break  # break out of inner (DataLoader) loop
                # Inner loop done (epoch complete OR break for iter cap).
                # Re-check the cap to also break the outer (epoch) loop —
                # `break` only escapes one level in Python.
                if it >= n_iters:
                    break
        except RuntimeError as ex:
            # Training diverged (sustained NaN streak, etc.). Before
            # giving up entirely, try to load the in-progress
            # checkpoint we saved every `ckpt_every` epochs — that's a
            # state from BEFORE the divergence and is usable for
            # prediction. Caller gets to decide whether to save those
            # predictions or skip them.
            pbar.close()
            log.warning(
                f"  [{pbar_desc}] TRAINING ABORTED: {ex.__class__.__name__}"
            )
            if ckpt_path is not None and Path(ckpt_path).exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location=device)
                    if ckpt.get("n_tasks") == n_tasks:
                        model.load_state_dict(ckpt["model_state"])
                        _restore_noise(likelihood, ckpt, device)
                        log.warning(
                            f"  [{pbar_desc}] recovered from "
                            f"{Path(ckpt_path).name} @ iter "
                            f"{ckpt.get('iter', 0)} / epoch "
                            f"{ckpt.get('epoch', 0)} — predictions will "
                            f"use this state (early stop). Caller still "
                            f"counts this as a failed batch."
                        )
                        model.eval()
                        likelihood.eval()
                        return model, likelihood, "early_stopped_from_ckpt"
                    else:
                        log.warning(
                            f"  [{pbar_desc}] checkpoint n_tasks "
                            f"mismatch — cannot recover from it"
                        )
                except Exception as load_err:
                    log.warning(
                        f"  [{pbar_desc}] could not load checkpoint "
                        f"{ckpt_path}: {load_err}"
                    )
            log.warning(
                f"  [{pbar_desc}] no usable checkpoint — no predictions "
                f"will be written for this batch"
            )
            raise  # re-raise so caller sees the failure
    pbar.set_postfix(recon=f"{last_recon:.3g}", kl=f"{last_kl:.3g}")
    pbar.close()
    fit_elapsed = time.time() - fit_t0
    log.info(
        f"  [{pbar_desc}] FIT DONE: final recon={last_recon:.4g} "
        f"kl={last_kl:.4g} in {fit_elapsed:.1f}s "
        f"({n_iters / max(fit_elapsed, 1e-9):.1f} it/s avg)"
    )

    model.eval()
    likelihood.eval()
    return model, likelihood, "ok"


def predict_batched(model, likelihood, x: torch.Tensor, n_tasks: int,
                    chunk: int = 20_000):
    """Forward in chunks. Returns (mean, var) of the **posterior over
    the latent f**, NOT the predictive over y. We then add the per-task
    observation noise back in at the end.

    Why split f vs y? Downstream metrics want both the latent-f mean and a
    y-comparable variance. The observation noise lives in the likelihood's
    per-task ``task_noises`` (the GaussianLikelihood analogue of the old
    ``log_var_n``); we add it to the latent variance to get the predictive
    variance over y (= q(f|x).variance + obs_noise).
    """
    means, vars_f = [], []
    n = x.shape[0]
    obs_var = likelihood.task_noises.detach().to(x.device)  # (n_tasks,)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            post = model(x[s:e])
            # MultitaskMVN: .mean is (batch, n_tasks), .variance same shape
            means.append(post.mean.detach().cpu())
            vars_f.append(post.variance.clamp(min=0).detach().cpu())
    mean = torch.cat(means).numpy()                       # (N, n_tasks)
    var_f = torch.cat(vars_f).numpy()
    var_y = var_f + obs_var.cpu().numpy()[None, :]        # broadcast (1, n_tasks)

    # NaN audit at predict time. The training loop bails on non-finite
    # loss, but the variational mean / variance can still be NaN at
    # query points outside the model's support (e.g. ill-conditioned
    # Cholesky of the inducing covariance). Log a one-line summary
    # rather than silently writing NaN-laden .npy files.
    n_nan_mean = int(np.isnan(mean).sum())
    n_nan_var = int(np.isnan(var_y).sum())
    if n_nan_mean or n_nan_var:
        logging.getLogger("per_lipid_gp").warning(
            f"  predict produced NaN: mean {n_nan_mean}/{mean.size} "
            f"({100*n_nan_mean/mean.size:.2f}%), var "
            f"{n_nan_var}/{var_y.size} ({100*n_nan_var/var_y.size:.2f}%). "
            f"The predictions for this batch will contain NaN."
        )
    return mean, var_y


# =============================================================================
# Spectral (weight-space SpectralLatentGP) per-lipid trainer
# -----------------------------------------------------------------------------
# SpectralLatentGP learns a diagonal variational posterior over the spectral
# weights w (one mean+var per Laplacian eigenmode per lipid) with prior
# N(0, Phi), Phi = the Matern spectral density. It has a custom kl_divergence()
# and NO variational_strategy, so gpytorch.mlls.VariationalELBO (used by
# train_lipid_batch) does not apply — hence this dedicated trainer. Prediction
# reuses predict_batched (forward returns .mean / .variance of shape (N, tasks)).
#
# Objective mirrors train_lipid_batch's loss scale: a minibatch estimate of the
# full-data expected log-likelihood (scaled by n_train/bs) minus the full KL.
# v1 is ML-II (the per-task noise still has the Interval constraint preventing
# collapse/explosion); the MAP-II hyperprior used by the analytic path is not
# folded in here.
# =============================================================================
def train_lipid_batch_spectral(
    *,
    coords_train: torch.Tensor,   # (N, 3) CPU, z-scored
    y_train: torch.Tensor,        # (N, B)
    manifold_kernel,
    config: MaldiConfig,
    args: dict,
    device: str,
    log: logging.Logger,
    pbar_desc: str,
):
    n_tasks = y_train.shape[1]
    n_train = y_train.shape[0]

    model = SpectralLatentGP(num_tasks=n_tasks, manifold_kernel=manifold_kernel).to(device)
    likelihood = _build_task_likelihood(n_tasks, device)

    model.train()
    likelihood.train()
    if args.get("lengthscale_no_decay", False):
        no_decay, decay = [], []
        for pname, p_ in model.named_parameters():
            (no_decay if ("raw_lengthscale" in pname or "raw_graphbandwidth" in pname
                          or "raw_diffusion_scale" in pname)
             else decay).append(p_)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-3},
             {"params": no_decay, "weight_decay": 0.0},
             {"params": list(likelihood.parameters()), "weight_decay": 0.0}],
            lr=config.learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=config.learning_rate, weight_decay=1e-3,
        )

    bs = min(int(config.batch_size), n_train)
    iters_per_epoch = max(1, n_train // bs)
    n_iters = int(config.epochs) * iters_per_epoch
    log_every = max(1, n_iters // 20)
    grad_clip = float(args.get("grad_clip", 10.0))
    skip_frac = float(args.get("bad_grad_skip_frac", 0.01))
    two_pi = math.log(2.0 * math.pi)

    # ---- resume from in-progress checkpoint ----
    ckpt_path = args.get("checkpoint_path", None)
    iter_start = 0
    if ckpt_path is not None and Path(ckpt_path).exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if ckpt.get("n_tasks") == n_tasks:
                model.load_state_dict(ckpt["model_state"])
                _restore_noise(likelihood, ckpt, device)
                iter_start = int(ckpt.get("iter", 0))
                log.info(f"  [{pbar_desc}] resumed spectral @ iter "
                         f"{iter_start}/{n_iters}")
        except Exception as ex:
            log.warning(f"  [{pbar_desc}] could not load spectral ckpt: {ex}")

    log.info(f"  [{pbar_desc}] spectral fit: n_train={n_train:,} bs={bs} "
             f"epochs={args['epochs']} iters={n_iters} n_tasks={n_tasks} "
             f"num_modes={model.num_modes}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(coords_train, y_train),
        batch_size=bs, shuffle=True,
        num_workers=int(args.get("dataloader_workers", 2)),
        pin_memory=device.startswith("cuda"),
        persistent_workers=int(args.get("dataloader_workers", 2)) > 0,
        drop_last=True,
    )

    ckpt_every = int(args.get("checkpoint_every_epochs", 5))
    pbar = tqdm(total=n_iters, initial=iter_start, desc=pbar_desc,
                leave=False, dynamic_ncols=True)
    bad_grad_count = 0
    it = iter_start
    last_loss = float("nan")

    def _save_spectral_ckpt(iter_idx, epoch):
        if ckpt_path is None:
            return
        try:
            tmp = Path(str(ckpt_path) + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(),
                        "likelihood_state": likelihood.state_dict(),
                        "iter": iter_idx, "epoch": epoch, "n_tasks": n_tasks}, tmp)
            os.replace(tmp, ckpt_path)
        except Exception as ex:
            log.warning(f"  [{pbar_desc}] spectral ckpt failed: {ex}")

    start_epoch = iter_start // iters_per_epoch
    for epoch in range(start_epoch, int(config.epochs)):
        for coord_b, y_b in train_loader:
            coord_b = coord_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            gp_out = model(coord_b)
            mean_z = gp_out.mean                       # (b, tasks)
            var_z = gp_out.variance.clamp(min=0)       # (b, tasks)
            sigma2 = likelihood.task_noises.clamp(min=1e-8)   # (tasks,)
            # Expected Gaussian log-likelihood under the diagonal posterior.
            ell = -0.5 * (
                two_pi + torch.log(sigma2)
                + ((y_b - mean_z) ** 2 + var_z) / sigma2
            ).sum()
            kl = model.kl_divergence()
            loss = -(n_train / bs) * ell + kl

            if not torch.isfinite(loss):
                bad_grad_count += 1
                _record_error(config, "nonfinite_loss", it,
                              f"loss={loss.item()}", bad_grad_count, log)
                _maybe_bail_on_nan_streak(bad_grad_count, it, args, skip_frac, log, pbar)
                it += 1
                pbar.update(1)
                continue

            loss.backward()
            # NaN/Inf gradient handling (mirror train_lipid_batch).
            n_bad = n_tot = 0
            for p_ in list(model.parameters()) + list(likelihood.parameters()):
                if p_.grad is None:
                    continue
                bad = ~torch.isfinite(p_.grad)
                n_bad += int(bad.sum().item())
                n_tot += p_.grad.numel()
            if n_bad:
                bad_grad_count += 1
                frac = n_bad / max(1, n_tot)
                if frac >= skip_frac:
                    _record_error(config, "grad_skip", it,
                                  f"frac={frac:.4f}", bad_grad_count, log)
                    _maybe_bail_on_nan_streak(bad_grad_count, it, args, skip_frac, log, pbar)
                    it += 1
                    pbar.update(1)
                    continue
                for p_ in list(model.parameters()) + list(likelihood.parameters()):
                    if p_.grad is not None:
                        torch.nan_to_num_(p_.grad, nan=0.0, posinf=0.0, neginf=0.0)
                _record_error(config, "grad_zero", it, f"frac={frac:.4f}",
                              bad_grad_count, log)
                _maybe_bail_on_nan_streak(bad_grad_count, it, args, skip_frac, log, pbar)
            else:
                bad_grad_count = 0

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(likelihood.parameters()), grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
            it += 1
            pbar.update(1)
            if it % log_every == 0:
                log.info(f"  [{pbar_desc}] it {it}/{n_iters} loss={last_loss:.3f} "
                         f"kl={float(kl.item()):.3f}")
        if ckpt_every and ((epoch + 1) % ckpt_every == 0):
            _save_spectral_ckpt(it, epoch + 1)
    pbar.close()
    return model, likelihood, "ok"


# =============================================================================
# Variational Nearest-Neighbour GP (VNNGP, Wu et al. 2022) — isolated path
# -----------------------------------------------------------------------------
# Single-output, Euclidean Matern. Inducing points = (a dense subset of) the
# TRAINING voxels, so "closer points per test point" is built in: each query
# conditions on its --nn-k nearest inducing neighbours. No O(M^3) Cholesky, so
# M can be the whole training set — this is the fix for "inducing points too
# scattered". Kept completely separate from the hardened analytic multitask
# path; it writes the SAME on-disk layout so analysis / the viewer are agnostic.
# =============================================================================
class VNNGPModel(gpytorch.models.ApproximateGP):
    """Single-task VNNGP with a Euclidean Matern kernel.

    Uses gpytorch's NNVariationalStrategy + MeanFieldVariationalDistribution.
    Inducing points are the training inputs themselves (or a large subset);
    the variational posterior only couples each inducing point to its k
    nearest inducing neighbours (Vecchia ordering), giving O(k^3) per-step
    cost independent of M.
    """

    def __init__(self, inducing_points, k=256, training_batch_size=256,
                 nu=2.5):
        m, d = inducing_points.shape
        self.m = m
        self.k = k
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)
        variational_strategy = gpytorch.variational.NNVariationalStrategy(
            self, inducing_points, variational_distribution,
            k=k, training_batch_size=training_batch_size,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=d),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        # x=None during training (NNVariationalStrategy minibatches over the
        # inducing points internally); x=test points at prediction time.
        if x is not None and x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=prior, **kwargs)


def train_one_lipid_vnngp(*, inducing_z, y_train_col, args, device, log, desc,
                          geo_nn=None):
    """Fit one VNNGP for a single lipid. ``inducing_z`` is (M, 3) z-scored
    training coords (== the inducing set); ``y_train_col`` is (M,) z-scored
    targets ALIGNED to ``inducing_z`` row order (required — VNNGP indexes y by
    the inducing-point minibatch it draws). ``geo_nn``, if given, is the
    precomputed (seq_nn, node_knn) shortest-path structure: when present the
    Euclidean NNUtil is swapped for GraphGeodesicNNUtil so neighbours are
    chosen by graph shortest-path distance. Returns (model, likelihood)."""
    inducing_z = inducing_z.to(device).contiguous()
    y = y_train_col.to(device).contiguous()
    M = inducing_z.shape[0]
    k = min(int(args["nn_k"]), M - 1)
    tbs = min(int(args["batch_size"]), M)

    model = VNNGPModel(inducing_z, k=k, training_batch_size=tbs,
                       nu=float(args["nu"])).to(device)
    if geo_nn is not None:
        # Replace gpytorch's Euclidean Vecchia structure with the precomputed
        # shortest-path one, then recompute the strategy's cached NN indices.
        seq_nn, node_knn = geo_nn
        model.variational_strategy.nn_util = GraphGeodesicNNUtil(
            k=k, dim=inducing_z.shape[1], seq_nn=seq_nn, node_knn=node_knn,
            inducing_coords=inducing_z, device=device,
        )
        model.variational_strategy._compute_nn()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model.train(); likelihood.train()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()},
         {"params": likelihood.parameters()}],
        lr=float(args["learning_rate"]),
    )
    # num_data = M: the ELBO's KL is over all inducing points; recon is the
    # minibatch the strategy drew. VariationalELBO rescales internally.
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=M)

    iters_per_epoch = max(1, M // tbs)
    n_iters = int(args["epochs"]) * iters_per_epoch
    log_every = max(1, n_iters // 20)
    log.info(f"  [{desc}] VNNGP fit: M={M:,} k={k} tbs={tbs} "
             f"iters_per_epoch={iters_per_epoch} total_iters={n_iters}")

    t0 = time.time()
    for it in range(n_iters):
        optimizer.zero_grad()
        output = model(x=None)
        # The inducing-point minibatch the strategy just drew; index y by it.
        idx = model.variational_strategy.current_training_indices
        y_batch = y[idx].to(output.mean.device)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
        if it % log_every == 0 or it == n_iters - 1:
            elapsed = time.time() - t0
            rate = (it + 1) / max(elapsed, 1e-9)
            log.info(f"  [{desc}] it {it+1:>6d}/{n_iters} "
                     f"loss={loss.item():.4g} {rate:.1f} it/s "
                     f"elapsed={elapsed:.0f}s")
    model.eval(); likelihood.eval()
    return model, likelihood


def predict_vnngp(model, likelihood, x, chunk=20_000):
    """Chunked predictive (mean, var of y) for a single-task VNNGP."""
    means, vars_y = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for s in range(0, x.shape[0], chunk):
            e = min(s + chunk, x.shape[0])
            pred = likelihood(model(x[s:e]))
            means.append(pred.mean.detach().cpu())
            vars_y.append(pred.variance.clamp(min=0).detach().cpu())
    return torch.cat(means).numpy(), torch.cat(vars_y).numpy()


# =============================================================================
# Geodesic neighbour selection for VNNGP
# -----------------------------------------------------------------------------
# Instead of gpytorch's built-in Euclidean kNN, rank each point's conditioning
# neighbours by SHORTEST-PATH distance on a kNN graph over the inducing voxels.
# We precompute the Vecchia structure (each point's k nearest among PRECEDING
# points) and a per-node geodesic kNN (for snapping out-of-sample queries), then
# inject them via a drop-in that mimics gpytorch's NNUtil 3-method interface.
# The kernel stays a Euclidean Matern, so the GP is PSD-safe.
# =============================================================================
def _faiss_knn_graph(coords_np: np.ndarray, k: int):
    """Build a symmetric k-NN graph over coords (M,D) with faiss (exact L2).
    Returns (edge_index (2,E) int64, edge_value (E,) float = squared L2)."""
    import faiss
    M, D = coords_np.shape
    coords_np = np.ascontiguousarray(coords_np, dtype=np.float32)
    index = faiss.IndexFlatL2(D)
    index.add(coords_np)
    dist, idx = index.search(coords_np, k + 1)        # +1: first hit is self
    rows = np.repeat(np.arange(M, dtype=np.int64), k)
    cols = idx[:, 1:].reshape(-1).astype(np.int64)
    vals = dist[:, 1:].reshape(-1).astype(np.float64)  # squared L2
    return np.stack([rows, cols]), vals


def _build_geodesic_nn(M: int, edge_index: np.ndarray, edge_value: np.ndarray,
                       k: int, log, batch: int = 256):
    """Shortest-path kNN on the graph. Returns:
      seq  (M-k, k) int64 — for point i in [k,M), its k geodesically-nearest
                            among the PRECEDING points 0..i-1 (Vecchia order).
      node (M, k)   int64 — each point's k geodesically-nearest overall (used
                            to give snapped out-of-sample queries their
                            neighbours).
    Unreachable pairs (disconnected components) are treated as +inf so the
    selection prefers reachable nodes and only falls back to far ones if it
    must."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg

    lengths = np.sqrt(np.maximum(edge_value, 0.0))
    G = sp.csr_matrix((lengths, (edge_index[0], edge_index[1])), shape=(M, M))
    G = G.maximum(G.T)                                  # undirected
    seq = np.empty((M - k, k), dtype=np.int64)
    node = np.empty((M, k), dtype=np.int64)
    BIG = np.float64(1e12)
    t0 = time.time()
    for s in range(0, M, batch):
        e = min(s + batch, M)
        D = csg.dijkstra(G, directed=False, indices=np.arange(s, e))  # (b, M)
        D[~np.isfinite(D)] = BIG
        for r, i in enumerate(range(s, e)):
            row = D[r]
            row[i] = BIG                                # exclude self
            sel = np.argpartition(row, k)[:k]
            node[i] = sel[np.argsort(row[sel])]
            if i >= k:
                rowp = row.copy()
                rowp[i:] = BIG                          # only j < i allowed
                selp = np.argpartition(rowp, k)[:k]
                seq[i - k] = selp[np.argsort(rowp[selp])]
        if (s // batch) % 20 == 0:
            log.info(f"    geodesic kNN: {e:,}/{M:,} nodes "
                     f"({time.time()-t0:.0f}s)")
    return seq, node


class GraphGeodesicNNUtil(torch.nn.Module):
    """Drop-in for gpytorch's NNUtil that serves PRECOMPUTED shortest-path
    neighbours. Single-task only (inducing batch shape must be empty)."""

    def __init__(self, k, dim, seq_nn, node_knn, inducing_coords, device="cpu"):
        super().__init__()
        self.k = int(k)
        self.dim = int(dim)
        self.batch_shape = torch.Size([])
        self.train_n = int(inducing_coords.shape[0])
        self.register_buffer("_seq", seq_nn.to(torch.long))        # (M-k, k)
        self.register_buffer("_node", node_knn.to(torch.long))     # (M, k)
        ind_np = inducing_coords.detach().cpu().numpy().astype(np.float32)
        import faiss
        self._snap = faiss.IndexFlatL2(ind_np.shape[1])
        self._snap.add(np.ascontiguousarray(ind_np))
        self.to(device)

    def set_nn_idx(self, train_x):  # structure is precomputed; just record n
        self.train_n = train_x.shape[-2]

    def build_sequential_nn_idx(self, x):
        # match NNUtil's (batch_numel, M-k, k) leading-dim convention
        return self._seq.unsqueeze(0)

    def find_nn_idx(self, test_x, k=None):
        kk = self.k if k is None else int(k)
        q = test_x.reshape(-1, self.dim).detach().cpu().numpy().astype(np.float32)
        _, idx = self._snap.search(np.ascontiguousarray(q), 1)   # snap to nearest node
        snapped = torch.from_numpy(idx[:, 0]).long().to(self._node.device)
        nn = self._node[snapped][:, :kk]
        return nn.to(test_x.device)


# =============================================================================
# Whole-brain renders — composite PNGs per trained lipid, mirroring the manifold
# GP's whole_brain_reconstruction -> render_reconstruction. Reuses the already-
# computed whole-brain node predictions (predictions/<slug>/graph_pred_raw.npy),
# so it costs no extra GPU work and honours --resume automatically.
# =============================================================================
def _graph_meta_voxel_layout(graph_meta, args, template_full):
    """Map graph nodes -> integer voxel indices and pick the volume shape +
    anatomy background to render into.

    Euclidean runs store full-resolution ``node_voxel_idx`` directly, so we
    render at the atlas's native resolution. Manifold runs store the strided
    ``reference_nodes_z`` (nodes live on the stride-subsampled grid); scattering
    those into the full-res atlas would leave every non-strided slice empty, so
    we render into the STRIDED template instead — dense at that resolution and
    the honest fidelity of the per-lipid manifold reconstruction.

    Returns (positions:int64 (N,3), out_shape:tuple, anatomy:np.ndarray).
    """
    if "node_voxel_idx" in graph_meta:
        positions = graph_meta["node_voxel_idx"].astype(np.int64)
        out_shape = tuple(int(s) for s in graph_meta["template_shape"])
        anatomy = template_full
    else:
        mm = (graph_meta["reference_nodes_z"] * graph_meta["coord_std"]
              + graph_meta["coord_mean"])
        scale = float(graph_meta["voxel_scale_mm"])   # strided scale (stride*0.025)
        positions = (mm / scale).round().astype(np.int64)
        stride = int(args["stride"])
        anatomy = template_full[::stride, ::stride, ::stride]
        out_shape = anatomy.shape
    positions = positions.copy()
    for ax in range(3):
        np.clip(positions[:, ax], 0, out_shape[ax] - 1, out=positions[:, ax])
    return positions, out_shape, anatomy


def render_trained_lipids(out_root, args, lipid_names_all, lipid_idx_to_fit, log):
    """Render composite PNGs of the whole-brain reconstruction for the lipids
    trained in this run, into ``<out_root>/renders/`` — the same panels and
    output layout as ``lgp_manifold_experiment``'s reconstruction render.

    Each lipid's per-node prediction (``graph_pred_raw.npy``) is scattered back
    into a template-shaped volume (NaN outside the brain) and handed to
    ``render_lipid_volumes.render_selected_lipids``. Capped at
    ``--render-max-lipids`` so a batch job produces a few sanity-check images
    rather than one PNG per lipid. Lipids without predictions on disk (e.g. a
    batch that hasn't run yet) are skipped.
    """
    from render_lipid_volumes import (
        render_selected_lipids, render_lipid_diagnostics, render_error_slice,
    )

    meta_path = out_root / "graph_meta.npz"
    if not meta_path.exists():
        log.warning(f"render: {meta_path} missing; skipping renders.")
        return
    graph_meta = np.load(meta_path)
    template_full = np.load(args["reference_file"])
    positions, out_shape, anatomy = _graph_meta_voxel_layout(
        graph_meta, args, template_full,
    )
    z, y, x = positions[:, 0], positions[:, 1], positions[:, 2]

    predictions_root = out_root / "predictions"
    render_dir = out_root / "renders"
    cap = int(args.get("render_max_lipids", 8) or 0)
    volume_dir = render_dir / "volume"
    volume_dir.mkdir(parents=True, exist_ok=True)

    # Per-lipid de-standardization params (fit-order columns), for the raw-scale
    # diagnostics. col_means[j] pairs with lipid_idx_to_fit[j].
    col_means = col_stds = None
    cm_path, cs_path = out_root / "col_means.pt", out_root / "col_stds.pt"
    if cm_path.exists() and cs_path.exists():
        col_means = torch.load(cm_path).cpu().numpy()
        col_stds = torch.load(cs_path).cpu().numpy()
    log_transform = bool(args.get("log_transform", False))

    rendered = []   # (name, fit_col_j)
    for j, g_idx in enumerate(lipid_idx_to_fit):
        if cap and len(rendered) >= cap:
            break
        name = lipid_names_all[g_idx]
        pred_file = predictions_root / safe_filename(name) / "graph_pred_raw.npy"
        if not pred_file.exists():
            continue
        vals = np.load(pred_file).astype(np.float32)
        if vals.shape[0] != positions.shape[0]:
            log.warning(
                f"render: {name} has {vals.shape[0]} node predictions but the "
                f"graph has {positions.shape[0]} nodes; skipping."
            )
            continue
        vol = np.full(out_shape, np.nan, dtype=np.float32)
        vol[z, y, x] = vals
        np.save(volume_dir / f"{name}_volume.npy", vol)
        rendered.append((name, j))

    if not rendered:
        log.warning("render: no per-lipid predictions found to render.")
        return

    rendered_names = [name for name, _ in rendered]
    log.info(f"render: writing {len(rendered_names)} composite PNG(s) to "
             f"{render_dir} (lipids: {rendered_names})")
    render_selected_lipids(
        template_volume=anatomy,
        volume_dir=volume_dir,
        output_dir=render_dir,
        selected_lipids_names=rendered_names,
        lipid_names=rendered_names,
        suffix="",
        # render_lipid_volumes lays out 10 anatomical-slice columns, so the
        # rotation row must also be 10 wide (same value the manifold GP uses).
        n_rotation_frames=10,
    )

    # Per-lipid diagnostics: value distribution + true-vs-pred scatter (linear
    # & log), from the held-out TEST points de-standardized back to raw
    # intensity units (exp'd when --log-transform, matching the reconstruction).
    if col_means is None:
        log.warning("render: col_means/col_stds missing; skipping diagnostics.")
    else:
        for name, j in rendered:
            lip_dir = predictions_root / safe_filename(name)
            tz, pz = lip_dir / "test_true_z.npy", lip_dir / "test_pred_z.npy"
            if not (tz.exists() and pz.exists()):
                continue
            cm, cs = float(col_means[j]), float(col_stds[j])
            true_v = np.load(tz).astype(np.float64) * cs + cm
            pred_v = np.load(pz).astype(np.float64) * cs + cm
            if log_transform:
                true_v = np.exp(true_v) - 1e-10
                pred_v = np.exp(pred_v) - 1e-10
            try:
                render_lipid_diagnostics(
                    true_v, pred_v, name,
                    render_dir / f"{name}_diagnostics.png",
                )
            except Exception as ex:
                log.warning(f"render: diagnostics for {name} failed ({ex}).")
            # One MALDI section as a reconstruction-error heatmap, from the same
            # held-out TEST points (physical mm coords saved alongside).
            coords_f = lip_dir / "test_coords_mm.npy"
            if coords_f.exists():
                try:
                    render_error_slice(
                        np.load(coords_f), true_v, pred_v, name,
                        render_dir / f"{name}_error_slice.png",
                        axis_labels=("xccf", "yccf", "zccf"),
                    )
                except Exception as ex:
                    log.warning(f"render: error-slice for {name} failed ({ex}).")

    log.info(f"render: done -> {render_dir}")


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    # Cluster-friendly logging: line-buffered stdout (so log lines
    # appear in real time, not at process exit) + explicit stream
    # to stdout (so it's interleaved with the rest of the job's
    # output rather than appearing on stderr).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    log = logging.getLogger("per_lipid_gp")
    args = parse_args()
    apply_faiss_cpu_args(args)  # pin FAISS phases to CPU before any graph build
    config = MaldiConfig.from_args(args)

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    # ---- output directory ----
    out_root = config.exp_path
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(exist_ok=True)
    (out_root / "checkpoints").mkdir(exist_ok=True)

    # ---- refuse to proceed on a previously-failed dir ----
    # A run that terminally diverged writes FAILED.txt; on the next
    # invocation (e.g. cluster auto-restart) we refuse to re-run
    # automatically. The operator must investigate and explicitly
    # remove FAILED.txt to retry. This is intentional — auto-retrying
    # a divergent config wastes cluster time and produces nothing.
    failure_marker = out_root / "FAILED.txt"
    if failure_marker.exists():
        log.error("=" * 72)
        log.error(f"Output dir contains FAILED.txt — refusing to run.")
        log.error(f"  Path: {failure_marker}")
        log.error(f"  Contents:\n{failure_marker.read_text()}")
        log.error(
            "Remove FAILED.txt and re-run if you want to retry, OR "
            "change --exp-name to start fresh in a different directory."
        )
        log.error("=" * 72)
        sys.exit(2)  # distinct exit code: 2 = previously-failed-refusal

    with open(out_root / "config.json", "w") as f:
        json.dump(args, f, indent=2, default=str)
    log.info("=" * 72)
    log.info(f"Per-lipid GP experiment — {config.exp_name}")
    log.info(f"  kernel family : {args['kernel_family']}")
    log.info(f"  output_dir    : {out_root}")
    log.info("=" * 72)

    # ---- Weights & Biases (optional) ----
    # One run for the whole job; train_lipid_batch logs per-iter metrics at a
    # global step (batch_index * n_iters + it), and the summary is written at
    # the end. Scoped to the analytic variational path: the VNNGP path
    # (--variational nngp) has its own loop and returns early, so it would
    # leave the run unfinished — skip init there.
    wandb_active = bool(args.get("wandb", False)) and args.get("variational") != "nngp"
    if args.get("wandb", False) and not wandb_active:
        log.warning("--wandb is ignored for --variational nngp "
                    "(W&B covers the analytic variational path only).")
    args["wandb"] = wandb_active  # propagated into train_lipid_batch via args
    if wandb_active:
        import wandb
        wandb.init(
            project=args.get("wandb_project", "l3di_maldi_per_lipid"),
            name=config.exp_name, config=args,
        )
        log.info(f"  W&B logging enabled (project={args.get('wandb_project')})")

    # ---- config / inducing points (same as lgp_*_experiment.py) ----
    log.info("Computing inducing points + coord normalization …")
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
    log.info(f"  got {inducing_points.shape[0]} inducing points")
    inducing_points = inducing_points.to(args["device"])

    # ---- manifold setup (manifold / eigenmap / spectral all need the graph
    # + eigenpairs: 'manifold' uses the Riemann kernel directly, 'eigenmap'
    # uses the eigenfunctions as input features, 'spectral' uses them as the
    # weight-space basis) ----
    # ---- reference-only parcel field (--parcel-field) ----
    # Loaded once here and shared by every lipid batch: the KD-tree build and the
    # membership arrays are per-field, not per-batch. Its node coordinates are in
    # the same standardized frame as coords_train (parcelgp.volume mirrors
    # utils.coord_norm_from_reference), so no re-scaling is needed.
    parcel_field = None
    if args.get("parcel_field"):
        parcel_field = ParcelField.load(args["parcel_field"])
        log.info(
            "parcel field: %s — %d nodes, %d parcels, built from %s (stride %s, "
            "features %s)", args["parcel_field"],
            parcel_field.node_coords.shape[0], parcel_field.n_parcels,
            parcel_field.meta.get("reference_file", "?"),
            parcel_field.meta.get("stride", "?"),
            parcel_field.meta.get("features", "?"))

    needs_graph = args["kernel_family"] in ("manifold", "eigenmap", "spectral")
    manifold_ctx = None
    if needs_graph:
        if args.get("eigenvector_dir") is None:
            log.error(f"--eigenvector-dir is required for "
                      f"--kernel-family={args['kernel_family']}")
            sys.exit(2)
        log.info("Building manifold kernel (graph + eigenvectors) …")
        manifold_ctx = setup_manifold_kernel(
            args, config, coord_mean, coord_std, log,
        )
        knn = manifold_ctx["knn"]
        want_maldi_inducing = bool(args.get("inducing_from_maldi_nodes", False))
        if want_maldi_inducing:
            # Blend the inducing points over the UNALTERED (strided) graph:
            # ~density_frac from the densest graph nodes + the rest from the
            # measured MALDI voxels that snap onto the graph most cheaply. The
            # KNN graph itself is NOT modified — this is independent of
            # --augment-maldi-nodes (graph stays at stride). Both sources resolve
            # to exact graph nodes, so K_uu stays on the exact-eigenvector path.
            from utils import inducing_blend_density_maldi, maldi_voxels_standardized
            graph_coords = knn.x.detach().cpu().numpy()
            maldi_coords = maldi_voxels_standardized(
                config.maldi_file, config.section_filter, coord_mean, coord_std,
            ).numpy()
            density_frac = float(args.get("inducing_density_frac", 0.8))
            sel = inducing_blend_density_maldi(
                graph_coords, maldi_coords, config.num_inducing,
                density_frac=density_frac,
                density_k=args.get("knn_k", 15),
                method=args.get("inducing_method", "kmeans_snap"),
                seed=args["seed"],
            )
            inducing_points = torch.tensor(sel, dtype=torch.float32)
            n_dense = int(round(density_frac * config.num_inducing))
            log.info(
                f"  Inducing points blended: ~{n_dense} from densest graph "
                f"nodes + ~{config.num_inducing - n_dense} from cheapest-to-snap "
                f"MALDI voxels (graph N={graph_coords.shape[0]:,}, "
                f"MALDI={maldi_coords.shape[0]:,}): {inducing_points.shape[0]} "
                f"unique (from {config.num_inducing} requested)"
            )
        else:
            # Snap the k-means inducing points to the nearest graph nodes so
            # that every K_uu evaluation uses the exact eigenvector lookup
            # branch (is_on_graph) rather than the numerically fragile Nyström
            # OOS path. Mirrors lgp_manifold_experiment.setup_experiment().
            with torch.no_grad():
                ind_gpu = inducing_points.to(args["device"])
                _, nn_idx = knn.search(ind_gpu, 1)       # (M, 1) nearest node
                nn_idx = nn_idx.squeeze(1).cpu()          # (M,)
                nn_idx_unique = torch.unique(nn_idx)
                inducing_points = knn.x[nn_idx_unique].cpu()
            log.info(
                f"  Inducing points snapped to graph nodes: "
                f"{inducing_points.shape[0]} unique (from {config.num_inducing} requested)"
            )
        config.num_inducing = inducing_points.shape[0]
        inducing_points = inducing_points.to(args["device"])

    # ---- load lipid names + section filters ----
    lipid_names_all = list(config.selected_lipids_names)
    log.info(f"Available lipids: {len(lipid_names_all)}")
    with open(out_root / "lipid_names.json", "w") as f:
        json.dump(lipid_names_all, f, indent=2)

    # Merge --lipids and --lipids-file into one spec list. Both are
    # optional; if both are given they're concatenated. resolve_lipids()
    # then deduplicates while preserving order.
    lipid_spec = list(args.get("lipids") or [])
    if args.get("lipids_file"):
        with open(args["lipids_file"]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lipid_spec.append(line)
        log.info(f"Loaded {len(lipid_spec)} lipid spec(s) from file "
                 f"(after merging with --lipids).")
    if not lipid_spec:
        lipid_spec = None  # explicit "all lipids"

    lipid_idx_to_fit = resolve_lipids(lipid_spec, lipid_names_all, log)
    if args["limit"]:
        lipid_idx_to_fit = lipid_idx_to_fit[:int(args["limit"])]
    log.info(f"Will fit {len(lipid_idx_to_fit)} lipids: "
             f"{[lipid_names_all[i] for i in lipid_idx_to_fit[:5]]}"
             + ("..." if len(lipid_idx_to_fit) > 5 else ""))

    # ---- load FULL coords + ALL needed lipid columns once ----
    # (read each parquet file once, not 173 times)
    # NOTE: MaldiConfig calls the TRAIN-side filter `section_filter`
    # (the test-side is `test_filter`). MaldiExperiment.__init__ renames
    # the train one to `self.train_filter`, but we're not going through
    # the Experiment wrapper here — read the config attribute directly.
    log.info("Loading MaLDI train + test data (subset of columns) …")
    coords_tr_mm, y_tr_raw = load_maldi_columns(
        args["maldi_file"], config.section_filter,
        lipid_idx_to_fit, lipid_names_all,
    )
    log.info(f"  train: {coords_tr_mm.shape[0]:,} pts × "
             f"{y_tr_raw.shape[1]} lipids")
    coords_te_mm, y_te_raw = load_maldi_columns(
        args["maldi_file"], config.test_filter,
        lipid_idx_to_fit, lipid_names_all,
    )
    log.info(f"  test : {coords_te_mm.shape[0]:,} pts × "
             f"{y_te_raw.shape[1]} lipids")

    # Pre-processing identical to MaldiExperiment.load_train_data()
    y_tr_raw = y_tr_raw.clamp(min=0)
    y_te_raw = y_te_raw.clamp(min=0)
    if args["log_transform"]:
        y_tr_raw = torch.log(y_tr_raw + 1e-10)
        y_te_raw = torch.log(y_te_raw + 1e-10)
    col_means = y_tr_raw.mean(dim=0)
    col_stds = y_tr_raw.std(dim=0).clamp(min=1e-6)
    y_tr_z = (y_tr_raw - col_means) / col_stds
    y_te_z = (y_te_raw - col_means) / col_stds
    torch.save(col_means, out_root / "col_means.pt")
    torch.save(col_stds, out_root / "col_stds.pt")

    # Z-score coords using the global (whole-brain) statistics.
    # coords_tr_z stays on CPU — the DataLoader inside train_lipid_batch
    # will move each minibatch to GPU asynchronously with pin_memory.
    # coords_te_z goes to GPU directly: prediction is one-shot, chunked
    # internally by predict_batched, no need for a DataLoader.
    coords_tr_z = (coords_tr_mm - coord_mean) / coord_std
    coords_te_z = ((coords_te_mm - coord_mean) / coord_std).to(args["device"])

    # ---- graph_meta for off-line reconstruction ----
    if manifold_ctx is not None:
        savez_safe(
            out_root / "graph_meta.npz",
            reference_nodes_z=manifold_ctx["reference_nodes"].cpu().numpy(),
            template_shape=np.array(manifold_ctx["template_shape"]),
            voxel_offset=manifold_ctx["voxel_offset"],
            voxel_scale_mm=manifold_ctx["voxel_scale_mm"],
            coord_mean=coord_mean.cpu().numpy(),
            coord_std=coord_std.cpu().numpy(),
        )

    # For the EUCLIDEAN case we still want a whole-brain reconstruction.
    # Reuse the same node set if we have it, else build voxel coords on
    # the fly from the template's non-zero voxels.
    if manifold_ctx is None:
        log.info("Building whole-brain voxel grid for reconstruction …")
        template_volume = np.load(args["reference_file"])
        # Same threshold logic as whole_brain_reconstruction()
        nz = np.argwhere(template_volume > args.get("threshold", 5)).astype(np.int32)
        # Convert voxel idx → mm. The Allen CCF template has shape
        # (AP, DV, LR), and the parquet's xccf/yccf/zccf are in the SAME
        # order — i.e. xccf is axis 0, yccf is axis 1, zccf is axis 2.
        # So no permutation is needed: voxel (i, j, k) maps directly to
        # mm (i*scale, j*scale, k*scale) and that aligns with the
        # parquet's (xccf, yccf, zccf) columns used for coord_mean.
        # (Earlier versions of this file applied an erroneous [2,1,0]
        # reorder; if you have old Euclidean graph_pred_z.npy files
        # they're plotted at wrong voxel positions and need re-running.)
        voxel_scale = 0.025
        coords_brain_mm = torch.from_numpy(nz.astype(np.float32) * voxel_scale)
        node_voxel_idx = nz
        savez_safe(
            out_root / "graph_meta.npz",
            node_voxel_idx=node_voxel_idx,
            template_shape=np.array(template_volume.shape),
            coord_mean=coord_mean.cpu().numpy(),
            coord_std=coord_std.cpu().numpy(),
        )
        brain_nodes_z = ((coords_brain_mm - coord_mean) / coord_std).to(args["device"])
        log.info(f"  brain grid: {brain_nodes_z.shape[0]:,} voxels")
    else:
        brain_nodes_z = manifold_ctx["reference_nodes"]

    # =====================================================================
    # Eigenmap: replace the 3D coordinates with the standardized leading-r
    # Laplacian-eigenfunction embedding, then fall through to the EUCLIDEAN
    # analytic path (the GP is a standard ARD Matern over e(x)). The embedding
    # is fixed (precomputed); the ARD kernel learns per-mode weighting.
    # =====================================================================
    if args["kernel_family"] == "eigenmap":
        r = int(args["embed_dim"])
        kern = manifold_ctx["kernel"]
        dev = args["device"]
        log.info(f"Eigenmap: embedding coordinates into leading {r} "
                 f"eigenfunctions (of {kern.num_modes} modes) …")
        emb_tr = embed_eigenmap(kern, coords_tr_z, r, dev)        # CPU (N_tr, r)
        emb_mean = emb_tr.mean(0)
        emb_std = emb_tr.std(0).clamp(min=1e-6)
        torch.save({"emb_mean": emb_mean, "emb_std": emb_std, "r": r},
                   out_root / "eigenmap_embed.pt")
        # Train stays on CPU (DataLoader moves minibatches to GPU); the rest go
        # to GPU for one-shot chunked prediction. Standardize with train stats.
        coords_tr_z = (emb_tr - emb_mean) / emb_std
        coords_te_z = ((embed_eigenmap(kern, coords_te_z, r, dev) - emb_mean)
                       / emb_std).to(dev)
        brain_nodes_z = ((embed_eigenmap(kern, brain_nodes_z, r, dev) - emb_mean)
                         / emb_std).to(dev)
        inducing_points = ((embed_eigenmap(kern, inducing_points, r, dev) - emb_mean)
                           / emb_std).to(dev)
        args["_embed_dim"] = r
        log.info(f"  embedded: train {tuple(coords_tr_z.shape)}, "
                 f"inducing {tuple(inducing_points.shape)}")

    # =====================================================================
    # VNNGP path (isolated). Euclidean only; fits one lipid at a time and
    # writes the SAME predictions/<slug>/*.npy + metrics.csv layout, then
    # returns before the analytic multitask loop below.
    # =====================================================================
    if args.get("variational") == "nngp":
        if manifold_ctx is not None:
            log.error(
                "--variational nngp is EUCLIDEAN ONLY (the manifold "
                "spectral kernel does not compose with the nearest-"
                "neighbour factorisation). Re-run with "
                "--kernel-family euclidean."
            )
            sys.exit(2)

        device = args["device"]
        predictions_root = out_root / "predictions"
        # Inducing set = (subset of) training voxels, shared across lipids
        # (all lipids live on the same voxels). y is indexed by the SAME
        # rows so inducing<->target alignment holds.
        N_tr = coords_tr_z.shape[0]
        cap = int(args.get("nngp_num_inducing", 0) or 0)
        if 0 < cap < N_tr:
            g = torch.Generator().manual_seed(int(args["seed"]))
            ind_rows = torch.randperm(N_tr, generator=g)[:cap]
        else:
            ind_rows = torch.arange(N_tr)
        inducing_z = coords_tr_z[ind_rows].to(device)
        log.info(f"VNNGP: {inducing_z.shape[0]:,} inducing voxels "
                 f"(of {N_tr:,} training), nn_metric={args.get('nn_metric','euclidean')}, "
                 f"k={args['nn_k']}, fitting {len(lipid_idx_to_fit)} lipids "
                 f"one at a time.")

        # ---- geodesic neighbour structure (shared across lipids; cached) ----
        geo_nn = None
        if args.get("nn_metric", "euclidean") == "geodesic":
            M = inducing_z.shape[0]
            k_geo = min(int(args["nn_k"]), M - 1)
            gk = int(args.get("geodesic_graph_k", 16))
            cache_f = out_root / f"geodesic_nn_M{M}_k{k_geo}_gk{gk}.npz"
            if cache_f.exists():
                log.info(f"VNNGP geodesic: loading cached NN structure "
                         f"{cache_f.name}")
                z = np.load(cache_f)
                seq_np, node_np = z["seq"], z["node"]
            else:
                ind_np = inducing_z.detach().cpu().numpy().astype(np.float32)
                log.info(f"VNNGP geodesic: faiss kNN graph (gk={gk}) over "
                         f"{M:,} inducing voxels + shortest-path kNN "
                         f"(k={k_geo}) …")
                ei, ev = _faiss_knn_graph(ind_np, gk)
                seq_np, node_np = _build_geodesic_nn(M, ei, ev, k_geo, log)
                savez_safe(cache_f, seq=seq_np, node=node_np)
                log.info(f"VNNGP geodesic: cached → {cache_f.name}")
            geo_nn = (torch.from_numpy(np.ascontiguousarray(seq_np)).long(),
                      torch.from_numpy(np.ascontiguousarray(node_np)).long())

        resume_mode = args.get("resume", "auto")
        metrics_rows = []
        if resume_mode == "auto" and (out_root / "metrics.csv").exists():
            try:
                prev = pd.read_csv(out_root / "metrics.csv")
                if "slug" in prev.columns:
                    metrics_rows = [
                        r for r in prev.to_dict("records")
                        if lipid_is_complete(predictions_root, str(r["slug"]))
                    ]
                    log.info(f"VNNGP resume: restored {len(metrics_rows)} "
                             f"metric row(s) from metrics.csv")
            except Exception as ex:
                log.warning(f"VNNGP resume: couldn't read metrics.csv: {ex}")

        grand_t0 = time.time()
        for j, g_idx in enumerate(lipid_idx_to_fit):
            name = lipid_names_all[g_idx]
            slug = safe_filename(name)
            desc = f"lipid {j+1}/{len(lipid_idx_to_fit)} {name}"
            if resume_mode == "auto" and lipid_is_complete(predictions_root, slug):
                log.info(f"[{desc}] SKIPPED — complete on disk (resume=auto)")
                continue
            log.info(f"[{desc}] training VNNGP")

            y_col = y_tr_z[ind_rows, j].contiguous()
            t0 = time.time()
            model, likelihood = train_one_lipid_vnngp(
                inducing_z=inducing_z, y_train_col=y_col,
                args=args, device=device, log=log, desc=desc,
                geo_nn=geo_nn,
            )
            fit_sec = time.time() - t0

            t0 = time.time()
            test_mean, test_var = predict_vnngp(model, likelihood, coords_te_z)
            brain_mean, brain_var = predict_vnngp(model, likelihood, brain_nodes_z)
            pred_sec = time.time() - t0
            log.info(f"  [{desc}] fit={fit_sec:.1f}s pred={pred_sec:.1f}s")

            mean_t = test_mean.astype(np.float32)
            std_t = np.sqrt(test_var).astype(np.float32)
            true_t = y_te_z[:, j].numpy().astype(np.float32)
            mean_b = brain_mean.astype(np.float32)
            std_b = np.sqrt(brain_var).astype(np.float32)

            cm = float(col_means[j].item())
            cs = float(col_stds[j].item())
            lip_dir = predictions_root / slug
            lip_dir.mkdir(parents=True, exist_ok=True)
            np.save(lip_dir / "test_coords_mm.npy",
                    coords_te_mm.numpy().astype(np.float32))
            np.save(lip_dir / "test_pred_z.npy", mean_t)
            np.save(lip_dir / "test_pred_raw.npy", mean_t * cs + cm)
            np.save(lip_dir / "test_std_z.npy", std_t)
            np.save(lip_dir / "test_true_z.npy", true_t)
            np.save(lip_dir / "graph_pred_z.npy", mean_b)
            np.save(lip_dir / "graph_pred_raw.npy", mean_b * cs + cm)
            np.save(lip_dir / "graph_std_z.npy", std_b)

            err = true_t - mean_t
            rmse = float(np.sqrt(np.mean(err ** 2)))
            if np.std(mean_t) < 1e-10 or np.std(true_t) < 1e-10:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(true_t, mean_t)[0, 1])
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((true_t - true_t.mean()) ** 2)) or 1.0
            metrics_rows.append({
                "lipid_global_idx": int(g_idx),
                "lipid_name": name, "slug": slug, "batch": int(j),
                "test_rmse_z": rmse, "test_corr": corr,
                "test_r2": 1.0 - ss_res / ss_tot,
                "mean_pred_std_z": float(std_t.mean()),
                "fit_sec": fit_sec,
            })
            pd.DataFrame(metrics_rows).to_csv(out_root / "metrics.csv", index=False)

            del model, likelihood
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        grand_t = time.time() - grand_t0
        log.info("=" * 72)
        log.info(f"VNNGP: all {len(lipid_idx_to_fit)} lipids done in "
                 f"{grand_t:.1f}s")
        if metrics_rows:
            df = pd.DataFrame(metrics_rows)
            with open(out_root / "summary.json", "w") as f:
                json.dump({
                    "n_lipids": int(len(df)),
                    "variational": "nngp",
                    "kernel_family": "euclidean",
                    "nn_k": int(args["nn_k"]),
                    "n_inducing": int(inducing_z.shape[0]),
                    "wall_time_sec": float(grand_t),
                    "test_corr_median": float(df["test_corr"].median(skipna=True)),
                    "test_r2_median": float(df["test_r2"].median()),
                }, f, indent=2)
            log.info(f"VNNGP summary: corr median="
                     f"{df['test_corr'].median(skipna=True):+.4f}  "
                     f"R2 median={df['test_r2'].median():+.4f}")
        if args.get("render", True):
            try:
                render_trained_lipids(
                    out_root, args, lipid_names_all, lipid_idx_to_fit, log,
                )
            except Exception as ex:
                log.warning(f"render: failed ({ex}); training outputs unaffected.")
        log.info(f"Outputs in: {out_root}")
        return

    # ---- batched training loop ----
    metrics_rows = []
    B = int(args["lipid_batch_size"])
    n_lipids = len(lipid_idx_to_fit)
    n_batches = (n_lipids + B - 1) // B

    # ---- Resume support -------------------------------------------------
    # On a clean run, both lists below are empty. On a resumed run, we
    # restore prior metrics rows AND identify which batches have already
    # produced all of their lipids' .npy predictions on disk; those
    # batches will be skipped to save GPU time. The skip granularity is
    # per-batch because a batch is one joint GP fit — half-finished
    # batches need a re-fit, not a partial salvage.
    resume_mode = args.get("resume", "auto")
    predictions_root = out_root / "predictions"
    resumed_metrics_csv = out_root / "metrics.csv"
    completed_batches = set()  # indices (into `range(n_batches)`)
    if resume_mode == "auto":
        if resumed_metrics_csv.exists():
            try:
                prev_df = pd.read_csv(resumed_metrics_csv)
                # Defensive: older runs may have written metrics.csv
                # without a 'slug' column. Derive it from lipid_name if
                # missing so backward compat with prior runs holds.
                if "slug" not in prev_df.columns:
                    if "lipid_name" in prev_df.columns:
                        prev_df["slug"] = prev_df["lipid_name"].apply(
                            safe_filename)
                    else:
                        raise ValueError(
                            "metrics.csv has neither 'slug' nor "
                            "'lipid_name' columns; cannot resume."
                        )
                # Drop any rows for lipids that aren't fully on disk —
                # those are stale entries from a crashed batch and will
                # be re-written when the batch retrains.
                completed_slugs = {
                    safe_filename(n) for n in lipid_names_all
                    if lipid_is_complete(predictions_root, safe_filename(n))
                }
                before = len(prev_df)
                prev_df = prev_df[prev_df["slug"].isin(completed_slugs)]
                # Keep only the latest row per slug (in case of duplicates
                # from previous crashed-then-restarted runs).
                prev_df = prev_df.drop_duplicates(
                    subset=["slug"], keep="last")
                metrics_rows = prev_df.to_dict("records")
                log.info(
                    f"  resume: restored {len(metrics_rows)}/{before} "
                    f"valid metric rows from {resumed_metrics_csv.name}"
                )
            except Exception as ex:
                log.warning(
                    f"  resume: couldn't read existing metrics.csv "
                    f"({ex}); starting metrics fresh"
                )
        for batch_i in range(n_batches):
            s = batch_i * B
            e = min(s + B, n_lipids)
            batch_global_ids = lipid_idx_to_fit[s:e]
            batch_names = [lipid_names_all[i] for i in batch_global_ids]
            if all(
                lipid_is_complete(predictions_root, safe_filename(name))
                for name in batch_names
            ):
                completed_batches.add(batch_i)
    if completed_batches:
        log.info(
            f"  resume: {len(completed_batches)}/{n_batches} batch(es) "
            f"already complete on disk and will be SKIPPED: "
            f"{sorted(completed_batches)}"
        )

    log.info("=" * 72)
    log.info(f"Training {n_lipids} lipids in {n_batches} batches of {B} "
             f"(or fewer for the tail).")
    log.info("=" * 72)

    grand_t0 = time.time()
    # Tracker for any lipid-batches that came back as
    # "early_stopped_from_ckpt" — we still save their predictions but
    # the whole run is marked failed at the end.
    early_stopped_batches = []
    for batch_i in range(n_batches):
        s = batch_i * B
        e = min(s + B, n_lipids)
        batch_lipid_global = lipid_idx_to_fit[s:e]
        batch_lipid_local = list(range(s, e))     # indices into y_tr_z columns
        batch_size_actual = len(batch_lipid_local)
        batch_names = [lipid_names_all[i] for i in batch_lipid_global]

        if batch_i in completed_batches:
            log.info(
                f"[batch {batch_i+1}/{n_batches}] SKIPPED — already "
                f"complete on disk ({batch_size_actual} lipids, "
                f"resume=auto)."
            )
            continue

        log.info(f"[batch {batch_i+1}/{n_batches}] lipids "
                 f"{batch_lipid_global[0]}..{batch_lipid_global[-1]} "
                 f"({batch_size_actual} tasks)")
        log.info(f"  → {', '.join(batch_names[:3])}"
                 + (", ..." if batch_size_actual > 3 else ""))

        # y_tr_z_batch stays on CPU; the DataLoader inside train_lipid_batch
        # moves minibatches to GPU asynchronously.
        y_tr_z_batch = y_tr_z[:, batch_lipid_local].contiguous()
        y_te_z_batch_np = y_te_z[:, batch_lipid_local].numpy()

        # Per-batch checkpoint path. Single file, overwritten as
        # training progresses (see _save_ckpt in train_lipid_batch).
        # Stored under "checkpoints/" so it's separate from final
        # per-lipid predictions.
        ckpt_path = out_root / "checkpoints" / f"batch_{batch_i:03d}_inprogress.pt"
        args_for_batch = dict(args, checkpoint_path=ckpt_path,
                              wandb_batch_index=batch_i)

        t0 = time.time()
        try:
            if args["kernel_family"] == "spectral":
                model, likelihood, train_status = train_lipid_batch_spectral(
                    coords_train=coords_tr_z,
                    y_train=y_tr_z_batch,
                    manifold_kernel=manifold_ctx["kernel"],
                    config=config,
                    args=args_for_batch,
                    device=args["device"],
                    log=log,
                    pbar_desc=f"batch {batch_i+1}/{n_batches}",
                )
            else:
                # 'manifold' uses the Riemann kernel; 'euclidean' and 'eigenmap'
                # take the kernel-free path (eigenmap's inputs are already the
                # eigenfunction embedding).
                mk = (manifold_ctx["kernel"]
                      if (manifold_ctx is not None
                          and args["kernel_family"] == "manifold") else None)
                model, likelihood, train_status = train_lipid_batch(
                    coords_train=coords_tr_z,
                    y_train=y_tr_z_batch,
                    inducing_points=inducing_points,
                    config=config,
                    args=args_for_batch,
                    manifold_kernel=mk,
                    parcel_field=parcel_field,
                    device=args["device"],
                    log=log,
                    pbar_desc=f"batch {batch_i+1}/{n_batches}",
                )
            if train_status == "early_stopped_from_ckpt":
                # Training diverged but we recovered the last checkpoint
                # so we can still save predictions. Mark the run as
                # failed — once the predictions land on disk, we'll
                # write FAILED.txt and abort.
                early_stopped_batches.append({
                    "batch": batch_i + 1,
                    "of_total": n_batches,
                    "lipid_names": batch_names,
                    "lipid_indices": list(batch_lipid_global),
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                })
                try:
                    with open(out_root / "ERRORS.txt", "a") as f:
                        f.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')}  "
                            f"early_stopped batch={batch_i+1}/{n_batches}  "
                            f"lipids={batch_names}  "
                            f"(predictions saved from in-progress ckpt)\n"
                        )
                except OSError:
                    pass
                log.warning(
                    f"[batch {batch_i+1}/{n_batches}] early-stopped — "
                    f"predictions will be from in-progress checkpoint"
                )
        except RuntimeError as ex:
            # Training diverged AND no checkpoint to recover from — no
            # predictions can be saved for this batch. This means the
            # divergence happened before the first --checkpoint-every-
            # epochs interval, OR the checkpoint itself was unloadable.
            # The minibatch-level failure threshold inside
            # train_lipid_batch (max_consecutive_bad_grads) already
            # exhausted, so the config is structurally broken — write
            # FAILED.txt and abort the whole run.
            failure_marker = out_root / "FAILED.txt"
            try:
                failure_marker.write_text(
                    f"Run failed at batch {batch_i+1}/{n_batches}\n"
                    f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Lipids in failing batch: {batch_names}\n"
                    f"Lipid global indices: {list(batch_lipid_global)}\n"
                    f"\n"
                    f"Error:\n{ex}\n"
                    f"\n"
                    f"No checkpoint existed when the training diverged "
                    f"— no predictions saved for this batch. Reduce "
                    f"--checkpoint-every-epochs to checkpoint earlier "
                    f"so at least partial predictions can be recovered "
                    f"next time.\n"
                    f"\n"
                    f"Full per-iter log: ERRORS.txt\n\n"
                    f"To retry: remove this file (FAILED.txt) and "
                    f"re-run with the same --exp-name. Previously-"
                    f"completed batches will be skipped via resume.\n"
                )
                log.error(
                    f"[batch {batch_i+1}/{n_batches}] FAILED, no ckpt "
                    f"— wrote {failure_marker}. Aborting."
                )
            except OSError as write_err:
                log.error(
                    f"could not write {failure_marker}: {write_err}"
                )
            try:
                with open(out_root / "ERRORS.txt", "a") as f:
                    f.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')}  "
                        f"batch_failed batch={batch_i+1}/{n_batches}  "
                        f"lipids={batch_names}\n  {ex}\n"
                    )
            except OSError:
                pass
            raise  # propagate the RuntimeError, killing the run
        fit_sec = time.time() - t0

        # Predict on test set + brain. predict_batched returns y-space
        # variance (latent f variance + obs noise from the likelihood), so the
        # std arrays we save are comparable to held-out y values.
        t0 = time.time()
        test_mean_z, test_var_z = predict_batched(
            model, likelihood, coords_te_z, n_tasks=batch_size_actual,
        )
        # faiss_cpu_recon() pins KNN searches to CPU iff --faiss-cpu-recon is set.
        with faiss_cpu_recon():
            brain_mean_z, brain_var_z = predict_batched(
                model, likelihood, brain_nodes_z, n_tasks=batch_size_actual,
            )
        pred_sec = time.time() - t0
        log.info(f"  fit={fit_sec:.1f}s pred={pred_sec:.1f}s")

        # Save state_dict + likelihood state (one file per batch).
        # The likelihood's per-task task_noises IS the observation noise model.
        torch.save({
            "model_state": model.state_dict(),
            "likelihood_state": likelihood.state_dict(),
            "lipid_global_idx": batch_lipid_global,
            "lipid_names": batch_names,
            "n_tasks": batch_size_actual,
            "kernel_family": args["kernel_family"],
            "args": args,
        }, out_root / "checkpoints" / f"batch_{batch_i:03d}.pt")

        # Per-lipid splits + metrics
        for k, (g_idx, name) in enumerate(zip(batch_lipid_global, batch_names)):
            slug = safe_filename(name)
            lip_dir = out_root / "predictions" / slug
            lip_dir.mkdir(exist_ok=True)

            mean_t = test_mean_z[:, k].astype(np.float32)
            std_t = np.sqrt(test_var_z[:, k]).astype(np.float32)
            true_t = y_te_z_batch_np[:, k].astype(np.float32)

            mean_b = brain_mean_z[:, k].astype(np.float32)
            std_b = np.sqrt(brain_var_z[:, k]).astype(np.float32)

            # Save z + raw (de-standardize using the per-lipid mean/std)
            cm = float(col_means[k + s].item())
            cs = float(col_stds[k + s].item())
            np.save(lip_dir / "test_coords_mm.npy",
                    coords_te_mm.numpy().astype(np.float32))
            np.save(lip_dir / "test_pred_z.npy", mean_t)
            np.save(lip_dir / "test_pred_raw.npy", mean_t * cs + cm)
            np.save(lip_dir / "test_std_z.npy", std_t)
            np.save(lip_dir / "test_true_z.npy", true_t)
            np.save(lip_dir / "graph_pred_z.npy", mean_b)
            np.save(lip_dir / "graph_pred_raw.npy", mean_b * cs + cm)
            np.save(lip_dir / "graph_std_z.npy", std_b)

            # Per-lipid metrics on z-scored test predictions
            err = true_t - mean_t
            rmse = float(np.sqrt(np.mean(err ** 2)))
            if np.std(mean_t) < 1e-10 or np.std(true_t) < 1e-10:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(true_t, mean_t)[0, 1])
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((true_t - true_t.mean()) ** 2)) or 1.0
            r2 = 1.0 - ss_res / ss_tot

            metrics_rows.append({
                "lipid_global_idx": int(g_idx),
                "lipid_name": name,
                "slug": slug,
                "batch": int(batch_i),
                "test_rmse_z": rmse,
                "test_corr": corr,
                "test_r2": r2,
                "mean_pred_std_z": float(std_t.mean()),
                "fit_sec": fit_sec / batch_size_actual,  # amortised
            })

        # Flush after each batch (a long run can be inspected partway through)
        pd.DataFrame(metrics_rows).to_csv(
            out_root / "metrics.csv", index=False,
        )

        # The batch is fully complete (.npy files written, metrics
        # flushed). The in-progress checkpoint is no longer needed —
        # remove it so we don't accumulate stale checkpoints on S3.
        try:
            if ckpt_path.exists():
                ckpt_path.unlink()
        except OSError as ex:
            log.warning(f"  cleanup of {ckpt_path.name} failed: {ex}")

        # Free memory before the next batch
        del model, likelihood, test_mean_z, test_var_z
        del brain_mean_z, brain_var_z, y_tr_z_batch
        torch.cuda.empty_cache() if args["device"].startswith("cuda") else None

        # If this batch was early-stopped (training diverged but we
        # recovered from a checkpoint), the predictions ARE on disk
        # now — but the run is failed. Write FAILED.txt and abort
        # AFTER the save completes so the partial-but-useful
        # predictions are preserved.
        if early_stopped_batches and early_stopped_batches[-1]["batch"] == batch_i + 1:
            b = early_stopped_batches[-1]
            failure_marker = out_root / "FAILED.txt"
            try:
                failure_marker.write_text(
                    f"Run failed: batch {b['batch']}/{b['of_total']} "
                    f"diverged during training.\n"
                    f"Timestamp: {b['timestamp']}\n"
                    f"Lipids in failing batch: {b['lipid_names']}\n"
                    f"Lipid global indices: {b['lipid_indices']}\n"
                    f"\n"
                    f"The minibatch failure threshold "
                    f"(--max-consecutive-bad-grads) was hit during "
                    f"SGD. Training was aborted, but we recovered the "
                    f"last in-progress checkpoint, so predictions for "
                    f"this batch ARE in predictions/<slug>/ — they "
                    f"reflect the model's state at the last checkpoint "
                    f"epoch BEFORE divergence (useful diagnostic, not "
                    f"fully trained).\n\n"
                    f"Full per-iter log: ERRORS.txt\n\n"
                    f"To retry: remove this file (FAILED.txt) and "
                    f"re-run with the same --exp-name. Successful "
                    f"earlier batches will be skipped via resume; "
                    f"this batch will be re-attempted.\n"
                )
                log.error(
                    f"WROTE FAILED.txt — predictions from in-progress "
                    f"checkpoint were preserved. Aborting."
                )
            except OSError as write_err:
                log.error(
                    f"could not write {failure_marker}: {write_err}"
                )
            raise RuntimeError(
                f"Aborting: batch {b['batch']}/{b['of_total']} "
                f"diverged; predictions saved from checkpoint."
            )

    grand_t = time.time() - grand_t0
    log.info("=" * 72)
    log.info(f"All {n_lipids} lipids trained in {grand_t:.1f}s "
             f"({grand_t / max(n_lipids, 1):.1f}s/lipid avg)")
    log.info("=" * 72)

    # ---- summary ----
    df = pd.DataFrame(metrics_rows)
    summary = {
        "n_lipids": int(len(df)),
        "n_batches": int(n_batches),
        "lipid_batch_size": int(B),
        "kernel_family": args["kernel_family"],
        "wall_time_sec": float(grand_t),
        "test_rmse_z": {
            "mean": float(df["test_rmse_z"].mean()),
            "median": float(df["test_rmse_z"].median()),
        },
        "test_corr": {
            "mean": float(df["test_corr"].mean(skipna=True)),
            "median": float(df["test_corr"].median(skipna=True)),
        },
        "test_r2": {
            "mean": float(df["test_r2"].mean()),
            "median": float(df["test_r2"].median()),
        },
        "hypers": {k: args[k] for k in
                   ("nu", "num_inducing", "learning_rate", "epochs",
                    "batch_size", "lipid_batch_size") if k in args},
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary: test corr mean={summary['test_corr']['mean']:+.4f} "
             f"median={summary['test_corr']['median']:+.4f}")
    log.info(f"         test RMSE(z) mean={summary['test_rmse_z']['mean']:.4f}")
    log.info(f"Outputs in: {out_root}")

    # ---- whole-brain renders (same output as lgp_manifold_experiment) ----
    if args.get("render", True):
        try:
            render_trained_lipids(
                out_root, args, lipid_names_all, lipid_idx_to_fit, log,
            )
        except Exception as ex:
            log.warning(f"render: failed ({ex}); training outputs unaffected.")

    if args.get("wandb", False):
        import wandb
        wandb.summary.update({
            "test_corr_mean": summary["test_corr"]["mean"],
            "test_corr_median": summary["test_corr"]["median"],
            "test_rmse_z_mean": summary["test_rmse_z"]["mean"],
            "test_r2_mean": summary["test_r2"]["mean"],
            "wall_time_sec": summary["wall_time_sec"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()