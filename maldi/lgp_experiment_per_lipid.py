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
from l3di.lgp_manifold import LatentRiemannGP
from utils import (
    get_inducing_points,
    get_data_inducing_points,
    crop_or_stride_volume,
    reference_ccf_from_subvolume,
)

# Manifold kernel + graph stack. Imports kept identical to
# ``lgp_manifold_experiment.py`` for compatibility.
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key,
)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
)
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel


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
    p.add_argument("--kernel-family", choices=["euclidean", "manifold"],
                   required=True,
                   help="'euclidean' → IndependentMultitaskGPModel "
                        "(reuse from l3di.lgp), 'manifold' → "
                        "LatentRiemannGP (reuse from l3di.lgp_manifold).")

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

    # ---- Manifold-only knobs (ignored when --kernel-family=euclidean) ----
    p.add_argument("--eigenvector-dir", default=None,
                   help="Required when --kernel-family=manifold.")
    p.add_argument("--knn-method", default="faiss",
                   choices=["faiss", "anatomical_atlas",
                            "faiss_atlas_weighted"],
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
                         "collapses for inflated d²."))
    p.add_argument("--cross-region-inflation", type=float, default=10.0,
                   help=("For --knn-method=faiss_atlas_weighted only. "
                         "Multiplier applied to squared Euclidean "
                         "distance on edges that connect two atlas "
                         "regions. Default 10 = mild prior; try 100 "
                         "for a strong prior, 1 for none. The "
                         "underlying graph topology is identical to "
                         "pure faiss; only the edge weights change."))
    p.add_argument("--laplacian-norm", default="randomwalk",
                   choices=["symmetric", "randomwalk"])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--n-list", type=int, default=1)
    p.add_argument("--graphbandwidth-init", type=float, default=0.1)
    p.add_argument("--bump-scale", type=float, default=20.0)
    p.add_argument("--bump-decay", type=float, default=0.01)
    p.add_argument("--num-modes", type=int, default=1300)
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
                         "direction. Default 0.01 (1% of params). Set "
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
        "nlist": args["n_list"],
        "bbox": None,   # kept (always None) so existing graph/eig cache keys match
    }
    if args["knn_method"] == "anatomical_atlas":
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
    elif args["knn_method"] == "faiss_atlas_weighted":
        # The base graph is pure faiss — we reuse that cache. The atlas
        # weighting is applied AFTER loading, so we only need to encode
        # the inflation factor in the eigenvector key (the kNN graph
        # cache key uses "faiss" so it's shareable with vanilla faiss
        # runs of the same k / stride / threshold).
        pass
    graph_key = make_graph_key(graph_key_parts)

    # Build / load KNN graph
    if args["knn_method"] == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss",
            coords=reference_nodes,
            k=args["knn_k"], nlist=args["n_list"],
            extra=graph_key_parts, device=args["device"],
        )
    elif args["knn_method"] == "anatomical_atlas":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas",
            volume=sub_volume, threshold=args["threshold"],
            atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes,
            k=args["knn_k"], nlist=args["n_list"],
            extra=graph_key_parts, device=args["device"],
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
            k=args["knn_k"], nlist=args["n_list"],
            extra=base_key_parts, device=args["device"],
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
        if node_labels.shape[0] != knn.x.shape[0]:
            raise RuntimeError(
                f"Mismatch between labelled voxels ({node_labels.shape[0]}) "
                f"and graph nodes ({knn.x.shape[0]}). The atlas and "
                f"reference template were probably cropped/strided "
                f"differently — check that they have the same shape."
            )

        # ---- 3. Inflate cross-region edges via library function -----
        # Topology is unchanged; only edge_value gets reweighted.
        inflation = float(args.get("cross_region_inflation", 10.0))
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, node_labels,
            inflation=inflation, treat_zero_as_cross=True,
        )

        # ---- 4. Re-key the eigenvector cache -----------------------
        # Same graph topology but different edge weights → different
        # Laplacian → different eigenmodes. Encode the inflation in
        # graph_key so the eigvec cache doesn't collide with vanilla
        # faiss runs.
        graph_key_parts["weighting"] = f"atlas_x{inflation:g}"
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"unknown knn_method: {args['knn_method']}")

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
    ncv_min = max(1500, 3 * args["num_modes"] + 20)
    solver = LaplacianEigensolver(
        num_modes=args["num_modes"], backend="cupy",
        tol=1e-4, ncv_min=ncv_min, verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op, cache_dir=eigvec_cache_dir, key=eigvec_key,
        graphbandwidth=args["graphbandwidth_init"],
        laplacian_normalization=args["laplacian_norm"],
        extra=eigvec_key_parts, device=args["device"],
    )

    # Kernel
    manifold_kernel = RiemannMaternKernel(
        nu=args["nu"], knn=knn, edge_index=edge_index, edge_value=edge_value,
        eigval=eigval, eigvec=eigvec,
        nearest_neighbors=args["knn_k"], num_modes=args["num_modes"],
        bump_scale=args["bump_scale"], bump_decay=args["bump_decay"],
        laplacian_normalization=args["laplacian_norm"],
        graphbandwidth_init=args["graphbandwidth_init"],
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


def train_lipid_batch(
    *,
    coords_train: torch.Tensor,   # (N, 3) on device, already z-scored
    y_train: torch.Tensor,        # (N, B)   B = lipid batch size
    inducing_points: torch.Tensor,
    config: MaldiConfig,
    args: dict,
    manifold_kernel=None,         # if None → Euclidean path
    device: str,
    log: logging.Logger,
    pbar_desc: str,
):
    """Build one multitask model with num_tasks=B, run variational SGD,
    return the trained model + per-task log-variance.

    Loss formulation mirrors ``LGP.loss_function`` / ``ManifoldLGP.loss_function``
    in the existing l3di code base, but uses the *analytic* expected-
    log-likelihood form rather than rsample-then-MC. Two things differ:

      1. For a Gaussian observation model (which is what LGP's
         ``log_var_n`` defines), the expected log-likelihood under
         q(f|x) has a closed form:
             E_q[(y - f)²] = (y - E_q[f])² + Var_q[f]
             recon = 0.5 * sum_n ( ((y - mean)² + var) / σ² + log σ² )
         where mean = q(f|x).mean, var = q(f|x).variance (diagonal),
         and σ² = exp(log_var_n). This is EXACT for the expectation, not
         an MC estimator — gradients have zero MC variance, which gives
         faster convergence than the rsample version used by LGP.
         The LGP framework can't use this because its MLP decoder
         breaks the closed form; we can because we go straight from
         the GP to the Gaussian loss.
      2. The KL is computed via
         ``model.variational_strategy.kl_divergence().sum()``
         (summed over tasks), as in LGP. No likelihood object is
         constructed because the noise lives in ``log_var_n``, not in
         a ``GaussianLikelihood``.

    Why not use ``gpytorch.mlls.VariationalELBO`` directly? It would do
    the same math, but it requires a ``GaussianLikelihood`` object that
    duplicates ``log_var_n``'s role. Inlining the formula keeps
    ``log_var_n`` as the single source of truth for noise (matching the
    LGP framework's design) while still getting the closed-form speedup.
    """
    n_tasks = y_train.shape[1]
    n_train = y_train.shape[0]

    if manifold_kernel is None:
        # Euclidean — same as run_final.sh's setup.
        voxel_size = 0.025
        # Equivalent to lgp_experiment.minimal_length_scale
        minimal_length_scale = config.n_pixels * voxel_size / 3.0
        model = IndependentMultitaskGPModel(
            inducing_points=inducing_points,
            num_tasks=n_tasks,
            kernel_type=config.kernel,
            nu=config.nu,
            minimal_length_scale=minimal_length_scale,
            input_dim=3,
        ).to(device)
    else:
        model = LatentRiemannGP(
            inducing_points=inducing_points,
            num_tasks=n_tasks,
            manifold_kernel=manifold_kernel,
        ).to(device)

    # Per-task learnable log-variance — exactly LGP.log_var_n with p = n_tasks.
    # Initialised to zero (variance = 1) to match LGP's nn.Parameter(torch.zeros(p)).
    log_var_n = torch.nn.Parameter(torch.zeros(n_tasks, device=device))

    model.train()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [log_var_n],
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
                with torch.no_grad():
                    log_var_n.copy_(ckpt["log_var_n"].to(device))
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
    # Clamp log_var_n to a sane range. The natural scale for z-scored
    # outputs is variance ~1 (so log_var_n ~0). Values <-10 mean noise
    # variance of <5e-5 — far below numerical precision for typical
    # mini-batch residuals, where (y-z)² / exp(-10) overflows fast.
    # Bounds for log observation noise variance on z-scored outputs (unit
    # variance by construction). -5.0 → σ²≈0.007 (tight but reachable);
    # +1.5 → σ²≈4.5 (allows genuinely noisy lipids). The old [-8, 4]
    # range let AdamW momentum drive log_var_n to exp(-8)≈3e-4, making
    # the recon term explode on any nonzero residual early in training.
    log_var_n_min, log_var_n_max = -5.0, 1.5

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
        log_var_n, current iter, epoch, and last losses. Atomic via
        write-to-temp-then-rename so a SIGKILL mid-save doesn't leave
        a corrupt checkpoint."""
        if ckpt_path is None:
            return
        try:
            tmp = Path(str(ckpt_path) + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "log_var_n": log_var_n.detach().cpu(),
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

                    # ---- Analytic expected log-likelihood --------------------
                    # For a Gaussian observation model with noise σ² = exp(log_var_n[t]),
                    # the expected log-lik under q(f|x) has a closed form that
                    # does NOT need samples:
                    #
                    #   E_q[(y - f)²]      = (y - E_q[f])² + Var_q[f]
                    #                      = (y - mean)²  + variance
                    #
                    #   E_q[-log p(y|f)]   = 0.5 * [ ((y-mean)² + variance) / σ²
                    #                                 + log(2π σ²) ]
                    #
                    # Dropping the y-independent 0.5 * log(2π) (constant, no
                    # gradient) leaves us with the LGP-style nll_loss but
                    # with `var` taking the place of the MC noise from a
                    # single rsample. This is *exact* for the expectation,
                    # not an estimator — gradients have zero MC variance.
                    # The LGP framework can't use this because its MLP
                    # decoder breaks the closed form; we can because we go
                    # straight from GP to Gaussian loss.
                    #
                    # gp_posterior is MultitaskMVN with shape (bs, n_tasks).
                    # .mean and .variance are both (bs, n_tasks). .variance
                    # extracts ONLY the diagonal of the posterior covariance,
                    # which avoids the full Cholesky that rsample requires
                    # — that's where the per-iter speedup comes from.
                    mean_f = gp_posterior.mean
                    var_f = gp_posterior.variance.clamp(min=0)
                    inv_sigma2 = torch.exp(-log_var_n).unsqueeze(0)  # (1, n_tasks)
                    recon = 0.5 * torch.sum(
                        ((y_b - mean_f).pow(2) + var_f) * inv_sigma2
                        + log_var_n.unsqueeze(0)
                    )

                    # ---- KL from the variational strategy ----
                    kl = model.variational_strategy.kl_divergence().sum()

                    # ELBO upweighting: the recon was summed over a mini-batch of
                    # bs points, but the KL is exact (total over all data). To
                    # keep the gradient unbiased w.r.t. the full-data ELBO, we
                    # rescale either the KL down by bs/n_train or scale recon up
                    # by n_train/bs. The LGP scripts use the latter implicitly
                    # by accumulating recon across all batches in an epoch and
                    # adding the KL once per batch — we just inline the
                    # equivalent: minimize (n_train/bs) * recon + kl.
                    loss = (n_train / bs) * recon + kl

                    # ---- Loss-NaN: treat same as any forward failure ----
                    # The forward + posterior-NaN check upstream already
                    # catches most kernel ill-conditioning. A NaN here is
                    # rarer — usually log_var_n drift or KL overflow. Route
                    # it through the same bad-grad streak counter so it
                    # contributes to the bail-out logic uniformly.
                    if not torch.isfinite(loss):
                        bad_grad_count += 1
                        _record_error(
                            config, "loss_nan", it,
                            f"loss={loss.item()}, "
                            f"recon={recon.item():.3g}, "
                            f"kl={kl.item():.3g}, "
                            f"lvn=[{float(log_var_n.min().item()):.2f},"
                            f"{float(log_var_n.max().item()):.2f}]",
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
                        list(model.named_parameters()) + [("log_var_n", log_var_n)]
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
                        for p in list(model.parameters()) + [log_var_n]:
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

                    # ---- gradient clipping ---------------------------------
                    # After NaN sanitization, gradients are finite — clipping
                    # caps any remaining large-but-finite values. Helps with
                    # the analytic ELBO's large gradients at init.
                    grad_clip = float(args.get("grad_clip", 10.0))
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + [log_var_n],
                        max_norm=grad_clip,
                    )

                    optimizer.step()
                    # Clamp log_var_n back into a sane range so the noise term
                    # can't explode/underflow. Done after the step so AdamW's
                    # update can move freely, then we project.
                    with torch.no_grad():
                        log_var_n.clamp_(log_var_n_min, log_var_n_max)

                    # ---- periodic log line (cluster-friendly) -----------------
                    # Visible even when tqdm output is suppressed (non-tty, log
                    # capture, runai, etc). Includes recon / KL / log_var_n
                    # range / elapsed / ETA so a cluster job log gives a
                    # full picture.
                    if it % log_every == 0 or it == n_iters - 1:
                        elapsed = time.time() - fit_t0
                        done = it + 1
                        rate = done / max(elapsed, 1e-9)
                        eta = (n_iters - done) / max(rate, 1e-9)
                        lvn_min = float(log_var_n.min().item())
                        lvn_max = float(log_var_n.max().item())
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
                        pbar.set_postfix(
                            recon=f"{last_recon:.3g}",
                            kl=f"{last_kl:.3g}",
                            lvn=f"[{log_var_n.min().item():.2f},"
                                f"{log_var_n.max().item():.2f}]",
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
                        with torch.no_grad():
                            log_var_n.copy_(ckpt["log_var_n"].to(device))
                        log.warning(
                            f"  [{pbar_desc}] recovered from "
                            f"{Path(ckpt_path).name} @ iter "
                            f"{ckpt.get('iter', 0)} / epoch "
                            f"{ckpt.get('epoch', 0)} — predictions will "
                            f"use this state (early stop). Caller still "
                            f"counts this as a failed batch."
                        )
                        model.eval()
                        return model, log_var_n.detach(), "early_stopped_from_ckpt"
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
    return model, log_var_n.detach(), "ok"


def predict_batched(model, log_var_n, x: torch.Tensor, n_tasks: int,
                    chunk: int = 20_000):
    """Forward in chunks. Returns (mean, var) of the **posterior over
    the latent f**, NOT the predictive over y. We then add the per-task
    observation noise back in at the end.

    Why split f vs y? Because the LGP / ManifoldLGP framework doesn't
    use a likelihood object — the observation noise lives in
    ``log_var_n``. To get a predictive variance that's directly
    comparable to held-out y values, we add ``exp(log_var_n)`` to the
    latent variance (i.e., predictive variance = q(f|x).variance +
    obs_noise).
    """
    means, vars_f = [], []
    n = x.shape[0]
    obs_var = torch.exp(log_var_n).to(x.device)  # (n_tasks,)
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

    # ---- manifold setup (only if needed) ----
    manifold_ctx = None
    if args["kernel_family"] == "manifold":
        if args.get("eigenvector_dir") is None:
            log.error("--eigenvector-dir is required for --kernel-family=manifold")
            sys.exit(2)
        log.info("Building manifold kernel (graph + eigenvectors) …")
        manifold_ctx = setup_manifold_kernel(
            args, config, coord_mean, coord_std, log,
        )
        # Snap the k-means inducing points to the nearest graph nodes so
        # that every K_uu evaluation uses the exact eigenvector lookup
        # branch (is_on_graph) rather than the numerically fragile Nyström
        # OOS path. Mirrors lgp_manifold_experiment.setup_experiment().
        with torch.no_grad():
            knn = manifold_ctx["knn"]
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
        args_for_batch = dict(args, checkpoint_path=ckpt_path)

        t0 = time.time()
        try:
            model, log_var_n, train_status = train_lipid_batch(
                coords_train=coords_tr_z,
                y_train=y_tr_z_batch,
                inducing_points=inducing_points,
                config=config,
                args=args_for_batch,
                manifold_kernel=(manifold_ctx["kernel"]
                                 if manifold_ctx is not None else None),
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
        # variance (latent f variance + obs noise from log_var_n), so the
        # std arrays we save are comparable to held-out y values.
        t0 = time.time()
        test_mean_z, test_var_z = predict_batched(
            model, log_var_n, coords_te_z, n_tasks=batch_size_actual,
        )
        brain_mean_z, brain_var_z = predict_batched(
            model, log_var_n, brain_nodes_z, n_tasks=batch_size_actual,
        )
        pred_sec = time.time() - t0
        log.info(f"  fit={fit_sec:.1f}s pred={pred_sec:.1f}s")

        # Save state_dict + log_var_n (one file per batch).
        # log_var_n IS the noise model — analogous to a GaussianLikelihood's
        # noise parameter but per-task and outside the gpytorch object.
        torch.save({
            "model_state": model.state_dict(),
            "log_var_n": log_var_n.cpu(),  # (n_tasks,)
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
        del model, log_var_n, test_mean_z, test_var_z
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


if __name__ == "__main__":
    main()