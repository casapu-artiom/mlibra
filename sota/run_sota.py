"""SOTA 3D-reconstruction models for MALDI, sharing the baselines harness.

This driver plugs the coordinate-regression SOTA models into the *exact same*
data-loading / normalization / reconstruction / render / metrics pipeline that
``maldi/experiment_baselines.py`` (and hence the manifold GP runs) use -- so
their whole-brain renders, per-lipid metrics.csv and held-out diagnostics are
directly comparable to run_manifold / run_baseline.

Models (``--model``):
    ntf    Neural Transcriptomic Field (hash-grid INR)      -> sota/ntf_model.py
    spa3d  Spatial-pattern-enhanced GCN (SPE + z-aware GCN)  -> sota/spa3d_model.py

DeepSpatial is NOT here: it is a within-specimen slice-interpolation method, not
a coordinate regressor, so it does not fit this harness. It lives as a separate
faithful implementation in ``sota/deepspatial_transport/`` (run via
``MODEL=deepspatial ./sota/run_sota.sh``, which delegates there).

It reuses ``experiment_baselines.main`` unchanged (all the reconstruction /
render / metrics logic) by (a) registering the model classes into its
``MODEL_REGISTRY`` and (b) swapping in a ``parse_args`` that accepts the extra
``--model`` choices and each model's hyper-parameters. Everything else --
splits, log-transform, per-lipid normalization, whole-brain voxel
reconstruction, renders, diagnostics -- is identical to the baselines.
"""
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

# experiment_baselines lives in ../maldi and imports its siblings (config, utils,
# manifold_kernel_builder, ...) by bare module name -> put maldi on sys.path.
_MALDI_DIR = Path(__file__).resolve().parent.parent / "maldi"
sys.path.insert(0, str(_MALDI_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_baselines as eb          # noqa: E402
from manifold_kernel_builder import add_manifold_args  # noqa: E402

from ntf_model import NTFModel             # noqa: E402
from spa3d_model import Spa3DModel         # noqa: E402

SOTA_MODELS = {
    "ntf": NTFModel,
    "spa3d": Spa3DModel,
}


def parse_args():
    """Mirror ``experiment_baselines.parse_args`` + the SOTA model choices/knobs.

    Kept in sync with the baselines parser for the shared flags so that
    ``MaldiConfig.from_args`` and ``experiment_baselines.main`` see everything
    they expect; the SOTA-specific groups are appended at the end.
    """
    parser = ArgumentParser(description="SOTA 3D-reconstruction models for MALDI.")
    # --- shared flags (one-for-one with experiment_baselines.parse_args) ---
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    parser.add_argument("--maldi-file", dest="maldi_file", type=str, required=True)
    parser.add_argument("--exp-name", dest="exp_name", type=str, required=True)
    parser.add_argument("--available-lipids-file", dest="available_lipids_file",
                        type=str, required=True)
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--slices-dataset-file", dest="slices_dataset_file",
                        type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-transform", dest="log_transform", action="store_true")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4096)
    parser.add_argument("--load-args", dest="load_args", action="store_true")
    parser.add_argument("--use-diffusion", dest="use_diffusion", action="store_true")
    parser.add_argument("--region-bbox", dest="region_bbox", type=int, nargs=6,
                        default=None,
                        metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"))
    # MaldiConfig wants these
    parser.add_argument("--num-inducing", dest="num_inducing", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=10)
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-3)
    # model selection
    parser.add_argument("--model", type=str, default="ntf",
                        choices=list(SOTA_MODELS.keys()))
    # reconstruction / rendering
    parser.add_argument("--template-name", dest="template_name", type=str, required=True)
    parser.add_argument("--reference-file", dest="reference_file", type=str, required=True)
    parser.add_argument("--annotations-file", dest="annotations_file", type=str)
    parser.add_argument("--reconstruct", type=str, default="auto",
                        choices=["none", "auto", "whole_brain", "region"])
    parser.add_argument("--reconstruct-threshold", type=float, default=5.0)
    # Mirror experiment_baselines: reconstruction forward-pass batch (not training)
    # and the render-voxels-only sparse reconstruction. Kept here because eb.main()
    # uses THIS parser (run_sota rebinds eb.parse_args), so a flag missing here is a
    # KeyError in eb.main, not just an unavailable option.
    parser.add_argument("--inference-batch-size", dest="inference_batch_size",
                        type=int, default=65536)
    parser.add_argument("--render-voxels-only", dest="render_voxels_only",
                        action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    # Early stopping (best-checkpoint on a val split carved from TRAIN; the
    # held-out test set is never used for model selection). Shared by all models.
    parser.add_argument("--val-frac", dest="val_frac", type=float, default=0.05,
                        help="Fraction of TRAIN held out as validation for early "
                             "stopping (0 disables; test stays clean either way).")
    parser.add_argument("--early-stop-patience", dest="early_stop_patience",
                        type=int, default=5,
                        help="Epochs without improvement before stopping; the "
                             "best weights are always restored (0 = keep best, "
                             "never terminate early).")
    parser.add_argument("--early-stop-monitor", dest="early_stop_monitor",
                        choices=["val", "test"], default="val",
                        help="Metric for best-checkpoint. 'val' (train-carved, no "
                             "leak) catches ordinary over-training but NOT the "
                             "cross-mouse overfit of the fold splits (val is the "
                             "same brains). 'test' picks the best held-out-mouse "
                             "epoch -- fixes the rising-test-MSE case, at the cost "
                             "of selecting the epoch on the test set.")
    parser.add_argument("--reconstruction-lipids", dest="reconstruction_lipids",
                        nargs="+", default=None)
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str,
                        default="sota_maldi", help="W&B project name.")
    # manifold flags (unused by the SOTA models, kept so shared launch scripts
    # and MaldiConfig do not choke on them)
    add_manifold_args(parser)

    # --- NTF knobs ---
    g = parser.add_argument_group("NTF")
    g.add_argument("--ntf-levels", type=int, default=16)
    g.add_argument("--ntf-features", type=int, default=2)
    g.add_argument("--ntf-log2-hashmap", type=int, default=19)
    g.add_argument("--ntf-base-res", type=int, default=16)
    g.add_argument("--ntf-max-res", type=int, default=256,
                   help="Finest hash-grid resolution. Kept moderate: on a sparse, "
                        "cross-mouse split too high a value lets the grid memorise "
                        "per-voxel detail and hurts held-out generalization.")
    g.add_argument("--ntf-hidden", type=int, nargs="+", default=[128, 128])
    g.add_argument("--ntf-grid-lr", type=float, default=1e-2,
                   help="Dedicated LR for the hash-grid embeddings (InstantNGP "
                        "recipe): much higher than the MLP LR, with a tiny Adam eps "
                        "and no weight decay, so the sparse grid gradients actually "
                        "move the near-zero-init tables off the constant field.")
    g.add_argument("--ntf-tv-weight", type=float, default=0.01,
                   help="Spatial smoothness / TV regulariser weight.")
    g.add_argument("--ntf-tv-eps", type=float, default=0.01,
                   help="Std of the coord perturbation for the TV term.")
    g.add_argument("--ntf-max-sections", type=int, default=256,
                   help="Max number of z (coronal) section bins for the bias.")
    g.add_argument("--ntf-zero-inflation", action="store_true")
    g.add_argument("--ntf-weight-decay", type=float, default=0.0,
                   help="Adam weight decay (L2) on the MLP heads only. The hash "
                        "embeddings are explicitly excluded (weight decay collapses "
                        "them to a constant field); regularise the grid via "
                        "--ntf-max-res and --ntf-tv-weight instead.")
    # ---- ported from the official NTF models.py ----
    g.add_argument("--ntf-features-z", type=int, default=16,
                   help="Latent-z width feeding the variance (sigma) net.")
    g.add_argument("--ntf-features-slice", type=int, default=8,
                   help="Slice-embedding width (shared by bias + variance nets).")
    g.add_argument("--ntf-levels-bias", type=int, default=4,
                   help="Low-frequency hash levels feeding the bias net (0=off).")
    g.add_argument("--ntf-aux-hidden", type=int, default=64,
                   help="Hidden width of the bias / variance nets.")
    g.add_argument("--ntf-bias-weight", type=float, default=0.01,
                   help="L2 regulariser weight on the (log-)section bias.")
    g.add_argument("--ntf-psf-samples", type=int, default=4,
                   help="PSF Monte-Carlo samples averaged before the loss (1=off).")
    g.add_argument("--ntf-psf-sigma", type=float, default=0.01,
                   help="PSF std in standardized-coord units (0=off).")

    # --- Spa3D knobs ---
    g = parser.add_argument_group("Spa3D")
    g.add_argument("--spa3d-spe", type=str, default="alft",
                   choices=["none", "hilbert", "alft"],
                   help="Spatial pattern enhancement operator.")
    g.add_argument("--spa3d-grid", type=int, default=128,
                   help="Per-section rasterization grid for SPE.")
    g.add_argument("--spa3d-sections", type=int, default=64,
                   help="Number of z (coronal) section bins for SPE.")
    g.add_argument("--spa3d-alft-keep", type=float, default=0.1,
                   help="ALFT: fraction of wavenumber coefficients to keep.")
    g.add_argument("--spa3d-z-weight", type=float, default=1.0,
                   help="Scale on the z (inter-slice) axis in the graph metric.")
    g.add_argument("--spa3d-knn-k", type=int, default=15)
    g.add_argument("--spa3d-graph-nodes", type=int, default=80000,
                   help="(spa3d) node count of the single global Gaussian graph "
                        "(fixed subsample of train coords; O(N^2) build cost).")
    g.add_argument("--spa3d-length-scale", type=float, default=0.0,
                   help="(spa3d) Gaussian-affinity bandwidth l in exp(-d^2/2l^2); "
                        "<=0 uses the median-KNN-distance heuristic.")
    g.add_argument("--spa3d-interp-k", type=int, default=8,
                   help="(spa3d) nearest graph nodes blended by inverse-distance "
                        "interpolation at read-out.")
    g.add_argument("--spa3d-hidden", type=int, nargs="+", default=[256, 256, 128])
    g.add_argument("--spa3d-dropout", type=float, default=0.1)

    args = vars(parser.parse_args())
    if args.get("reconstruction_lipids"):
        try:
            args["reconstruction_lipids"] = [int(v) for v in args["reconstruction_lipids"]]
        except ValueError:
            pass
    return args


def main():
    # Register the SOTA models into the shared harness and reuse its main()
    # verbatim (reconstruction / render / metrics parity).
    eb.MODEL_REGISTRY.update(SOTA_MODELS)
    eb.parse_args = parse_args        # eb.main() looks this up as a module global

    # Optionally open a W&B run BEFORE eb.main() so the models' wandb_log() calls
    # (guarded by an active run) log during fit; close it after.
    args = parse_args()
    run = None
    if args.get("wandb"):
        try:
            import wandb
            run = wandb.init(project=args.get("wandb_project", "sota_maldi"),
                             name=args["exp_name"], config=args)
            logging.info(f"W&B enabled (project={args.get('wandb_project')}, "
                         f"model={args['model']})")
        except Exception as e:  # noqa: BLE001
            logging.warning(f"wandb init failed ({e}); continuing without it")
    try:
        eb.main()
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
