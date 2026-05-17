"""This script sets up and runs a MALDI experiment using the l3di library.

This is the Euclidean baseline — the GP uses a vanilla Matern/RBF kernel on
standardized 3D CCF coordinates. Pair it with `lgp_manifold_experiment.py`
(the Riemann manifold version) for the kernel comparison.
"""
from l3di.lgp import LGP, IndependentMultitaskGPModel
from experiment import MaldiExperiment
from config import MaldiConfig
from argparse import ArgumentParser
from utils import (
    get_inducing_points,
    get_bbox_inducing_points,
    apply_region_to_config,
)
import logging


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run MALDI experiment with l3di.")
    parser.add_argument("--mode", type=str, required=True, help="Experiment mode (e.g., 'train', 'test').")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi-file", dest="maldi_file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--exp-name", dest="exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available-lipids-file", dest="available_lipids_file", type=str, required=True, help="File with available lipids.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--slices-dataset-file", dest="slices_dataset_file", type=str, required=True, help="File for slices dataset.")
    parser.add_argument("--num-inducing", dest="num_inducing", type=int, default=100, help="Number of inducing points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=10, help="Dimensionality of the latent space.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on (e.g., 'cpu', 'cuda').")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for the GP model.")
    parser.add_argument("--log-transform", dest="log_transform", action='store_true', help="Apply log transformation to the data.")
    parser.add_argument("--nu", type=float, default=1.5, help="Parameter for the GP model.")
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10, help="Number of pixels to consider in the experiment.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2000, help="Batch size for training")
    parser.add_argument("--load-args", dest="load_args", action='store_true', help="Load arguments from a file instead of command line.")
    parser.add_argument("--use-diffusion", dest="use_diffusion", action='store_true', help="Use diffusion model in the experiment.")

    # ---- region restriction ----
    parser.add_argument(
        "--region-bbox", dest="region_bbox", type=int, nargs=6, default=None,
        metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"),
        help=("Optional bbox in voxel coords of the full-res 25um atlas. "
              "If set: inducing points are placed inside the bbox via k-means, "
              "and MALDI train/test points are filtered to the same bbox in mm. "
              "The standardized coordinate space (coord_mean / coord_std) is "
              "kept global, so the region maps to the same standardized space "
              "as a whole-brain run."),
    )
    return vars(parser.parse_args())

def print_kernel_param_statistics(experiment, n_sample: int = 5000, header: str = ""):
    """Diagnostics for the GP front-end of an LGP after (or before) training.

    Prints:
      [1] Variational distribution: |m| and ‖m‖² (whitened-basis KL contributor)
      [2] Kernel hyperparameters: outputscale, lengthscales, floor check
      [3] log_var_n: per-lipid learnable noise — spread tells you if the model
          gave up on hard lipids
      [4] Inducing point stats
      [5] GP latent at training points: std per task — if << 1 the GP is
          producing near-constant features and the decoder is doing everything

    Call after experiment.run() (or twice — before and after — to see what
    training moved). Wrap with torch.no_grad context internally.
    """
    import torch

    lgp = experiment.lgp_model
    gp = lgp.gp_model
    device = experiment.config.device

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"KERNEL / GP PARAMETER STATISTICS  {header}")
    print(bar)

    # --- [1] Variational distribution -------------------------------------
    print("\n[1] Variational distribution q(u)")
    try:
        vs = gp.variational_strategy
        # IndependentMultitaskVariationalStrategy wraps a VariationalStrategy
        base_vs = getattr(vs, "base_variational_strategy", vs)
        vd = base_vs._variational_distribution
        m = vd.variational_mean.detach()
        m_sq = (m ** 2)
        print(f"  variational_mean shape: {tuple(m.shape)}   "
              f"(num_tasks, num_inducing)")
        print(f"  |m|   per-element:  mean={m.abs().mean():.4f}  "
              f"med={m.abs().median():.4f}  max={m.abs().max():.4f}")
        print(f"  ‖m‖² per task:      "
              f"{[f'{v:.2f}' for v in m_sq.sum(dim=-1).cpu().tolist()]}")
        print(f"  ‖m‖² total:         {m_sq.sum().item():.2f}   "
              f"(whitened+S≈I → KL ≈ ½·‖m‖² ≈ {0.5*m_sq.sum().item():.1f})")
        if hasattr(vd, "chol_variational_covar"):
            chol = vd.chol_variational_covar.detach()
            diag = chol.diagonal(dim1=-2, dim2=-1)
            print(f"  S=LLᵀ diag(L):      min={diag.min():.4f}  "
                  f"med={diag.median():.4f}  max={diag.max():.4f}")
    except Exception as e:
        print(f"  [could not access variational distribution: {e}]")

    # --- [2] Kernel hyperparameters ---------------------------------------
    print("\n[2] Kernel hyperparameters")
    try:
        cov = gp.covar_module  # ScaleKernel
        out = cov.outputscale.detach().cpu()
        print(f"  outputscale per task: "
              f"{[f'{v:.4f}' for v in out.tolist()]}")

        # ScaleKernel -> (maybe Custom3DKernel) -> MaternKernel/RBFKernel
        bk = cov.base_kernel
        if hasattr(bk, "base_kernel") and hasattr(bk.base_kernel, "lengthscale"):
            inner = bk.base_kernel
            kind = f"{type(bk).__name__} > {type(inner).__name__}"
        else:
            inner = bk
            kind = type(bk).__name__
        ls = inner.lengthscale.detach().cpu()
        print(f"  base kernel: {kind}    lengthscale shape: {tuple(ls.shape)}")
        ls_flat = ls.squeeze(-2)   # (num_tasks, ard_num_dims)
        for t, row in enumerate(ls_flat.tolist()):
            print(f"    task {t} lengthscale (xccf, yccf, zccf): "
                  f"{[f'{v:.4f}' for v in row]}")

        # Floor check
        try:
            cons = inner.raw_lengthscale_constraint
            if cons is not None and hasattr(cons, "lower_bound"):
                lb = cons.lower_bound
                if lb is not None:
                    floor = float(lb.min().item())
                    n_close = (ls_flat < floor * 1.05).sum().item()
                    print(f"  lengthscale floor (min over axes): {floor:.4f}    "
                          f"{n_close}/{ls_flat.numel()} elements within 5% of floor")
                    if n_close > 0:
                        print("    ↳ kernel is pinned at the minimal_length_scale "
                              "constraint; try lowering --n-pixels")
        except Exception:
            pass
    except Exception as e:
        print(f"  [could not access kernel: {e}]")

    # --- [3] log_var_n ----------------------------------------------------
    print("\n[3] log_var_n  (per-lipid Gaussian noise in the recon loss)")
    try:
        lvn = lgp.log_var_n.detach().cpu()
        var = lvn.exp()
        print(f"  log_var_n: min={lvn.min():.3f}  med={lvn.median():.3f}  "
              f"max={lvn.max():.3f}")
        print(f"  variance:  min={var.min():.4f}  med={var.median():.4f}  "
              f"max={var.max():.4f}")
        spread = (var.max() / var.min()).item()
        print(f"  variance spread (max/min): {spread:.1f}x")
        if spread > 50:
            print("    ↳ large spread: the model is down-weighting some lipids "
                  "via log_var_n. Try plain MSE or a single scalar noise.")
        k = min(10, lvn.numel())
        top_idx = var.topk(k).indices.tolist()
        bot_idx = var.topk(k, largest=False).indices.tolist()
        print(f"  top {k} HIGHEST-noise lipid indices: {top_idx}")
        print(f"  top {k} LOWEST-noise lipid indices:  {bot_idx}")
    except Exception as e:
        print(f"  [could not access log_var_n: {e}]")

    # --- [4] Inducing points ---------------------------------------------
    print("\n[4] Inducing points")
    try:
        base_vs = getattr(gp.variational_strategy,
                          "base_variational_strategy", gp.variational_strategy)
        ip = base_vs.inducing_points.detach().cpu()
        # ip shape: (num_tasks, num_inducing, 3) or (num_inducing, 3)
        if ip.dim() == 3:
            print(f"  inducing_points shape: {tuple(ip.shape)}  (per-task)")
            radii = ip.norm(dim=-1)  # (num_tasks, num_inducing)
            print(f"  ‖z‖ in standardized space:  mean={radii.mean():.3f}  "
                  f"max={radii.max():.3f}")
            # per-axis spread
            for axis, name in enumerate(["xccf", "yccf", "zccf"]):
                print(f"    {name}: min={ip[..., axis].min():.3f}  "
                      f"max={ip[..., axis].max():.3f}  "
                      f"std={ip[..., axis].std():.3f}")
        else:
            print(f"  inducing_points shape: {tuple(ip.shape)}")
    except Exception as e:
        print(f"  [could not access inducing points: {e}]")

    # --- [5] GP latent at training points --------------------------------
    print(f"\n[5] GP latent at training points (sample n={n_sample})")
    try:
        if not hasattr(experiment, "coordinates_train") or experiment.coordinates_train is None:
            print("  experiment.coordinates_train not populated; skipping.")
        else:
            was_training = lgp.training
            lgp.eval()
            n = min(n_sample, experiment.coordinates_train.shape[0])
            with torch.no_grad():
                coords = experiment.coordinates_train[:n].to(device).contiguous()
                post = gp(coords)
                latent_mean = post.mean.detach().cpu()      # (n, num_tasks)
                latent_var = (post.variance.detach().cpu()
                              if hasattr(post, "variance") else None)
            print(f"  latent shape: {tuple(latent_mean.shape)}")
            stds = latent_mean.std(dim=0)
            means = latent_mean.mean(dim=0)
            print(f"  latent mean per task: {[f'{v:+.4f}' for v in means.tolist()]}")
            print(f"  latent std  per task: {[f'{v:.4f}'  for v in stds.tolist()]}")
            if latent_var is not None:
                print(f"  posterior var per task (avg over points): "
                      f"{[f'{v:.4f}' for v in latent_var.mean(dim=0).tolist()]}")
            if stds.max().item() < 0.1:
                print("    ↳ WARNING: all task stds < 0.1. The GP latent is "
                      "near-constant; the decoder is doing all the work.")
            elif stds.min().item() < 0.05:
                n_dead = (stds < 0.05).sum().item()
                print(f"    ↳ WARNING: {n_dead} task(s) have std < 0.05 "
                      "(dead dimensions of the latent).")
            if was_training:
                lgp.train()
    except Exception as e:
        print(f"  [could not sample latent: {e}]")

    print(bar + "\n")


def setup_experiment(args):
    config = MaldiConfig.from_args(args)
    logging.info("Configuration created successfully")

    region_bbox = args.get("region_bbox", None)

    # 1. Patch the config's MALDI parquet filters so train/test only sees
    #    points inside the bbox. No-op when region_bbox is None.
    apply_region_to_config(config, region_bbox)

    # 2. Inducing points: bbox-restricted k-means when a bbox is set, else
    #    the original whole-brain symmetric k-means. Both routines return
    #    coord_mean / coord_std using the *global* normalization.
    logging.info("Getting inducing points")
    if region_bbox is not None:
        inducing_points, coord_mean, coord_std = get_bbox_inducing_points(
            config.exp_path, config.dataset_path, config.num_inducing, region_bbox,
        )
        # Keep config in sync if k-means was clamped to fewer voxels.
        config.num_inducing = inducing_points.shape[0]
    else:
        inducing_points, coord_mean, coord_std = get_inducing_points(
            config.exp_path, config.dataset_path, config.num_inducing,
        )
    logging.info(f"Got {inducing_points.shape[0]} inducing points")

    logging.info("Creating GP model")
    voxel_size = 0.025
    n_pixel = args["n_pixels"]
    minimal_length_scale = args["n_pixels"] * voxel_size / (coord_std.sum() / 3)
    logging.info(f"minimal length scale in um: {n_pixel * voxel_size}")
    gp_model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        kernel_type=config.kernel,
        nu=config.nu,
        minimal_length_scale=minimal_length_scale,
        input_dim=3,
    )
    logging.info("GP model created successfully")

    logging.info("Creating LGP instance")
    lgp_model = LGP(
        gp_model=gp_model,
        p=len(config.selected_lipids_names),
        d=config.latent_dim,
        n_neurons=[256, 256, 128],
        dropout=[0.1, 0.1, 0.1],
        activation='silu',
        device=config.device,
    )
    return MaldiExperiment(config, lgp_model, coord_mean, coord_std), region_bbox


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting MALDI experiment (Euclidean Matern baseline)")
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")
    experiment, region_bbox = setup_experiment(args)
    print_kernel_param_statistics(experiment, header="(before training)")
    experiment.run()
    print_kernel_param_statistics(experiment, header="(after training)")
    if region_bbox is not None:
        # Skip whole-brain reconstruction in region mode -- a GP trained
        # only on points inside the bbox will extrapolate poorly outside it.
        experiment.region_reconstruction(region_bbox)
    else:
        experiment.whole_brain_reconstruction()