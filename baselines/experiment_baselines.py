"""baselines.py — non-GP baselines, same I/O contract as lgp_manifold_experiment.py.

Reads MALDI parquet, applies the same train/test filters + region bbox patching
+ log transform + per-lipid normalization that experiment.py uses, trains one
baseline model on coords -> lipids, then writes the predictions to disk in the
same layout the GP runs produce. Optionally runs whole-brain or regional
reconstruction (atlas voxel grid -> per-lipid volumes) producing batch_*.pth
files in the same format experiment.load_whole_brain_reconstruction reads.

Per-split outputs:
    <exp_path>/train/predictions.npy              (n_train, p)  un-log, un-norm
    <exp_path>/train/true_values.npy              (n_train, p)  un-log, un-norm
    <exp_path>/train/coordinates.pth              (n_train, 3)  standardized mm
    <exp_path>/train/coordinates_pixel_index.pth  (n_train, 3)  voxel indices
    <exp_path>/test/predictions.npy               (n_test , p)
    <exp_path>/test/true_values.npy               (n_test , p)
    <exp_path>/test/coordinates.pth               (n_test , 3)
    <exp_path>/test/coordinates_pixel_index.pth   (n_test , 3)

Reconstruction outputs (when --reconstruct {whole_brain,region}):
    <exp_path>/volume/template_volume.npy           atlas template
    <exp_path>/volume/batch_{i}.pth                 {coordinates,indices,predictions}
    or
    <exp_path>/volume_region_<bbox>/template_volume.npy
    <exp_path>/volume_region_<bbox>/batch_{i}.pth

Persistence outputs (so reconstruction can be re-run later from CLI):
    <exp_path>/model.pth          sklearn pickle / xgb-bundle / torch state_dict
    <exp_path>/coord_mean.pth     for reconstruction-time coord standardization
    <exp_path>/coord_std.pth
    <exp_path>/lipid_means.pth    per-lipid (log) mean
    <exp_path>/lipid_stds.pth     per-lipid (log) std

Available baselines (--model):
    mean       per-lipid training mean. The floor.
    linear     Ridge regression (closed form).
    xgboost    p independent XGBRegressors with early stopping.
    mlp        configurable-depth MLP (default 128x128x128, SiLU, dropout).
    mlp_eigen  mlp with the points' manifold-eigenbasis projection concatenated
               onto the coords (needs the eigenpair pipeline; reference frame).
    gcn        Graph Conv Net over a per-batch KNN graph of the coords.
    gcn_faiss  Graph Conv Net over the FAISS reference-node manifold graph (same
               graph as the manifold GP); reads out per nearest node.

Reconstruction/render/diagnostics parity with the GP runs: after reconstruction
this writes the composite renders AND the per-lipid diagnostics PNGs (value
distribution + true-vs-pred scatter, linear & log) from the held-out test split,
matching MaldiExperiment.render_reconstruction.

All shared CLI flags mirror lgp_manifold_experiment.py so launches feel
symmetric. Region semantics also mirror it: when --region-bbox is set, parquet
filters get patched to restrict to the bbox.

Usage examples:
    # whole-brain MLP with whole-brain reconstruction
    python baselines.py --mode train --model mlp [other flags] --reconstruct auto

    # region run; reconstruction will be region-restricted automatically
    python baselines.py [...] --region-bbox 200 250 150 200 200 250 --reconstruct auto

    # re-run reconstruction only, against a previously-trained baseline
    python baselines.py --mode train --model mlp [...] \\
        --skip-training --reconstruct whole_brain
"""

# --- repo path bootstrap (this file moved out of maldi/) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_), str(_REPO_ / "maldi"), str(_REPO_ / "manifold"),
           str(_Path(__file__).resolve().parent),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import logging
import os
import pickle
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from tqdm import tqdm

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

from config import MaldiConfig
from utils import coord_norm_from_reference
from manifold_kernel_builder import (
    add_manifold_args, build_manifold_kernel, build_manifold_graph,
)

try:
    from torch_geometric.nn import GCNConv
    HAVE_PYG = True
except ImportError:
    HAVE_PYG = False


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    parser = ArgumentParser(description="Non-GP baselines for MALDI lipid prediction.")
    # Shared flags (one-for-one with lgp_manifold_experiment.py where applicable)
    parser.add_argument("--mode", type=str, required=True,
                        help="Experiment mode (e.g., 'train').")
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
    # Reconstruction is a pure forward pass over millions of voxels, so it wants a
    # far larger batch than training's SGD minibatch — but --batch-size drives BOTH
    # (and is baked into EXP_NAME), so raising that to speed up inference would
    # silently change the optimization. Hence a separate knob.
    parser.add_argument("--inference-batch-size", dest="inference_batch_size",
                        type=int, default=65536,
                        help="Batch size for the reconstruction forward pass only "
                             "(training minibatch is --batch-size).")
    parser.add_argument("--load-args", dest="load_args", action="store_true")
    parser.add_argument("--use-diffusion", dest="use_diffusion", action="store_true")
    parser.add_argument(
        "--region-bbox", dest="region_bbox", type=int, nargs=6, default=None,
        metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"),
    )

    # MaldiConfig wants these even though baselines don't use them
    parser.add_argument("--num-inducing", dest="num_inducing", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=10)
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-3)

    # Baseline-specific knobs
    parser.add_argument("--model", type=str, default="linear",
                        choices=["mean", "linear", "xgboost", "mlp", "mlp_eigen",
                                 "gcn", "gcn_faiss", "euclid"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--xgb-n-estimators", type=int, default=400)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    # (--model mlp_eigen) Where the on-disk feature memmap is staged. Must be a
    # LOCAL filesystem: mmap is unsupported on the S3/FUSE mounts (OSError 95),
    # and training reads it in random order every epoch, so network storage would
    # crawl even where mmap works. Default: TMPDIR (never --output-dir).
    parser.add_argument("--feat-scratch-dir", dest="feat_scratch_dir", type=str, default=None,
                        help="(--model mlp_eigen) local dir for the feature memmap "
                             "(default: TMPDIR). Needs ~N*(3+num_modes)*4 bytes free.")
    # GCN knobs (--model gcn): per-batch KNN graph over coords.
    parser.add_argument("--gcn-hidden", dest="gcn_hidden", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--gcn-dropout", dest="gcn_dropout", type=float, default=0.1)
    parser.add_argument("--gcn-knn-k", dest="gcn_knn_k", type=int, default=15,
                        help="Neighbours for the per-batch GCN KNN graph.")
    # gcn_faiss is full-graph (1 optimizer step per forward), so it needs its own
    # iteration budget rather than the shared --epochs (tuned for minibatch models).
    parser.add_argument("--gcn-faiss-iters", dest="gcn_faiss_iters", type=int, default=2000,
                        help="(--model gcn_faiss) number of full-graph gradient steps.")
    parser.add_argument("--gcn-faiss-node-batch", dest="gcn_faiss_node_batch", type=int, default=65536,
                        help="(--model gcn_faiss) training measurements sampled per step.")
    parser.add_argument("--gcn-faiss-interp-k", dest="gcn_faiss_interp_k", type=int, default=8,
                        help="(--model gcn_faiss) nearest reference nodes blended by "
                             "inverse-distance (Shepard) interpolation at read-out. "
                             "1 = the old nearest-node lookup.")

    # EUCLID knobs (--model euclid). EUCLID's own settings are the defaults and
    # there is nothing else to tune: the grid, radius, exp(-d) weights, the leaf
    # gate and the int(ccf*10)-1 index map all come from their code and their
    # shipped 100um volumes.
    parser.add_argument("--euclid-repo", dest="euclid_repo", type=str, default=None,
                        help="Path to the EUCLID checkout (default: <repo>/euclid). "
                             "Its anatomical_interpolation is executed as-is.")
    parser.add_argument("--euclid-w", dest="euclid_w", type=float, default=50,
                        help="EUCLID's `w`: low-intensity voxels are dropped below "
                             "this (their anatomical_interpolation default is 50).")
    parser.add_argument("--euclid-reference", dest="euclid_reference", type=str,
                        default=None,
                        help="Reference volume for EUCLID's `reference < 4` "
                             "background mask. Default: reference_image100um.npy "
                             "from --euclid-repo. 25um volumes are subsampled "
                             "[::4] onto their 100um grid.")
    parser.add_argument("--euclid-annotation", dest="euclid_annotation", type=str,
                        default=None,
                        help="Annotation volume for EUCLID's same-structure donor "
                             "gate. Default: annotation_image100um.npy from "
                             "--euclid-repo (the Allen 672-label leaf volume). "
                             "A coarser atlas weakens the gate: level_15annot's "
                             "root label alone covers 57%% of tissue.")
    parser.add_argument("--euclid-jobs", dest="euclid_jobs", type=int, default=1,
                        help="Processes to fan EUCLID's per-lipid kernel across "
                             "(~242 s/lipid single-threaded; 173 lipids = 11.6 h at 1). "
                             "0 or negative = auto: joblib.cpu_count(), which "
                             "respects cgroup quotas and CPU affinity. Always "
                             "capped at the number of lipids left to run.")
    parser.add_argument("--euclid-verify-reduction", dest="euclid_verify_reduction",
                        action="store_true",
                        help="Check the per-voxel row reduction against EUCLID's "
                             "unreduced iterrows() path on one lipid. Slow.")

    # Reconstruction
    parser.add_argument("--template-name", dest="template_name", type=str, required=True, help="Template name.")
    parser.add_argument("--reference-file", dest="reference_file", type=str, required=True, help="The reference image npy.")
    parser.add_argument("--annotations-file", dest="annotations_file", type=str, help="The annotations if needed.")
    parser.add_argument("--reconstruct", type=str, default="auto",
                        choices=["none", "auto", "whole_brain", "region"],
                        help="auto = whole_brain if no bbox, region if bbox.")
    parser.add_argument("--reconstruct-threshold", type=float, default=5.0)
    # Opt-in: reconstruct only the voxels the composite render reads (the slice
    # planes + the 3D MIP's stride) instead of the whole brain — ~5.5x fewer
    # voxels, near-identical figure. Off by default so the dense volumes (which
    # napari/analysis scripts consume) are still produced unless asked otherwise.
    parser.add_argument("--render-voxels-only", dest="render_voxels_only",
                        action="store_true",
                        help="Reconstruct only the voxels the render reads "
                             "(slice planes + MIP stride). Writes sparse volumes "
                             "to volume_sparse/ with a _sparse suffix; the dense "
                             "volume/ dir is not produced.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip fitting; only run reconstruction from a saved model.")
    parser.add_argument(
        "--reconstruction-lipids", dest="reconstruction_lipids",
        nargs="+", default=None,
        help=("Restrict reconstruction to these lipids. Accepts indices "
              "(0 5 10) or names ('PA 36:4' 'PE 40:7'). Default: all."),
    )

    # Manifold graph/eigenbasis flags (used by --model mlp_eigen).
    add_manifold_args(parser)

    args = vars(parser.parse_args())
    # Coerce to int where possible so the resolver gets a homogeneous list.
    if args.get("reconstruction_lipids"):
        try:
            args["reconstruction_lipids"] = [int(v) for v in args["reconstruction_lipids"]]
        except ValueError:
            pass  # keep as strings; resolver handles names
    return args

# ===========================================================================
# Data
# ===========================================================================
def _load_split(config, train: bool):
    """Returns (y_logged, y_original, coords, pixel_index)."""
    flt = config.section_filter if train else config.test_filter
    lipid_cols = [str(i) for i in config.selected_lipids_names]

    logging.info(f"Loading {'train' if train else 'test'} parquet...")
    t0 = time.time()
    y = pd.read_parquet(config.maldi_file, columns=lipid_cols, filters=flt).values
    coords = pd.read_parquet(
        config.maldi_file, columns=["xccf", "yccf", "zccf"], filters=flt,
    ).values
    pixel_index = pd.read_parquet(
        config.maldi_file,
        columns=["x_index", "y_index", "z_index"], filters=flt,
    ).values
    logging.info(f"  loaded {y.shape[0]:,} rows in {time.time()-t0:.1f}s")
    if y.shape[0] == 0:
        raise RuntimeError("Empty split. Bbox probably has no MALDI rows in it.")

    y = torch.tensor(y, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.float32)
    pixel_index = torch.tensor(pixel_index, dtype=torch.float32)

    n_neg = int((y < 0).sum())
    if n_neg:
        logging.info(f"  imputing {n_neg} negative values to 0")
        y[y < 0] = 0
    y_original = y.clone()

    if config.log_transform:
        logging.info("  applying log(x + 1e-10)")
        y = torch.log(y + 1e-10)

    return y, y_original, coords, pixel_index


def _normalize_with_train_stats(y, coords, col_means, col_stds, coord_mean, coord_std):
    return (y - col_means) / col_stds, (coords - coord_mean) / coord_std


def _denormalize(pred_norm, col_means, col_stds, log_transform):
    pred = pred_norm * col_stds + col_means
    if log_transform:
        pred = torch.exp(pred) - 1e-10
    return pred


# ===========================================================================
# Baseline wrappers: each has fit / predict / save / load.
# predict() takes standardized coords and returns normalized predictions
# (lipid log-mean/std space). Reconstruction calls predict() and denormalizes
# once at the end, mirroring how MaldiExperiment denormalizes GP outputs.
# ===========================================================================
class MeanBaseline:
    kind = "mean"
    def __init__(self):
        self.mean_vec: torch.Tensor | None = None

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        self.mean_vec = y_train.mean(dim=0, keepdim=True)
        return (self.mean_vec.expand_as(y_train).clone(),
                self.mean_vec.expand_as(y_test).clone())

    def predict(self, coords_std):
        return self.mean_vec.expand(coords_std.shape[0], -1).clone()

    def save(self, path):
        torch.save({"mean_vec": self.mean_vec}, path)

    def load(self, path, p, args):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.mean_vec = d["mean_vec"]


class LinearBaseline:
    kind = "linear"
    def __init__(self):
        self.model = None

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        self.model = Ridge(alpha=args["ridge_alpha"])
        self.model.fit(coords_train.numpy(), y_train.numpy())
        pred_tr = torch.tensor(self.model.predict(coords_train.numpy()), dtype=torch.float32)
        pred_te = torch.tensor(self.model.predict(coords_test.numpy()),  dtype=torch.float32)
        return pred_tr, pred_te

    def predict(self, coords_std):
        out = self.model.predict(coords_std.cpu().numpy())
        return torch.tensor(out, dtype=torch.float32)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path, p, args):
        with open(path, "rb") as f:
            self.model = pickle.load(f)


class XGBoostBaseline:
    kind = "xgboost"
    def __init__(self):
        self.models: list | None = None

    def _params(self, args):
        params = dict(
            n_estimators=args["xgb_n_estimators"],
            max_depth=args["xgb_max_depth"],
            learning_rate=args["xgb_lr"],
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=30,
            verbosity=0,
        )
        if args["device"] == "cuda":
            params["device"] = "cuda"
        return params

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        if not HAVE_XGB:
            raise RuntimeError("xgboost not installed.")
        params = self._params(args)
        X_tr, X_te = coords_train.numpy(), coords_test.numpy()
        Y_tr, Y_te = y_train.numpy(), y_test.numpy()
        n_lipids = Y_tr.shape[1]
        pred_tr = np.zeros_like(Y_tr)
        pred_te = np.zeros_like(Y_te)

        n_tr = X_tr.shape[0]
        perm = np.random.RandomState(args["seed"]).permutation(n_tr)
        cut = int(0.9 * n_tr)
        tr_idx, val_idx = perm[:cut], perm[cut:]

        self.models = []
        t0 = time.time()
        for j in range(n_lipids):
            if j > 0 and j % 25 == 0:
                elapsed = time.time() - t0
                eta = elapsed / j * (n_lipids - j)
                logging.info(f"  xgb lipid {j}/{n_lipids}  ({elapsed:.0f}s, ETA {eta:.0f}s)")
            m = xgb.XGBRegressor(**params)
            m.fit(
                X_tr[tr_idx], Y_tr[tr_idx, j],
                eval_set=[(X_tr[val_idx], Y_tr[val_idx, j])],
                verbose=False,
            )
            self.models.append(m)
            pred_tr[:, j] = m.predict(X_tr)
            pred_te[:, j] = m.predict(X_te)
        logging.info(f"  xgb total: {time.time()-t0:.0f}s")
        return (torch.tensor(pred_tr, dtype=torch.float32),
                torch.tensor(pred_te, dtype=torch.float32))

    def predict(self, coords_std):
        X = coords_std.cpu().numpy()
        out = np.stack([m.predict(X) for m in self.models], axis=1)
        return torch.tensor(out, dtype=torch.float32)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    def load(self, path, p, args):
        with open(path, "rb") as f:
            self.models = pickle.load(f)


class MLPBaseline:
    kind = "mlp"
    def __init__(self):
        self.model: nn.Sequential | None = None
        self.device: torch.device | None = None

    def _build(self, p, hidden, dropout, device, in_dim=3):
        dims = [in_dim] + list(hidden) + [p]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers).to(device)

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        self.device = torch.device(args["device"])
        p = y_train.shape[1]
        self.model = self._build(p, args["mlp_hidden"], args["mlp_dropout"], self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=args["learning_rate"])

        Xtr = coords_train.to(self.device); Ytr = y_train.to(self.device)
        Xte = coords_test.to(self.device);   Yte = y_test.to(self.device)
        n = Xtr.shape[0]
        bs = args["batch_size"]
        pbar = tqdm(range(args["epochs"]), desc="mlp")
        for epoch in pbar:
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            loss_sum = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                pred = self.model(Xtr[idx])
                loss = F.mse_loss(pred, Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item() * idx.shape[0]
            pbar.set_postfix(train=f"{loss_sum/n:.4g}")
            if (epoch + 1) % 25 == 0 or epoch == args["epochs"] - 1:
                self.model.eval()
                with torch.no_grad():
                    te = F.mse_loss(self.model(Xte), Yte).item()
                logging.info(f"  mlp epoch {epoch+1}/{args['epochs']}  "
                             f"train={loss_sum/n:.6g}  test={te:.6g}")
        self.model.eval()
        with torch.no_grad():
            pred_tr = self.model(Xtr).cpu()
            pred_te = self.model(Xte).cpu()
        return pred_tr, pred_te

    def predict(self, coords_std):
        self.model.eval()
        with torch.no_grad():
            return self.model(coords_std.to(self.device)).cpu()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, p, args):
        self.device = torch.device(args["device"])
        self.model = self._build(p, args["mlp_hidden"], args["mlp_dropout"], self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


class MLPEigenBaseline(MLPBaseline):
    """MLP whose inputs are the 3D coords CONCATENATED with the point's
    projection onto the manifold eigenbasis (the Laplacian harmonics).

    Same MLP as ``MLPBaseline`` but with input dim ``3 + num_modes``. The
    eigen-projection ``raw_eigenvectors(x)`` is only valid in the reference
    coordinate frame the graph was built in, so ``main`` standardizes coords
    with ``coord_norm_from_reference`` for this model (see ``needs_eigen``); the
    coords reaching ``predict`` are therefore already in that frame.

    ``attach_kernel`` must be called (in ``main``) before ``fit``/``predict``/
    ``load`` so the eigenbasis is available; the kernel is rebuilt from the same
    caches on the ``--skip-training`` path.
    """
    kind = "mlp_eigen"

    def __init__(self):
        super().__init__()
        self.kernel = None
        self.num_modes = None

    def attach_kernel(self, kernel):
        self.kernel = kernel
        self.num_modes = int(kernel.num_modes)

    # Row-chunk for the eigenvector interpolation. The Nyström out_of_sample step
    # materializes a (rows, k, num_modes) transient, which OOMs on the full
    # multi-million-row train set — so we interpolate in chunks and concatenate.
    eigen_chunk = 50_000
    # Rows per model forward pass for the whole-split eval/prediction passes, which
    # would OOM if run in one shot regardless of the training batch size.
    forward_chunk = 100_000

    def _eigenvectors(self, x):
        outs = []
        for i in range(0, x.shape[0], self.eigen_chunk):
            outs.append(self.kernel.raw_eigenvectors(x[i:i + self.eigen_chunk]))
        return torch.cat(outs, dim=0)

    def _inputs(self, coords_std):
        """concat[coords (3), eigenfeatures (num_modes)] on ``self.device``."""
        if self.kernel is None:
            raise RuntimeError("MLPEigenBaseline needs attach_kernel() before use.")
        x = coords_std.to(self.device)
        with torch.no_grad():
            U = self._eigenvectors(x)                    # (B, num_modes), chunked
        return torch.cat([x, U], dim=1)

    def _predict_coords(self, coords_std):
        """Model forward over raw coords, featurizing + forwarding in chunks and
        returning predictions on the CPU. The full ``(N, 3+num_modes)`` feature
        matrix is never materialized — only one ``eigen_chunk`` slice at a time —
        so this stays bounded in both VRAM and host RAM for multi-million-row
        splits (train/test eval and whole_brain reconstruction)."""
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, coords_std.shape[0], self.eigen_chunk):
                xb = self._inputs(coords_std[i:i + self.eigen_chunk])
                outs.append(self.model(xb).cpu())
        return torch.cat(outs, dim=0)

    def _featurize_to_memmap(self, coords_std, path):
        """Featurize the whole split ONCE into an on-disk float32 memmap
        ``(N, 3+num_modes)`` and return it. The eigenbasis is frozen during MLP
        training, so per-epoch re-featurization is wasted work; precomputing once
        removes it. The matrix is far too big for VRAM/RAM at thousands of modes,
        so it lives on disk — training streams minibatches back, RAM bounded by the
        OS page cache and it scales to any mode count (limited by disk, not RAM)."""
        n = coords_std.shape[0]
        width = 3 + self.num_modes
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, width))
        with tqdm(total=n, desc="mlp_eigen featurize", unit="row", unit_scale=True) as pbar:
            for i in range(0, n, self.eigen_chunk):
                feats = self._inputs(coords_std[i:i + self.eigen_chunk]).cpu().numpy()
                mm[i:i + feats.shape[0]] = feats
                pbar.update(feats.shape[0])
        mm.flush()
        return mm

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        self.device = torch.device(args["device"])
        p = y_train.shape[1]
        in_dim = 3 + self.num_modes
        self.model = self._build(p, args["mlp_hidden"], args["mlp_dropout"], self.device, in_dim=in_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=args["learning_rate"])

        # Precompute train eigenfeatures ONCE to a disk-backed memmap, then stream
        # minibatches — instead of re-featurizing every epoch. Test features are
        # only needed at the (infrequent) eval, so those stay on the fly.
        Ytr = y_train.cpu().numpy()
        n = coords_train.shape[0]
        bs = args["batch_size"]
        # Stage the memmap on LOCAL disk: --output-dir is typically an S3/FUSE
        # mount, where mmap fails outright (OSError 95).
        scratch_dir = args.get("feat_scratch_dir") or None
        if scratch_dir:
            os.makedirs(scratch_dir, exist_ok=True)
        feat_dir = tempfile.mkdtemp(prefix="mlp_eigen_feat_", dir=scratch_dir)
        feat_path = os.path.join(feat_dir, "train_feats.f32")
        need = n * (3 + self.num_modes) * 4
        logging.info(f"  mlp_eigen feature memmap: {need/1e9:.1f} GB -> {feat_path}")
        Xtr = None
        try:
            Xtr = self._featurize_to_memmap(coords_train.cpu(), feat_path)
            pbar = tqdm(range(args["epochs"]), desc="mlp_eigen")
            for epoch in pbar:
                self.model.train()
                perm = np.random.permutation(n)
                loss_sum = 0.0
                for i in range(0, n, bs):
                    # Sort within the batch so the memmap reads are near-sequential
                    # (order within a batch is irrelevant to the mean-reduced loss).
                    idx = np.sort(perm[i:i + bs])
                    xb = torch.from_numpy(np.ascontiguousarray(Xtr[idx])).to(self.device)
                    yb = torch.from_numpy(Ytr[idx]).to(self.device)
                    loss = F.mse_loss(self.model(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                    loss_sum += loss.item() * idx.shape[0]
                pbar.set_postfix(train=f"{loss_sum/n:.4g}")
                if (epoch + 1) % 25 == 0 or epoch == args["epochs"] - 1:
                    te = F.mse_loss(self._predict_coords(coords_test), y_test.cpu()).item()
                    logging.info(f"  mlp_eigen epoch {epoch+1}/{args['epochs']}  "
                                 f"train={loss_sum/n:.6g}  test={te:.6g}")
            # Final predictions: train from the memmap (already computed), test on the fly.
            pred_tr = self._forward_memmap(Xtr)
            pred_te = self._predict_coords(coords_test)
        finally:
            # Guard the del: if _featurize_to_memmap raised, Xtr never bound and an
            # UnboundLocalError here would mask the real exception.
            del Xtr
            try:
                os.remove(feat_path)
            except OSError:
                pass
            try:
                os.rmdir(feat_dir)
            except OSError:
                pass
        return pred_tr, pred_te

    def _forward_memmap(self, Xmm):
        """Model forward over a memmap feature matrix, chunked, returned on CPU."""
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, Xmm.shape[0], self.forward_chunk):
                xb = torch.from_numpy(np.ascontiguousarray(Xmm[i:i + self.forward_chunk])).to(self.device)
                outs.append(self.model(xb).cpu())
        return torch.cat(outs, dim=0)

    def predict(self, coords_std):
        return self._predict_coords(coords_std)

    def load(self, path, p, args):
        self.device = torch.device(args["device"])
        in_dim = 3 + self.num_modes
        self.model = self._build(p, args["mlp_hidden"], args["mlp_dropout"], self.device, in_dim=in_dim)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


def _knn_edge_index(x, k):
    """Symmetric KNN edge_index (2, E) over the batch point cloud ``x`` (B, 3).

    Built per batch (inductive) from Euclidean distances; self is excluded
    (GCNConv re-adds self-loops). k is clamped to B-1 so small final batches
    don't error.
    """
    n = x.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)
    d = torch.cdist(x, x)
    idx = d.topk(k + 1, largest=False).indices[:, 1:]        # (n, k), drop self
    src = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


class _GCNNet(nn.Module):
    """Stacked GCNConv layers + a linear read-out (features -> lipids).

    Two anti-oversmoothing measures: (1) a residual connection on every
    width-preserving layer, so repeated neighbourhood averaging can't wash the
    signal out to the graph mean; (2) an optional per-edge ``edge_weight`` so
    message passing follows the manifold graph's affinities instead of a plain
    unweighted neighbour mean. ``edge_weight=None`` recovers the original
    behaviour (used by the per-batch ``gcn`` baseline)."""
    def __init__(self, in_dim, hidden, p, dropout):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = dropout
        self.head = nn.Linear(dims[-1], p)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            h = F.silu(conv(x, edge_index, edge_weight))
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            # Residual only where the layer preserves width (every layer but
            # the first, which lifts in_dim -> hidden).
            x = h + x if h.shape[-1] == x.shape[-1] else h
        return self.head(x)


class GCNBaseline:
    """Graph Convolutional Network over a per-batch KNN graph of the coords.

    Inductive: each train/test/reconstruction batch builds its own KNN graph
    (neighbourhoods are batch-local), so it slots into the existing batched
    ``predict``/``reconstruct`` contract with no refactor. Node features are the
    3D coords; the graph supplies the manifold structure.
    """
    kind = "gcn"

    def __init__(self):
        self.model: _GCNNet | None = None
        self.device: torch.device | None = None
        self.k = None
        self.hidden = None
        self.dropout = None

    def _predict_batch(self, coords):
        edge_index = _knn_edge_index(coords, self.k)
        return self.model(coords, edge_index)

    def _predict_split(self, X, bs):
        """Batched prediction: the per-batch KNN uses O(batch^2) cdist, so the
        full split (millions of rows) must NOT be passed to _predict_batch at
        once. Batches also keep the KNN graph at the same scale as training."""
        self.model.eval()
        with torch.no_grad():
            return torch.cat([self._predict_batch(X[i:i + bs])
                              for i in range(0, X.shape[0], bs)])

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        if not HAVE_PYG:
            raise RuntimeError("torch_geometric not installed (needed for --model gcn).")
        self.device = torch.device(args["device"])
        self.k = args["gcn_knn_k"]
        self.hidden = args["gcn_hidden"]
        self.dropout = args["gcn_dropout"]
        p = y_train.shape[1]
        self.model = _GCNNet(3, self.hidden, p, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=args["learning_rate"])

        Xtr = coords_train.to(self.device); Ytr = y_train.to(self.device)
        Xte = coords_test.to(self.device);   Yte = y_test.to(self.device)
        n = Xtr.shape[0]
        bs = args["batch_size"]
        pbar = tqdm(range(args["epochs"]), desc="gcn")
        for epoch in pbar:
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            loss_sum = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = Xtr[idx]
                pred = self.model(xb, _knn_edge_index(xb, self.k))
                loss = F.mse_loss(pred, Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item() * idx.shape[0]
            pbar.set_postfix(train=f"{loss_sum/n:.4g}")
            if (epoch + 1) % 25 == 0 or epoch == args["epochs"] - 1:
                te = F.mse_loss(self._predict_split(Xte, bs), Yte).item()
                logging.info(f"  gcn epoch {epoch+1}/{args['epochs']}  "
                             f"train={loss_sum/n:.6g}  test={te:.6g}")
        # Predict each split in batches so the KNN graphs match training scale.
        pred_tr = self._predict_split(Xtr, bs).cpu()
        pred_te = self._predict_split(Xte, bs).cpu()
        return pred_tr, pred_te

    def predict(self, coords_std):
        self.model.eval()
        with torch.no_grad():
            return self._predict_batch(coords_std.to(self.device)).cpu()

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(),
                    "k": self.k, "hidden": self.hidden, "dropout": self.dropout}, path)

    def load(self, path, p, args):
        if not HAVE_PYG:
            raise RuntimeError("torch_geometric not installed (needed for --model gcn).")
        self.device = torch.device(args["device"])
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.k, self.hidden, self.dropout = d["k"], d["hidden"], d["dropout"]
        self.model = _GCNNet(3, self.hidden, p, self.dropout).to(self.device)
        self.model.load_state_dict(d["state_dict"])
        self.model.eval()


class GCNFaissBaseline:
    """GCN over the FAISS reference-node manifold graph (the same graph the
    manifold GP uses), as opposed to ``GCNBaseline``'s per-batch KNN graph.

    Node features are the graph-node coords; the fixed KNN graph supplies the
    manifold structure. Each measurement / reconstruction voxel reads out from
    its nearest graph node (``knn.search``). Training is transductive full-graph:
    one forward over all nodes per step, MSE on the training measurements mapped
    to their nearest node.

    ``attach_graph`` must be called (in ``main``) before fit/predict/load, and
    coords must arrive in the reference frame (``main`` standardizes with
    ``coord_norm_from_reference``) so nearest-node search matches the graph.
    """
    kind = "gcn_faiss"

    def __init__(self):
        self.model: _GCNNet | None = None
        self.device: torch.device | None = None
        self.hidden = None
        self.dropout = None
        self.knn = None            # KnnGraph: .x (N,3), .search(x, k)
        self.edge_index = None     # (2, E)
        self.edge_weight = None    # (E,) heat-kernel affinities from edge_value
        self.node_x = None         # (N, 3) node features
        self.interp_k = 8          # nearest nodes averaged at read-out
        self._node_preds = None    # cached (N, p) node predictions (eval)

    def attach_graph(self, knn, edge_index, edge_value, device, interp_k=8):
        self.device = torch.device(device)
        self.knn = knn
        self.edge_index = edge_index.to(self.device)
        self.node_x = knn.x.to(self.device).float()
        self.interp_k = int(interp_k)
        # edge_value holds SQUARED distances; turn them into affinities with a
        # heat kernel whose bandwidth is the median edge (self-scaling, so the
        # weights sit in a sane range regardless of stride/coordinate units).
        # GCNConv applies its own symmetric degree normalization on top.
        ev = edge_value.to(self.device).float()
        sigma2 = torch.median(ev).clamp_min(1e-12)
        self.edge_weight = torch.exp(-ev / sigma2)

    def _interp_weights(self, coords, k=None):
        """Shepard (inverse-distance) interpolation over the k nearest reference
        nodes. Returns (idx (B,k) long, w (B,k) float) with weights normalized to
        sum 1 per query. Guards the approximate FAISS search's -1 padding."""
        k = self.interp_k if k is None else k
        # faiss's torch search requires a contiguous input; sliced/normalized
        # coord tensors often aren't. .contiguous() is a no-op when already so.
        q = coords.to(self.node_x.device).contiguous()
        d2, idx = self.knn.search(q, k)          # d2: squared distances
        valid = idx >= 0
        idx = idx.long().clamp_min(0)
        d2 = d2.clamp_min(0.0)
        w = torch.where(valid, 1.0 / (torch.sqrt(d2) + 1e-8),
                        torch.zeros_like(d2))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return idx, w

    @staticmethod
    def _readout(node_preds, idx, w):
        """Weighted gather: (B,k) node indices + (B,k) weights -> (B, p). Loops
        over the (small) k to avoid materializing the (B, k, p) intermediate."""
        out = node_preds.new_zeros((idx.shape[0], node_preds.shape[1]))
        for j in range(idx.shape[1]):
            out = out + w[:, j:j + 1] * node_preds[idx[:, j]]
        return out

    def _refresh_node_preds(self):
        self.model.eval()
        with torch.no_grad():
            self._node_preds = self.model(self.node_x, self.edge_index,
                                          self.edge_weight)

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        if not HAVE_PYG:
            raise RuntimeError("torch_geometric not installed (needed for --model gcn_faiss).")
        if self.knn is None:
            raise RuntimeError("GCNFaissBaseline needs attach_graph() before fit().")
        self.hidden = args["gcn_hidden"]
        self.dropout = args["gcn_dropout"]
        p = y_train.shape[1]
        self.model = _GCNNet(3, self.hidden, p, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=args["learning_rate"])

        # Each measurement reads out from its k nearest reference nodes (Shepard
        # interpolation) instead of snapping to a single nearest node, so the
        # supervision — and the reconstruction — vary smoothly below node scale.
        tr_idx, tr_w = self._interp_weights(coords_train)
        te_idx, te_w = self._interp_weights(coords_test)
        Ytr = y_train.to(self.device); Yte = y_test.to(self.device)

        # Full-graph transductive training. Each step is ONE full-graph forward +
        # ONE optimizer step, so the shared per-epoch count (tuned for the
        # minibatch baselines that take n/batch_size steps per epoch) would leave
        # this drastically undertrained. Instead run a dedicated iteration budget
        # (gcn_faiss_iters), each on a stochastic minibatch of training
        # measurements (interpolated from their nearest nodes) for SGD noise +
        # bounded memory on the (mb, p) read-out.
        n_tr = tr_idx.shape[0]
        n_iters = int(args.get("gcn_faiss_iters", 2000))
        mb = min(int(args.get("gcn_faiss_node_batch", 65536)), n_tr)
        log_every = max(1, n_iters // 20)
        pbar = tqdm(range(n_iters), desc="gcn_faiss")
        for it in pbar:
            self.model.train()
            node_preds = self.model(self.node_x, self.edge_index,
                                    self.edge_weight)               # (N, p)
            sel = torch.randint(0, n_tr, (mb,), device=self.device)
            pred = self._readout(node_preds, tr_idx[sel], tr_w[sel])
            loss = F.mse_loss(pred, Ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(train=f"{loss.item():.4g}")
            if (it + 1) % log_every == 0 or it == n_iters - 1:
                self.model.eval()
                with torch.no_grad():
                    node_preds_e = self.model(self.node_x, self.edge_index,
                                              self.edge_weight)
                    te = F.mse_loss(
                        self._readout(node_preds_e, te_idx, te_w), Yte
                    ).item()
                logging.info(f"  gcn_faiss iter {it+1}/{n_iters}  "
                             f"train={loss.item():.6g}  test={te:.6g}")

        self._refresh_node_preds()
        return (self._readout(self._node_preds, tr_idx, tr_w).cpu(),
                self._readout(self._node_preds, te_idx, te_w).cpu())

    def predict(self, coords_std):
        if self._node_preds is None:
            self._refresh_node_preds()
        idx, w = self._interp_weights(coords_std)
        return self._readout(self._node_preds, idx, w).cpu()

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(),
                    "hidden": self.hidden, "dropout": self.dropout}, path)

    def load(self, path, p, args):
        if not HAVE_PYG:
            raise RuntimeError("torch_geometric not installed (needed for --model gcn_faiss).")
        self.device = torch.device(args["device"])
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.hidden, self.dropout = d["hidden"], d["dropout"]
        self.model = _GCNNet(3, self.hidden, p, self.dropout).to(self.device)
        self.model.load_state_dict(d["state_dict"])
        self.model.eval()
        self._node_preds = None   # recomputed lazily (graph re-attached in main)


class EuclidBaseline:
    """EUCLID's `anatomical_interpolation`, run unmodified (see euclid_kernel).

    Transductive: nothing is trained. `fit` hands the fold's TRAIN rows to
    EUCLID's own driver, which builds the 100um donor volume and interpolates it;
    `predict` reads those volumes. Held-out rows are never passed in, so the fold
    is respected by construction.

    Predictions come back in EUCLID's `normalize_to_255(log(x))` units, so a
    per-lipid affine fitted on TRAIN maps them into the harness's (log - mean)/std
    space. Correlation — the headline metric — is invariant to that; it only makes
    r2/rmse readable.
    """
    kind = "euclid"

    def __init__(self):
        self.volumes = None
        self.lipids = None
        self.volume_dir = None
        self.exp_path = None
        self.affine = None            # (p, 2): slope, intercept
        self.coord_mean = self.coord_std = None
        self.col_means = self.col_stds = None
        self.log_transform = True

    # -- context from main() ----------------------------------------------
    def set_fit_context(self, coord_mean, coord_std, col_means, col_stds,
                        log_transform, lipid_names, exp_path):
        """fit() receives standardized log values, but EUCLID's driver applies its
        own log to raw intensities — so we need the stats to invert the harness's
        normalization, the coordinate frame to recover CCF mm, and the lipid names
        in the column order _load_split read them."""
        self.coord_mean, self.coord_std = coord_mean, coord_std
        self.col_means, self.col_stds = col_means, col_stds
        self.log_transform = log_transform
        self.lipids = [str(n) for n in lipid_names]
        # MaldiConfig rewrites exp_name (it appends the epoch count), so the run
        # dir is NOT output_dir/exp_name -- take the resolved path from main() or
        # the volumes land in a sibling directory.
        self.exp_path = Path(exp_path)

    def _mm(self, coords_std):
        c = coords_std if torch.is_tensor(coords_std) else torch.tensor(coords_std)
        return (c.cpu().float() * self.coord_std + self.coord_mean).numpy()

    def _raw(self, y_norm):
        return _denormalize(y_norm, self.col_means, self.col_stds, self.log_transform).numpy()

    def _apply_affine(self, sampled):
        out = sampled * self.affine[:, 0] + self.affine[:, 1]
        return np.nan_to_num(out, nan=0.0)   # 0 == the per-lipid train mean

    # -- fit ---------------------------------------------------------------
    def fit(self, coords_train, y_train, coords_test, y_test, args):
        from euclid_kernel import (run_anatomical_interpolation, sample_volumes,
                                   verify_row_reduction)
        if self.coord_mean is None:
            raise RuntimeError("main() must call set_fit_context() before fit()")

        _t = time.time()
        def _phase(msg):
            logging.info(f"[euclid] {msg}  (+{time.time() - _t:.0f}s)")

        mm_train, mm_test = self._mm(coords_train), self._mm(coords_test)
        raw_train = self._raw(y_train)
        w = float(args.get("euclid_w", 50))
        self.volume_dir = self.exp_path / "euclid_volumes"
        _phase(f"phase 1/5 inputs ready: {len(mm_train):,} train / {len(mm_test):,} "
               f"test pixels, {len(self.lipids)} lipids, w={w}, "
               f"volumes -> {self.volume_dir}")

        if args.get("euclid_verify_reduction"):
            verify_row_reduction(mm_train, raw_train, self.lipids[0],
                                 self.exp_path / "euclid_verify",
                                 w=w, euclid_repo=args.get("euclid_repo"))

        self.volumes = run_anatomical_interpolation(
            mm_train, raw_train, self.lipids, self.volume_dir,
            w=w, euclid_repo=args.get("euclid_repo"),
            n_jobs=int(args.get("euclid_jobs", 1)),
            reference_file=args.get("euclid_reference"),
            annotation_file=args.get("euclid_annotation"),
        )
        _phase(f"phase 2/5 interpolation done: {len(self.volumes)} volumes")

        s_train = sample_volumes(self.volumes, self.lipids, mm_train)
        s_test = sample_volumes(self.volumes, self.lipids, mm_test)
        _phase("phase 3/5 volumes sampled at train + test pixels")

        # Per-lipid affine, fitted on TRAIN only. Closed form in float64 rather
        # than np.polyfit: on ~5M float32 points polyfit's Vandermonde is
        # ill-conditioned and silently returns a slope of the WRONG SIGN
        # (observed: corr +0.52, polyfit slope -0.08, with a RankWarning).
        yt = y_train.numpy()
        self.affine = np.zeros((len(self.lipids), 2), dtype=np.float64)
        for j in range(len(self.lipids)):
            m = np.isfinite(s_train[:, j]) & np.isfinite(yt[:, j])
            if m.sum() <= 10:
                continue
            x = s_train[m, j].astype(np.float64)
            y = yt[m, j].astype(np.float64)
            xm, ym = x.mean(), y.mean()
            var = np.mean((x - xm) ** 2)
            if var <= 0:
                continue
            a = np.mean((x - xm) * (y - ym)) / var
            self.affine[j] = (a, ym - a * xm)
        neg = int((self.affine[:, 0] < 0).sum())
        _phase(f"phase 4/5 calibrated {int((self.affine[:, 0] != 0).sum())}"
               f"/{len(self.lipids)} lipids onto harness scale"
               + (f"  WARNING: {neg} with NEGATIVE slope" if neg else ""))

        pred_tr = self._apply_affine(s_train)
        pred_te = self._apply_affine(s_test)

        # Held-out corr right here, so the log carries the headline number
        # without waiting for metrics.csv at the end of the run.
        yte = y_test.numpy()
        cs = []
        for j in range(len(self.lipids)):
            m = np.isfinite(pred_te[:, j]) & np.isfinite(yte[:, j])
            if m.sum() > 10 and np.std(pred_te[m, j]) > 0:
                cs.append(np.corrcoef(pred_te[m, j], yte[m, j])[0, 1])
        if cs:
            cs = np.array(cs)
            _phase(f"phase 5/5 held-out corr (normalized log space): "
                   f"mean {cs.mean():.4f}  median {np.median(cs):.4f}  "
                   f"min {cs.min():.4f}  max {cs.max():.4f}  over {len(cs)} lipids")

        return (torch.tensor(pred_tr, dtype=torch.float32),
                torch.tensor(pred_te, dtype=torch.float32))

    # -- predict -----------------------------------------------------------
    def predict(self, coords_std):
        from euclid_kernel import sample_volumes
        s = sample_volumes(self.volumes, self.lipids, self._mm(coords_std))
        return torch.tensor(self._apply_affine(s), dtype=torch.float32)

    # -- persistence -------------------------------------------------------
    def save(self, path):
        # The volumes are already on disk where EUCLID wrote them (~10 MB each);
        # store the pointer, not a second copy.
        torch.save({"volume_dir": str(self.volume_dir), "lipids": self.lipids,
                    "affine": self.affine, "coord_mean": self.coord_mean,
                    "coord_std": self.coord_std}, path)

    def load(self, path, p, args):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.volume_dir = Path(d["volume_dir"])
        self.lipids, self.affine = d["lipids"], d["affine"]
        self.coord_mean, self.coord_std = d["coord_mean"], d["coord_std"]
        self.volumes = {lip: np.load(self.volume_dir / f"{lip}_interpolation_log.npy")
                        for lip in self.lipids}


MODEL_REGISTRY = {
    "mean":      MeanBaseline,
    "euclid":    EuclidBaseline,
    "linear":    LinearBaseline,
    "xgboost":   XGBoostBaseline,
    "mlp":       MLPBaseline,
    "mlp_eigen": MLPEigenBaseline,
    "gcn":       GCNBaseline,
    "gcn_faiss": GCNFaissBaseline,
}

# Models that consume the manifold eigenbasis (need build_manifold_kernel).
EIGEN_MODELS = {"mlp_eigen"}
# Models that consume the manifold graph topology (need build_manifold_graph).
GRAPH_MODELS = {"gcn_faiss"}
# Both families require coords in the reference frame the graph was built in.
REFERENCE_FRAME_MODELS = EIGEN_MODELS | GRAPH_MODELS

def _resolve_lipid_filter(selected_lipids_names, lipid_indices=None, lipid_names=None):
    """Resolve a lipid filter spec to an int array of column indices.

    Returns None to mean "all lipids". Hard-errors on miss or ambiguity.
    Mirrors MaldiExperiment._resolve_lipid_filter so behavior matches the
    LGP/Manifold runs.
    """
    if lipid_indices is None and lipid_names is None:
        return None
    all_names = [str(n) for n in selected_lipids_names]

    if lipid_names is not None:
        resolved = []
        for name in lipid_names:
            target = name.strip()
            exact = [i for i, n in enumerate(all_names) if n == target]
            if len(exact) == 1:
                resolved.append(exact[0]); continue
            if len(exact) > 1:
                raise ValueError(f"{name!r} matches multiple lipids exactly: {exact}")
            sub = [i for i, n in enumerate(all_names) if target.lower() in n.lower()]
            if len(sub) == 0:
                raise ValueError(f"No lipid matches {name!r}. First 10: {all_names[:10]}")
            if len(sub) > 1:
                raise ValueError(
                    f"Ambiguous lipid name {name!r}; matches: "
                    + ", ".join(f"[{i}]{all_names[i]}" for i in sub)
                )
            resolved.append(sub[0])
        idx = np.asarray(resolved, dtype=np.int64)
    else:
        idx = np.asarray(lipid_indices, dtype=np.int64)
        if idx.min() < 0 or idx.max() >= len(all_names):
            raise ValueError(f"lipid index out of range [0, {len(all_names)})")

    # Dedupe but preserve order
    _, keep = np.unique(idx, return_index=True)
    idx = idx[np.sort(keep)]

    selected = [all_names[i] for i in idx]
    logging.info(f"Reconstruction restricted to {len(idx)} lipid(s): {selected}")
    return idx


def _write_per_lipid_volumes(volume_path, predictions, indices, template_shape,
                              all_names, col_indices, suffix=""):
    """Write {lipid_name}_volume{suffix}.npy and its 255-normalized sibling.

    predictions: (N, K) where K = len(col_indices). col_indices maps each
    column to its global lipid index in selected_lipids_names.
    """
    if not isinstance(indices, np.ndarray):
        raise TypeError(
            f"_write_per_lipid_volumes expected indices: ndarray, "
            f"got {type(indices).__name__}. Check argument order at call site."
        )
    z, y, x = indices[:, 0], indices[:, 1], indices[:, 2]

    for col_pos, lipid_global_idx in enumerate(
            tqdm(col_indices, desc="per-lipid volumes")):
        lipid_name = all_names[int(lipid_global_idx)]
        out_path = volume_path / f"{lipid_name}_volume{suffix}.npy"
        out_255  = volume_path / f"{lipid_name}_volume255{suffix}.npy"
        if out_path.exists() and out_255.exists():
            continue
        vol = np.full(template_shape, np.nan, dtype=np.float32)
        vol[z, y, x] = predictions[:, col_pos]
        np.save(out_path, vol)
        vmin = np.nanmin(vol); vmax = np.nanmax(vol)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            np.save(out_255, 255.0 * (vol - vmin) / (vmax - vmin))
        else:
            np.save(out_255, vol)

def reconstruct(model, config, template_volume, coord_mean, coord_std,
                col_means, col_stds, mode: str, region_bbox, threshold: float,
                batch_size: int, lipid_filter=None, render_voxels_only: bool = False):
    """One-pass reconstruction. Accumulates filtered predictions in RAM,
    writes consolidated `predictions{tag}.npy` + per-lipid volumes.

    Returns (volume_path, suffix) so the caller can render from disk.
    """
    all_names = [str(n) for n in config.selected_lipids_names]
    n_lipids_total = len(all_names)

    if lipid_filter is None:
        col_indices = np.arange(n_lipids_total, dtype=np.int64)
        filter_tag = ""
    else:
        col_indices = np.asarray(lipid_filter, dtype=np.int64)
        if len(col_indices) <= 4:
            filter_tag = "_lipids_" + "_".join(str(i) for i in col_indices)
        else:
            import hashlib
            h = hashlib.sha1(col_indices.tobytes()).hexdigest()[:8]
            filter_tag = f"_lipids_{len(col_indices)}_{h}"
    n_lipids_out = len(col_indices)

    # --- Resolve voxel set + paths ---
    if mode == "whole_brain":
        keep = template_volume > threshold
        n_full = int(keep.sum())
        if render_voxels_only:
            # Sparse volumes go to their own dir AND carry their own suffix: the
            # dense volume/*_volume.npy files are read by napari/analysis scripts
            # that would silently get a hollow volume if we overwrote them.
            from render_lipid_volumes import render_voxel_mask
            keep = keep & render_voxel_mask(template_volume.shape)
            volume_path = config.exp_path / "volume_sparse"
            suffix = "_sparse"
        else:
            volume_path = config.exp_path / "volume"
            suffix = ""
        volume_path.mkdir(parents=True, exist_ok=True)
        non_zero_indices = np.argwhere(keep).astype(np.int32)
        n_keep = non_zero_indices.shape[0]
        if render_voxels_only:
            logging.info(
                f"Render-voxels-only reconstruction: {n_keep:,} voxels "
                f"({n_full / max(n_keep, 1):.1f}x fewer than whole-brain's {n_full:,}). "
                f"Renders read {suffix!r} volumes; dense volume/ is NOT written."
            )
        else:
            logging.info(f"Whole-brain reconstruction: {n_keep:,} voxels")
    elif mode == "region":
        if region_bbox is None:
            raise ValueError("--reconstruct region requires --region-bbox")
        bbox_str = "_".join(str(int(b)) for b in region_bbox)
        volume_path = config.exp_path / f"volume_region_{bbox_str}"
        suffix = f"_region_{bbox_str}"
        volume_path.mkdir(parents=True, exist_ok=True)
        zmin, zmax, ymin, ymax, xmin, xmax = (int(b) for b in region_bbox)
        sub = template_volume[zmin:zmax, ymin:ymax, xmin:xmax]
        z, y, x = np.where(sub > threshold)
        if z.shape[0] == 0:
            logging.warning(f"No voxels >{threshold} in bbox {region_bbox}. Skipping.")
            return volume_path, suffix
        non_zero_indices = np.stack(
            [z + zmin, y + ymin, x + xmin], axis=1,
        ).astype(np.int32)
        logging.info(f"Region reconstruction: {non_zero_indices.shape[0]:,} voxels")
    else:
        raise ValueError(f"unknown reconstruct mode: {mode}")

    n_voxels = non_zero_indices.shape[0]

    preds_file       = volume_path / f"predictions{filter_tag}.npy"
    indices_file     = volume_path / "predictions_indices.npy"
    filter_meta_file = volume_path / f"predictions{filter_tag}.lipidcols.npy"

    # --- Resume: same voxel set AND same filter ---
    if preds_file.exists() and indices_file.exists():
        saved_indices = np.load(indices_file)
        same_voxels = (saved_indices.shape == non_zero_indices.shape and
                       np.array_equal(saved_indices, non_zero_indices))
        same_filter = True
        if filter_meta_file.exists():
            saved_cols = np.load(filter_meta_file)
            same_filter = (saved_cols.shape == col_indices.shape and
                           np.array_equal(saved_cols, col_indices))
        if same_voxels and same_filter:
            logging.info(f"Reusing cached predictions at {preds_file}")
            all_preds = np.load(preds_file, mmap_mode="r")
            _write_per_lipid_volumes(
                volume_path, all_preds, non_zero_indices,
                template_volume.shape, all_names, col_indices, suffix,
            )
            return volume_path, suffix
        else:
            logging.warning(
                f"Cache mismatch (same_voxels={same_voxels}, "
                f"same_filter={same_filter}); recomputing."
            )

    # --- Coords -> standardized -> dataloader ---
    non_zero_ccf = non_zero_indices.astype(np.float32) * 0.025
    non_zero_ccf = torch.tensor(non_zero_ccf, dtype=torch.float32)
    non_zero_ccf = (non_zero_ccf - coord_mean) / coord_std

    # Slice the coord tensor directly rather than going through a
    # DataLoader/TensorDataset: that calls __getitem__ once PER VOXEL and then
    # collates, which at 34M voxels costs more than the forward pass itself (and
    # gets *worse* with a bigger batch, since the collate stacks more tensors).
    # The dataset's second column (indices) was built and collated but never read.

    # --- Pre-allocate (filtered width only) ---
    pred_bytes = n_voxels * n_lipids_out * 4
    logging.info(
        f"Allocating {pred_bytes / 1e9:.2f} GB for predictions "
        f"({n_voxels:,} voxels × {n_lipids_out} lipid{'s' if n_lipids_out != 1 else ''}); "
        f"inference batch {batch_size:,} -> {(n_voxels + batch_size - 1) // batch_size:,} batches"
    )
    all_preds = np.empty((n_voxels, n_lipids_out), dtype=np.float32)

    # Slice de-normalization params once
    col_means_np = col_means.cpu().numpy()[col_indices]
    col_stds_np  = col_stds.cpu().numpy()[col_indices]

    cursor = 0
    n_batches = (n_voxels + batch_size - 1) // batch_size
    for start in tqdm(range(0, n_voxels, batch_size), total=n_batches,
                      desc=f"reconstruct[{mode}]"):
        coords_batch = non_zero_ccf[start:start + batch_size]
        pred_norm = model.predict(coords_batch)        # (B, n_lipids_total)
        full_preds_np = pred_norm.detach().cpu().numpy()

        # Slice BEFORE de-normalization so the in-loop array stays at (B, K)
        sliced = full_preds_np[:, col_indices]
        sliced = sliced * col_stds_np + col_means_np
        if config.log_transform:
            sliced = np.exp(sliced) - 1e-10

        n_b = sliced.shape[0]
        all_preds[cursor:cursor + n_b] = sliced.astype(np.float32, copy=False)
        cursor += n_b

    # --- Save consolidated ---
    np.save(preds_file, all_preds)
    np.save(indices_file, non_zero_indices)
    np.save(filter_meta_file, col_indices)
    logging.info(
        f"Saved predictions: {all_preds.shape} -> {preds_file}  "
        f"(lipids: {[all_names[i] for i in col_indices]})"
    )

    # --- Per-lipid volumes ---
    _write_per_lipid_volumes(
        volume_path, all_preds, non_zero_indices,
        template_volume.shape, all_names, col_indices, suffix,
    )

    return volume_path, suffix

def _render_diagnostics(config, renders_dir, lipid_filter):
    """Per-lipid diagnostics from the held-out TEST split (mirror of
    MaldiExperiment.render_reconstruction's diagnostics block).

    Reads the raw (de-standardized) test predictions/true values written by the
    training path — so a measurement and a prediction exist for each point — and
    writes ``{lipid}_diagnostics.png`` for each reconstructed lipid. Non-fatal:
    a plotting hiccup must never invalidate a completed reconstruction.
    """
    try:
        from render_lipid_volumes import render_lipid_diagnostics
        test_dir = config.exp_path / "test"
        true_path = test_dir / "true_values.npy"
        pred_path = test_dir / "predictions.npy"
        if not (true_path.exists() and pred_path.exists()):
            logging.warning("Skipping diagnostics: test predictions/true values "
                            "not found (run training first).")
            return
        true_te = np.load(true_path)
        pred_te = np.load(pred_path)
        names = [str(n) for n in config.selected_lipids_names]
        filt = (list(lipid_filter) if lipid_filter is not None
                else list(range(len(names))))
        renders_dir.mkdir(parents=True, exist_ok=True)
        for gi in filt:
            render_lipid_diagnostics(
                true_te[:, gi], pred_te[:, gi], names[gi],
                renders_dir / f"{names[gi]}_diagnostics.png",
            )
        logging.info(f"Diagnostics written to {renders_dir}")
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Reconstruction diagnostics plotting failed: {e}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logging.info(f"Args: {args}")

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    config = MaldiConfig.from_args(args)
    region_bbox = args.get("region_bbox")
    if region_bbox is not None:
        # The bbox parquet-filter helpers (apply_region_to_config / bbox_to_mm_bounds)
        # were removed upstream ("Remove bbox" commit); only whole-brain remains.
        raise NotImplementedError(
            "--region-bbox is no longer supported (its parquet-filter helpers were "
            "removed upstream). Run whole-brain by omitting --region-bbox."
        )
    config.exp_path.mkdir(parents=True, exist_ok=True)

    rec_mode = args["reconstruct"]
    if rec_mode == "auto":
        rec_mode = "region" if region_bbox is not None else "whole_brain"
    elif rec_mode == "region" and region_bbox is None:
        raise ValueError("--reconstruct region requires --region-bbox")

    model_cls = MODEL_REGISTRY[args["model"]]
    model = model_cls()

    # Manifold-aware models (mlp_eigen / gcn_faiss): build the eigenbasis and/or
    # the FAISS graph and attach it, in both the training and --skip-training
    # paths (cheap — same graph/eigvec caches as the manifold runners). Their
    # coords must live in the reference frame the graph was built in, so we
    # standardize with coord_norm_from_reference.
    needs_eigen = args["model"] in EIGEN_MODELS
    needs_graph = args["model"] in GRAPH_MODELS
    needs_ref_frame = args["model"] in REFERENCE_FRAME_MODELS
    ref_mean = ref_std = None
    if needs_ref_frame:
        from manifold_gp.utils.nearest_neighbors import apply_faiss_cpu_args
        apply_faiss_cpu_args(args)
        ref_mean, ref_std = coord_norm_from_reference(config.reference_file)
    if needs_eigen:
        manifold_kernel, _knn = build_manifold_kernel(args, config, ref_mean, ref_std)
        model.attach_kernel(manifold_kernel)
    if needs_graph:
        knn, edge_index, edge_value, _gk = build_manifold_graph(
            args, config, ref_mean, ref_std,
        )
        model.attach_graph(knn, edge_index, edge_value, args["device"],
                           interp_k=args["gcn_faiss_interp_k"])

    model_path = config.exp_path / "model.pth"
    coord_mean_path = config.exp_path / "coord_mean.pth"
    coord_std_path = config.exp_path / "coord_std.pth"
    means_path = config.exp_path / "lipid_means.pth"
    stds_path = config.exp_path / "lipid_stds.pth"
    p_lipids = len(config.selected_lipids_names)

    if args["skip_training"]:
        if not all(p.exists() for p in [model_path, coord_mean_path, coord_std_path,
                                         means_path, stds_path]):
            raise FileNotFoundError(
                "--skip-training requires a prior fit. Missing one of: "
                f"{[str(p) for p in [model_path, coord_mean_path, coord_std_path, means_path, stds_path]]}"
            )
        logging.info("Skipping training; loading saved model + stats")
        model.load(model_path, p_lipids, args)
        coord_mean = torch.load(coord_mean_path)
        coord_std = torch.load(coord_std_path)
        col_means = torch.load(means_path)
        col_stds = torch.load(stds_path)
    else:
        # ---- Training path: load splits, fit, write train/test outputs ----
        y_train_logged, y_train_original, coords_train, pixel_idx_train = _load_split(
            config, train=True,
        )
        col_means = y_train_logged.mean(dim=0)
        col_stds  = y_train_logged.std(dim=0).clamp(min=1e-8)
        if needs_ref_frame:
            # Reference frame so raw_eigenvectors / nearest-node search is valid;
            # saved below so reconstruction standardizes voxels in the same frame.
            coord_mean, coord_std = ref_mean, ref_std
        else:
            coord_mean = coords_train.mean(dim=0)
            coord_std  = coords_train.std(dim=0).clamp(min=1e-8)
        y_train_norm, coords_train_std = _normalize_with_train_stats(
            y_train_logged, coords_train, col_means, col_stds, coord_mean, coord_std,
        )

        # Persist stats *before* training so a partial fit can still be picked up.
        torch.save(col_means, means_path)
        torch.save(col_stds, stds_path)
        torch.save(coord_mean, coord_mean_path)
        torch.save(coord_std, coord_std_path)
        logging.info(f"train: {y_train_norm.shape[0]:,} rows, {y_train_norm.shape[1]} lipids")

        # Transductive models (euclid) run on raw intensities in CCF mm, so they
        # need the frames fit() cannot recover from its standardized inputs.
        if hasattr(model, "set_fit_context"):
            model.set_fit_context(coord_mean, coord_std, col_means, col_stds,
                                  config.log_transform, config.selected_lipids_names,
                                  config.exp_path)

        y_test_logged, y_test_original, coords_test, pixel_idx_test = _load_split(
            config, train=False,
        )
        y_test_norm, coords_test_std = _normalize_with_train_stats(
            y_test_logged, coords_test, col_means, col_stds, coord_mean, coord_std,
        )
        logging.info(f"test:  {y_test_norm.shape[0]:,} rows")

        logging.info(f"Fitting baseline: {args['model']}")
        t0 = time.time()
        pred_train_norm, pred_test_norm = model.fit(
            coords_train_std, y_train_norm,
            coords_test_std,  y_test_norm,
            args,
        )
        logging.info(f"  fit completed in {time.time()-t0:.1f}s")
        model.save(model_path)
        logging.info(f"  saved model -> {model_path}")

        pred_train = _denormalize(pred_train_norm, col_means, col_stds, config.log_transform)
        pred_test  = _denormalize(pred_test_norm,  col_means, col_stds, config.log_transform)

        train_dir = config.exp_path / "train"
        test_dir  = config.exp_path / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        np.save(train_dir / "predictions.npy", pred_train.numpy().astype(np.float32))
        np.save(train_dir / "true_values.npy", y_train_original.numpy().astype(np.float32))
        torch.save(coords_train_std, train_dir / "coordinates.pth")
        torch.save(pixel_idx_train,  train_dir / "coordinates_pixel_index.pth")

        np.save(test_dir / "predictions.npy",  pred_test.numpy().astype(np.float32))
        np.save(test_dir / "true_values.npy",  y_test_original.numpy().astype(np.float32))
        torch.save(coords_test_std, test_dir / "coordinates.pth")
        torch.save(pixel_idx_test,  test_dir / "coordinates_pixel_index.pth")

        logging.info("Per-split outputs written.")

    # Per-lipid held-out metrics table (read by lgp_report.py). Runs in both the
    # training and --skip-training paths (the npy already exist either way).
    try:
        from lgp_metrics import write_metrics
        for _split in ("test", "train"):
            write_metrics(config.exp_path, _split,
                          lipid_names=config.selected_lipids_names)
        logging.info("Wrote per-lipid metrics.csv")
    except Exception as _e:  # noqa: BLE001
        logging.warning(f"metrics.csv generation failed: {_e}")

    recon_lipids = args.get("reconstruction_lipids", None)
    lipid_filter = None
    if recon_lipids:
        if all(isinstance(v, int) for v in recon_lipids):
            lipid_filter = _resolve_lipid_filter(
                config.selected_lipids_names, lipid_indices=recon_lipids,
            )
        else:
            lipid_filter = _resolve_lipid_filter(
                config.selected_lipids_names, lipid_names=recon_lipids,
            )
    if rec_mode == "none":
        logging.info("Skipping reconstruction (--reconstruct none).")
    else:
        template_volume=np.load(args["reference_file"])
        volume_path, suffix = reconstruct(
            model=model, config=config, template_volume=template_volume,
            coord_mean=coord_mean, coord_std=coord_std,
            col_means=col_means, col_stds=col_stds,
            mode=rec_mode, region_bbox=region_bbox,
            threshold=args["reconstruct_threshold"],
            # .get(): external callers rebind parse_args (e.g. sota/run_sota.py) and
            # a mirror missing these keys must fall back to defaults, not KeyError
            # AFTER training has already run.
            batch_size=args.get("inference_batch_size") or args["batch_size"],
            lipid_filter=lipid_filter,
            render_voxels_only=args.get("render_voxels_only", False),
        )

        # ---- Render per-lipid composites + diagnostics ----
        # Parity with MaldiExperiment.render_reconstruction: the composite PNGs
        # AND the per-lipid diagnostics (value distribution + true-vs-pred
        # scatter, linear & log) computed from the held-out TEST points.
        output_dir = config.exp_path / "renders"
        try:
            from render_lipid_volumes import render_selected_lipids
            render_selected_lipids(
                template_volume=template_volume,
                volume_dir=volume_path,
                output_dir=output_dir,
                selected_lipids_names=config.selected_lipids_names,
                lipid_indices=(list(lipid_filter)
                                if lipid_filter is not None else None),
                suffix=suffix,
                n_rotation_frames=10,
            )
            logging.info(f"Renders written to {output_dir}")
        except Exception as e:
            logging.error(f"Rendering failed (reconstruction still saved): {e}")

        _render_diagnostics(config, output_dir, lipid_filter)

    logging.info("Done.")


if __name__ == "__main__":
    main()