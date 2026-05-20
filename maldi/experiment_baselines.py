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
    mean      per-lipid training mean. The floor.
    linear    Ridge regression (closed form).
    xgboost   p independent XGBRegressors with early stopping.
    mlp       configurable-depth MLP (default 128x128x128, SiLU, dropout).

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
import logging
import pickle
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
from utils import apply_region_to_config


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
                        choices=["mean", "linear", "xgboost", "mlp"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--xgb-n-estimators", type=int, default=400)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-lr", type=float, default=0.05)

    # Reconstruction
    parser.add_argument("--template-name", dest="template_name", type=str, required=True, help="Template name.")
    parser.add_argument("--reference-file", dest="reference_file", type=str, required=True, help="The reference image npy.")
    parser.add_argument("--annotations-file", dest="annotations_file", type=str, help="The annotations if needed.")
    parser.add_argument("--reconstruct", type=str, default="auto",
                        choices=["none", "auto", "whole_brain", "region"],
                        help="auto = whole_brain if no bbox, region if bbox.")
    parser.add_argument("--reconstruct-threshold", type=float, default=5.0)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip fitting; only run reconstruction from a saved model.")
    parser.add_argument(
        "--reconstruction-lipids", dest="reconstruction_lipids",
        nargs="+", default=None,
        help=("Restrict reconstruction to these lipids. Accepts indices "
              "(0 5 10) or names ('PA 36:4' 'PE 40:7'). Default: all."),
    )

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

    def _build(self, p, hidden, dropout, device):
        dims = [3] + list(hidden) + [p]
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
        for epoch in range(args["epochs"]):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            loss_sum = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                pred = self.model(Xtr[idx])
                loss = F.mse_loss(pred, Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item() * idx.shape[0]
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


MODEL_REGISTRY = {
    "mean":    MeanBaseline,
    "linear":  LinearBaseline,
    "xgboost": XGBoostBaseline,
    "mlp":     MLPBaseline,
}

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
                batch_size: int, lipid_filter=None):
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
        volume_path = config.exp_path / "volume"
        suffix = ""
        volume_path.mkdir(parents=True, exist_ok=True)
        non_zero_indices = np.argwhere(template_volume > threshold).astype(np.int32)
        logging.info(f"Whole-brain reconstruction: {non_zero_indices.shape[0]:,} voxels")
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

    ccf_dataset = torch.utils.data.TensorDataset(
        non_zero_ccf, torch.tensor(non_zero_indices, dtype=torch.int32),
    )
    ccf_loader = torch.utils.data.DataLoader(
        ccf_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # --- Pre-allocate (filtered width only) ---
    pred_bytes = n_voxels * n_lipids_out * 4
    logging.info(
        f"Allocating {pred_bytes / 1e9:.2f} GB for predictions "
        f"({n_voxels:,} voxels × {n_lipids_out} lipid{'s' if n_lipids_out != 1 else ''})"
    )
    all_preds = np.empty((n_voxels, n_lipids_out), dtype=np.float32)

    # Slice de-normalization params once
    col_means_np = col_means.cpu().numpy()[col_indices]
    col_stds_np  = col_stds.cpu().numpy()[col_indices]

    cursor = 0
    for batch in tqdm(ccf_loader, desc=f"reconstruct[{mode}]"):
        coords_batch = batch[0]
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
    apply_region_to_config(config, region_bbox)
    config.exp_path.mkdir(parents=True, exist_ok=True)

    rec_mode = args["reconstruct"]
    if rec_mode == "auto":
        rec_mode = "region" if region_bbox is not None else "whole_brain"
    elif rec_mode == "region" and region_bbox is None:
        raise ValueError("--reconstruct region requires --region-bbox")

    model_cls = MODEL_REGISTRY[args["model"]]
    model = model_cls()

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
            batch_size=args["batch_size"],
            lipid_filter=lipid_filter,
        )

        # ---- Render per-lipid composites ----
        try:
            from render_lipid_volumes import render_selected_lipids
            output_dir = config.exp_path / "renders"
            render_selected_lipids(
                template_volume=template_volume,
                volume_dir=volume_path,
                output_dir=output_dir,
                selected_lipids_names=config.selected_lipids_names,
                lipid_indices=(list(lipid_filter)
                                if lipid_filter is not None else None),
                suffix=suffix,
            )
            logging.info(f"Renders written to {output_dir}")
        except Exception as e:
            logging.error(f"Rendering failed (reconstruction still saved): {e}")

    logging.info("Done.")


if __name__ == "__main__":
    main()