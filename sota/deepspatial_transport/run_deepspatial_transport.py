"""Faithful DeepSpatial (transport mode) on MALDI.

Runs the OFFICIAL ``deepspatial`` package as the paper intends -- within-specimen
slice interpolation -- adapted to MALDI:

  * Train the GiT flow-matching transport model on the TRAIN-fold mice, using
    UOT-coupled cross-section cell (voxel) pairs. Multiple mice are handled by
    building WITHIN-mouse adjacent-section pairs with a shared (global) atlas-
    region label set, so no section pair ever crosses a mouse boundary.
  * For each held-out TEST-fold mouse, reconstruct the full 3D brain volume by
    solving the probability-flow ODE between its adjacent sections, rasterize the
    synthesized cells onto the CCF grid, and render per-lipid volumes with the
    same renderer the manifold / run_sota experiments use.
  * Quantitative eval: leave-one-section-out interpolation on the test mice
    (drop an interior section, reconstruct its gap, match synthesized cells to
    the held section by nearest in-plane position, score per-lipid corr / RMSE).

This is the sole DeepSpatial implementation (the earlier harness-plugged
regression stand-in was removed). Launch via ``MODEL=deepspatial
./sota/run_sota.sh`` (which delegates here) or ``run_deepspatial_transport.sh``.
See README.md.
"""
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

_HERE = Path(__file__).resolve().parent
_MALDI = _HERE.parent.parent / "maldi"
sys.path.insert(0, str(_MALDI))
sys.path.insert(0, str(_HERE.parent))          # sota/ (sota_utils)
sys.path.insert(0, str(_HERE))                 # adapter

from config import MaldiConfig                                   # noqa: E402
from experiment_baselines import _write_per_lipid_volumes, _resolve_lipid_filter  # noqa: E402
from sota_utils import wandb_log                                 # noqa: E402
import adapter                                                   # noqa: E402

from deepspatial import DeepSpatial                              # noqa: E402
from deepspatial.data_utils.uot_solver import compute_uot_coupling  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = ArgumentParser(description="Faithful DeepSpatial (transport) on MALDI.")
    p.add_argument("--mode", default="lgp")
    p.add_argument("--dataset-path", dest="dataset_path", required=True)
    p.add_argument("--maldi-file", dest="maldi_file", required=True)
    p.add_argument("--exp-name", dest="exp_name", required=True)
    p.add_argument("--available-lipids-file", dest="available_lipids_file", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--slices-dataset-file", dest="slices_dataset_file", required=True)
    p.add_argument("--template-name", dest="template_name", default="reference")
    p.add_argument("--reference-file", dest="reference_file", required=True)
    p.add_argument("--annotations-file", dest="annotations_file", required=True,
                   help="Atlas annotation volume (level_15annot.npy) for region labels.")
    p.add_argument("--seed", type=int, default=416465)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-transform", dest="log_transform", action="store_true")
    # MaldiConfig requires these
    p.add_argument("--num-inducing", dest="num_inducing", type=int, default=100)
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=5)
    p.add_argument("--kernel", default="matern")
    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--n-pixels", dest="n_pixels", type=int, default=10)
    p.add_argument("--learning-rate", dest="learning_rate", type=float, default=2e-4)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--reconstruction-lipids", dest="reconstruction_lipids",
                   nargs="+", default=None)
    p.add_argument("--reconstruct", default="whole_brain",
                   choices=["none", "whole_brain"])
    p.add_argument("--reconstruct-threshold", dest="reconstruct_threshold",
                   type=float, default=5.0)
    # DeepSpatial model / transport knobs
    p.add_argument("--ds-hidden-size", dest="ds_hidden_size", type=int, default=256)
    p.add_argument("--ds-depth", dest="ds_depth", type=int, default=6)
    p.add_argument("--ds-heads", dest="ds_heads", type=int, default=8)
    p.add_argument("--ds-patch", dest="ds_patch", type=int, default=8)
    # NOTE: the upstream defaults are lambda_g=0.1, lambda_c=10 -- weighting the
    # loss 100:1 toward CELL-TYPE over gene expression (cell type is the paper's
    # product). For MALDI the LIPIDS are the product and the atlas region is only
    # auxiliary conditioning, so we FLIP it: prioritise lipid fidelity, keep a
    # small region term. Without this the model learns per-lipid means but not
    # spatial structure (per-voxel corr ~0).
    p.add_argument("--ds-lambda-g", dest="ds_lambda_g", type=float, default=1.0)
    p.add_argument("--ds-lambda-c", dest="ds_lambda_c", type=float, default=0.1)
    p.add_argument("--ds-steps", dest="ds_steps", type=int, default=20)
    p.add_argument("--ds-thickness", dest="ds_thickness", type=float, default=0.02,
                   help="Inter-plane spacing in z_coord (xccf, mm) units. Smaller "
                        "=> denser reconstruction. target_cells ~ n_sec*(gap/thickness); "
                        "MALDI section gaps are ~0.15-2 mm, so ~0.02 fills the volume.")
    p.add_argument("--ds-alpha-spatial", dest="ds_alpha_spatial", type=float, default=0.5)
    p.add_argument("--ds-uot-reg", dest="ds_uot_reg", type=float, default=0.8)
    p.add_argument("--ds-uot-tau", dest="ds_uot_tau", type=float, default=0.05)
    p.add_argument("--ds-max-cells", dest="ds_max_cells", type=int, default=3000,
                   help="Voxels per section subsampled before UOT (train). Kept "
                        "small because the UOT coupling is O(N^2).")
    p.add_argument("--ds-max-cells-recon", dest="ds_max_cells_recon", type=int,
                   default=30000,
                   help="Source voxels per section at reconstruction. This sets the "
                        "IN-PLANE reconstruction density: every synthesized cell is a "
                        "transported source cell, so coverage ~ this many points per "
                        "section (measured sections are 60-90k). Higher = denser but "
                        "more target cells => more memory/time (total scales with this "
                        "/ thickness); 0 = use ALL voxels. No UOT at reconstruction.")
    p.add_argument("--ds-recon-batch", dest="ds_recon_batch", type=int, default=8000,
                   help="Source voxels per reconstruct() CALL. Each gap's sections "
                        "are chunked into batches of this size (cycling to cover all "
                        "requested source cells) so ONE call never allocates "
                        "batch*(gap/thickness)*n_lipids on the GPU -- this is what lets "
                        "--ds-max-cells-recon=0 (all voxels) run without OOM. Lower if "
                        "you still OOM; higher for fewer/faster calls.")
    p.add_argument("--ds-recon-chunk", dest="ds_recon_chunk", type=int, default=32768,
                   help="ODE-integration batch (cells processed in parallel per "
                        "solver step). The upstream default (2048) badly under-uses "
                        "the GPU; larger is much faster until the card saturates. "
                        "Lower only if the ODE step itself OOMs.")
    p.add_argument("--ds-n-samples", dest="ds_n_samples", type=int, default=50000,
                   help="Trajectory pairs sampled per mouse (n_samples_base).")
    p.add_argument("--ds-recon-scope", dest="ds_recon_scope", default="follow",
                   choices=["follow", "per-mouse", "cross-mouse"],
                   help="Scope of the full-volume reconstruction. 'per-mouse': "
                        "transport only within each test mouse's own adjacent "
                        "sections -> one (sparse, partial-AP) volume per mouse. "
                        "'cross-mouse': pool ALL test sections, sort by CCF AP and "
                        "transport across adjacent sections regardless of animal -> "
                        "one canonical whole-brain volume (mirrors cross-mouse "
                        "training pairing; sound only when mice register well). "
                        "'follow' (default): cross-mouse iff --ds-pairing=cross-mouse, "
                        "else per-mouse -- so training and reconstruction agree.")
    p.add_argument("--ds-pairing", dest="ds_pairing", default="within-mouse",
                   choices=["within-mouse", "cross-mouse"],
                   help="How training UOT section pairs are formed. 'within-mouse' "
                        "(default, faithful): adjacent sections of the SAME mouse -- "
                        "real within-specimen tissue continuity. 'cross-mouse': pool "
                        "every section across mice into one AP-ordered stack and pair "
                        "adjacent sections regardless of animal (treats the cohort as "
                        "serial sections of one canonical brain -- denser AP sampling + "
                        "cross-mouse gap-filling). Only sound when mice register well "
                        "to the common CCF frame.")
    p.add_argument("--force-retrain", dest="force_retrain", action="store_true",
                   help="Ignore any existing checkpoint and train from scratch "
                        "(default: resume from the checkpoint if one is present).")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", dest="wandb_project", default="sota_maldi")
    args = vars(p.parse_args())
    if args.get("reconstruction_lipids"):
        try:
            args["reconstruction_lipids"] = [int(v) for v in args["reconstruction_lipids"]]
        except ValueError:
            pass
    return args


# ---------------------------------------------------------------------------
# Trajectory dataset (within-mouse pairs, global labels, official UOT solver)
# ---------------------------------------------------------------------------
class _TrajDataset(Dataset):
    def __init__(self, tensors):
        self.t = tensors
        self.n = tensors["x0"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.t.items()}


def _apply_norm(sections, stats):
    """Write spatial_norm / z_norm using PRE-COMPUTED (training) stats, so test
    mice land in the same frame the model was trained in."""
    for a in sections:
        c = a.obsm["spatial"].copy().astype(np.float32)
        c[:, 0] = (c[:, 0] - stats["x_min"]) / stats["x_range"]
        c[:, 1] = (c[:, 1] - stats["y_min"]) / stats["y_range"]
        a.obsm["spatial_norm"] = c
        z = float(a.obs["z_coord"].iloc[0])
        a.obs["z_norm"] = (z - stats["z_min"]) / stats["z_range"]


def build_trajectories(train_by_mouse, le, num_classes, args):
    """UOT trajectory pairs -> materialized tensor dict.

    Two pairing modes (``--ds-pairing``):
      * ``within-mouse`` (default, faithful): adjacent sections OF THE SAME mouse
        -- real within-specimen tissue continuity; no cross-animal correspondence.
        Each mouse gets the full ``--ds-n-samples`` budget.
      * ``cross-mouse``: pool every section from every mouse into one stack ordered
        by CCF AP coordinate (``z_norm``) and pair adjacent sections regardless of
        animal -- treats the cohort as serial sections of one canonical brain
        (denser AP sampling + cross-mouse gap-filling). The budget is scaled by the
        mouse count so the total trajectory volume matches within-mouse. Only sound
        when the mice register well to the common frame.
    """
    def onehot(labels):
        idx = le.transform(np.asarray(labels).astype(str))
        return np.eye(num_classes, dtype=np.float32)[idx]

    pairing = args.get("ds_pairing", "within-mouse")
    n_mice = len(train_by_mouse)
    if pairing == "within-mouse":
        # One group per mouse; each spends the full n_samples budget on its own
        # adjacent-section pairs (order is the per-mouse section stacking order).
        groups = [(secs, args["ds_n_samples"]) for secs in train_by_mouse.values()]
    elif pairing == "cross-mouse":
        # Pool ALL sections, sort by CCF AP (z_norm), pair adjacent across animals.
        # Scale the budget by n_mice so the total trajectory count matches the
        # within-mouse mode (which spends n_samples per mouse).
        all_secs = sorted(
            (a for secs in train_by_mouse.values() for a in secs),
            key=lambda a: float(a.obs["z_norm"].iloc[0]))
        groups = [(all_secs, args["ds_n_samples"] * max(n_mice, 1))]
    else:
        raise ValueError(
            f"unknown --ds-pairing {pairing!r} (within-mouse|cross-mouse)")

    acc = {k: [] for k in ("x0", "g0", "c0", "z0", "x1", "g1", "c1", "z1", "delta_z")}
    for secs, budget in groups:
        pairs = [(secs[k], secs[k + 1]) for k in range(len(secs) - 1)]
        if not pairs:
            continue
        weights = [a0.n_obs * a1.n_obs for a0, a1 in pairs]
        tot = sum(weights)
        for (a0, a1), w in zip(pairs, weights):
            n = int(budget * w / max(tot, 1))
            if n <= 0:
                continue
            z0 = float(a0.obs["z_norm"].iloc[0]); z1 = float(a1.obs["z_norm"].iloc[0])
            # Skip degenerate pairs with no AP extent (can arise in cross-mouse
            # when two animals have a section at ~the same AP): a zero-gap flow
            # target is undefined. within-mouse consecutive sections never hit this.
            if abs(z1 - z0) < 1e-6:
                continue
            x0, g0 = a0.obsm["spatial_norm"], np.asarray(a0.X, np.float32)
            x1, g1 = a1.obsm["spatial_norm"], np.asarray(a1.X, np.float32)
            c0 = onehot(a0.obs["cell_class"]); c1 = onehot(a1.obs["cell_class"])
            pi = compute_uot_coupling(x0, g0, c0, x1, g1, c1,
                                      alpha_spatial=args["ds_alpha_spatial"],
                                      uot_reg=args["ds_uot_reg"],
                                      uot_tau=args["ds_uot_tau"])
            pf = pi.ravel()
            s = pf.sum()
            if s <= 0:
                continue
            sel = np.random.choice(len(pf), size=n, p=pf / s, replace=True)
            i0, i1 = np.unravel_index(sel, pi.shape)
            acc["x0"].append(x0[i0]); acc["g0"].append(g0[i0]); acc["c0"].append(c0[i0])
            acc["z0"].append(np.full((n, 1), z0, np.float32))
            acc["x1"].append(x1[i1]); acc["g1"].append(g1[i1]); acc["c1"].append(c1[i1])
            acc["z1"].append(np.full((n, 1), z1, np.float32))
            acc["delta_z"].append(np.full((n, 1), z1 - z0, np.float32))
    tensors = {k: torch.from_numpy(np.concatenate(v, 0)) for k, v in acc.items() if v}
    if not tensors:
        raise RuntimeError("No trajectory pairs built (need >=2 sections; "
                           "cross-mouse needs >=2 sections total).")
    return tensors


# ---------------------------------------------------------------------------
# Leave-one-section-out interpolation metric
# ---------------------------------------------------------------------------
def _subsample(a, n, rng):
    """Random-subsample an AnnData section to <=n rows (for cheap LOSO recon)."""
    if a.n_obs <= n:
        return a
    return a[rng.choice(a.n_obs, n, replace=False)].copy()


def loso_predictions(ds, by_mouse, args, max_secs=6):
    """Leave-one-section-out interpolation over the given mice: drop an interior
    section, reconstruct its gap from the two neighbours, and predict each held
    voxel from its nearest synthesized cell. Returns (true, pred, pixel_index,
    ccf_coords) stacked over all evaluated held voxels -- the harness-layout
    per-split predictions (regeneratable, so they are stored to disk).

    ``max_secs`` caps interior sections evaluated per mouse (bounds cost on the
    dense atlas mice). LOSO neighbours are capped so a 2-gap reconstruction stays
    cheap regardless of --ds-max-cells-recon.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(args["seed"])
    cap = min(args["ds_max_cells_recon"] or 4000, 4000)
    T, P, PIX, C = [], [], [], []
    for mouse, secs in by_mouse.items():
        if len(secs) < 3:
            continue
        n_done = 0
        for k in range(1, len(secs) - 1):
            if n_done >= max_secs:
                break
            a0 = _subsample(secs[k - 1], cap, rng)
            a2 = _subsample(secs[k + 1], cap, rng)
            _apply_norm([a0, a2], ds.spatial_stats)
            seg = ds.reconstruct_between_slices(
                a0, a2, thickness=args["ds_thickness"], steps=args["ds_steps"],
                chunk_size=int(args.get("ds_recon_chunk", 32768)), device=args["device"])
            held = secs[k]
            # Match each held voxel to its nearest synthesized cell in 3D
            # (yccf, zccf, xccf-depth). A 2D (y,z)-only match pulls values from
            # cells at ARBITRARY depths across the gap -- the held section sits at
            # one depth, so ignoring it mismatches values and tanks correlation.
            syn3 = np.column_stack([seg.obsm["spatial"].astype(np.float64),
                                    np.asarray(seg.obs["z_coord"], np.float64)])
            held3 = np.column_stack([held.obsm["spatial"].astype(np.float64),
                                     held.obs["xccf"].to_numpy(np.float64)])
            _, nn = cKDTree(syn3).query(held3, k=1)
            synX = seg.X.toarray() if hasattr(seg.X, "toarray") else np.asarray(seg.X)
            P.append(synX[nn].astype(np.float32))
            T.append(np.asarray(held.X, np.float32))
            PIX.append(held.obs[["x_index", "y_index", "z_index"]].to_numpy(np.int64))
            C.append(held.obs[["xccf", "yccf", "zccf"]].to_numpy(np.float32))
            n_done += 1
    if not T:
        return None, None, None, None
    return (np.concatenate(T), np.concatenate(P),
            np.concatenate(PIX), np.concatenate(C))


def _write_split(exp_path, split, true, pred, pix, coords):
    """Write per-split predictions in the harness layout so the shared reporting
    tools (lgp_metrics, lgp_report, diagnostics) can consume them."""
    d = exp_path / split
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "predictions.npy", pred.astype(np.float32))
    np.save(d / "true_values.npy", true.astype(np.float32))
    torch.save(torch.from_numpy(coords), d / "coordinates.pth")
    torch.save(torch.from_numpy(pix), d / "coordinates_pixel_index.pth")


def _accumulate(seg, ccf2idx, tshape, col_indices, K, run):
    """Rasterize a segment's K reconstruction lipids to unique CCF voxels and
    merge into the running (flat, sum, count) accumulator. Memory stays bounded
    by unique_voxels x K, independent of the number of synthesized cells."""
    fl = adapter.syn_flat_indices(seg, ccf2idx, tshape)
    X = seg.X.toarray() if hasattr(seg.X, "toarray") else np.asarray(seg.X)
    X = X[:, col_indices].astype(np.float32)
    u, inv = np.unique(fl, return_inverse=True)
    s = np.zeros((len(u), K), np.float32); c = np.zeros(len(u), np.float32)
    np.add.at(s, inv, X); np.add.at(c, inv, 1.0)
    rf, rs, rc = run
    if rf is None:
        return u, s, c
    mf = np.concatenate([rf, u]); ms = np.concatenate([rs, s]); mc = np.concatenate([rc, c])
    u2, i2 = np.unique(mf, return_inverse=True)
    rs2 = np.zeros((len(u2), K), np.float32); rc2 = np.zeros(len(u2), np.float32)
    np.add.at(rs2, i2, ms); np.add.at(rc2, i2, mc)
    return u2, rs2, rc2


def reconstruct_mouse(ds, secs, args, ccf2idx, template, col_indices, K, rng):
    """Dense full-volume reconstruction for one mouse, memory-safe.

    For each adjacent-section gap, the two sections' source voxels are chunked
    into ``--ds-recon-batch`` pieces (cycling the shorter one) and each chunk-pair
    is reconstructed separately, so a single reconstruct() call never allocates
    batch*(gap/thickness)*n_lipids on the GPU. This decouples IN-PLANE density
    (total source cells covered across chunks) from per-call memory -- the fix
    that lets --ds-max-cells-recon=0 fill the volume without OOM. Results are
    incrementally deduped into a running CCF-grid accumulator.
    """
    from tqdm import tqdm
    batch = max(1, int(args["ds_recon_batch"]))
    ode_chunk = int(args.get("ds_recon_chunk", 32768))
    cap = int(args["ds_max_cells_recon"])          # 0 => all voxels
    # Pre-count chunks for a progress bar (dense =0 runs can be many calls).
    plan = []
    for k in range(len(secs) - 1):
        u0 = secs[k].n_obs if cap <= 0 else min(secs[k].n_obs, cap)
        u1 = secs[k + 1].n_obs if cap <= 0 else min(secs[k + 1].n_obs, cap)
        plan.append((k, u0, u1, max(1, int(np.ceil(max(u0, u1) / batch)))))
    total_calls = sum(p[3] for p in plan)
    run = (None, None, None)
    pbar = tqdm(total=total_calls, desc="  recon (gap-chunks)", unit="call", leave=False)
    for k, use0, use1, nch in plan:
        a0, a1 = secs[k], secs[k + 1]
        p0 = rng.permutation(a0.n_obs)[:use0]
        p1 = rng.permutation(a1.n_obs)[:use1]
        for ci in range(nch):
            off = ci * batch
            c0 = p0[np.arange(off, off + batch) % use0]   # cycle shorter section
            c1 = p1[np.arange(off, off + batch) % use1]
            s0, s1 = a0[c0].copy(), a1[c1].copy()
            seg = ds.reconstruct_between_slices(
                s0, s1, thickness=args["ds_thickness"], steps=args["ds_steps"],
                chunk_size=ode_chunk, device=args["device"])
            run = _accumulate(seg, ccf2idx, template.shape, col_indices, K, run)
            del seg, s0, s1
            pbar.update(1)
            pbar.set_postfix(voxels=f"{0 if run[0] is None else len(run[0]):,}")
    pbar.close()
    run_flat, run_sum, run_cnt = run
    preds = (run_sum / run_cnt[:, None]).astype(np.float32)
    indices = adapter.flat_to_indices(run_flat, template.shape)
    return preds, indices


def _find_checkpoint(ckpt_dir):
    """Return (ckpt_path, config_path) for the newest checkpoint in ``ckpt_dir``
    (with its config.json), else (None, None)."""
    ckpt_dir = Path(ckpt_dir)
    cfg = ckpt_dir / "config.json"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if ckpts and cfg.exists():
        return ckpts[-1], cfg
    return None, None


def _load_ds_checkpoint(ds, ckpt, cfg_path, args):
    """Rebuild the model from the saved config.json and load the checkpoint
    weights + metadata (spatial_stats, categories, dims). Builds the module
    first (with clean build_model kwargs) so load_checkpoint just restores state
    -- avoiding the upstream build_model(**train_config) kwarg mismatch."""
    import json
    cfg = json.load(open(cfg_path))
    ds.gene_dim = cfg["gene_dim"]
    ds.num_classes = cfg["num_classes"]
    ds.build_model(**cfg["model_config"], lr=args["learning_rate"],
                   lambda_g=args["ds_lambda_g"], lambda_c=args["ds_lambda_c"],
                   sampling_method="euler")
    ds.load_checkpoint(str(ckpt), str(cfg_path), sampling_method="euler")


# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    np.random.seed(args["seed"]); torch.manual_seed(args["seed"])
    config = MaldiConfig.from_args(args)
    exp_path = config.exp_path
    lipid_names = [str(n) for n in config.selected_lipids_names]

    run = None
    if args.get("wandb"):
        try:
            import wandb
            run = wandb.init(project=args["wandb_project"], name=args["exp_name"],
                             config=args)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"wandb init failed ({e})")

    # ---- Train sections (train-fold mice) ----
    logging.info("Loading TRAIN sections...")
    train_by_mouse = adapter.load_sections(
        args["maldi_file"], lipid_names, config.section_filter,
        args["annotations_file"], max_cells=args["ds_max_cells"], seed=args["seed"])

    ds = DeepSpatial()
    ds.spatial_key, ds.z_key, ds.label_key = "spatial", "z_coord", "cell_class"

    # ---- Resume from checkpoint if present (skip training from scratch) ----
    ckpt, cfg_path = _find_checkpoint(exp_path / "checkpoints")
    if ckpt is not None and not args.get("force_retrain"):
        logging.info(f"Found checkpoint {ckpt.name} -> resuming (skipping "
                     f"training). Use --force-retrain to override.")
        _load_ds_checkpoint(ds, ckpt, cfg_path, args)  # restores model + stats + categories
        logging.info(f"  resumed: gene_dim={ds.gene_dim} classes={ds.num_classes}")
    else:
        # Global normalization + label set, then within-mouse trajectory pairs.
        all_train = [a for secs in train_by_mouse.values() for a in secs]
        ds._normalize_spatial(all_train)                   # sets ds.spatial_stats
        # Fit the region label encoder on the FULL atlas vocabulary (every region
        # id in the annotation volume), NOT just training-present regions --
        # otherwise a held-out mouse can carry a region unseen in training, whose
        # Categorical code is -1, and one_hot(-1) triggers a CUDA device-side
        # assert at reconstruction.
        atlas_regions = np.unique(np.load(args["annotations_file"])).astype(str)
        le = LabelEncoder().fit(atlas_regions)
        num_classes = len(le.classes_)
        logging.info(f"  atlas regions (classes, full vocabulary): {num_classes}")

        tensors = build_trajectories(train_by_mouse, le, num_classes, args)
        ds.dataset = _TrajDataset(tensors)
        ds.categories = pd.Index(le.classes_)
        ds.gene_dim = tensors["g0"].shape[1]
        ds.num_classes = num_classes
        ds.train_loader = DataLoader(ds.dataset, batch_size=args["batch_size"],
                                     shuffle=True, num_workers=4)
        logging.info(f"  trajectory pairs: {len(ds.dataset):,}  gene_dim={ds.gene_dim}")

        ds.build_model(patch_size=args["ds_patch"], hidden_size=args["ds_hidden_size"],
                       depth=args["ds_depth"], num_heads=args["ds_heads"],
                       lr=args["learning_rate"], lambda_g=args["ds_lambda_g"],
                       lambda_c=args["ds_lambda_c"], sampling_method="euler")
        logging.info("Training DeepSpatial transport model...")
        ds.fit(max_epochs=args["epochs"], save_dir=str(exp_path / "checkpoints"),
               accelerator="gpu" if args["device"].startswith("cuda") else "cpu",
               save_ckpt=True)

    # ---- Test sections (held-out mice) ----
    logging.info("Loading TEST sections...")
    test_by_mouse = adapter.load_sections(
        args["maldi_file"], lipid_names, config.test_filter,
        args["annotations_file"], max_cells=args["ds_max_cells_recon"],
        seed=args["seed"])

    # LOSO interpolation predictions for BOTH splits, stored in the harness layout
    # (<exp>/{train,test}/{predictions,true_values}.npy + coords/pixel_index) so
    # lgp_metrics / lgp_report / diagnostics can consume them; metrics.csv is
    # (re)generated from them via the shared write_metrics.
    from lgp_metrics import write_metrics
    loso_true = loso_pred = None
    for split, by_mouse, max_secs in [("test", test_by_mouse, 6),
                                      ("train", train_by_mouse, 3)]:
        try:
            true, pred, pix, coords = loso_predictions(ds, by_mouse, args, max_secs)
            if true is None:
                continue
            _write_split(exp_path, split, true, pred, pix, coords)
            mdf = write_metrics(exp_path, split, pred=pred, true=true,
                                lipid_names=lipid_names)
            r2 = float(mdf["r2"].mean()); corr = float(mdf["corr"].mean())
            logging.info(f"  LOSO {split}: {len(true):,} voxels  mean r2={r2:.4f} "
                         f"corr={corr:.4f} -> {split}/ + metrics")
            wandb_log({f"loso/{split}_r2": r2, f"loso/{split}_corr": corr})
            if split == "test":
                loso_true, loso_pred = true, pred
        except Exception as e:  # noqa: BLE001
            logging.warning(f"LOSO {split} failed: {e}")

    # ---- Full-volume reconstruction + render ----
    if args["reconstruct"] != "none":
        template = np.load(args["reference_file"])
        ccf2idx = adapter.fit_ccf_to_index(args["maldi_file"], config.test_filter)
        lipid_filter = None
        if args.get("reconstruction_lipids"):
            rl = args["reconstruction_lipids"]
            lipid_filter = _resolve_lipid_filter(
                lipid_names,
                lipid_indices=rl if all(isinstance(v, int) for v in rl) else None,
                lipid_names=None if all(isinstance(v, int) for v in rl) else rl)
        volume_path = exp_path / "volume"; volume_path.mkdir(parents=True, exist_ok=True)
        col_indices = (np.asarray(lipid_filter, np.int64) if lipid_filter is not None
                       else np.arange(len(lipid_names), dtype=np.int64))
        try:
            from render_lipid_volumes import render_selected_lipids
        except Exception as e:  # noqa: BLE001
            render_selected_lipids = None
            logging.error(f"renderer unavailable: {e}")

        # Per-lipid true-vs-pred scatter + value-distribution diagnostics from the
        # held-out LOSO sections -- parity with the harness models' diagnostics.
        if loso_true is not None:
            try:
                from render_lipid_volumes import render_lipid_diagnostics
                diag_dir = exp_path / "renders" / "diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)
                for gi in col_indices:
                    render_lipid_diagnostics(
                        loso_true[:, gi], loso_pred[:, gi], lipid_names[gi],
                        diag_dir / f"{lipid_names[gi]}_diagnostics.png")
                logging.info(f"LOSO diagnostics (scatter) -> {diag_dir}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"diagnostics plotting failed: {e}")
        K = len(col_indices)
        rng = np.random.default_rng(args["seed"])

        # Resolve reconstruction scope. 'follow' ties it to the training pairing so
        # a cross-mouse run reconstructs the canonical cross-mouse brain (the two
        # were previously mismatched: pairing could be cross-mouse while the
        # reconstruction always ran per-mouse).
        scope = args.get("ds_recon_scope", "follow")
        if scope == "follow":
            scope = ("cross-mouse" if args["ds_pairing"] == "cross-mouse"
                     else "per-mouse")

        # Normalize every test section into the training frame BEFORE grouping, so
        # z_norm (CCF AP) is comparable across mice for the cross-mouse sort.
        for secs in test_by_mouse.values():
            _apply_norm(secs, ds.spatial_stats)

        if scope == "cross-mouse":
            # Pool ALL test sections into one AP-ordered stack; reconstruct_mouse
            # then transports across adjacent sections regardless of animal, and
            # _accumulate averages overlapping-AP voxels from different mice into
            # the shared CCF grid -> one canonical whole-brain volume.
            pooled = sorted(
                (a for secs in test_by_mouse.values() for a in secs),
                key=lambda a: float(a.obs["z_norm"].iloc[0]))
            groups = [("crossmouse", pooled)]
        else:
            groups = [(str(m), secs) for m, secs in test_by_mouse.items()]

        for tag, secs in groups:
            if len(secs) < 2:
                logging.warning(f"Skipping reconstruction '{tag}': "
                                f"only {len(secs)} section(s).")
                continue
            logging.info(f"Reconstructing full volume [{scope}]: {tag} "
                         f"({len(secs)} sections)")
            preds, indices = reconstruct_mouse(
                ds, secs, args, ccf2idx, template, col_indices, K, rng)
            _write_per_lipid_volumes(volume_path, preds, indices,
                                     template.shape, lipid_names, col_indices,
                                     suffix=f"_{tag}")
            logging.info(f"  wrote {K} lipid volumes for {tag} "
                         f"({len(indices):,} voxels)")
            # Render into a per-tag subdir (PNG names carry no suffix, so separate
            # dirs prevent collisions across reconstructions).
            if render_selected_lipids is not None:
                try:
                    render_selected_lipids(
                        template_volume=template, volume_dir=volume_path,
                        output_dir=exp_path / "renders" / tag,
                        selected_lipids_names=lipid_names,
                        lipid_indices=(list(lipid_filter) if lipid_filter is not None else None),
                        suffix=f"_{tag}", n_rotation_frames=10)
                    logging.info(f"  renders -> {exp_path / 'renders' / tag}")
                except Exception as e:  # noqa: BLE001
                    logging.error(f"Rendering failed for {tag} (volumes saved): {e}")

    if run is not None:
        run.finish()
    logging.info("Done.")


if __name__ == "__main__":
    main()
