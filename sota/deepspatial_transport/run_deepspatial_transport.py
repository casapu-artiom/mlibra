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
    (drop an interior section, reconstruct its gap, estimate each held voxel from
    the mean of its --ds-loso-k nearest synthesized cells, score per-lipid
    corr / RMSE).

No gene data is involved anywhere: upstream's ``gene_dim`` / ``lambda_g`` / ``g0``
names are the package's term for the expression channel, which here carries the
173 lipids.

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
    p.add_argument("--ds-loso-k", dest="ds_loso_k", type=int, default=32,
                   help="Neighbours averaged per held voxel in the LOSO metric. "
                        "The flow is GENERATIVE: one nearest cell (k=1) is a single "
                        "Monte-Carlo draw, whose variance inflates RMSE and wrecks R^2 "
                        "even when the marginals are right. Averaging k of them "
                        "estimates the conditional mean instead. k=1 restores the "
                        "old single-sample behaviour.")
    p.add_argument("--ds-loso-max-cells", dest="ds_loso_max_cells", type=int,
                   default=-1,
                   help="Source voxels per neighbour section in the LOSO metric; "
                        "sets its IN-PLANE density exactly as --ds-max-cells-recon "
                        "does for the full volume. 0 = ALL voxels, -1 = follow "
                        "--ds-max-cells-recon. Chunked through --ds-recon-batch, so "
                        "raising it costs time, not memory.")
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
def _chunk_pairs(a0, a1, cap, batch, rng):
    """Yield (idx0, idx1) source-voxel index blocks covering ``cap`` voxels per
    section (<=0 => ALL), in blocks of ``batch``, cycling the shorter section.

    Chunking decouples IN-PLANE density (total source cells covered, = cap) from
    per-call memory (= batch): one reconstruct() call allocates
    batch*(gap/thickness)*n_lipids on the GPU regardless of how dense the
    reconstruction is overall.
    """
    u0 = a0.n_obs if cap <= 0 else min(a0.n_obs, cap)
    u1 = a1.n_obs if cap <= 0 else min(a1.n_obs, cap)
    p0 = rng.permutation(a0.n_obs)[:u0]
    p1 = rng.permutation(a1.n_obs)[:u1]
    for ci in range(max(1, int(np.ceil(max(u0, u1) / batch)))):
        off = ci * batch
        yield (p0[np.arange(off, off + batch) % u0],
               p1[np.arange(off, off + batch) % u1])


def _n_chunks(a0, a1, cap, batch):
    """Number of blocks ``_chunk_pairs`` will yield (for progress bars)."""
    u0 = a0.n_obs if cap <= 0 else min(a0.n_obs, cap)
    u1 = a1.n_obs if cap <= 0 else min(a1.n_obs, cap)
    return max(1, int(np.ceil(max(u0, u1) / batch)))


def _knn_mean(seg, held3, k):
    """Mean of the ``k`` synthesized cells nearest each held voxel in 3D
    (yccf, zccf, xccf-depth).

    Depth must be in the match: a 2D (y,z)-only match pulls values from cells at
    ARBITRARY depths across the gap, while the held section sits at one depth.

    k>1 is what makes this an estimate of the conditional mean rather than a
    single draw. The flow transports position AND expression jointly and both are
    generated, so one nearest cell is a Monte-Carlo sample whose variance adds to
    the error instead of cancelling. Accumulated one neighbour at a time so peak
    memory is n_held x n_lipids, not n_held x k x n_lipids.
    """
    from scipy.spatial import cKDTree
    if seg.n_obs == 0:
        return None
    syn3 = np.column_stack([seg.obsm["spatial"].astype(np.float64),
                            np.asarray(seg.obs["z_coord"], np.float64)])
    kk = max(1, min(int(k), len(syn3)))
    _, nn = cKDTree(syn3).query(held3, k=kk, workers=-1)
    nn = nn.reshape(len(held3), kk)
    synX = seg.X.toarray() if hasattr(seg.X, "toarray") else np.asarray(seg.X)
    acc = np.zeros((len(held3), synX.shape[1]), dtype=np.float32)
    for j in range(kk):
        acc += synX[nn[:, j]]
    return acc / kk


def loso_predictions(ds, by_mouse, args, max_secs=6):
    """Leave-one-section-out interpolation over the given mice: drop an interior
    section, reconstruct its gap from the two neighbours, and estimate each held
    voxel from the mean of its ``--ds-loso-k`` nearest synthesized cells. Returns
    (true, pred, pixel_index, ccf_coords) stacked over all evaluated held voxels
    -- the harness-layout per-split predictions (regeneratable, so they are stored
    to disk).

    ``max_secs`` caps interior sections evaluated per mouse (bounds cost on the
    dense atlas mice). Source density follows --ds-loso-max-cells and is covered
    in --ds-recon-batch blocks, exactly as ``reconstruct_mouse`` does for the full
    volume; each block's k-NN estimate is averaged, so the effective neighbourhood
    is k per block and memory is independent of the density.
    """
    rng = np.random.default_rng(args["seed"])
    cap = args["ds_loso_max_cells"]
    if cap < 0:                                    # -1 => follow the volume knob
        cap = args["ds_max_cells_recon"]
    batch = max(1, int(args["ds_recon_batch"]))
    k_nn = int(args["ds_loso_k"])
    T, P, PIX, C = [], [], [], []
    for mouse, secs in by_mouse.items():
        if len(secs) < 3:
            continue
        # Normalize whole sections once (chunks inherit spatial_norm / z_norm).
        # Needed here as well as in main() because a resumed run restores
        # spatial_stats from the checkpoint without re-running _normalize_spatial.
        _apply_norm(secs, ds.spatial_stats)
        n_done = 0
        for k in range(1, len(secs) - 1):
            if n_done >= max_secs:
                break
            a0, a2, held = secs[k - 1], secs[k + 1], secs[k]
            held3 = np.column_stack([held.obsm["spatial"].astype(np.float64),
                                     held.obs["xccf"].to_numpy(np.float64)])
            acc, n_acc = None, 0
            nch = _n_chunks(a0, a2, cap, batch)
            for c0, c2 in _chunk_pairs(a0, a2, cap, batch, rng):
                seg = ds.reconstruct_between_slices(
                    a0[c0].copy(), a2[c2].copy(), thickness=args["ds_thickness"],
                    steps=args["ds_steps"],
                    chunk_size=int(args.get("ds_recon_chunk", 32768)),
                    device=args["device"])
                est = _knn_mean(seg, held3, k_nn)
                del seg
                if est is None:
                    continue
                acc = est if acc is None else acc + est
                n_acc += 1
            if acc is None:
                logging.warning(f"  LOSO {mouse} section {k}: no synthesized "
                                f"cells over {nch} chunk(s) -- skipped.")
                continue
            # Clip at zero: the ODE is unconstrained and overshoots below the data
            # support (measured intensities are clipped at 0 on load), so negative
            # synthesized values are pure error. Eval-only -- no training stats.
            P.append(np.clip(acc / n_acc, 0.0, None).astype(np.float32))
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
    total_calls = sum(_n_chunks(secs[k], secs[k + 1], cap, batch)
                      for k in range(len(secs) - 1))
    run = (None, None, None)
    pbar = tqdm(total=total_calls, desc="  recon (gap-chunks)", unit="call", leave=False)
    for k in range(len(secs) - 1):
        a0, a1 = secs[k], secs[k + 1]
        for c0, c1 in _chunk_pairs(a0, a1, cap, batch, rng):
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
