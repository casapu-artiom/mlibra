"""NTF -- Neural Transcriptomic Field, adapted for MALDI lipid volumes.

Paper: Gong et al., "Ultra-efficient High Resolution 3D Reconstruction of
Spatial Omics Data with Neural Transcriptomic Field" (bioRxiv 2026.05.28.726140).

Core idea (paper): a coordinate-based neural field  f(x,y,z) -> expression,
built on a *multiresolution hash-grid* encoding (InstantNGP, Mueller et al. 2022)
decoded by a small MLP. Training uses a composite objective that decouples the
true biological signal from measurement artefacts:
  * a heteroscedastic Gaussian likelihood on the (non-zero) expression, so the
    field can down-weight noisy sections,
  * a per-section learnable bias embedding that absorbs batch effects, and
  * a spatial smoothness / total-variation regulariser that encourages
    neighbouring coordinates to share expression (suppresses technical noise
    while preserving anatomy).
The paper also adds a zero-inflation (dropout) BCE head for the heavy zeros of
transcriptomics; MALDI intensities are continuous and the harness already
imputes negatives to 0 + log-transforms, so that head is off by default (it can
be enabled with --ntf-zero-inflation).

MALDI adaptation notes:
  * "section identity" == coronal slice, derived from the z coordinate (the
    coronal axis). The per-section bias corrects slice-to-slice acquisition
    offsets; at *reconstruction* time the bias is dropped (bias=0) so we query
    the clean continuous field -- exactly the paper's "decouple signal from
    section artefact" intent.
  * The multiresolution hash grid is implemented in pure PyTorch (no tinycudann
    / CUDA compile), so it runs in the existing container. It is fully
    differentiable (gradients flow into the looked-up table entries).

I/O contract mirrors the baselines in ``maldi/experiment_baselines.py``:
``fit(coords_train, y_train, coords_test, y_test, args) -> (pred_tr, pred_te)``,
``predict(coords_std) -> pred``, ``save(path)``, ``load(path, p, args)``.
coords are standardized (n,3); y is per-lipid log-mean/std-normalized.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sota_utils import wandb_log, train_val_split, EarlyStopper


# ---------------------------------------------------------------------------
# Multiresolution hash-grid encoding (InstantNGP, pure torch)
# ---------------------------------------------------------------------------
# Large primes for the spatial hash (Mueller et al. 2022, eq. 4). The first is
# left as 1 so the x dimension is un-permuted.
_HASH_PRIMES = (1, 2654435761, 805459861)


class HashGridEncoder(nn.Module):
    """3D multiresolution hash-grid positional encoding.

    L levels geometrically spaced between resolutions ``base_res`` and
    ``max_res``. Each level owns a hash table of ``2**log2_hashmap`` rows of
    ``features_per_level`` floats. A query point is trilinearly interpolated from
    its 8 surrounding grid corners at every level; the per-level features are
    concatenated, giving an ``L * features_per_level`` embedding.

    Inputs are expected in the unit cube [0, 1]^3 (the wrapper maps standardized
    coords into it and clamps).
    """

    def __init__(self, n_levels=16, features_per_level=2, log2_hashmap=19,
                 base_res=16, max_res=1024):
        super().__init__()
        self.n_levels = n_levels
        self.features_per_level = features_per_level
        self.table_size = 2 ** log2_hashmap
        # Per-level growth factor b so that res_l = base_res * b**l reaches max_res.
        if n_levels > 1:
            b = np.exp((np.log(max_res) - np.log(base_res)) / (n_levels - 1))
        else:
            b = 1.0
        resolutions = [int(round(base_res * (b ** l))) for l in range(n_levels)]
        self.register_buffer("resolutions",
                             torch.tensor(resolutions, dtype=torch.long))
        # One embedding table per level, stacked: (L, T, F).
        self.embeddings = nn.Parameter(
            torch.empty(n_levels, self.table_size, features_per_level)
        )
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)
        # 8 corner offsets of the unit cube.
        corners = torch.tensor(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
            dtype=torch.long,
        )                                                        # (8, 3)
        self.register_buffer("corners", corners)
        primes = torch.tensor(_HASH_PRIMES, dtype=torch.long)
        self.register_buffer("primes", primes)

    @property
    def out_dim(self):
        return self.n_levels * self.features_per_level

    def _hash(self, coords_int, res, level):
        """Map integer corner coords -> table row index.

        Dense indexing when (res+1)^3 fits the table (coarse levels have no
        collisions); spatial hash otherwise.
        coords_int: (..., 3) long in [0, res].
        """
        grid = res + 1
        if grid ** 3 <= self.table_size:
            x, y, z = coords_int.unbind(-1)
            return (x * grid + y) * grid + z
        # Spatial xor-hash (Mueller et al. 2022, eq. 4).
        h = (coords_int[..., 0] * self.primes[0]) \
            ^ (coords_int[..., 1] * self.primes[1]) \
            ^ (coords_int[..., 2] * self.primes[2])
        return h % self.table_size

    def forward(self, x):
        """x: (B, 3) in [0, 1]. Returns (B, L*F)."""
        B = x.shape[0]
        feats = []
        for l in range(self.n_levels):
            res = int(self.resolutions[l].item())
            scaled = x * res                                     # (B,3) in [0,res]
            base = torch.floor(scaled).long()
            frac = scaled - base.float()                         # (B,3) in [0,1)
            acc = x.new_zeros(B, self.features_per_level)
            for c in range(8):
                off = self.corners[c]                            # (3,)
                ci = (base + off).clamp_(0, res)                 # (B,3) corner ints
                idx = self._hash(ci, res, l)                     # (B,)
                emb = self.embeddings[l][idx]                    # (B,F)
                # Trilinear weight = prod_d (frac_d if off_d==1 else 1-frac_d).
                w = torch.where(off.bool().unsqueeze(0), frac, 1.0 - frac)
                w = w.prod(dim=1, keepdim=True)                  # (B,1)
                acc = acc + w * emb
            feats.append(acc)
        return torch.cat(feats, dim=1)


# ---------------------------------------------------------------------------
# Field network: hash encoding -> MLP -> mean + auxiliary heads
# ---------------------------------------------------------------------------
# This mirrors the official NTF ``models.py`` structure (GYQ-form/NTF):
#   * an expression trunk -> mean (p) + latent z (n_features_z),
#   * a slice embedding shared by the bias and variance branches,
#   * a BIAS network on the LOW-FREQUENCY hash levels + slice embedding (the
#     paper applies a multiplicative log-bias to correct section batch effects;
#     we apply it ADDITIVELY because the MALDI targets are z-scored and can be
#     negative, where a multiplicative log-bias is ill-defined),
#   * a heteroscedastic variance sigma_net(z, slice) + per-slice log_var_slice.
class NTFNet(nn.Module):
    def __init__(self, p, n_slices, cfg):
        super().__init__()
        self.encoder = HashGridEncoder(
            n_levels=cfg["n_levels"],
            features_per_level=cfg["features_per_level"],
            log2_hashmap=cfg["log2_hashmap"],
            base_res=cfg["base_res"],
            max_res=cfg["max_res"],
        )
        Fpl = cfg["features_per_level"]
        self.n_levels_bias = int(cfg["n_levels_bias"])
        self.low_dim = self.n_levels_bias * Fpl          # low-freq slice for bias
        n_z = int(cfg["n_features_z"])
        n_s = int(cfg["n_features_slice"])
        aux_h = int(cfg["aux_hidden"])
        self.n_z, self.n_s, self.p = n_z, n_s, p

        # Expression trunk -> mean + latent z (raw coords concatenated as a
        # low-frequency channel the hash grid can't cheaply represent).
        in_dim = self.encoder.out_dim + 3
        layers, d = [], in_dim
        for h in cfg["hidden"]:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(d, p)
        self.z_head = nn.Linear(d, n_z)

        # Slice embedding shared by the bias + variance branches.
        self.slice_emb = nn.Embedding(max(n_slices, 1), n_s)
        nn.init.zeros_(self.slice_emb.weight)

        # Bias network on low-frequency levels + slice embedding (zero-init the
        # output so it starts as identity / no correction).
        if self.n_levels_bias > 0:
            self.bias_net = nn.Sequential(
                nn.Linear(self.low_dim + n_s, aux_h), nn.SiLU(),
                nn.Linear(aux_h, p),
            )
            nn.init.zeros_(self.bias_net[-1].weight)
            nn.init.zeros_(self.bias_net[-1].bias)
        else:
            self.bias_net = None

        # Variance network on latent z + slice embedding, plus a per-slice
        # log-variance that lets whole noisy sections be down-weighted.
        self.sigma_net = nn.Sequential(
            nn.Linear(n_z + n_s, aux_h), nn.SiLU(),
            nn.Linear(aux_h, p),
        )
        self.log_var_slice = nn.Embedding(max(n_slices, 1), p)
        nn.init.zeros_(self.log_var_slice.weight)

    def forward(self, x_unit, coords, slice_idx=None):
        """Returns (mean, logvar, log_bias). ``slice_idx=None`` is the clean
        field: no section bias, variance from a zero slice embedding."""
        enc = self.encoder(x_unit)
        h = self.trunk(torch.cat([enc, coords], dim=1))
        mean = self.mean_head(h)
        z = self.z_head(h)
        log_bias = None
        if slice_idx is not None:
            se = self.slice_emb(slice_idx)
            if self.bias_net is not None:
                log_bias = self.bias_net(torch.cat([enc[:, :self.low_dim], se], dim=1))
                mean = mean + log_bias
            logvar = self.sigma_net(torch.cat([z, se], dim=1)) + self.log_var_slice(slice_idx)
        else:
            se0 = mean.new_zeros(mean.shape[0], self.n_s)
            logvar = self.sigma_net(torch.cat([z, se0], dim=1))
        return mean, logvar, log_bias


class NTFModel:
    """NTF wrapper implementing the baseline model contract."""
    kind = "ntf"

    def __init__(self):
        self.net = None
        self.device = None
        self.cfg = None
        # unit-cube mapping (buffers persisted in the checkpoint)
        self.lo = None
        self.hi = None
        self.slice_edges = None      # bucketize edges along z (for section bias)
        self.n_slices = None
        self.p = None
        self.psf_samples = 1         # set in fit; predict path never uses PSF
        self.psf_sigma = 0.0

    # ---- helpers ----------------------------------------------------------
    def _cfg_from_args(self, args):
        return dict(
            n_levels=args.get("ntf_levels", 16),
            features_per_level=args.get("ntf_features", 2),
            log2_hashmap=args.get("ntf_log2_hashmap", 19),
            base_res=args.get("ntf_base_res", 16),
            max_res=args.get("ntf_max_res", 256),
            hidden=args.get("ntf_hidden", [128, 128]),
            tv_weight=args.get("ntf_tv_weight", 0.01),
            tv_eps=args.get("ntf_tv_eps", 0.01),
            zero_inflation=args.get("ntf_zero_inflation", False),
            weight_decay=args.get("ntf_weight_decay", 0.0),
            grid_lr=args.get("ntf_grid_lr", 1e-2),
            # ---- ported from the official NTF models.py ----
            n_features_z=args.get("ntf_features_z", 16),      # latent for sigma_net
            n_features_slice=args.get("ntf_features_slice", 8),
            n_levels_bias=args.get("ntf_levels_bias", 4),     # low-freq levels -> bias net
            aux_hidden=args.get("ntf_aux_hidden", 64),        # bias/sigma net width
            bias_weight=args.get("ntf_bias_weight", 0.01),    # L2 reg on log-bias
            psf_samples=args.get("ntf_psf_samples", 4),       # PSF Monte-Carlo samples
            psf_sigma=args.get("ntf_psf_sigma", 0.01),        # PSF std (std-coord units)
        )

    def _to_unit(self, coords):
        """Map standardized coords -> [0,1]^3 using stored train bounds."""
        return ((coords - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)

    # The MALDI sectioning axis is xccf == coords column 0 (per-section mean xccf
    # marches monotonically; y/z stay ~constant within the stack). Sections are
    # therefore binned along axis 0, NOT z.
    SECTION_AXIS = 0

    def _slice_idx(self, coords):
        """Section index in [0, n_slices-1] for each row, from the sectioning
        (AP / xccf) axis, via ``torch.bucketize`` on the stored bin edges --
        O(B log S) and no (B, S) materialization (the axis has millions of
        distinct values, so an argmin over all of them would OOM)."""
        z = coords[:, self.SECTION_AXIS].contiguous()
        return torch.bucketize(z, self.slice_edges).clamp_(0, self.n_slices - 1)

    # ---- contract ---------------------------------------------------------
    def fit(self, coords_train, y_train, coords_test, y_test, args):
        self.device = torch.device(args["device"])
        self.cfg = self._cfg_from_args(args)
        self.p = y_train.shape[1]

        Xtr = coords_train.to(self.device)
        Ytr = y_train.to(self.device)
        Xte = coords_test.to(self.device)
        Yte = y_test.to(self.device)

        # Unit-cube bounds from train coords (small margin so test/recon voxels
        # near the border still map inside).
        lo = Xtr.min(dim=0).values
        hi = Xtr.max(dim=0).values
        margin = 0.05 * (hi - lo).clamp(min=1e-6)
        self.lo = (lo - margin)
        self.hi = (hi + margin)

        # Sections along the AP sectioning axis (xccf == column 0). It is
        # continuous mm, so bin it into at most ``ntf_max_sections`` groups: use
        # the unique values when few, else quantile bins. Store bucketize edges
        # (midpoints / quantile boundaries); the per-section bias absorbs slice
        # batch effects.
        max_sections = int(args.get("ntf_max_sections", 256))
        z = Xtr[:, self.SECTION_AXIS]          # xccf = sectioning axis (not z)
        uz = torch.unique(z)
        if uz.numel() <= max_sections:
            centers = uz.sort().values
            self.slice_edges = (centers[1:] + centers[:-1]) * 0.5
            self.n_slices = centers.numel()
        else:
            qs = torch.linspace(0, 1, max_sections + 1, device=z.device)[1:-1]
            self.slice_edges = torch.quantile(z.float(), qs)
            self.n_slices = max_sections
        logging.info(f"  NTF: {self.n_slices} sections (z-binned), "
                     f"hash levels={self.cfg['n_levels']}")

        self.net = NTFNet(self.p, self.n_slices, self.cfg).to(self.device)
        # Two param groups (InstantNGP recipe). The hash-grid embeddings train at a
        # higher LR with a tiny Adam eps and NO weight decay: their per-entry
        # gradients are sparse and small, so a shared 1e-3 LR barely moves them, and
        # any weight decay drags the near-zero-init table back to a constant field
        # -- collapsing NTF to the per-lipid mean (the dead runs). Generalization is
        # instead regularised by the grid resolution (``max_res``) and the TV term;
        # weight decay applies only to the MLP heads, which can safely be shrunk.
        enc_params = list(self.net.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other_params = [p for p in self.net.parameters() if id(p) not in enc_ids]
        opt = torch.optim.Adam(
            [
                {"params": enc_params, "lr": float(self.cfg["grid_lr"]),
                 "weight_decay": 0.0, "eps": 1e-15},
                {"params": other_params, "lr": args["learning_rate"],
                 "weight_decay": float(self.cfg["weight_decay"])},
            ]
        )

        self.psf_samples = int(self.cfg["psf_samples"])
        self.psf_sigma = float(self.cfg["psf_sigma"])
        bias_w = float(self.cfg["bias_weight"])

        s_tr = self._slice_idx(Xtr)
        n = Xtr.shape[0]
        bs = args["batch_size"]
        tv_w = self.cfg["tv_weight"]
        tv_eps = self.cfg["tv_eps"]

        # Val split carved from TRAIN (test stays clean); monitor it for
        # best-checkpoint early stopping -- a high-capacity field on a cross-mouse
        # split overfits, so the last epoch is usually NOT the best one.
        tr_idx, va_idx = train_val_split(n, args.get("val_frac", 0.05),
                                         args["seed"], self.device)
        ntr = tr_idx.numel()
        stopper = EarlyStopper(patience=int(args.get("early_stop_patience", 5)))

        E = args["epochs"]
        gstep = 0
        for epoch in range(E):
            self.net.train()
            perm = tr_idx[torch.randperm(ntr, device=self.device)]
            loss_sum = nll_sum = tv_sum = bias_sum = lv_sum = 0.0
            inner = tqdm(range(0, ntr, bs), desc=f"ntf e{epoch+1}/{E}", leave=False)
            for i in inner:
                idx = perm[i:i + bs]
                xb, yb, sb = Xtr[idx], Ytr[idx], s_tr[idx]
                # PSF: average the field over Monte-Carlo perturbations of the
                # coordinate (models the physical capture volume) BEFORE the loss.
                mean, logvar, log_bias = self._psf_forward(xb, sb)
                # Heteroscedastic Gaussian NLL (0.5*(exp(-s)(y-mu)^2 + s)).
                logvar = logvar.clamp(-8.0, 8.0)
                nll = (0.5 * (torch.exp(-logvar) * (yb - mean) ** 2 + logvar)).mean()
                loss = nll
                bias_r = loss.new_zeros(())
                if log_bias is not None and bias_w > 0:
                    # Keep the section bias small (paper's mean(log_bias)^2 reg).
                    bias_r = bias_w * (log_bias ** 2).mean()
                    loss = loss + bias_r
                tv_r = loss.new_zeros(())
                if tv_w > 0:
                    # Spatial smoothness: perturb coords, penalize field change
                    # (the paper's neighbour-perturbation expression regulariser).
                    noise = torch.randn_like(xb) * tv_eps
                    mean2 = self.net(self._to_unit(xb + noise), xb + noise, None)[0]
                    mean_clean = self.net(self._to_unit(xb), xb, None)[0]
                    tv_r = tv_w * F.mse_loss(mean2, mean_clean)
                    loss = loss + tv_r
                opt.zero_grad()
                loss.backward()
                opt.step()
                nb = idx.shape[0]
                loss_sum += loss.item() * nb
                nll_sum += nll.item() * nb
                tv_sum += float(tv_r.detach()) * nb
                bias_sum += float(bias_r.detach()) * nb
                lv_sum += logvar.mean().item() * nb
                gstep += 1
                inner.set_postfix(loss=f"{loss.item():.4g}", nll=f"{nll.item():.4g}",
                                  tv=f"{float(tv_r.detach()):.3g}",
                                  bias=f"{float(bias_r.detach()):.3g}")
                if gstep % 50 == 0:
                    wandb_log({"ntf/step_loss": loss.item(),
                               "ntf/step_nll": nll.item()}, step=gstep)
            inner.close()
            val = (self._eval_mse(Xtr[va_idx], Ytr[va_idx], bs)
                   if va_idx.numel() else loss_sum / ntr)
            te = self._eval_mse(Xte, Yte, bs)
            logging.info(f"  ntf epoch {epoch+1}/{E}  loss={loss_sum/ntr:.6g}  "
                         f"nll={nll_sum/ntr:.6g}  tv={tv_sum/ntr:.4g}  "
                         f"bias={bias_sum/ntr:.4g}  val_mse={val:.6g}  test_mse={te:.6g}")
            wandb_log({"epoch": epoch + 1, "ntf/train_loss": loss_sum / ntr,
                       "ntf/train_nll": nll_sum / ntr, "ntf/tv_reg": tv_sum / ntr,
                       "ntf/bias_reg": bias_sum / ntr, "ntf/logvar_mean": lv_sum / ntr,
                       "ntf/val_mse": val, "ntf/test_mse": te}, step=gstep)
            mon = te if args.get("early_stop_monitor", "val") == "test" else val
            if stopper.step(mon, self.net, epoch + 1):
                logging.info(f"  ntf early stop at epoch {epoch+1} "
                             f"(best epoch {stopper.best_epoch}, {stopper.best:.6g})")
                break
        stopper.restore(self.net)
        logging.info(f"  ntf restored best epoch {stopper.best_epoch} "
                     f"({args.get('early_stop_monitor','val')}_mse={stopper.best:.6g})")

        pred_tr = self._predict_mean(Xtr, bs).cpu()
        pred_te = self._predict_mean(Xte, bs).cpu()
        return pred_tr, pred_te

    def _psf_forward(self, coords, slice_idx):
        """PSF Monte-Carlo forward: average the field over ``psf_samples``
        Gaussian perturbations (std ``psf_sigma``) of each coordinate, modelling
        the acquisition capture volume. Returns (mean, logvar, log_bias), each
        averaged over samples. Used at TRAIN time only -- inference queries the
        deblurred field with a single evaluation."""
        ns, ps = self.psf_samples, self.psf_sigma
        if ns <= 1 or ps <= 0:
            return self.net(self._to_unit(coords), coords, slice_idx)
        B = coords.shape[0]
        ce = coords.unsqueeze(1) + ps * torch.randn(B, ns, 3, device=coords.device)
        cf = ce.reshape(B * ns, 3)
        sf = slice_idx.repeat_interleave(ns) if slice_idx is not None else None
        mean, logvar, log_bias = self.net(self._to_unit(cf), cf, sf)
        mean = mean.reshape(B, ns, -1).mean(1)
        var = torch.exp(logvar.clamp(-8.0, 8.0)).reshape(B, ns, -1).mean(1)
        logvar = torch.log(var + 1e-8)
        if log_bias is not None:
            log_bias = log_bias.reshape(B, ns, -1).mean(1)
        return mean, logvar, log_bias

    def _predict_mean(self, X, bs):
        """Clean-field prediction (no section bias, no PSF blur -> the deblurred
        super-resolution field the paper reconstructs) in batches."""
        self.net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], bs):
                xb = X[i:i + bs]
                mean = self.net(self._to_unit(xb), xb, None)[0]
                outs.append(mean)
        return torch.cat(outs) if outs else X.new_zeros(0, self.p)

    def _eval_mse(self, X, Y, bs):
        with torch.no_grad():
            pred = self._predict_mean(X, bs)
            return F.mse_loss(pred, Y).item()

    def predict(self, coords_std):
        bs = 65536
        X = coords_std.to(self.device)
        return self._predict_mean(X, bs).cpu()

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "cfg": self.cfg,
            "lo": self.lo.cpu(),
            "hi": self.hi.cpu(),
            "slice_edges": self.slice_edges.cpu(),
            "p": self.p,
            "n_slices": self.n_slices,
        }, path)

    def load(self, path, p, args):
        self.device = torch.device(args["device"])
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.cfg = d["cfg"]
        self.p = d["p"]
        self.lo = d["lo"].to(self.device)
        self.hi = d["hi"].to(self.device)
        self.slice_edges = d["slice_edges"].to(self.device)
        self.n_slices = d["n_slices"]
        self.net = NTFNet(self.p, self.n_slices, self.cfg).to(self.device)
        self.net.load_state_dict(d["state_dict"])
        self.net.eval()
