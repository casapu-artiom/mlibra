"""Spa3D -- spatial-pattern-enhanced GCN for 3D reconstruction, for MALDI.

Paper: Tang et al., "3D reconstruction of spatial transcriptomics with spatial
pattern enhanced graph convolutional neural network" (Briefings in
Bioinformatics 2026, bbag060).

Two ingredients from the paper, adapted to the MALDI coords->lipids task:

1. Spatial Pattern Enhancement (SPE) -- denoise each 2D section before graph
   learning, isolating coherent spatial structure from technical noise. The
   paper offers two SPE operators (Methods, eqs 1-8):
     * 'hilbert' -- analytic-signal envelope  s_e = |s + j*Hilbert(s)|  (eqs 1-2),
       generalizing local expression to instantaneous (extremal) expression.
     * 'alft' -- anti-leakage Fourier transform: iteratively extract the
       highest-energy wavenumber components until the residual falls below a
       threshold (eqs 3-8), keeping only dominant spatial frequencies.
   We apply the chosen operator per coronal section, per lipid channel, on a
   rasterized (x,y) grid built from the (standardized) coordinates, then map the
   enhanced values back onto the measured points. SPE only smooths the *training
   target*; predictions are compared against the raw held-out signal.

2. A GCN over a genuine 3D graph that *explicitly encodes true z-axis distances*
   (the paper's central contrast with PASTE's 2.5D stacking). We build ONE global
   graph -- matching the original Spa3D ``calculate_adj_3D`` adjacency -- over a
   fixed node set (a subsample of the training coords) with SpaGCN-style
   Gaussian-affinity edge weights ``exp(-d^2 / 2 l^2)`` on an anisotropic metric
   where the z (inter-slice) axis is scaled by ``--spa3d-z-weight``. Training is
   transductive full-graph; any query coordinate (test / whole-brain
   reconstruction) reads out by inverse-distance interpolation over its nearest
   graph nodes, so the readout stays consistent and scales to millions of voxels.
   This replaces an earlier per-batch unweighted KNN, whose graph was rebuilt
   from a fresh random subsample every step (non-physical and unstable).

Requires ``torch_geometric`` (same dependency the ``gcn`` baseline uses).
Contract mirrors ``maldi/experiment_baselines.py`` model wrappers.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sota_utils import wandb_log, train_val_split, EarlyStopper

try:
    from torch_geometric.nn import GCNConv
    HAVE_PYG = True
except ImportError:
    HAVE_PYG = False


# ---------------------------------------------------------------------------
# Spatial Pattern Enhancement operators
# ---------------------------------------------------------------------------
def _hilbert_envelope_2d(img):
    """Analytic-signal envelope of a 2D image (eqs 1-2).

    Hilbert transform applied along both spatial axes then combined into the
    envelope magnitude. Pure-numpy FFT implementation (no scipy dependency).
    """
    def hilbert_1d(a, axis):
        n = a.shape[axis]
        Fa = np.fft.fft(a, axis=axis)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[0] = 1
            h[1:(n + 1) // 2] = 2
        shape = [1] * a.ndim
        shape[axis] = n
        return np.fft.ifft(Fa * h.reshape(shape), axis=axis)

    a = img.astype(np.float64)
    analytic = hilbert_1d(a, 0) + hilbert_1d(a, 1) - a  # combine both axes
    return np.abs(analytic)


def _alft_denoise_2d(img, keep_frac=0.1, max_iters=64, energy_stop=0.05):
    """Anti-leakage Fourier transform denoise (eqs 3-8, wavenumber-domain form).

    Iteratively remove the highest-magnitude wavenumber component from the
    residual spectrum, accumulating the kept components, until the residual
    energy drops below ``energy_stop`` of the original (or ``keep_frac`` of the
    coefficients are kept). A single inverse FFT maps the kept spectrum back.
    """
    F0 = np.fft.fft2(img.astype(np.float64))
    resid = F0.copy()
    kept = np.zeros_like(F0)
    total_e = np.sum(np.abs(F0) ** 2) + 1e-12
    n_coef = F0.size
    max_keep = max(1, int(keep_frac * n_coef))
    for _ in range(min(max_iters, max_keep)):
        flat = np.abs(resid).ravel()
        k = int(flat.argmax())
        idx = np.unravel_index(k, resid.shape)
        kept[idx] = resid[idx]
        resid[idx] = 0.0
        if np.sum(np.abs(resid) ** 2) / total_e < energy_stop:
            break
    return np.real(np.fft.ifft2(kept))


def apply_spe(coords, y, mode, grid=128, z_axis=0, n_sections=64, **kw):
    """Enhance the per-section spatial pattern of every lipid channel.

    coords: (n,3) standardized; y: (n,p). ``z_axis`` is the SECTIONING axis
    (xccf == column 0 for MALDI -- per-section mean marches monotonically while
    the in-plane y/z stay ~constant). It is continuous mm, so it is binned into
    ``n_sections`` groups (quantile bins). Each section is rasterized onto a
    ``grid``x``grid`` in-plane image per lipid, the SPE operator runs on it, and
    the enhanced value is scattered back to each point (nearest cell). Returns a
    new (n,p) tensor.
    """
    if mode == "none":
        return y
    dev = y.device
    coords_np = coords.cpu().numpy()
    y_np = y.cpu().numpy()
    z = coords_np[:, z_axis]
    xy_axes = [a for a in range(3) if a != z_axis]
    out = y_np.copy()

    # Bin z into bounded sections (unique values when few, else quantile bins).
    uz = np.unique(z)
    if uz.size <= n_sections:
        edges = (uz[1:] + uz[:-1]) * 0.5
    else:
        edges = np.quantile(z, np.linspace(0, 1, n_sections + 1)[1:-1])
    sec = np.digitize(z, edges)

    for s in np.unique(sec):
        m = np.where(sec == s)[0]
        if m.size < 4:
            continue
        xy = coords_np[np.ix_(m, xy_axes)]
        lo = xy.min(0)
        span = np.clip(xy.max(0) - lo, 1e-9, None)
        gi = np.clip(((xy - lo) / span * (grid - 1)).astype(int), 0, grid - 1)
        cell = gi[:, 0] * grid + gi[:, 1]
        for c in range(y_np.shape[1]):
            img = np.zeros((grid, grid))
            cnt = np.zeros((grid, grid))
            np.add.at(img, (gi[:, 0], gi[:, 1]), y_np[m, c])
            np.add.at(cnt, (gi[:, 0], gi[:, 1]), 1.0)
            occ = cnt > 0
            img[occ] /= cnt[occ]
            if mode == "hilbert":
                enh = _hilbert_envelope_2d(img)
            elif mode == "alft":
                enh = _alft_denoise_2d(img, **kw)
            else:
                raise ValueError(f"unknown SPE mode {mode!r}")
            out[m, c] = enh.reshape(-1)[cell]
    return torch.tensor(out, dtype=y.dtype, device=dev)


# ---------------------------------------------------------------------------
# z-aware 3D KNN graph + GCN
# ---------------------------------------------------------------------------
def _build_global_gaussian_graph(coords, k, z_weight, l_scale=None, z_axis=0,
                                 chunk=2048):
    """One fixed, symmetric KNN graph over ALL nodes ``coords`` (N,3) with a
    z-weighted distance metric and SpaGCN-style Gaussian-affinity edge weights
    ``exp(-d^2 / (2 l^2))``. Mirrors the original Spa3D ``calculate_adj_3D``
    adjacency (a single global graph with distance decay), replacing the
    per-batch unweighted KNN.

    The SECTIONING axis (``z_axis``; xccf == column 0 for MALDI) is scaled by
    ``z_weight`` in the metric: <1 makes inter-section neighbours *cheaper*
    (encourages true-3D links), >1 keeps sections apart. ``l_scale`` sets the
    Gaussian bandwidth ``l``; when None/<=0 it defaults to the median KNN
    (squared) distance -- a self-scaling median heuristic. cdist runs in row
    chunks so N up to ~1e5 nodes stays within memory.

    Returns (edge_index (2,E) long, edge_weight (E,) float).
    """
    N = coords.shape[0]
    k = min(k, N - 1)
    scale = torch.ones(3, device=coords.device, dtype=coords.dtype)
    scale[z_axis] = z_weight
    xs = coords * scale
    knn_idx = torch.empty(N, k, dtype=torch.long, device=coords.device)
    knn_d2 = torch.empty(N, k, dtype=coords.dtype, device=coords.device)
    for i in range(0, N, chunk):
        d = torch.cdist(xs[i:i + chunk], xs)          # (c, N) z-weighted dist
        dk, ik = d.topk(k + 1, largest=False)          # col 0 is self (d=0)
        knn_idx[i:i + chunk] = ik[:, 1:]
        knn_d2[i:i + chunk] = dk[:, 1:] ** 2
    if l_scale is not None and l_scale > 0:
        l2 = float(l_scale) ** 2
    else:
        l2 = float(knn_d2.median().clamp_min(1e-12))   # median heuristic
    w = torch.exp(-knn_d2 / (2.0 * l2))                # (N, k) Gaussian affinity
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = knn_idx.reshape(-1)
    wv = w.reshape(-1)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_weight = torch.cat([wv, wv])
    return edge_index, edge_weight


class _GCNNet(nn.Module):
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
            x = F.silu(conv(x, edge_index, edge_weight))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class Spa3DModel:
    """Spatial-pattern-enhanced GCN over a single global z-aware Gaussian graph.

    The graph is built ONCE over a fixed node set (a subsample of the training
    coords, capped at ``--spa3d-graph-nodes``) with SpaGCN-style Gaussian-affinity
    edge weights on a z-weighted metric -- matching the original Spa3D adjacency
    and replacing the earlier per-batch KNN. Training is transductive full-graph
    (one forward over all nodes per step); any query coordinate reads out by
    inverse-distance interpolation over its nearest graph nodes, so train / test /
    whole-brain reconstruction stay consistent and scale to millions of voxels."""
    kind = "spa3d"

    def __init__(self):
        self.model = None
        self.device = None
        self.k = None
        self.z_weight = None
        self.l_scale = None
        self.interp_k = None
        self.hidden = None
        self.dropout = None
        self.p = None
        self.node_x = None          # (Nn, 3) graph-node coords == node features
        self.node_scaled = None     # (Nn, 3) z-weighted, for nearest-node search
        self.edge_index = None      # (2, E)
        self.edge_weight = None     # (E,) Gaussian affinities
        self._node_preds = None     # cached (Nn, p) node predictions (eval)

    # --- graph read-out -----------------------------------------------------
    def _interp_weights(self, coords, chunk=4096):
        """Shepard (inverse-distance) weights over the ``interp_k`` nearest graph
        nodes, in the z-weighted metric. Returns (idx (B,ik) long, w (B,ik)).
        cdist is row-chunked so millions of query coords stay within memory."""
        scale = torch.ones(3, device=self.node_x.device, dtype=self.node_x.dtype)
        scale[0] = self.z_weight
        q = (coords.to(self.node_x.device) * scale).contiguous()
        ik = min(self.interp_k, self.node_scaled.shape[0])
        idx_out = torch.empty(q.shape[0], ik, dtype=torch.long, device=q.device)
        w_out = torch.empty(q.shape[0], ik, dtype=q.dtype, device=q.device)
        for i in range(0, q.shape[0], chunk):
            d = torch.cdist(q[i:i + chunk], self.node_scaled)   # (c, Nn)
            dk, ii = d.topk(ik, largest=False)
            w = 1.0 / (dk + 1e-8)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
            idx_out[i:i + chunk] = ii
            w_out[i:i + chunk] = w
        return idx_out, w_out

    @staticmethod
    def _readout(node_preds, idx, w):
        """Weighted gather: (B,ik) node indices + (B,ik) weights -> (B, p). Loops
        over the (small) ik to avoid materializing the (B, ik, p) intermediate."""
        out = node_preds.new_zeros((idx.shape[0], node_preds.shape[1]))
        for j in range(idx.shape[1]):
            out = out + w[:, j:j + 1] * node_preds[idx[:, j]]
        return out

    def _forward_nodes(self):
        return self.model(self.node_x, self.edge_index, self.edge_weight)

    def _refresh_node_preds(self):
        self.model.eval()
        with torch.no_grad():
            self._node_preds = self._forward_nodes()

    def _set_nodes(self, node_x):
        self.node_x = node_x.contiguous()
        scale = torch.ones(3, device=node_x.device, dtype=node_x.dtype)
        scale[0] = self.z_weight
        self.node_scaled = (self.node_x * scale).contiguous()

    def fit(self, coords_train, y_train, coords_test, y_test, args):
        if not HAVE_PYG:
            raise RuntimeError("torch_geometric not installed (needed for spa3d).")
        self.device = torch.device(args["device"])
        self.k = args.get("spa3d_knn_k", 15)
        self.z_weight = args.get("spa3d_z_weight", 1.0)
        self.l_scale = args.get("spa3d_length_scale", 0.0)
        self.interp_k = int(args.get("spa3d_interp_k", 8))
        self.hidden = args.get("spa3d_hidden", [256, 256, 128])
        self.dropout = args.get("spa3d_dropout", 0.1)
        self.p = y_train.shape[1]

        spe = args.get("spa3d_spe", "alft")
        if spe != "none":
            logging.info(f"  Spa3D: applying SPE ({spe}) to training targets...")
            y_train = apply_spe(coords_train, y_train, spe,
                                grid=args.get("spa3d_grid", 128),
                                n_sections=args.get("spa3d_sections", 64),
                                keep_frac=args.get("spa3d_alft_keep", 0.1))

        Xtr = coords_train.to(self.device); Ytr = y_train.to(self.device)
        Xte = coords_test.to(self.device);  Yte = y_test.to(self.device)
        n = Xtr.shape[0]

        # --- one global graph over a fixed node subsample of the train coords --
        max_nodes = int(args.get("spa3d_graph_nodes", 80000))
        if n > max_nodes:
            g = torch.Generator(device=self.device).manual_seed(int(args["seed"]))
            node_sel = torch.randperm(n, generator=g, device=self.device)[:max_nodes]
        else:
            node_sel = torch.arange(n, device=self.device)
        self._set_nodes(Xtr[node_sel])
        logging.info(f"  Spa3D: building global Gaussian graph over "
                     f"{self.node_x.shape[0]:,} nodes (k={self.k}, "
                     f"z_weight={self.z_weight}, l={self.l_scale or 'median'})...")
        self.edge_index, self.edge_weight = _build_global_gaussian_graph(
            self.node_x, self.k, self.z_weight, self.l_scale)

        self.model = _GCNNet(3, self.hidden, self.p, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=args["learning_rate"])

        # Read-out interpolation for supervision / eval, precomputed once.
        tr_idx, tr_w = self._interp_weights(Xtr)
        te_idx, te_w = self._interp_weights(Xte)

        # Val split from TRAIN for best-checkpoint early stopping (test stays clean).
        tr_i, va_i = train_val_split(n, args.get("val_frac", 0.05),
                                     args["seed"], self.device)
        ntr = tr_i.numel()
        bs = args["batch_size"]
        stopper = EarlyStopper(patience=int(args.get("early_stop_patience", 5)))
        E = args["epochs"]
        gstep = 0
        for epoch in range(E):
            self.model.train()
            perm = tr_i[torch.randperm(ntr, device=self.device)]
            loss_sum = 0.0
            nsteps = 0
            inner = tqdm(range(0, ntr, bs), desc=f"spa3d e{epoch+1}/{E}", leave=False)
            for i in inner:
                sel = perm[i:i + bs]
                # Transductive: one full-graph forward, supervise a minibatch of
                # measurements via the interpolated read-out.
                node_preds = self._forward_nodes()
                pred = self._readout(node_preds, tr_idx[sel], tr_w[sel])
                loss = F.mse_loss(pred, Ytr[sel])
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item(); nsteps += 1
                gstep += 1
                inner.set_postfix(mse=f"{loss.item():.4g}",
                                  run=f"{loss_sum/nsteps:.4g}")
                if gstep % 50 == 0:
                    wandb_log({"spa3d/step_mse": loss.item()}, step=gstep)
            inner.close()
            self.model.eval()
            with torch.no_grad():
                np_e = self._forward_nodes()
                val = (F.mse_loss(self._readout(np_e, tr_idx[va_i], tr_w[va_i]),
                                  Ytr[va_i]).item()
                       if va_i.numel() else loss_sum / max(nsteps, 1))
                te = F.mse_loss(self._readout(np_e, te_idx, te_w), Yte).item()
            logging.info(f"  spa3d epoch {epoch+1}/{E}  "
                         f"train_mse={loss_sum/max(nsteps,1):.6g}  "
                         f"val_mse={val:.6g}  test_mse={te:.6g}")
            wandb_log({"epoch": epoch + 1, "spa3d/train_mse": loss_sum / max(nsteps, 1),
                       "spa3d/val_mse": val, "spa3d/test_mse": te}, step=gstep)
            mon = te if args.get("early_stop_monitor", "val") == "test" else val
            if stopper.step(mon, self.model, epoch + 1):
                logging.info(f"  spa3d early stop at epoch {epoch+1} "
                             f"(best {stopper.best_epoch}, {stopper.best:.6g})")
                break
        stopper.restore(self.model)
        logging.info(f"  spa3d restored best epoch {stopper.best_epoch} "
                     f"({args.get('early_stop_monitor','val')}_mse={stopper.best:.6g})")
        self._refresh_node_preds()
        pred_tr = self._readout(self._node_preds, tr_idx, tr_w).cpu()
        pred_te = self._readout(self._node_preds, te_idx, te_w).cpu()
        return pred_tr, pred_te

    def predict(self, coords_std):
        if self._node_preds is None:
            self._refresh_node_preds()
        idx, w = self._interp_weights(coords_std)
        return self._readout(self._node_preds, idx, w).cpu()

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(), "k": self.k,
                    "z_weight": self.z_weight, "l_scale": self.l_scale,
                    "interp_k": self.interp_k, "hidden": self.hidden,
                    "dropout": self.dropout, "p": self.p,
                    "node_x": self.node_x.cpu(),
                    "edge_index": self.edge_index.cpu(),
                    "edge_weight": self.edge_weight.cpu()}, path)

    def load(self, path, p, args):
        if not HAVE_PYG:
            raise RuntimeError("torch_geometric not installed (needed for spa3d).")
        self.device = torch.device(args["device"])
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.k, self.z_weight = d["k"], d["z_weight"]
        self.l_scale, self.interp_k = d["l_scale"], d["interp_k"]
        self.hidden, self.dropout, self.p = d["hidden"], d["dropout"], d["p"]
        self._set_nodes(d["node_x"].to(self.device))
        self.edge_index = d["edge_index"].to(self.device)
        self.edge_weight = d["edge_weight"].to(self.device)
        self.model = _GCNNet(3, self.hidden, self.p, self.dropout).to(self.device)
        self.model.load_state_dict(d["state_dict"])
        self.model.eval()
        self._node_preds = None
