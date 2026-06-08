#!/usr/bin/env python
# encoding: utf-8
"""
toy_manifold_gp_compare.py
==========================

Predict the toy-manifold signal with a Euclidean GP and a manifold GP and compare
held-out RMSE + correlation. The manifold path uses the real manifold_gp stack:

    manifold_gp.utils.nearest_neighbors.NearestNeighbors        (faiss kNN graph)
    manifold_gp.utils.anatomical_knn.inflate_cross_region_edges (atlas prior)
    manifold_gp.operators.graph_laplacian_operator.GraphLaplacianOperator
    manifold_gp.utils.compute_eigenvectors.LaplacianEigensolver
    manifold_gp.kernels.riemann_matern_kernel.RiemannMaternKernel

wired as in lgp_manifold_experiment.py, wrapped in a single-output gpytorch
ExactGP (we only need the kernel for this comparison, not LatentRiemannGP /
ManifoldLGP).

The toy analog of an atlas region is the ROLL LAYER (turn index). The
"faiss_atlas_weighted" run inflates cross-layer edges via inflate_cross_region_edges,
which is what lets the manifold kernel beat Euclidean on the folded sheet.

This script REQUIRES manifold_gp (and its deps: cupy on GPU / faiss / torch_sparse).
It is meant to run on the GPU box, not the CPU sandbox.

Compared runs:
  euclidean (3D Matern)                  gpytorch Matern on raw coords
  manifold (plain faiss)                 RiemannMaternKernel, no atlas prior
  manifold (faiss_atlas_weighted xF)     RiemannMaternKernel + cross-layer inflation
  geodesic ORACLE (intrinsic Matern)     Matern on true (s,h) -- the ceiling

Usage:
  python toy_manifold_gp_compare.py --signal geodesic --inflation 50 \
      --dump-predictions preds_geodesic.npz
  python toy_manifold_pred_napari.py preds_geodesic.npz      # render predictions
"""
from __future__ import annotations
import argparse
import inspect
import tempfile
import numpy as np
import torch
import gpytorch
from scipy.stats import pearsonr, spearmanr

from toy_manifold import make_folded_manifold


def gen_manifold(signal="geodesic", **kw):
    """Call make_folded_manifold tolerant of its exact signature: pass signal_kind
    only if supported, drop kwargs it doesn't accept (e.g. shell_cycles), and fail
    loudly if --signal euclidean is requested against a geodesic-only version."""
    params = inspect.signature(make_folded_manifold).parameters
    if "signal_kind" in params:
        kw["signal_kind"] = signal
    elif signal != "geodesic":
        raise SystemExit("This make_folded_manifold has no signal_kind: only the "
                         "geodesic signal is available. Re-add signal_kind to use "
                         "--signal euclidean.")
    return make_folded_manifold(**{k: v for k, v in kw.items() if k in params})

from manifold_gp.utils.nearest_neighbors import NearestNeighbors
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

torch.set_default_dtype(torch.float32)            # matches the manifold_gp pipeline


def layer_labels(data):
    """Toy 'atlas region' = roll layer (turn index)."""
    return np.floor(data["t"] / (2 * np.pi)).astype(np.int64)


# =============================================================================
# Build a RiemannMaternKernel from coords (+ optional atlas-style cross-layer
# inflation), exactly as the brain pipeline does.
# =============================================================================
def build_riemann_kernel(coords_np, labels, inflation, k, n_modes, nu,
                         lap_norm, graphbandwidth, device):
    coords = torch.as_tensor(coords_np, dtype=torch.float32, device=device).contiguous()
    knn = NearestNeighbors(coords)                       # faiss kNN
    edge_index, edge_value = knn.graph(k)                # (2,E),(E,) squared dists

    # bandwidth must be FIXED for the inflation to bite (recomputing it from the
    # inflated distances cancels the effect). Auto = median original edge distance.
    bw = graphbandwidth if graphbandwidth and graphbandwidth > 0 \
        else float(np.sqrt(np.median(edge_value.detach().cpu().numpy())))

    if inflation and inflation != 1.0:
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, labels,
            inflation=inflation, treat_zero_as_cross=False,  # layer 0 is a real layer
        )

    lap = GraphLaplacianOperator(
        edge_value, edge_index, coords.shape[0],
        torch.tensor(bw, device=device), lap_norm, True,
    )
    solver = LaplacianEigensolver(
        num_modes=n_modes,
        backend="cupy" if device.type == "cuda" else "scipy",
        ncv_min=max(1500, 3 * n_modes + 20), verbose=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        eigval, eigvec = solver.compute_or_load(
            lap, cache_dir=tmp, key="toy",
            graphbandwidth=bw, laplacian_normalization=lap_norm,
            force_recompute=True, device=device,
        )

    kernel = RiemannMaternKernel(
        nu=nu, knn=knn, edge_index=edge_index, edge_value=edge_value,
        eigval=eigval, eigvec=eigvec, nearest_neighbors=k, num_modes=n_modes,
        bump_scale=0.1, bump_decay=0.01, laplacian_normalization=lap_norm,
        graphbandwidth_init=bw,
    ).to(device)
    kernel.eval()
    # freeze graph bandwidth: eigvecs are precomputed/frozen, so its gradient is biased
    for name, p in kernel.named_parameters():
        if "graphbandwidth" in name:
            p.requires_grad_(False)

    # --- diagnostics for the OOS-misrouting failure mode -------------------
    # (1) Are our nodes actually detected as on-graph? features() uses
    #     faiss_sqdist < 1e-8, but float32 faiss self-distances are ~1e-6, so
    #     nodes can be misrouted to the OOS Nystrom path.
    with torch.no_grad():
        self_d, _ = knn.search(coords, 1)
        self_d0 = self_d[:, 0]
        frac_on = float((self_d0 < 1e-8).double().mean())
    # (2) The OOS path divides by (1 - bw^2*lambda)^2; if bw^2*lambda_max -> 1 it blows up.
    bw2lam_max = float((bw ** 2) * eigval.max())
    print(f"  [diag] node self-dist^2: max={float(self_d0.max()):.2e} "
          f"median={float(self_d0.median()):.2e}  frac<1e-8 (on-graph)={frac_on:.1%}"
          + ("  <-- nodes MISROUTED to OOS path!" if frac_on < 0.999 else ""))
    print(f"  [diag] bw={bw:.4g}  bw^2*lambda_max={bw2lam_max:.3g}"
          + ("  <-- OOS denom (1-bw^2 lam)^2 near-singular; predictions can explode"
             if bw2lam_max > 0.8 else ""))
    return kernel, coords


# =============================================================================
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def fit_gp(base_kernel, train_x, train_y, iters=80, lr=0.01, noise=1e-2):
    dev = train_x.device
    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    lik.noise = noise
    # ExactGP adds a ConstantMean + ScaleKernel whose params default to CPU; move
    # the whole assembled model to the data's device so mean/covar/targets agree.
    model = ExactGP(train_x, train_y, lik, base_kernel).to(dev)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    # The manifold kernel is exactly low-rank (rank = num_modes); CG conditions
    # poorly on it. At toy scale force exact Cholesky (set high enough to cover N).
    with gpytorch.settings.max_cholesky_size(1_000_000):
        for _ in range(iters):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik


def predict_gp(model, lik, x):
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(1_000_000), \
            gpytorch.settings.fast_pred_var(False):
        return lik(model(x)).mean


def metrics(pred, true):
    p = pred.detach().cpu().numpy(); t = true.detach().cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    pear = float(pearsonr(p, t)[0])
    spear = float(spearmanr(p, t).correlation)
    return rmse, pear, spear


def suggest_lengthscale(eigval, eigvec, y_centered, nu, q=0.9):
    """Spectral-matching lengthscale: put the prior's corner lam* = 2nu/ell^2 at the
    eigenvalue where the signal reaches q of its spectral energy. This is the
    PRINCIPLED init -- gpytorch's default ell is on the wrong scale for a Laplacian
    whose eigenvalues run to ~1/bw^2, which over-smooths the prior to the mean."""
    c = eigvec.t() @ y_centered                 # (M,) signal coeffs in eigenbasis
    E = (c ** 2)
    cum = torch.cumsum(E, 0) / E.sum().clamp(min=1e-30)
    idx = int((cum < q).sum().clamp(max=eigval.numel() - 1))
    lam_q = float(eigval[idx].clamp(min=1e-6))
    return float((2 * nu / lam_q) ** 0.5), lam_q


def _ell_of(kernel):
    try:
        return float(kernel.lengthscale.reshape(-1)[0].item())
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--cycles-per-turn", type=float, default=1.5)
    ap.add_argument("--height", type=float, default=20.0)
    ap.add_argument("--signal", choices=["geodesic", "euclidean"], default="geodesic")
    ap.add_argument("--thickness", type=float, default=1.5)
    ap.add_argument("--shell-cycles", type=float, default=4.0)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--num-modes", type=int, default=300)
    ap.add_argument("--nu", type=int, default=2)
    ap.add_argument("--inflation", type=float, default=50.0,
                    help="cross-layer edge inflation for faiss_atlas_weighted")
    ap.add_argument("--laplacian-norm", choices=["symmetric", "randomwalk"],
                    default="randomwalk")
    ap.add_argument("--graphbandwidth", type=float, default=0,
                    help="0 => auto (median edge distance)")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump-predictions", default=None,
                    help="path to save an .npz of full-field predictions for napari")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.num_modes < 50:
        print(f"WARNING: --num-modes={args.num_modes} is very small. The manifold "
              f"kernel is EXACTLY rank num_modes, so it can't represent a signal that "
              f"needs more modes; the manifold rows will look artificially bad. Use "
              f"~250-300 unless you specifically want a rank study.")

    data = gen_manifold(signal=args.signal, n=args.n, gap=args.gap, height=args.height,
                        cycles_per_turn=args.cycles_per_turn,
                        shell_cycles=args.shell_cycles,
                        thickness=args.thickness, seed=args.seed)
    X, intr, y = data["X"], data["intrinsic"], data["signal"]
    labels = layer_labels(data)
    Xs = ((X - X.mean(0)) / X.std()).astype(np.float32)
    Is = ((intr - intr.mean(0)) / intr.std()).astype(np.float32)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(args.n)
    ntr = int(args.train_frac * args.n)
    tr, te = perm[:ntr], perm[ntr:]
    tr_t, te_t = torch.as_tensor(tr, device=device), torch.as_tensor(te, device=device)
    Xs_t = torch.as_tensor(Xs, device=device)
    Is_t = torch.as_tensor(Is, device=device)

    print(f"N={args.n} train={ntr} test={len(te)} modes={args.num_modes} "
          f"k={args.knn_k} nu={args.nu} signal={args.signal} inflation={args.inflation} "
          f"gap={args.gap} cycles_per_turn={args.cycles_per_turn} "
          f"thickness={args.thickness} lap_norm={args.laplacian_norm}")

    results = {}          # name -> (rmse, pearson, spearman)
    pred_full = {}        # name -> full-field prediction (N,) np  (for dump)
    yc = (yt - yt.mean())

    def evaluate(name, kernel, full_inputs, train_inputs):
        model, lik = fit_gp(kernel, train_inputs[tr_t], yt[tr_t], iters=args.iters)
        tr_rmse = float(((predict_gp(model, lik, train_inputs[tr_t]) - yt[tr_t]) ** 2)
                        .mean().sqrt())
        results[name] = metrics(predict_gp(model, lik, train_inputs[te_t]), yt[te_t])
        print(f"  [{name}] train_rmse={tr_rmse:.4f}  test_rmse={results[name][0]:.4f}  "
              f"learned_ell={_ell_of(kernel):.3g}")
        if args.dump_predictions:
            pred_full[name] = predict_gp(model, lik, full_inputs).detach().cpu().numpy()

    def init_manifold_ell(kernel, tag):
        """Set the manifold kernel's lengthscale to the spectral-matching value
        (a principled init in the right regime for this Laplacian's eigenvalues)."""
        ell0, lam_q = suggest_lengthscale(kernel.eigval, kernel.eigvec, yc, args.nu)
        lo, hi = float(kernel.eigval.min()), float(kernel.eigval.max())
        print(f"  [{tag}] eigval range [{lo:.4g}, {hi:.4g}]  signal lam90={lam_q:.4g}  "
              f"-> spectral-match ell*={ell0:.3g} (init)")
        #kernel.lengthscale = ell0
        return kernel

    # 1. EUCLIDEAN Matern on 3D coords
    evaluate("euclidean (3D Matern)",
             gpytorch.kernels.MaternKernel(nu=args.nu + 0.5).to(device), Xs_t, Xs_t)

    # 2. MANIFOLD, plain faiss graph (no atlas prior)
    k_plain, coords = build_riemann_kernel(Xs, labels, 1.0, args.knn_k, args.num_modes,
                                           args.nu, args.laplacian_norm,
                                           args.graphbandwidth, device)
    evaluate("manifold (plain faiss)", init_manifold_ell(k_plain, "plain"), coords, coords)

    # 3. MANIFOLD, faiss_atlas_weighted (inflate cross-layer edges)
    k_atlas, coords = build_riemann_kernel(Xs, labels, args.inflation, args.knn_k,
                                           args.num_modes, args.nu, args.laplacian_norm,
                                           args.graphbandwidth, device)
    evaluate(f"manifold (faiss_atlas_weighted x{args.inflation:g})",
             init_manifold_ell(k_atlas, "atlas"), coords, coords)

    # 4. GEODESIC oracle: Matern on true intrinsic coords
    evaluate("geodesic ORACLE (intrinsic Matern)",
             gpytorch.kernels.MaternKernel(nu=args.nu + 0.5).to(device), Is_t, Is_t)

    print("\n=== held-out metrics (signal range ~[-1,1]) ===")
    print(f"  {'method':<44s} {'RMSE':>8} {'Pearson':>9} {'Spearman':>9}")
    for name, (r, p, s) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:<44s} {r:>8.4f} {p:>9.4f} {s:>9.4f}")
    base = results["euclidean (3D Matern)"][0]
    plain = results["manifold (plain faiss)"][0]
    atlas = results[f"manifold (faiss_atlas_weighted x{args.inflation:g})"][0]
    print(f"\nplain faiss          vs euclidean RMSE: ratio {plain/base:.2f}  "
          f"({'WINS' if plain < base else 'loses'})")
    print(f"faiss_atlas_weighted vs euclidean RMSE: ratio {atlas/base:.2f}  "
          f"({'WINS' if atlas < base else 'loses'})")

    if args.dump_predictions:
        names = list(pred_full.keys())
        np.savez(
            args.dump_predictions,
            X=X.astype(np.float32), y_true=y.astype(np.float32),
            train_idx=tr.astype(np.int64), test_idx=te.astype(np.int64),
            names=np.array(names),
            preds=np.stack([pred_full[n] for n in names]).astype(np.float32),
            signal_kind=np.array(args.signal),
        )
        print(f"\nsaved full-field predictions -> {args.dump_predictions}\n"
              f"  render with: python toy_manifold_pred_napari.py {args.dump_predictions}")


if __name__ == "__main__":
    main()