#!/usr/bin/env python
# encoding: utf-8
"""
toy_box_cylinder_gp_compare.py
================================

Tests the theory:
  "Implicit Manifold GP is useful when data lies on a manifold (geodesic ≠
   Euclidean), but gives no advantage in dense structures where Euclidean
   distance is a good proxy for geodesic distance."

Geometry: a box [-L,L]^3 with a cylinder inside (radius R, half-height Hc).

  --mode surface  Points only on the 6 box faces + cylinder lateral + caps.
                  Antiphase geodesic signal: the gap between cylinder and box
                  wall creates Euclidean-near / value-opposite pairs.
                  → manifold GP (with atlas inflation) should WIN.

  --mode volume   Points fill the box interior uniformly.
                  Same signal extended smoothly in 3D (no discontinuity).
                  → Euclidean GP should win or tie.

Running both modes (default --mode both) gives a head-to-head table.

Compared methods (same as toy_manifold_gp_compare.py):
  euclidean          Matern GP on 3-D coords
  manifold plain     RiemannMaternKernel, no atlas inflation
  manifold atlas     RiemannMaternKernel + inflate_cross_region_edges
                     (inflates cylinder↔box-wall edges; region labels come
                      from the surface generator: LABEL_CYL=0, LABEL_BOX=1)
  oracle             Matern GP on intrinsic surface coords (arc, z)
                     -- the ceiling for the surface mode

Usage:
  python toy_box_cylinder_gp_compare.py
  python toy_box_cylinder_gp_compare.py --mode surface --signal geodesic
  python toy_box_cylinder_gp_compare.py --mode volume  --signal geodesic
  python toy_box_cylinder_gp_compare.py --num-modes 250 --inflation 50
  python toy_box_cylinder_gp_compare.py --variational --num-inducing 200
  python toy_box_cylinder_gp_compare.py --variational --num-inducing 200 \\
      --dump-predictions preds_boxcyl
"""
from __future__ import annotations
import argparse
import tempfile
import numpy as np
import torch
import gpytorch
from scipy.stats import pearsonr, spearmanr

from toy_box_cylinder import make_box_cylinder, LABEL_CYL, LABEL_BOX

from manifold_gp.utils.nearest_neighbors import NearestNeighbors
from manifold_gp.utils.anatomical_knn import inflate_cross_region_edges
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

torch.set_default_dtype(torch.float32)


# =============================================================================
# RiemannMaternKernel builder (identical to toy_manifold_gp_compare.py)
# =============================================================================

def build_riemann_kernel(coords_np, labels, inflation, k, n_modes, nu,
                         lap_norm, graphbandwidth, device):
    coords = torch.as_tensor(coords_np, dtype=torch.float32,
                             device=device).contiguous()
    knn = NearestNeighbors(coords)
    edge_index, edge_value = knn.graph(k)

    bw = (graphbandwidth if graphbandwidth and graphbandwidth > 0
          else float(np.sqrt(np.median(edge_value.detach().cpu().numpy()))))

    if inflation and inflation != 1.0:
        edge_index, edge_value, _info = inflate_cross_region_edges(
            edge_index, edge_value, labels,
            inflation=inflation, treat_zero_as_cross=False,
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
            lap, cache_dir=tmp, key="boxcyl",
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
    for name, p in kernel.named_parameters():
        if "graphbandwidth" in name:
            p.requires_grad_(False)

    with torch.no_grad():
        self_d, _ = knn.search(coords, 1)
        self_d0 = self_d[:, 0]
        frac_on = float((self_d0 < 1e-8).double().mean())
    bw2lam_max = float((bw ** 2) * eigval.max())
    print(f"  [diag] node self-dist^2: max={float(self_d0.max()):.2e} "
          f"median={float(self_d0.median()):.2e}  frac<1e-8(on-graph)={frac_on:.1%}"
          + ("  <-- nodes MISROUTED to OOS path!" if frac_on < 0.999 else ""))
    print(f"  [diag] bw={bw:.4g}  bw^2*lambda_max={bw2lam_max:.3g}"
          + ("  <-- OOS denom near-singular" if bw2lam_max > 0.8 else ""))
    return kernel, coords


# =============================================================================
# GP models
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
    model = ExactGP(train_x, train_y, lik, base_kernel).to(dev)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
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


# =============================================================================
# Sparse variational GP (mirrors LatentRiemannGP)
# For the manifold kernel the inducing points must be GRAPH NODES and must NOT
# be learned (they drift off-graph into the unstable OOS path); and
# num_inducing <= num_modes or Kuu is rank-deficient.
# =============================================================================

class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, base_kernel, learn_inducing):
        vd = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        vs = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=learn_inducing)
        super().__init__(vs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def fit_svgp(base_kernel, train_x, train_y, num_inducing, learn_inducing,
             iters=80, lr=0.1, noise=1e-2):
    dev = train_x.device
    m   = min(num_inducing, train_x.size(0))
    sel = torch.randperm(train_x.size(0), device=dev)[:m]
    inducing = train_x[sel].clone()
    lik   = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    lik.noise = noise
    model = SVGP(inducing, base_kernel, learn_inducing).to(dev)
    model.train(); lik.train()
    opt = torch.optim.Adam(list(model.parameters()) + list(lik.parameters()), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=train_y.size(0))
    with gpytorch.settings.max_cholesky_size(1_000_000):
        for _ in range(iters):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik


def metrics(pred, true):
    p = pred.detach().cpu().numpy(); t = true.detach().cpu().numpy()
    rmse  = float(np.sqrt(np.mean((p - t) ** 2)))
    pear  = float(pearsonr(p, t)[0])
    spear = float(spearmanr(p, t).correlation)
    return rmse, pear, spear


def suggest_lengthscale(eigval, eigvec, y_centered, nu, q=0.9):
    c   = eigvec.t() @ y_centered
    E   = c ** 2
    cum = torch.cumsum(E, 0) / E.sum().clamp(min=1e-30)
    idx = int((cum < q).sum().clamp(max=eigval.numel() - 1))
    lam_q = float(eigval[idx].clamp(min=1e-6))
    return float((2 * nu / lam_q) ** 0.5), lam_q


def _ell_of(kernel):
    try:
        return float(kernel.lengthscale.reshape(-1)[0].item())
    except Exception:
        return float("nan")


# =============================================================================
# Run one mode (surface or volume)
# =============================================================================

def run_mode(args, mode: str, device: torch.device):
    infer_mode = (f"variational(M={args.num_inducing})"
                  if args.variational else "exact")
    print(f"\n{'='*70}")
    print(f"  MODE: {mode.upper()}  |  signal: {args.signal}  |  "
          f"N={args.n}  modes={args.num_modes}  k={args.knn_k}  "
          f"inflation={args.inflation}  infer={infer_mode}")
    print(f"{'='*70}")

    data = make_box_cylinder(
        n=args.n,
        box_half=args.box_half,
        cyl_radius=args.cyl_radius,
        cyl_half_height=args.cyl_half_height,
        mode=mode,
        signal_kind=args.signal,
        n_azimuthal_cycles=args.n_azimuthal_cycles,
        wavelength_gaps=args.wavelength_gaps,
        seed=args.seed,
    )

    X, intr, y, labels = (data["X"], data["intrinsic"],
                          data["signal"], data["labels"])

    # ---- descriptive stats -------------------------------------------------
    n_cyl = int((labels == LABEL_CYL).sum())
    n_box = int((labels == LABEL_BOX).sum())
    print(f"  Points: {n_cyl} cylinder  {n_box} box  |  "
          f"signal in [{y.min():.2f}, {y.max():.2f}]")

    # ---- preprocessing -----------------------------------------------------
    Xs = ((X - X.mean(0)) / X.std()).astype(np.float32)
    Is = ((intr - intr.mean(0)) / (intr.std() + 1e-8)).astype(np.float32)
    yt   = torch.as_tensor(y, dtype=torch.float32, device=device)
    Xs_t = torch.as_tensor(Xs, device=device)
    Is_t = torch.as_tensor(Is, device=device)

    rng  = np.random.default_rng(args.seed)
    perm = rng.permutation(args.n)
    ntr  = int(args.train_frac * args.n)
    tr, te = perm[:ntr], perm[ntr:]
    tr_t = torch.as_tensor(tr, device=device)
    te_t = torch.as_tensor(te, device=device)
    yc   = yt - yt.mean()

    results   = {}
    pred_full = {}   # full-field predictions for --dump-predictions

    def evaluate(name, kernel, full_inputs, train_inputs, is_manifold=False):
        if args.variational:
            # manifold kernel: inducing points must be graph nodes (no learning)
            # and capped at num_modes to keep Kuu full-rank.
            m_ind = (min(args.num_inducing, args.num_modes)
                     if is_manifold else args.num_inducing)
            model, lik = fit_svgp(
                kernel, train_inputs[tr_t], yt[tr_t], m_ind,
                learn_inducing=(not is_manifold), iters=args.iters)
        else:
            model, lik = fit_gp(kernel, train_inputs[tr_t], yt[tr_t],
                                iters=args.iters)
        tr_rmse = float(((predict_gp(model, lik, train_inputs[tr_t])
                          - yt[tr_t]) ** 2).mean().sqrt())
        results[name] = metrics(
            predict_gp(model, lik, train_inputs[te_t]), yt[te_t])
        print(f"  [{name}]  train_rmse={tr_rmse:.4f}  "
              f"test_rmse={results[name][0]:.4f}  "
              f"ell={_ell_of(kernel):.3g}")
        if getattr(args, "dump_predictions", None):
            pred_full[name] = predict_gp(
                model, lik, full_inputs).detach().cpu().numpy()

    def init_manifold_ell(kernel, tag):
        ell0, lam_q = suggest_lengthscale(
            kernel.eigval, kernel.eigvec, yc, args.nu)
        lo, hi = float(kernel.eigval.min()), float(kernel.eigval.max())
        print(f"  [{tag}] eigval [{lo:.4g}, {hi:.4g}]  "
              f"signal lam90={lam_q:.4g}  -> ell*={ell0:.3g}")
        kernel.lengthscale = ell0
        return kernel

    # 1. Euclidean GP
    evaluate("euclidean (3D Matern)",
             gpytorch.kernels.MaternKernel(nu=args.nu + 0.5).to(device),
             Xs_t, Xs_t)

    # 2. Manifold GP -- plain (no inflation)
    k_plain, coords = build_riemann_kernel(
        Xs, labels, 1.0, args.knn_k, args.num_modes,
        args.nu, args.laplacian_norm, args.graphbandwidth, device)
    evaluate("manifold (plain)",
             init_manifold_ell(k_plain, "plain"),
             coords, coords, is_manifold=True)

    # 3. Manifold GP -- atlas weighted (inflate cylinder↔box edges)
    k_atlas, coords = build_riemann_kernel(
        Xs, labels, args.inflation, args.knn_k, args.num_modes,
        args.nu, args.laplacian_norm, args.graphbandwidth, device)
    atlas_name = f"manifold (atlas x{args.inflation:g})"
    evaluate(atlas_name,
             init_manifold_ell(k_atlas, "atlas"),
             coords, coords, is_manifold=True)

    # 4. Oracle GP -- Matern on intrinsic surface coords
    evaluate("oracle (intrinsic)",
             gpytorch.kernels.MaternKernel(nu=args.nu + 0.5).to(device),
             Is_t, Is_t)

    print(f"\n  --- {mode} held-out metrics ---")
    print(f"  {'method':<40s} {'RMSE':>8} {'Pearson':>9} {'Spearman':>9}")
    for name, (r, p, s) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:<40s} {r:>8.4f} {p:>9.4f} {s:>9.4f}")

    return results, pred_full, k_plain, k_atlas, coords, data, tr, te


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    # Geometry
    ap.add_argument("--n",               type=int,   default=40_000)
    ap.add_argument("--box-half",        type=float, default=4.0,
                    help="box half-side; radial gap = box_half - cyl_radius")
    ap.add_argument("--cyl-radius",      type=float, default=3.9,
                    help="cylinder radius; default gap=0.5 units (tight, creates fold)")
    ap.add_argument("--cyl-half-height", type=float, default=3.9)
    ap.add_argument("--n-azimuthal-cycles", type=int, default=2,
                    help="full cycles of azimuthal signal around the cylinder")
    ap.add_argument("--wavelength-gaps", type=float, default=3.0)
    ap.add_argument("--signal",
                    choices=["geodesic", "ambient"], default="geodesic",
                    help="geodesic: antiphase surface signal (manifold-favourable); "
                         "ambient: smooth 3D function (Euclidean-favourable everywhere)")
    # Which mode(s) to run
    ap.add_argument("--mode", choices=["surface", "volume", "both"], default="both",
                    help="surface: only surfaces; volume: filled box; "
                         "both: run both and compare")
    # Manifold kernel
    ap.add_argument("--knn-k",      type=int,   default=15)
    ap.add_argument("--num-modes",  type=int,   default=200)
    ap.add_argument("--nu",         type=int,   default=2)
    ap.add_argument("--inflation",  type=float, default=50.0,
                    help="cross-region edge inflation for atlas GP")
    ap.add_argument("--laplacian-norm", choices=["symmetric", "randomwalk"],
                    default="randomwalk")
    ap.add_argument("--graphbandwidth", type=float, default=0.0)
    # GP training
    ap.add_argument("--train-frac",    type=float, default=0.5)
    ap.add_argument("--iters",         type=int,   default=150)
    ap.add_argument("--variational",   action="store_true",
                    help="use sparse variational GP (inducing points) instead of "
                         "exact GP -- mirrors the real LatentRiemannGP setup")
    ap.add_argument("--num-inducing",  type=int,   default=256,
                    help="inducing points for --variational (clamped to <= num_modes "
                         "for the manifold kernel, else Kuu is rank-deficient)")
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump-predictions", default=None,
                    help="base path for .npz prediction dumps; surface mode saves "
                         "<base>_surface.npz, volume mode saves <base>_volume.npz. "
                         "Load with toy_box_cylinder_pred_napari.py for inspection.")
    ap.add_argument("--dump-eig-modes", type=int, default=96,
                    help="# eigenmodes stored in the dump for the napari viewer")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    modes = ["surface", "volume"] if args.mode == "both" else [args.mode]
    all_results = {}
    for m in modes:
        res, pred_full, k_plain, k_atlas, coords, data, tr, te = run_mode(
            args, m, device)
        all_results[m] = res

        if args.dump_predictions and pred_full:
            suffix = f"_{m}" if args.mode == "both" else ""
            out_path = args.dump_predictions + suffix + ".npz"
            ne = min(args.dump_eig_modes, args.num_modes)

            def _eig_pack(k):
                ev = k.eigvec.detach().cpu().numpy()[:, :ne].astype(np.float32)
                ea = k.eigval.detach().cpu().numpy()[:ne].astype(np.float32)
                return ev, ea, float(_ell_of(k))

            ev_p, ea_p, ell_p = _eig_pack(k_plain)
            ev_a, ea_a, ell_a = _eig_pack(k_atlas)
            names = list(pred_full.keys())
            np.savez(
                out_path,
                X=data["X"].astype(np.float32),
                y_true=data["signal"].astype(np.float32),
                intrinsic=data["intrinsic"].astype(np.float32),
                theta=data["theta"].astype(np.float32),
                labels=data["labels"].astype(np.int64),
                train_idx=tr.astype(np.int64),
                test_idx=te.astype(np.int64),
                names=np.array(names),
                preds=np.stack([pred_full[n] for n in names]).astype(np.float32),
                signal_kind=np.array(args.signal),
                mode=np.array(m),
                eig_names=np.array(["plain", "atlas"]),
                eig_vec=np.stack([ev_p, ev_a]),
                eig_val=np.stack([ea_p, ea_a]),
                eig_ell=np.array([ell_p, ell_a], dtype=np.float32),
                eig_nu=np.array(args.nu),
            )
            print(f"\n  saved predictions + eigenbases -> {out_path}"
                  f"\n  render with: python toy_box_cylinder_pred_napari.py {out_path}")

    if args.mode == "both":
        print(f"\n{'='*70}")
        print("  COMPARISON: manifold advantage by mode")
        print(f"  signal={args.signal}  N={args.n}  inflation={args.inflation}")
        print(f"{'='*70}")
        print(f"  {'method':<40s} {'surface RMSE':>13} {'volume RMSE':>12} {'ratio':>7}")
        methods = ["euclidean (3D Matern)",
                   "manifold (plain)",
                   f"manifold (atlas x{args.inflation:g})",
                   "oracle (intrinsic)"]
        for m in methods:
            sr = all_results["surface"].get(m, (float("nan"),)*3)[0]
            vr = all_results["volume"].get(m, (float("nan"),)*3)[0]
            ratio = sr / vr if vr > 0 else float("nan")
            print(f"  {m:<40s} {sr:>13.4f} {vr:>12.4f} {ratio:>7.2f}x")

        print("\nInterpretation:")
        print("  ratio >> 1 → method struggles in surface mode but not volume mode")
        print("  ratio ≈ 1  → method equally good (or bad) in both modes")
        euc_s = all_results["surface"].get("euclidean (3D Matern)", (1,))[0]
        atl_s = all_results["surface"].get(
            f"manifold (atlas x{args.inflation:g})", (1,))[0]
        print(f"\n  atlas vs euclidean in SURFACE mode: "
              f"{atl_s:.4f} vs {euc_s:.4f} = "
              f"{'manifold WINS' if atl_s < euc_s else 'euclidean wins'} "
              f"(ratio {atl_s/euc_s:.2f})")


if __name__ == "__main__":
    main()
