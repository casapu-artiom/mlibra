"""Behavioral comparison: fit upstream Pyro GPLFR and this GPyTorch port on the
same synthetic dataset and compare held-out accuracy.

Unlike ``compare_upstream.py`` (which checks the shared closed-form math is
identical), this trains both models end-to-end. The GP layers differ by design
(full-GP MAP vs inducing-point SVGP), so results are expected to be *close*, not
identical. We report test RMSE vs the noisy targets and R^2 vs the true signal.

Usage:
    GPLFR_UPSTREAM_DIR=/path/to/GPLFR python gplfr/tests/compare_upstream_fit.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def _load_mine():
    here = Path(__file__).resolve().parent.parent  # <repo>/gplfr
    spec = importlib.util.spec_from_file_location("mine_gplfr_model", here / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GPLFR


def _metrics(pred, Y_obs, Y_sig):
    rmse = float(np.sqrt(np.mean((pred - Y_obs) ** 2)))
    ss_res = np.sum((Y_sig - pred) ** 2)
    ss_tot = np.sum((Y_sig - Y_sig.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return rmse, r2


def main() -> int:
    up_dir = os.environ.get("GPLFR_UPSTREAM_DIR")
    if not up_dir or not Path(up_dir).exists():
        print("GPLFR_UPSTREAM_DIR not set / missing.")
        return 2
    sys.path.insert(0, up_dir)

    from gplfr.model import GPLFR as UpstreamGPLFR
    from gplfr.synthetic import create_synthetic_data
    MyGPLFR = _load_mine()
    from l3di.lgp import IndependentMultitaskGPModel

    torch.manual_seed(0)
    np.random.seed(0)

    # Low-N, high-output synthetic problem (GPLFR's regime). Dx=3 to match the
    # latent GP's hard-coded 3D input / linear mean.
    # Signal clearly above the spatially-correlated nuisance so both models have
    # something learnable; sigma_sig >> sigma_nuis.
    data = create_synthetic_data(N=400, Dx=3, H=12, W=12, D_sig=4,
                                 kernel="rbf", ell=1.0, sigma_sig=0.6,
                                 sigma_nuis=0.1, sigma_eps=0.02, seed=0)
    X, Y, Y_sig = data["X"], data["Y"], data["Y_sig"]
    n_tr = 300

    # Standardize on the train split (both models assume standardized I/O).
    x_mu, x_sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-8
    y_mu, y_sd = Y[:n_tr].mean(0), Y[:n_tr].std(0) + 1e-8
    Xs = (X - x_mu) / x_sd
    Ys = (Y - y_mu) / y_sd
    Xtr, Xte = Xs[:n_tr], Xs[n_tr:]
    Ytr = Ys[:n_tr]
    Yte_obs, Yte_sig = Y[n_tr:], Y_sig[n_tr:]   # compare in original units

    q = 6

    # Matched objective weighting so the comparison is fair (both trust the data).
    inverse_temperature = 1.0

    # ---- upstream ----
    up = UpstreamGPLFR(latent_dim=q, kernel="rbf", lengthscale_grouping="per_latent",
                       inverse_temperature=inverse_temperature)
    up.fit(Xtr, Ytr, num_steps=3000, verbose=False, seed=0)
    pred_up = up.predict(Xte) * y_sd + y_mu
    rmse_up, r2_up = _metrics(pred_up, Yte_obs, Yte_sig)

    # ---- mine ----
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    inducing = Xtr_t[torch.randperm(n_tr)[:80]].clone()
    gp = IndependentMultitaskGPModel(inducing_points=inducing, num_tasks=q,
                                     kernel_type="rbf", input_dim=3)
    mine = MyGPLFR(gp_model=gp, p=Ytr.shape[1], d=q, device="cpu",
                   inverse_temperature=inverse_temperature)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Ytr_t, Xtr_t), batch_size=100, shuffle=True)
    opt = torch.optim.Adam(mine.parameters(), lr=1e-2)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "checkpoints").mkdir()
        mine.train_model(Path(tmp), loader, opt, epochs=400, current_epoch=0, print_every=10000)
    with torch.no_grad():
        pred_mine = mine.predict(Xte_t)[0].cpu().numpy() * y_sd + y_mu
    rmse_mine, r2_mine = _metrics(pred_mine, Yte_obs, Yte_sig)

    # Baseline: predict the (train) mean everywhere.
    base = np.broadcast_to(Y[:n_tr].mean(0), Yte_obs.shape)
    rmse_base, r2_base = _metrics(base, Yte_obs, Yte_sig)

    # Agreement: do the two models predict the same structure on the test set?
    agree = float(np.corrcoef(pred_up.ravel(), pred_mine.ravel())[0, 1])

    print(f"{'model':<14}{'test RMSE (obs)':>18}{'R^2 vs signal':>16}")
    print(f"{'mean-baseline':<14}{rmse_base:>18.4f}{r2_base:>16.3f}")
    print(f"{'upstream':<14}{rmse_up:>18.4f}{r2_up:>16.3f}")
    print(f"{'mine':<14}{rmse_mine:>18.4f}{r2_mine:>16.3f}")
    print(f"\nprediction agreement (corr upstream vs mine) = {agree:.3f}")

    # Correct success criterion: both clearly learn the signal AND agree on the
    # predicted structure. A tight RMSE match is NOT expected (different
    # estimators: full-GP MAP vs inducing-point variational).
    ok = (r2_up > 0.6) and (r2_mine > 0.6) and (agree > 0.85)
    print("->", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
