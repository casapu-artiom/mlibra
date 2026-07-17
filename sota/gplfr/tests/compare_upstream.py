"""Numerical equivalence check: this GPyTorch GPLFR core vs upstream Pyro GPLFR.

The GP / inference layers differ by design (upstream: full-GP MAP in Pyro; here:
inducing-point SVGP in GPyTorch), so end-to-end outputs are *not* expected to
match. What must match exactly is the shared collapsed-decoder math we ported:
the collapsed log-likelihood (training objective), the decoder posterior
(predictive mean), and the predictive-variance formula. This script feeds both
implementations identical inputs in float64 (jitter=0) and reports the worst
absolute difference.

Usage:
    git clone https://github.com/edstevenson/GPLFR.git /path/to/GPLFR
    pip install pyro-ppl beartype jaxtyping
    GPLFR_UPSTREAM_DIR=/path/to/GPLFR python gplfr/tests/compare_upstream.py

Exits nonzero if the worst difference exceeds the tolerance.
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch

TOL = 1e-9


def _load_mine():
    """Import this repo's GPLFR by file path, so it can't clash with the
    upstream package (also named ``gplfr``) we put on ``sys.path``."""
    here = Path(__file__).resolve().parent.parent  # <repo>/gplfr
    spec = importlib.util.spec_from_file_location("mine_gplfr_model", here / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GPLFR


def main() -> int:
    up_dir = os.environ.get("GPLFR_UPSTREAM_DIR")
    if not up_dir or not Path(up_dir).exists():
        print("GPLFR_UPSTREAM_DIR not set / missing; clone "
              "https://github.com/edstevenson/GPLFR.git and point this at it.")
        return 2
    sys.path.insert(0, up_dir)

    torch.set_default_dtype(torch.float64)
    from gplfr.model import GPLFR as UpstreamGPLFR  # from the clone on sys.path
    MyGPLFR = _load_mine()
    from l3di.lgp import IndependentMultitaskGPModel

    torch.manual_seed(1)
    n, q, p = 25, 4, 7
    Z = torch.randn(n, q, dtype=torch.float64)
    Y = torch.randn(n, p, dtype=torch.float64)
    sigma = torch.tensor(0.37, dtype=torch.float64)

    up = UpstreamGPLFR(latent_dim=q, jitter=0.0, dtype=torch.float64)
    gp = IndependentMultitaskGPModel(
        inducing_points=torch.randn(6, 3), num_tasks=q, kernel_type="rbf", input_dim=3,
    )
    mine = MyGPLFR(gp_model=gp, p=p, d=q, device="cpu", jitter=0.0)
    mine.double()

    diffs = {}

    # A. collapsed log-likelihood (training objective)
    diffs["collapsed_loglik"] = (
        up._collapsed_loglikelihood(Y, Z, sigma)
        - mine._collapsed_loglikelihood(Y, Z, sigma)
    ).abs().item()

    # B. decoder posterior (predictive mean)
    muW_up, SigW_up = up._decoder_posterior(Z, Y, sigma)
    muW_mine, SigW_mine = mine._decoder_posterior(Z, Y, sigma)
    diffs["decoder_mu_W"] = (muW_up - muW_mine).abs().max().item()
    diffs["decoder_Sigma_W"] = (SigW_up - SigW_mine).abs().max().item()

    # C. predictive-moment formula (variance combination), with chosen latents
    t = 12
    z_mean = torch.randn(t, q, dtype=torch.float64)
    z_var = torch.rand(t, q, dtype=torch.float64) + 0.1
    sigma_sq = sigma ** 2
    up._state_ = {"mu_W": muW_up, "Sigma_W": SigW_up, "sigma_sq": sigma_sq}
    up._gp_predict_latents = lambda Xn, state: (z_mean, z_var)
    mine.mu_W, mine.Sigma_W, mine.sigma_sq = muW_mine, SigW_mine, sigma_sq
    mine._latent_moments = lambda coords: (z_mean, z_var, None)
    for include_noise in (False, True):
        mean_up, var_up = up._predict_moments(torch.zeros(t, 3), include_noise=include_noise)
        mean_mine, std_mine = mine.predict_moments(torch.zeros(t, 3), include_noise=include_noise)
        tag = "noise" if include_noise else "nonoise"
        diffs[f"predict_mean_{tag}"] = (mean_up - mean_mine).abs().max().item()
        diffs[f"predict_var_{tag}"] = (var_up - std_mine ** 2).abs().max().item()

    for k, v in diffs.items():
        print(f"  {k:24s} max|diff| = {v:.2e}")
    worst = max(diffs.values())
    ok = worst < TOL
    print(f"\nWORST DIFF = {worst:.2e}  (tol {TOL:.0e})  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
