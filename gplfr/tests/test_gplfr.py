"""Tests for the GPyTorch GPLFR port.

  * correctness: the collapsed log-likelihood equals a brute-force evaluation of
    log N(Y_j; 0, Z Zᵀ + σ² I) summed over output columns (the analytic
    marginalization of the linear decoder W ~ N(0, I)).
  * smoke: a tiny model trains, predicts, and yields finite, correctly-shaped
    moments.
"""
import torch
from torch.distributions import MultivariateNormal

from l3di.lgp import IndependentMultitaskGPModel
from gplfr import GPLFR


def _tiny_model(q=3, p=5, n_inducing=8, jitter=0.0, device="cpu"):
    inducing = torch.randn(n_inducing, 3)
    gp = IndependentMultitaskGPModel(
        inducing_points=inducing, num_tasks=q, kernel_type="rbf", input_dim=3,
    )
    return GPLFR(gp_model=gp, p=p, d=q, device=device, jitter=jitter)


def test_collapsed_loglikelihood_matches_bruteforce():
    torch.manual_seed(0)
    n, q, p = 12, 3, 4
    Z = torch.randn(n, q, dtype=torch.float64)
    Y = torch.randn(n, p, dtype=torch.float64)
    sigma = torch.tensor(0.5, dtype=torch.float64)

    model = _tiny_model(q=q, p=p, jitter=0.0)
    ll = model._collapsed_loglikelihood(Y, Z, sigma)

    # Marginal over W ~ N(0, I): each column Y_j ~ N(0, Z Zᵀ + σ² I).
    cov = Z @ Z.T + sigma ** 2 * torch.eye(n, dtype=torch.float64)
    mvn = MultivariateNormal(torch.zeros(n, dtype=torch.float64), covariance_matrix=cov)
    ll_ref = mvn.log_prob(Y.T).sum()

    assert torch.allclose(ll, ll_ref, rtol=1e-5, atol=1e-5), (ll.item(), ll_ref.item())


def test_decoder_posterior_shapes_and_psd():
    torch.manual_seed(0)
    n, q, p = 20, 4, 6
    Z = torch.randn(n, q, dtype=torch.float64)
    Y = torch.randn(n, p, dtype=torch.float64)
    model = _tiny_model(q=q, p=p)
    mu_W, Sigma_W = model._decoder_posterior(Z, Y, torch.tensor(0.3, dtype=torch.float64))
    assert mu_W.shape == (q, p)
    assert Sigma_W.shape == (q, q)
    # Posterior covariance must be symmetric positive-definite.
    torch.linalg.cholesky(0.5 * (Sigma_W + Sigma_W.T))


def test_train_predict_smoke(tmp_path):
    torch.manual_seed(0)
    n, q, p = 64, 3, 5
    coords = torch.randn(n, 3)
    Y = torch.randn(n, p)
    dataset = torch.utils.data.TensorDataset(Y, coords)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model = _tiny_model(q=q, p=p, jitter=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    (tmp_path / "checkpoints").mkdir()

    model.train_model(tmp_path, loader, optimizer, epochs=3, current_epoch=0, print_every=10)
    assert (tmp_path / "model.pth").exists()
    assert bool(model.state_ready)

    new_coords = torch.randn(10, 3)
    preds, gp_posterior = model.predict(new_coords)
    assert preds.shape == (10, p)
    assert torch.isfinite(preds).all()

    mean, std = model.predict_moments(new_coords, include_noise=True)
    assert mean.shape == (10, p) and std.shape == (10, p)
    assert torch.isfinite(std).all() and (std >= 0).all()
