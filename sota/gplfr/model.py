"""GPLFR model: inducing-point latent GP + analytically collapsed linear decoder.

This is a GPyTorch port of the GPLFR model (https://github.com/edstevenson/GPLFR,
arXiv:2606.06576). The decoder is GPLFR's linear-Gaussian map ``Y = Z W + eps``
with ``W`` marginalized in closed form; the latent GP that produces ``Z`` is
pluggable. GPLFR only touches its ``gp_model`` through two hooks — the per-point
marginal moments ``gp_model(coords).mean/.variance`` and the latent KL (see
``_gp_kl_divergence``) — so any multitask GP with ``num_tasks = latent_dim`` drops
in. The runner (``sota/gplfr_experiment.py --base-gp``) currently wires three:
the Euclidean inducing-point ``IndependentMultitaskGPModel`` (default, as used by
``l3di.lgp.LGP``), the manifold inducing-point ``LatentRiemannGP``, and the
weight-space ``SpectralLatentGP`` over the manifold spectrum.

Pixel-as-sample mapping for MALDI:
  X = standardized 3D CCF coordinates,  Y = p lipid channels,  Z = q latent factors.

Differences from upstream (deliberate, to scale to many pixels):
  * Latent values ``Z`` come from the variational GP posterior rather than being
    free MAP variables. During training we draw ``Z`` from the *marginal*
    (per-point) posterior so the full-batch collapsed likelihood never needs an
    O(N^3) joint sample.
  * Observation noise ``sigma`` is a single learnable scalar, matching the
    upstream collapsed-likelihood derivation (homoscedastic across outputs).

The public surface (``train_model`` / ``predict`` / ``loss_function``) mirrors
``l3di.lgp.LGP`` so it slots into ``maldi.experiment.MaldiExperiment`` unchanged.
"""

import math

import torch
from torch import nn
from tqdm import tqdm

try:  # wandb is initialised by MaldiExperiment.run(); guard for tests/standalone use.
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _wandb_log(payload):
    """Log to W&B only when a run is active (so tests don't spin one up)."""
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(payload)


class GPLFR(nn.Module):
    """Latent Gaussian Process with a collapsed (marginalized) linear decoder.

    Args:
        gp_model: a multitask inducing-point GP whose ``forward`` returns a
            ``q``-task posterior over the latent factors (e.g.
            ``l3di.lgp.IndependentMultitaskGPModel(num_tasks=d)``).
        p (int): number of output channels (lipids).
        d (int): latent dimension ``q`` (must equal ``gp_model``'s ``num_tasks``).
        device: torch device.
        inverse_temperature (float): scales the collapsed log-likelihood relative
            to the variational KL (GPLFR's ``inverse_temperature``).
        jitter (float): added to ``sigma^2`` for Cholesky stability.
        use_rsample (bool): draw a marginal posterior sample for ``Z`` during
            training (True) or use the posterior mean (False, closer to upstream
            MAP).
    """

    def __init__(self, gp_model, p, d, device, inverse_temperature=0.1,
                 jitter=1e-6, use_rsample=True):
        super().__init__()
        self.mode = "gplfr"
        self.p = p
        self.d = d  # latent dimension q
        self.gp_model = gp_model
        self.inverse_temperature = float(inverse_temperature)
        self.jitter = float(jitter)
        self._eps = 1e-12
        self.use_rsample = use_rsample

        # Single scalar observation noise (homoscedastic), learnable in log-space.
        self.log_sigma = nn.Parameter(torch.zeros(()))

        # Collapsed-decoder posterior, filled by ``build_state`` after training and
        # persisted in the state_dict so prediction works after a checkpoint reload.
        self.register_buffer("mu_W", torch.zeros(d, p))
        self.register_buffer("Sigma_W", torch.eye(d))
        self.register_buffer("sigma_sq", torch.ones(()))
        self.register_buffer("state_ready", torch.zeros((), dtype=torch.bool))

        self.float_type = torch.float32
        self.device = device
        self.to(device)

    # ------------------------------------------------------------------
    # Latent factors
    # ------------------------------------------------------------------
    def _latent_moments(self, coords):
        """Per-point marginal posterior over the q latent factors.

        Returns ``(mean, var)`` each of shape ``(N, q)``. Uses the *marginal*
        (diagonal) posterior, never the O(N^3) joint covariance.
        """
        gp_posterior = self.gp_model(coords)
        return gp_posterior.mean, gp_posterior.variance, gp_posterior

    def _sample_latent(self, coords):
        mean, var, gp_posterior = self._latent_moments(coords)
        if self.training and self.use_rsample:
            z = mean + var.clamp_min(0.0).sqrt() * torch.randn_like(mean)
        else:
            z = mean
        return z, gp_posterior

    # ------------------------------------------------------------------
    # Collapsed linear decoder (ported from upstream GPLFR.model)
    # ------------------------------------------------------------------
    def _collapsed_loglikelihood(self, Y, Z, sigma):
        """Log marginal likelihood of ``Y = Z W + eps`` with ``W`` marginalized.

        ``Z`` is ``(N, q)``, ``Y`` is ``(N, p)``, ``sigma`` a scalar tensor.
        Direct port of upstream ``_collapsed_loglikelihood`` (W ~ N(0, I) prior,
        eps ~ N(0, sigma^2 I)).
        """
        # The likelihood depends on Z only through the sufficient statistics
        # ZᵀZ, ZᵀY and ΣYᵀY, so we route through the stats form (which is also
        # what lets training chunk over minibatches — see ``train_model``).
        n = Y.shape[0]
        return self._collapsed_ll_from_stats(
            Z.T @ Z, Z.T @ Y, torch.sum(Y * Y), n, sigma)

    def _collapsed_ll_from_stats(self, ZtZ, ZtY, sum_YY, n, sigma):
        """Collapsed log-likelihood from sufficient statistics.

        ``ZtZ`` is ``(q,q)``, ``ZtY`` is ``(q,p)``, ``sum_YY`` a scalar, ``n`` the
        number of rows. Exact given the (full-data) statistics; identical to the
        upstream port when the statistics span the whole training set.
        """
        q, p = ZtY.shape
        sigma_sq = sigma ** 2 + ZtZ.new_tensor(self.jitter + self._eps)
        inv_sigma_sq = 1.0 / sigma_sq

        eye = torch.eye(q, device=ZtZ.device, dtype=ZtZ.dtype)
        Psi = eye + inv_sigma_sq * ZtZ
        L_Psi = torch.linalg.cholesky(Psi)
        logdet_Psi = 2.0 * torch.sum(torch.log(torch.diagonal(L_Psi)))

        Psi_inv_ZtY = torch.cholesky_solve(ZtY, L_Psi)
        return (
            -0.5 * p * n * math.log(2.0 * math.pi)
            - 0.5 * p * n * torch.log(sigma_sq)
            - 0.5 * p * logdet_Psi
            - 0.5 * inv_sigma_sq * sum_YY
            + 0.5 * inv_sigma_sq ** 2 * torch.sum(ZtY * Psi_inv_ZtY)
        )

    def _decoder_posterior(self, Z, Y, sigma):
        """Posterior over the decoder ``W``: returns ``(mu_W (q,p), Sigma_W (q,q))``."""
        return self._decoder_posterior_from_stats(Z.T @ Z, Z.T @ Y, sigma)

    def _decoder_posterior_from_stats(self, ZtZ, ZtY, sigma):
        """Decoder posterior from sufficient statistics ``ZᵀZ`` / ``ZᵀY``."""
        q = ZtZ.shape[0]
        inv_sigma_sq = 1.0 / (sigma ** 2 + ZtZ.new_tensor(self.jitter + self._eps))
        eye = torch.eye(q, device=ZtZ.device, dtype=ZtZ.dtype)
        precision = eye + inv_sigma_sq * ZtZ
        L = torch.linalg.cholesky(precision)
        mu_W = inv_sigma_sq * torch.cholesky_solve(ZtY, L)
        Sigma_W = torch.cholesky_inverse(L)
        return mu_W, Sigma_W

    # ------------------------------------------------------------------
    # Latent-GP KL (base-GP agnostic)
    # ------------------------------------------------------------------
    def _gp_kl_divergence(self):
        """KL(q||p) of the latent GP, summed to a scalar.

        Inducing-point bases (``IndependentMultitaskGPModel``, ``LatentRiemannGP``)
        expose it through ``variational_strategy.kl_divergence()``; the weight-space
        ``SpectralLatentGP`` parametrizes q(w) directly and exposes
        ``kl_divergence()`` on the model itself. Duck-type over the two so any base
        GP drops in.
        """
        vs = getattr(self.gp_model, "variational_strategy", None)
        if vs is not None:
            return vs.kl_divergence().sum()
        return self.gp_model.kl_divergence().sum()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss_function(self, Y, Z, beta=1.0, scale=1.0):
        """ELBO-style objective: -inverse_temperature * scale * collapsed_ll + beta * KL.

        ``scale`` (= n_total / n_batch) rescales a minibatch's collapsed
        log-likelihood up to dataset magnitude so the KL weighting is independent
        of batch size (the standard doubly-stochastic SVGP scaling).
        """
        sigma = self.log_sigma.exp()
        collapsed_ll = self._collapsed_loglikelihood(Y, Z, sigma)
        kl_gp = self._gp_kl_divergence()
        recon_loss = -self.inverse_temperature * scale * collapsed_ll
        total_loss = recon_loss + beta * kl_gp
        return total_loss, recon_loss, kl_gp

    # ------------------------------------------------------------------
    # Decoder-posterior cache for prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_state(self, dataloader):
        """Cache the global decoder posterior ``mu_W`` / ``Sigma_W``.

        Accumulates the sufficient statistics ``ZᵀZ`` / ``ZᵀY`` over the whole
        training set chunk-by-chunk (no autograd graph), so memory stays at one
        minibatch's GP forward — the decoder posterior is still exact and global.
        """
        was_training = self.training
        self.eval()
        ZtZ = torch.zeros(self.d, self.d, device=self.device, dtype=self.mu_W.dtype)
        ZtY = torch.zeros(self.d, self.p, device=self.device, dtype=self.mu_W.dtype)
        for Y_b, coord_b in dataloader:
            Y_b = Y_b.to(self.device)
            coord_b = coord_b.to(self.device)
            z = self._latent_moments(coord_b)[0]
            ZtZ += z.T @ z
            ZtY += z.T @ Y_b
        sigma = self.log_sigma.exp()
        mu_W, Sigma_W = self._decoder_posterior_from_stats(ZtZ, ZtY, sigma)
        self.mu_W.copy_(mu_W)
        self.Sigma_W.copy_(Sigma_W)
        self.sigma_sq.copy_(sigma ** 2)
        self.state_ready.fill_(True)
        if was_training:
            self.train()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def forward(self, coords):
        z, gp_posterior = self._sample_latent(coords)
        x_reconstructed = z @ self.mu_W
        return x_reconstructed, gp_posterior

    def predict(self, coords):
        """Posterior predictive mean. Returns ``(preds (N,p), gp_posterior)``.

        Signature matches ``LGP.predict`` so ``MaldiExperiment`` is unchanged.
        """
        with torch.no_grad():
            mean, _, gp_posterior = self._latent_moments(coords)
            preds = mean @ self.mu_W
            return preds, gp_posterior

    @torch.no_grad()
    def predict_moments(self, coords, include_noise=False):
        """Predictive mean and standard deviation (ported from upstream).

        Combines GP latent uncertainty with the collapsed-decoder posterior, and
        optionally observation noise. Returns ``(mean (N,p), std (N,p))``.
        """
        z_mean, z_var, _ = self._latent_moments(coords)
        mu_W, Sigma_W = self.mu_W, self.Sigma_W
        mean = z_mean @ mu_W
        var = (
            (z_mean @ Sigma_W * z_mean).sum(-1, keepdim=True)
            + z_var @ (mu_W ** 2)
            + z_var @ torch.diagonal(Sigma_W).unsqueeze(-1)
        )
        if include_noise:
            var = var + self.sigma_sq
        return mean, var.clamp_min(0.0).sqrt()

    # ------------------------------------------------------------------
    # Training (minibatched: each step optimizes the batch's collapsed
    # marginal likelihood, rescaled to dataset magnitude, plus the GP KL).
    # ------------------------------------------------------------------
    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch,
                    print_every=1000):
        self.to(self.device)
        self.train()
        n_total = len(dataloader.dataset)

        for epoch in tqdm(range(current_epoch, epochs)):
            mean_loss = recon_loss_sum = kl_loss_sum = 0.0
            n_batches = 0
            for Y_b, coord_b in dataloader:
                Y_b = Y_b.to(self.device)
                coord_b = coord_b.to(self.device)
                optimizer.zero_grad()
                z, _ = self._sample_latent(coord_b)
                scale = n_total / Y_b.shape[0]
                loss, recon_loss, kl_div = self.loss_function(Y_b, z, beta=1.0, scale=scale)
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_div.item()
                n_batches += 1

            torch.save(self.state_dict(), exp_path / f"checkpoints/model_{epoch}.pth")
            _wandb_log({"loss": mean_loss / n_batches,
                        "reconstruction_loss": recon_loss_sum / n_batches,
                        "kl_loss": kl_loss_sum / n_batches,
                        "sigma": self.log_sigma.exp().item()})
            if epoch % max(1, print_every) == 0:
                print(f"Epoch {epoch} loss: {mean_loss / n_batches:.4f} "
                      f"(recon {recon_loss_sum / n_batches:.4f}, "
                      f"kl {kl_loss_sum / n_batches:.4f}, "
                      f"sigma {self.log_sigma.exp().item():.4g})")

        # Cache the collapsed-decoder posterior (global, chunked) for prediction.
        self.build_state(dataloader)
        torch.save(self.state_dict(), exp_path / "model.pth")