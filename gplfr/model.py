"""GPLFR model: inducing-point latent GP + analytically collapsed linear decoder.

This is a GPyTorch port of the GPLFR model (https://github.com/edstevenson/GPLFR,
arXiv:2606.06576). The latent GP is the same variational inducing-point multitask
GP used by ``l3di.lgp.LGP`` (``IndependentMultitaskGPModel`` with
``num_tasks = latent_dim``); the decoder is GPLFR's linear-Gaussian map
``Y = Z W + eps`` with ``W`` marginalized in closed form.

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
        n, p = Y.shape
        q = Z.shape[1]
        sigma_sq = sigma ** 2 + Y.new_tensor(self.jitter + self._eps)
        inv_sigma_sq = 1.0 / sigma_sq

        eye = torch.eye(q, device=Y.device, dtype=Y.dtype)
        Psi = eye + inv_sigma_sq * (Z.T @ Z)
        L_Psi = torch.linalg.cholesky(Psi)
        logdet_Psi = 2.0 * torch.sum(torch.log(torch.diagonal(L_Psi)))

        ZTY = Z.T @ Y
        Psi_inv_ZTY = torch.cholesky_solve(ZTY, L_Psi)
        return (
            -0.5 * p * n * math.log(2.0 * math.pi)
            - 0.5 * p * n * torch.log(sigma_sq)
            - 0.5 * p * logdet_Psi
            - 0.5 * inv_sigma_sq * torch.sum(Y * Y)
            + 0.5 * inv_sigma_sq ** 2 * torch.sum(ZTY * Psi_inv_ZTY)
        )

    def _decoder_posterior(self, Z, Y, sigma):
        """Posterior over the decoder ``W``: returns ``(mu_W (q,p), Sigma_W (q,q))``."""
        q = Z.shape[1]
        inv_sigma_sq = 1.0 / (sigma ** 2 + Y.new_tensor(self.jitter + self._eps))
        eye = torch.eye(q, device=Y.device, dtype=Y.dtype)
        precision = eye + inv_sigma_sq * (Z.T @ Z)
        L = torch.linalg.cholesky(precision)
        mu_W = inv_sigma_sq * torch.cholesky_solve(Z.T @ Y, L)
        Sigma_W = torch.cholesky_inverse(L)
        return mu_W, Sigma_W

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss_function(self, Y, Z, beta=1.0):
        """ELBO-style objective: -inverse_temperature * collapsed_ll + beta * KL."""
        sigma = self.log_sigma.exp()
        collapsed_ll = self._collapsed_loglikelihood(Y, Z, sigma)
        kl_gp = self.gp_model.variational_strategy.kl_divergence().sum()
        recon_loss = -self.inverse_temperature * collapsed_ll
        total_loss = recon_loss + beta * kl_gp
        return total_loss, recon_loss, kl_gp

    # ------------------------------------------------------------------
    # Decoder-posterior cache for prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_state(self, coords, Y):
        """Compute and cache ``mu_W`` / ``Sigma_W`` from the full training set."""
        was_training = self.training
        self.eval()
        mean, _, _ = self._latent_moments(coords)
        sigma = self.log_sigma.exp()
        mu_W, Sigma_W = self._decoder_posterior(mean, Y, sigma)
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
    # Training (full-batch — the collapsed likelihood couples all points)
    # ------------------------------------------------------------------
    @staticmethod
    def _materialize_full_batch(dataloader):
        """Return ``(Y, coords)`` for the entire training set.

        Fast path for the ``TensorDataset`` built by ``MaldiExperiment``
        (``tensors == (train_data, coordinates_train)``); otherwise concatenate
        the loader.
        """
        dataset = dataloader.dataset
        tensors = getattr(dataset, "tensors", None)
        if tensors is not None and len(tensors) == 2:
            return tensors[0], tensors[1]
        ys, coords = [], []
        for y, coord in dataloader:
            ys.append(y)
            coords.append(coord)
        return torch.cat(ys, dim=0), torch.cat(coords, dim=0)

    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch,
                    print_every=1000):
        self.to(self.device)
        self.train()

        Y_full, coord_full = self._materialize_full_batch(dataloader)
        Y_full = Y_full.to(self.device)
        coord_full = coord_full.to(self.device)

        for epoch in tqdm(range(current_epoch, epochs)):
            optimizer.zero_grad()
            z, _ = self._sample_latent(coord_full)
            loss, recon_loss, kl_div = self.loss_function(Y_full, z, beta=1.0)
            loss.backward()
            optimizer.step()

            torch.save(self.state_dict(), exp_path / f"checkpoints/model_{epoch}.pth")
            _wandb_log({"loss": loss.item(),
                        "reconstruction_loss": recon_loss.item(),
                        "kl_loss": kl_div.item(),
                        "sigma": self.log_sigma.exp().item()})
            if epoch % max(1, print_every) == 0:
                print(f"Epoch {epoch} loss: {loss.item():.4f} "
                      f"(recon {recon_loss.item():.4f}, kl {kl_div.item():.4f}, "
                      f"sigma {self.log_sigma.exp().item():.4g})")

        # Cache the collapsed-decoder posterior for prediction, then save.
        self.build_state(coord_full, Y_full)
        torch.save(self.state_dict(), exp_path / "model.pth")