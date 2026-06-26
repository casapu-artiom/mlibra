import math

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from gpytorch.distributions import (MultitaskMultivariateNormal,
                                    MultivariateNormal)
from gpytorch.kernels import (MaternKernel, MultitaskKernel, RBFKernel,
                              ScaleKernel, ProductKernel, PeriodicKernel)
from gpytorch.means import ConstantMean, MultitaskMean, LinearMean, ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  IndependentMultitaskVariationalStrategy,
                                  MultitaskVariationalStrategy,
                                  NaturalVariationalDistribution,
                                  VariationalStrategy)
from gpytorch.likelihoods.multitask_gaussian_likelihood import MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from torch import nn
from tqdm import tqdm
import wandb
from linear_operator.operators import (DiagLinearOperator, RootLinearOperator,
                                        LowRankRootLinearOperator, MatmulLinearOperator)

# --------------------------------------------------------------------------- #
# Hyperparameter priors (MAP-II) for the decoder-ELBO manifold models.
#
# Unlike the per-lipid path (lgp_experiment_per_lipid.py), these models put an
# MLP decoder between the GP latent f and the observations, so the likelihood
# p(x | f) = N(x | decode(f), σ²) is NOT Gaussian in f. That's why training uses
# rsample()+decode (a Monte-Carlo ELBO) and CANNOT use gpytorch.VariationalELBO
# (whose expected_log_prob assumes a likelihood directly Gaussian in f). The one
# piece that transfers from the per-lipid refactor is the hyperparameter priors:
# we add log p(θ) to the ELBO so kernel/noise hyperparameters get MAP-II
# regularization, matching what manifold_informed_train applies to the exact
# model.
#
# The prior objects are stored OUTSIDE the nn.Module graph (in a plain list, not
# as submodules) and evaluated manually in loss_function — so the model's
# state_dict is byte-for-byte unchanged and existing checkpoints still load.
# --------------------------------------------------------------------------- #
def _build_kernel_hyperpriors(gp_model, lengthscale_prior_mean, device):
    """Priors on the GP kernel hyperparameters whose natural scale is known for
    z-scored data: outputscale (signal variance ~1) → Gamma(2,2); lengthscale
    only if a center is given (its scale is data/constraint-dependent).

    Returns a list of (prior, closure) pairs; ``closure()`` returns the current
    constrained value. Priors are moved to ``device`` here (they're not module
    submodules, so .to() on the model won't move them)."""
    from gpytorch.priors import GammaPrior
    terms = []
    covar = getattr(gp_model, "covar_module", None)
    if isinstance(covar, ScaleKernel):
        terms.append((GammaPrior(2.0, 2.0).to(device), lambda c=covar: c.outputscale))
        base = getattr(covar, "base_kernel", None)
        if (lengthscale_prior_mean is not None and base is not None
                and hasattr(base, "raw_lengthscale")):
            conc = 3.0
            terms.append((
                GammaPrior(conc, conc / float(lengthscale_prior_mean)).to(device),
                lambda b=base: b.lengthscale,
            ))
    return terms


def _make_noise_prior(device):
    """Soft prior on the observation-noise variance exp(log_var_n) for z-scored
    outputs: a SmoothedBox over [e^-3, e^0.5] that discourages noise collapse /
    explosion without a hard clamp."""
    from gpytorch.priors import SmoothedBoxPrior
    return SmoothedBoxPrior(math.exp(-3.0), math.exp(0.5), sigma=0.1).to(device)


def _eval_hyperprior_logprob(terms, ref):
    """Sum of prior log-probs over the (prior, closure) ``terms``. ``ref`` is any
    tensor on the right device/dtype used to seed the accumulator."""
    lp = ref.new_zeros(())
    for prior, closure in terms:
        lp = lp + prior.log_prob(closure()).sum()
    return lp


class BatchedRiemannWrapper(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel):
        super().__init__()
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        # 1. Track if we are in a batched/multi-task context
        is_batch = x1.dim() == 3
        
        # 2. Strip the batch dimension for the Riemann Kernel
        if is_batch: 
            x1 = x1[0]
        if x2.dim() == 3: 
            x2 = x2[0]
            
        # 3. Compute the kernel [N x M]
        res = self.base_kernel.forward(x1, x2, diag=diag, **params)
        
        # 4. CRITICAL FIX: Put the batch dimension back [1 x N x M]
        if is_batch and not diag:
            res = res.unsqueeze(0)
            
        return res

class PerTaskRiemannWrapper(gpytorch.kernels.Kernel):
    """Like ``BatchedRiemannWrapper`` but each task has its OWN
    ``RiemannMaternKernel`` (sharing the eigenpairs / graph tensors, with an
    independent learnable lengthscale). Produces a batched covariance
    ``(T, n1, n2)`` by stacking per-task feature maps from the validated
    ``RiemannKernel.features()`` — so per-task lengthscale comes for free without
    touching the kernel internals.

    ``kernels`` is a length-T iterable of ``RiemannMaternKernel`` whose
    ``eigval``/``eigvec``/``knn``/edges are the same tensor objects (only the
    lengthscale parameter differs per task).
    """
    def __init__(self, kernels):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)

    def _features(self, x):
        # x: (T, n, d) — same points per task is fine — or (n, d)
        if x.dim() == 3:
            return torch.stack([k.features(x[i]) for i, k in enumerate(self.kernels)], 0)
        return torch.stack([k.features(x) for k in self.kernels], 0)        # (T, n, M)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise NotImplementedError("PerTaskRiemannWrapper: last_dim_is_batch unsupported")
        z1 = self._features(x1)
        z2 = z1 if torch.equal(x1, x2) else self._features(x2)
        if diag:
            return (z1 * z2).sum(-1)                                        # (T, n)
        if torch.equal(x1, x2):
            if z1.size(-1) < z1.size(-2):
                return LowRankRootLinearOperator(z1)
            return RootLinearOperator(z1)
        return MatmulLinearOperator(z1, z2.transpose(-1, -2))


class LatentRiemannGP(ApproximateGP):
    """
    Gaussian Process model for the latent space, anchored to a Riemann Manifold graph.

    ``manifold_kernel`` may be a single ``RiemannMaternKernel`` (one lengthscale
    shared across all tasks, via ``BatchedRiemannWrapper``) or a list/ModuleList
    of T kernels (one per task, via ``PerTaskRiemannWrapper``) for per-task
    lengthscales.
    """
    def __init__(self, inducing_points, num_tasks, manifold_kernel,
                 product_ard_matern=False, product_ard_nu=2.5,
                 learn_inducing_locations=False):
        # Ensure inducing points are correctly batched for multi-task
        if inducing_points.dim() == 2:
            inducing_points = inducing_points.unsqueeze(0).repeat(num_tasks, 1, 1)

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        # NOTE: learn_inducing_locations defaults to False — the points stay
        # anchored to the exact FAISS graph nodes, so every K_uu evaluation
        # uses the Riemann kernel's exact-eigenvector branch. Setting it True
        # lets the optimizer move inducing points off-graph; they then go
        # through the kernel's Nyström out-of-sample path (less stable for
        # small graphbandwidth — see RiemannKernel.features). Opt in only
        # deliberately (via --learn-inducing).
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # Each latent task gets its own mean and variance scale
        #self.mean_module = ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.mean_module = LinearMean(input_size=3, batch_shape=torch.Size([num_tasks]))
        
        # Wrap the RiemannMaternKernel(s) so each task gets its own amplitude
        # (outputscale via ScaleKernel). A list/ModuleList of kernels additionally
        # gives each task its own lengthscale (PerTaskRiemannWrapper); a single
        # kernel shares one lengthscale across tasks (BatchedRiemannWrapper).
        if isinstance(manifold_kernel, (list, tuple, nn.ModuleList)):
            assert len(manifold_kernel) == num_tasks, (
                f"per-task kernels ({len(manifold_kernel)}) != num_tasks ({num_tasks})")
            base = PerTaskRiemannWrapper(manifold_kernel)
        else:
            base = BatchedRiemannWrapper(manifold_kernel)

        # Optional Route-B expansion: multiply the (isotropic, geodesic) Riemann
        # kernel by an ARD Euclidean Matern so the model regains the per-axis
        # ambient anisotropy the Riemann kernel structurally lacks, while the
        # geodesic factor still suppresses covariance across manifold gaps:
        #   k(a,b) = k_geo(a,b) * k_eucl-ARD(a,b).
        # The Matern carries per-task (batched) ARD lengthscales, so each task
        # gets its own 3 per-axis scales.
        if product_ard_matern:
            euclid_ard = MaternKernel(
                nu=product_ard_nu, ard_num_dims=3,
                batch_shape=torch.Size([num_tasks]),
            )
            base = ProductKernel(base, euclid_ard)

        self.covar_module = ScaleKernel(base, batch_shape=torch.Size([num_tasks]))

        # Learned inducing points need a DIFFERENTIABLE feature map: the default
        # exact-eigenvector lookup gathers by node index (zero gradient w.r.t.
        # coordinates), so the inducing-point parameters would never move. Flip
        # the Riemann kernel(s) into differentiable-inducing mode so K_uu
        # gradients reach them. Duck-typed (hasattr) to cover both the batched
        # and per-task wrappers without importing the kernel class here.
        if learn_inducing_locations:
            for m in self.modules():
                if hasattr(m, "differentiable_inducing"):
                    m.differentiable_inducing = True

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ManifoldLGP(nn.Module):
    """
    End-to-End architecture:
    Manifold Graph Coordinates -> Latent Riemann GP -> MLP -> 172 Channels
    """
    def __init__(self, p, d, n_neurons, dropout, activation, device, gp_model, use_rsample=True,
                 hyperpriors=True, lengthscale_prior_mean=None):
        super().__init__()
        self.mode = "manifold_lgp"
        self.use_rsample = use_rsample
        self.p = p  # number of channels (e.g., 172)
        self.d = d  # latent dimension (e.g., 10)

        self.log_var_n = nn.Parameter(torch.zeros(p))

        # The LatentRiemannGP instance
        self.gp_model = gp_model

        # MLP-DECODER
        self.decoder_layers = self.build_decoder(n_neurons[::-1], dropout[::-1], activation, d)
        self.output_layer = nn.Linear(n_neurons[0], p)

        self.float_type = torch.float32
        self.device = device
        self.to(device)

        # Hyperparameter priors (MAP-II). Stored as a plain list — NOT module
        # submodules — so state_dict / checkpoints are unchanged. See the
        # helper docstrings above for why VariationalELBO can't be used here.
        self.hyperpriors = hyperpriors
        if hyperpriors:
            terms = _build_kernel_hyperpriors(self.gp_model, lengthscale_prior_mean, device)
            terms.append((_make_noise_prior(device), lambda: self.log_var_n.exp()))
            self._hyperprior_terms = terms
        else:
            self._hyperprior_terms = []

    def build_decoder(self, n_neurons, dropout, activation, input_dim):
        layers = []
        for i in range(len(n_neurons)):
            layers.append(nn.Linear(input_dim, n_neurons[i]))
            # SiLU often performs better than ReLU for continuous spatial decoding
            layers.append(nn.SiLU() if activation == 'silu' else nn.ReLU())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_dim = n_neurons[i]
        return nn.Sequential(*layers)

    def decode(self, z):
        h = self.decoder_layers(z)
        x_mu = self.output_layer(h)
        return x_mu

    def forward(self, coords):
        gp_posterior = self.gp_model(coords)
        
        # REPARAMETERIZATION TRICK
        # rsample() keeps gradients attached during training. 
        # mean is used for deterministic output during evaluation.
        if self.training and self.use_rsample:
            latent_forward = gp_posterior.rsample() 
        else:
            latent_forward = gp_posterior.mean    
        x_reconstructed = self.decode(latent_forward)
        return x_reconstructed, gp_posterior

    def predict(self, coords):
        self.eval()
        with torch.no_grad():
            gp_posterior = self.gp_model(coords)
            x_reconstructed = self.decode(gp_posterior.mean)
            return x_reconstructed, gp_posterior

    def loss_function(self, x, x_reconstructed, beta=1.0):
        recon_loss = self.nll_loss(x, x_reconstructed, self.log_var_n)
        kl_gp = self.gp_model.variational_strategy.kl_divergence().sum()
        total_loss = recon_loss + beta * kl_gp
        # MAP-II: subtract log p(θ). Added once per batch at full scale, exactly
        # like the (full) KL term above, so the prior is weighted consistently
        # with the KL in this minibatch objective.
        if self.hyperpriors and self._hyperprior_terms:
            total_loss = total_loss - _eval_hyperprior_logprob(
                self._hyperprior_terms, self.log_var_n)
        return total_loss, recon_loss, kl_gp

    def nll_loss(self, x, x_reconstructed, log_var_x):
        return 0.5 * torch.sum((x - x_reconstructed).pow(2) / torch.exp(log_var_x) + log_var_x)

    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch, print_every=1000):
        # This remains entirely identical to your existing LGP.train_model logic.
        self.to(self.device)
        self.train()

        for epoch in range(current_epoch, epochs):
            mean_loss, reconstr_loss, kl_loss, mse_loss = 0, 0, 0, 0
            
            for i, data in enumerate(tqdm(dataloader)):
                # Adjust unpacking based on your dataloader
                if len(data) == 2:
                    x, coord = data
                else:
                    x, dummies, coord, sections, pixel_coord = data
                    
                x = x.to(self.device)
                coord = coord.to(self.device)
                
                optimizer.zero_grad()
                x_reconstructed, gp_posterior = self(coord)

                loss, recon_loss, kl_div = self.loss_function(x, x_reconstructed, beta=1.0)
                loss.backward()
                optimizer.step()
                
                mean_loss += loss.item()
                reconstr_loss += recon_loss.item()
                kl_loss += kl_div.item()
                
                mse_loss_batch = F.mse_loss(x_reconstructed.detach(), x.detach()).item()
                mse_loss += mse_loss_batch
                
                if i % 10 == 0:
                    wandb.log({"loss_batch": loss.item(), "mse_loss_batch": mse_loss_batch})

            torch.save(self.state_dict(), exp_path / f"checkpoints/model_{epoch}.pth")
            wandb.log({
                "loss": mean_loss / len(dataloader),
                "reconstruction_loss": reconstr_loss / len(dataloader),
                "kl_loss": kl_loss / len(dataloader),
                "mse_loss": mse_loss / len(dataloader)
            })
            print(f"Epoch {epoch} loss: {mean_loss / len(dataloader):.4f} | MSE: {mse_loss / len(dataloader):.4f}")
            
        torch.save(self.state_dict(), exp_path / "model.pth")

class SpectralLatentGP(gpytorch.models.GP):
    """
    A Variational GP that uses the Manifold Spectrum (Eigenvectors) as basis functions.
    Instead of spatial inducing points, it learns variational weights for the harmonics.
    """
    def __init__(self, num_tasks, manifold_kernel):
        super().__init__()
        self.num_tasks = num_tasks
        self.kernel = manifold_kernel # RiemannMaternKernel
        
        # Ensure the manifold is computed (U and Lambda)
        if not hasattr(self.kernel, 'eigvec'):
            self.kernel.eval()
            
        self.num_modes = self.kernel.num_modes
        N = self.kernel.eigvec.shape[0]

        # Variational Parameters: q(w) ~ N(mu, S) 
        # mu: (num_tasks, num_modes) -> The 'Spectral Weights' for each latent dim
        self.register_parameter("q_mu", nn.Parameter(torch.randn(num_tasks, self.num_modes) * 0.01))
        
        # S: Cholesky of covariance (num_tasks, num_modes, num_modes)
        self.register_parameter("q_log_diag_S", nn.Parameter(torch.zeros(num_tasks, self.num_modes)))

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))

    def kl_divergence(self):
        """
        Computes KL(q(w) || p(w)) where p(w) ~ N(0, Phi)
        Phi is the Matérn spectral density (the 'prior power' of each mode).
        """
        Phi = self.kernel.spectral_density() # (m,)
        mu = self.q_mu # (num_tasks, m)
        S_diag = self.q_log_diag_S.exp() # (num_tasks, m)
        
        # KL for diagonal Gaussians in spectral space:
        # 0.5 * sum [ S/Phi + mu^2/Phi - 1 + log(Phi/S) ]
        term1 = S_diag / Phi
        term2 = mu.pow(2) / Phi
        term3 = -1.0
        term4 = torch.log(Phi) - self.q_log_diag_S
        
        kl = 0.5 * torch.sum(term1 + term2 + term3 + term4)
        return kl

    def forward(self, x):
        # 1. Get interpolated eigenvectors: (Batch, m)
        U_star = self.kernel.raw_eigenvectors(x)
        
        # 2. Compute Mean: (Batch, m) @ (m, tasks) -> (Batch, tasks)
        # ConstantMean(batch_shape=[tasks]).constant is (tasks, 1); flatten to
        # (tasks,) so it broadcasts over the Batch dim of mean_z (Batch, tasks).
        mean_z = torch.matmul(U_star, self.q_mu.T) + self.mean_module.constant.view(-1)
        
        # 3. Compute Variance: (Batch, m) @ (m, tasks) -> (Batch, tasks)
        var_z = torch.matmul(U_star.pow(2), self.q_log_diag_S.exp().T)
        var_z = var_z + 1e-4 # Jitter
        
        # Now returns (Batch, tasks)
        return gpytorch.distributions.MultivariateNormal(mean_z, DiagLinearOperator(var_z))

class SpectralManifoldLGP(nn.Module):
    """
    End-to-End architecture:
    Manifold Graph Coordinates -> Latent Riemann GP -> MLP -> 172 Channels
    """
    def __init__(self, p, d, n_neurons, dropout, activation, device, gp_model,
                 hyperpriors=True, lengthscale_prior_mean=None):
        super().__init__()
        self.mode = "spectral_manifold_lgp"
        self.p = p  # number of channels (e.g., 172)
        self.d = d  # latent dimension (e.g., 10)

        self.log_var_n = nn.Parameter(torch.zeros(p))

        # The LatentRiemannGP instance
        self.gp_model = gp_model

        # MLP-DECODER
        self.decoder_layers = self.build_decoder(n_neurons[::-1], dropout[::-1], activation, d)
        self.output_layer = nn.Linear(n_neurons[0], p)

        self.float_type = torch.float32
        self.device = device
        self.to(device)

        # Hyperparameter priors (MAP-II), stored outside the module graph.
        # For a SpectralLatentGP gp_model there is no ScaleKernel/outputscale, so
        # _build_kernel_hyperpriors returns [] and only the noise prior applies.
        self.hyperpriors = hyperpriors
        if hyperpriors:
            terms = _build_kernel_hyperpriors(self.gp_model, lengthscale_prior_mean, device)
            terms.append((_make_noise_prior(device), lambda: self.log_var_n.exp()))
            self._hyperprior_terms = terms
        else:
            self._hyperprior_terms = []

    def build_decoder(self, n_neurons, dropout, activation, input_dim):
        layers = []
        for i in range(len(n_neurons)):
            layers.append(nn.Linear(input_dim, n_neurons[i]))
            # SiLU often performs better than ReLU for continuous spatial decoding
            layers.append(nn.SiLU() if activation == 'silu' else nn.ReLU())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_dim = n_neurons[i]
        return nn.Sequential(*layers)

    def decode(self, z):
        h = self.decoder_layers(z)
        x_mu = self.output_layer(h)
        return x_mu

    def forward(self, coords):
        gp_posterior = self.gp_model(coords)
        # Samples/Mean are now already (Batch, latent_dim)
        latent_samples = gp_posterior.rsample() if self.training else gp_posterior.mean
        return self.decode(latent_samples), gp_posterior

    def predict(self, coords):
        self.eval()
        with torch.no_grad():
            gp_posterior = self.gp_model(coords)
            # Already (Batch, latent_dim), no transpose needed
            return self.decode(gp_posterior.mean), gp_posterior
    
    def nll_loss(self, x, x_reconstructed, log_var_x):
        return 0.5 * torch.sum((x - x_reconstructed).pow(2) / torch.exp(log_var_x) + log_var_x)

    def loss_function(self, x, x_reconstructed, beta=1.0):
        recon_loss = self.nll_loss(x, x_reconstructed, self.log_var_n)

        # Check if using Spectral GP or Approximate GP
        if hasattr(self.gp_model, 'kl_divergence'):
            kl_gp = self.gp_model.kl_divergence()
        else:
            kl_gp = self.gp_model.variational_strategy.kl_divergence().sum()

        total_loss = recon_loss + beta * kl_gp
        # MAP-II: subtract log p(θ), weighted like the (full) KL term.
        if self.hyperpriors and self._hyperprior_terms:
            total_loss = total_loss - _eval_hyperprior_logprob(
                self._hyperprior_terms, self.log_var_n)
        return total_loss, recon_loss, kl_gp
    
    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch, print_every=1000):
        # This remains entirely identical to your existing LGP.train_model logic.
        self.to(self.device)
        self.train()

        for epoch in range(current_epoch, epochs):
            mean_loss, reconstr_loss, kl_loss, mse_loss = 0, 0, 0, 0
            
            for i, data in enumerate(tqdm(dataloader)):
                # Adjust unpacking based on your dataloader
                if len(data) == 2:
                    x, coord = data
                else:
                    x, dummies, coord, sections, pixel_coord = data
                    
                x = x.to(self.device)
                coord = coord.to(self.device)
                
                optimizer.zero_grad()
                x_reconstructed, gp_posterior = self(coord)

                loss, recon_loss, kl_div = self.loss_function(x, x_reconstructed, beta=1.0)
                loss.backward()
                optimizer.step()
                
                mean_loss += loss.item()
                reconstr_loss += recon_loss.item()
                kl_loss += kl_div.item()
                
                mse_loss_batch = F.mse_loss(x_reconstructed.detach(), x.detach()).item()
                mse_loss += mse_loss_batch
                
                if i % 10 == 0:
                    wandb.log({"loss_batch": loss.item(), "mse_loss_batch": mse_loss_batch})

            torch.save(self.state_dict(), exp_path / f"checkpoints/model_{epoch}.pth")
            wandb.log({
                "loss": mean_loss / len(dataloader),
                "reconstruction_loss": reconstr_loss / len(dataloader),
                "kl_loss": kl_loss / len(dataloader),
                "mse_loss": mse_loss / len(dataloader)
            })
            print(f"Epoch {epoch} loss: {mean_loss / len(dataloader):.4f} | MSE: {mse_loss / len(dataloader):.4f}")
            
        torch.save(self.state_dict(), exp_path / "model.pth")
