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
from linear_operator.operators import DiagLinearOperator

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

class LatentRiemannGP(ApproximateGP):
    """
    Gaussian Process model for the latent space, anchored to a Riemann Manifold graph.
    """
    def __init__(self, inducing_points, num_tasks, manifold_kernel):
        # Ensure inducing points are correctly batched for multi-task
        if inducing_points.dim() == 2:
            inducing_points = inducing_points.unsqueeze(0).repeat(num_tasks, 1, 1)
            
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        # CRITICAL: learn_inducing_locations MUST be False.
        # The points must stay anchored to the specific nodes on the FAISS graph.
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, 
                learn_inducing_locations=True  
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # Each latent task gets its own mean and variance scale
        #self.mean_module = ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.mean_module = LinearMean(input_size=3, batch_shape=torch.Size([num_tasks]))
        
        # Wrap the initialized RiemannMaternKernel so each latent dimension 
        # can learn its own amplitude (outputscale)
        self.covar_module = ScaleKernel(
            BatchedRiemannWrapper(manifold_kernel),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ManifoldLGP(nn.Module):
    """
    End-to-End architecture:
    Manifold Graph Coordinates -> Latent Riemann GP -> MLP -> 172 Channels
    """
    def __init__(self, p, d, n_neurons, dropout, activation, device, gp_model, use_rsample=True):
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
        # Broadcasting the constant (tasks,) works automatically here
        mean_z = torch.matmul(U_star, self.q_mu.T) + self.mean_module.constant
        
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
    def __init__(self, p, d, n_neurons, dropout, activation, device, gp_model):
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
