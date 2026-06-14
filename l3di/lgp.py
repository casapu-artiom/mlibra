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


def symmetrize_inputs(x):
    """
    Symmetrizes the input tensor in the coronal plane (x, y, z) -> (x, y, |z|).
    """
    x_sym = x.clone()
    x_sym[..., 2] = x_sym[..., 2].abs()
    return x_sym

class Custom3DKernel(gpytorch.kernels.Kernel):
    """ Custom 3D kernel that symmetrizes the inputs in the coronal plane in the brain and uses a Matern kernel."""
    def __init__(self, nu=2.5, batch_shape=torch.Size(), minimal_length_scale=0.1, **kwargs):
        """
        Args:
            nu (float): The smoothness parameter of the Matern kernel.
            batch_shape (torch.Size): The batch shape of the kernel in case of multitask GP.
            minimal_length_scale (float): Minimal length scale for the kernel. Constrained in case of slices from the brain.
        """
        super().__init__(batch_shape=batch_shape, **kwargs)

        lengthscale_constraint = GreaterThan([minimal_length_scale,0,0])
        self.nu= nu
        self.base_kernel = MaternKernel(
            nu=self.nu,
            ard_num_dims=3,
            batch_shape=batch_shape,
            lengthscale_constraint=lengthscale_constraint
        )

    def forward(self, x1, x2, **params):
        x1_sym = symmetrize_inputs(x1)
        x2_sym = symmetrize_inputs(x2)
        return self.base_kernel(x1, x2, **params)


class IndependentMultitaskGPModel(ApproximateGP):
    """
    Gaussian Process model for the latent space.

    We define a GP prior for the latent space. The GP prior is defined by a mean and a covariance function.
    """

    def __init__(self, inducing_points, num_tasks, kernel_type="rbf", nu=1.5, minimal_length_scale=1, input_dim=174):
        """
        Construct the GPModel class.

        Args:
            inducing_points (torch.Tensor): Inducing points for the GP
            num_tasks (int): Number of tasks
        """
        # Let's use a different set of inducing points for each task
        # for each num_task we have a set of inducing points
        if inducing_points.dim() == 2:
            inducing_points = inducing_points.unsqueeze(0).repeat(num_tasks, 1, 1)
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = LinearMean(input_size=3, batch_shape=torch.Size([num_tasks]))
        self.kernel_type = kernel_type
        self.nu = nu
        if kernel_type == "rbf":
            print("Using RBF kernel")
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=torch.Size([num_tasks]),
                          ard_num_dims=3),
                batch_shape=torch.Size([num_tasks])
            )
        else:
            if kernel_type == "matern":
                print(f"Using Matern kernel with nu={self.nu}")
                self.covar_module = ScaleKernel(
                    MaternKernel(batch_shape=torch.Size([num_tasks]),
                                 nu=self.nu,
                                 ard_num_dims=3,
                                 lengthscale_constraint=gpytorch.constraints.GreaterThan(torch.tensor([minimal_length_scale,0,0], dtype=torch.float32))),
                    batch_shape=torch.Size([num_tasks])
                )
            elif kernel_type == "symmetric":
                print("Using symmetric kernel")
                self.covar_module = ScaleKernel(
                    Custom3DKernel(batch_shape=torch.Size([num_tasks]),
                                   nu=self.nu,
                                   minimal_length_scale=minimal_length_scale),
                    batch_shape=torch.Size([num_tasks])
                )

    def forward(self, x):
        """
        Forward pass of the GP model.

        Get the mean and covariance of the GP posterior.
        """
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        # x has shape (batch, num_tasks, 3)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class LGP(nn.Module):
    """
    Latent Gaussian Process model with a conditional likelihood on the latent space.
    """
    def __init__(self, p, d, n_neurons, dropout, activation,  device, gp_model, use_rsample=True):
        super(LGP, self).__init__()
        self.mode = "lgp"
        self.use_rsample = use_rsample
        self.p = p  # number of channels
        self.d = d  # latent dimension

        self.log_var_n = nn.Parameter(torch.zeros(p))

        # GP
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
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Tanh())
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
        if self.training and self.use_rsample:
            latent_forward = gp_posterior.rsample()
        else:
            latent_forward = gp_posterior.mean
        x_reconstructed = self.decode(latent_forward)
        return x_reconstructed, gp_posterior

    def encode(self, coords):
        """
        Encode the coordinates using the GP model.
        This method is not used in the current implementation but can be useful for future extensions.
        """
        gp_posterior = self.gp_model(coords)
        return gp_posterior.mean, gp_posterior.variance

    def reparametrize(self, mu, logvar):
        """
        We only return the mean of the GP posterior.
        """
        return mu

    def predict(self, coords):
        with torch.no_grad():
            gp_posterior = self.gp_model(coords)
            x_reconstructed = self.decode(gp_posterior.mean)
            return x_reconstructed, gp_posterior

    def loss_function(self, x, x_reconstructed, beta=1.0):
        recon_loss = self.nll_loss(x, x_reconstructed, self.log_var_n)
        kl_gp = self.gp_model.variational_strategy.kl_divergence().sum()
        kl_loss = kl_gp
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def nll_loss(self, x, x_reconstructed, log_var_x):
        nll_loss = 0.5 * torch.sum((x - x_reconstructed).pow(2) / torch.exp(log_var_x) + log_var_x)
        return nll_loss

    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch, print_every=1000):
        self.to(self.device)
        self.train()

        for epoch in range(current_epoch, epochs):
            mean_loss = 0
            reconstr_loss = 0
            kl_loss = 0
            mse_loss = 0
            for i, data in enumerate(tqdm(dataloader)):
                x,  coord = data
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
                    wandb.log({"loss_batch": loss.item()})
                    wandb.log({"mse_loss_batch": mse_loss_batch})

            torch.save(self.state_dict(), exp_path / f"checkpoints/model_{epoch}.pth")
            wandb.log({"loss": mean_loss / len(dataloader)})
            wandb.log({"reconstruction_loss": reconstr_loss / len(dataloader)})
            wandb.log({"kl_loss": kl_loss / len(dataloader)})
            wandb.log({"mse_loss": mse_loss / len(dataloader)})
            print(f"Epoch {epoch} loss: {mean_loss / len(dataloader)}")
            print(f"Epoch {epoch} reconstruction loss: {reconstr_loss / len(dataloader)}")
            print(f"Epoch {epoch} mse loss: {mse_loss / len(dataloader)}")
        torch.save(self.state_dict(), exp_path / "model.pth")


class GPVAE(nn.Module):
    """
    Gaussian Process Variational Autoencoder.
    This model uses an MLP for both encoder and decoder, and samples from the GP posterior for reconstruction.
    """

    def __init__(self, p, d, n_neurons, dropout, activation, sigma_n, device, gp_model):
        super(GPVAE, self).__init__()
        self.p = p
        self.d = d

        # MLP-ENCODER
        self.encoder_layers = self.build_encoder(n_neurons, dropout, activation, p)
        self.mean_layer = nn.Linear(n_neurons[-1], d)
        self.log_var_param = nn.Parameter(torch.zeros(d))
        self.log_var_layer = lambda x: self.log_var_param.expand(x.size(0), -1)

        # GP
        self.gp_model = gp_model

        # SAMPLE and MLP-DECODER
        self.decoder_layers = self.build_decoder(n_neurons[::-1], dropout[::-1], activation, d)
        self.output_layer = nn.Linear(n_neurons[0], p)
        self.log_var_n_layer = nn.Linear(n_neurons[0], p)
        self.log_var_n = None

        self.float_type = torch.float32
        self.device = device
        self.to(device)

    def build_encoder(self, n_neurons, dropout, activation, input_dim):
        layers = []
        for i in range(len(n_neurons)):
            layers.append(nn.Linear(input_dim, n_neurons[i]))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_dim = n_neurons[i]
        return nn.Sequential(*layers)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def build_decoder(self, n_neurons, dropout, activation, input_dim):
        layers = []
        for i in range(len(n_neurons)):
            layers.append(nn.Linear(input_dim, n_neurons[i]))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_dim = n_neurons[i]
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder_layers(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        z = self.reparametrize(mean, log_var)
        return mean, log_var, z

    def decode(self, z):
        h = self.decoder_layers(z)
        x_mu = self.output_layer(h)
        self.log_var_n = self.log_var_n_layer(h)
        return x_mu, x_mu

    def forward(self, x, coords):
        mean, log_var, z = self.encode(x)
        gp_posterior = self.gp_model(coords)
        latent_forward = z
        x_reconstructed, x_mu = self.decode(latent_forward)
        return x_reconstructed, mean, log_var, x_mu, gp_posterior

    def predict(self, x, coords):
        with torch.no_grad():
            mean, log_var, _ = self.encode(x)
            gp_sample = self.sample_from_gp(coords)
            x_reconstructed, x_mu = self.decode(gp_sample)
            return x_reconstructed, mean, log_var, x_mu, gp_posterior

    def sample_from_gp(self, coords):
        with torch.no_grad():
            gp_posterior = self.gp_model(coords)
            gp_sample = gp_posterior.sample()
            return gp_sample

    def _compute_kl_z(self, mu_z, std_z, p_z_mean, p_z_stddev):
        kl_per_dim = []
        for i in range(self.d):
            p_z_i = torch.distributions.Normal(p_z_mean[:, i], p_z_stddev[:, i])
            q_z_i = torch.distributions.Normal(mu_z[:, i], std_z[:, i])
            kl_per_dim.append(torch.distributions.kl_divergence(q_z_i, p_z_i).sum())
        return torch.stack(kl_per_dim).sum()

    def loss_function(self, x, x_reconstructed, mean, log_var, coords, p_z_mean, p_z_stddev, beta=1.0):
        recon_loss = self.nll_loss(x, x_reconstructed, self.log_var_n)
        kl_z = self._compute_kl_z(mean, torch.exp(0.5 * log_var), p_z_mean, p_z_stddev)
        kl_gp = self.gp_model.variational_strategy.kl_divergence().sum()
        kl_loss = kl_z + kl_gp
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def nll_loss(self, x, x_reconstructed, log_var_x):
        nll_loss = 0.5 * torch.sum((x - x_reconstructed).pow(2) / torch.exp(log_var_x) + log_var_x)
        return nll_loss

    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch, print_every=1000):
        self.to(self.device)
        self.train()
        wandb.log({"minimal_length_scale": self.gp_model.minimal_length_scale})

        for epoch in range(current_epoch, epochs):
            mean_loss = 0
            reconstr_loss = 0
            kl_loss = 0
            mse_loss = 0
            for i, data in enumerate(tqdm(dataloader)):
                x, dummies, coord, sections, pixel_coord = data
                x = x.to(self.device)
                coord = coord.to(self.device)
                optimizer.zero_grad()
                x_reconstructed, gp_posterior = self(x, coord)

                loss, recon_loss, kl_div = self.loss_function(x, x_reconstructed, gp_posterior.mean, gp_posterior.log_var, coord, gp_posterior.mean, gp_posterior.stddev, beta=1.0)
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
                reconstr_loss += recon_loss.item()
                kl_loss += kl_div.item()
                mse_loss_batch = F.mse_loss(x_reconstructed.detach(), x).item()
                mse_loss += mse_loss_batch
                if i % 10 == 0:
                    wandb.log({"loss_batch": loss.item()})
                    wandb.log({"mse_loss_batch": mse_loss_batch})

            torch.save(self.state_dict(), exp_path / f"checkpoints/model_{epoch}.pth")
            wandb.log({"loss": mean_loss / len(dataloader)})
            wandb.log({"reconstruction_loss": reconstr_loss / len(dataloader)})
            wandb.log({"kl_loss": kl_loss / len(dataloader)})
            wandb.log({"mse_loss": mse_loss / len(dataloader)})
            print(f"Epoch {epoch} loss: {mean_loss / len(dataloader)}")
            print(f"Epoch {epoch} reconstruction loss: {reconstr_loss / len(dataloader)}")
            print(f"Epoch {epoch} mse loss: {mse_loss / len(dataloader)}")
        torch.save(self.state_dict(), exp_path / "model.pth")
