import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb as wandb

def extract(a, t, x_shape):
    """
    Extract values from a 1D tensor (array) based on specified indices, and then reshape the result to match a target shape.

    1. It takes three parameters:
       - `a`: A tensor from which values will be extracted
       - `t`: Tensor of indices to extract from `a`
       - `x_shape`: Target shape for the output

    2. It first gets the batch size `b` from the shape of `t`

    3. It uses PyTorch's `gather` method to extract values from the last dimension of `a` based on indices in `t`

    4. It reshapes the result to have shape `[b, 1, 1, ...]` where the number of 1's matches the dimensionality of `x_shape` minus 1

    This function is primarily used in the diffusion model to extract specific values from schedules (like beta or alpha values) at particular timesteps, and then broadcast them to match the shape of input data.
    This allows for batch processing where each item in the batch might be at a different timestep in the diffusion process.

    """
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class LGPVoxelUNet(nn.Module):
    """
    A simplified UNet architecture for noise prediction in LGP Voxel Diffusion with 1D data.
    Adapted for the l3di framework.
    """
    
    def __init__(self, in_channels=1, model_channels=64, out_channels=1, num_res_blocks=2, z_dim=64):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsample blocks
        self.down1 = nn.Sequential(
            nn.Conv1d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, model_channels*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(model_channels*2),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv1d(model_channels*2, model_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_channels*2),
            nn.SiLU(),
            nn.Conv1d(model_channels*2, model_channels*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(model_channels*4),
            nn.SiLU()
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv1d(model_channels*4, model_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_channels*4),
            nn.SiLU(),
            nn.Conv1d(model_channels*4, model_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_channels*4),
            nn.SiLU()
        )
        
        # Upsample blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(model_channels*4, model_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_channels*2),
            nn.SiLU(),
            nn.Conv1d(model_channels*2, model_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_channels*2),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(model_channels*2, model_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_channels),
            nn.SiLU()
        )
        
        # Output convolution
        self.conv_out = nn.Conv1d(model_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
        
        # Z embedding
        self.z_dim = z_dim
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
    
    def forward(self, x, t, low_res, z):
        # Assert inputs are available
        assert low_res is not None, "Conditioning (low_res) must be provided"
        assert z is not None, "Z vector must be provided"
        
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.time_mlp(t)
        
        # Z embedding
        z_emb = self.z_proj(z)
        # Combine time and z embeddings
        emb = t_emb + z_emb
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Add combined embedding
        h = h + emb.view(-1, h.shape[1], 1)
        
        # Downsample
        h1 = self.down1(h)
        h2 = self.down2(h1)
        
        # Middle
        h = self.middle(h2)
        
        # Upsample
        h = self.up1(h)
        h = self.up2(h)
        
        # Output
        return self.conv_out(h)


class LGPVoxelDiffusion(nn.Module):
    """
    LGP Voxel Diffusion Probabilistic Model.
    
    A DDPM implementation adapted for the l3di framework with training and inference loops.
    Always uses conditional generation with low-res conditioning and z-vector.
    
    Key differences from standard DDPM:
    1. Conditional generation: Always uses both low-res conditioning and z-vector
    2. Modified forward process: x_t = x_start * sqrt_alpha_bar + low_res + eps * minus_sqrt_alpha_bar
    3. Modified posterior calculation: Includes conditioning in the mean calculation
    4. Classifier-free guidance: Can interpolate between conditional and unconditional outputs
    5. Final sample processing: Removes conditioning bias in the final step
    """
    
    def __init__(
        self,
        decoder,
        beta_1=1e-4,
        beta_2=0.02,
        T=1000,
        var_type="fixedlarge",
    ):
        super().__init__()
        
        # The UNet model that predicts noise
        self.decoder = decoder
        
        # Number of diffusion steps and beta values
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.var_type = var_type
        
        # Define beta schedule
        self.register_buffer(
            "betas", torch.linspace(self.beta_1, self.beta_2, steps=self.T).double()
        )
        
        # Main constants
        dev = self.betas.device
        alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_shifted = torch.cat([torch.tensor([1.0], device=dev), alpha_bar[:-1]])
        
        # Pre-calculate diffusion parameters
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_shifted", alpha_bar_shifted)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("minus_sqrt_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_bar - 1)
        )
        
        # Posterior q(x_t-1|x_t,x_0,t) covariance of the forward process
        self.register_buffer(
            "post_variance", self.betas * (1.0 - alpha_bar_shifted) / (1.0 - alpha_bar)
        )
        # Clipping because post_variance is 0 before the chain starts
        self.register_buffer(
            "post_log_variance_clipped",
            torch.log(
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=dev),
                        self.post_variance[1:],
                    ]
                )
            ),
        )

        # q(x_t-1 | x_t, x_0) mean coefficients
        self.register_buffer(
            "post_coeff_1",
            self.betas * torch.sqrt(alpha_bar_shifted) / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "post_coeff_2",
            torch.sqrt(alphas) * (1 - alpha_bar_shifted) / (1 - alpha_bar),
        )
        self.register_buffer(
            "post_coeff_3",
            1 - self.post_coeff_2,
        )
    
    def _predict_xstart_from_eps(self, x_t, t, eps, cond):
        """
        Predict x_0 from noise eps, incorporating conditioning
        
        Modified from standard DDPM to include conditioning:
        x_0 = sqrt_recip_alphas_cumprod * x_t - sqrt_recip_alphas_cumprod * cond - sqrt_recipm1_alphas_cumprod * eps
        """

        assert x_t.shape == eps.shape
        assert cond is not None, "Conditioning must be provided"
        assert cond.shape == x_t.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * cond
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def get_posterior_mean_covariance(
        self, x_t, t, clip_denoised=False, cond=None, z_vae=None, guidance_weight=0.0
    ):
        """
        Compute posterior mean and covariance for the backward diffusion process
        
        Modified from standard DDPM to include conditioning in posterior mean:
        post_mean = post_coeff_1 * x_recons + post_coeff_2 * x_t + post_coeff_3 * cond
        """
        B = x_t.size(0)
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        assert t_.shape == torch.Size([B,])
        
        # Assert conditioning and z_vae are provided
        assert cond is not None, "Conditioning must be provided"
        assert z_vae is not None, "Z vector must be provided"
        
        # Compute updated score with classifier-free guidance if needed
        if guidance_weight == 0:
            eps_score = self.decoder(x_t, t_, low_res=cond, z=z_vae)
        else:
            # Classifier-free guidance: interpolate between conditional and unconditional outputs
            eps_score = (1 + guidance_weight) * self.decoder(
                x_t, t_, low_res=cond, z=z_vae
            ) - guidance_weight * self.decoder(
                x_t,
                t_,
                low_res=torch.zeros_like(cond),
                z=torch.zeros_like(z_vae),
            )
        eps_score = eps_score[:,:, :cond.shape[-1]]  # Ensure eps_score matches cond shape
        # Generate the reconstruction from x_t
        x_recons = self._predict_xstart_from_eps(x_t, t_, eps_score, cond=cond)
        
        # Clip
        if clip_denoised:
            x_recons.clamp_(-1.0, 1.0)
        
        # Compute posterior mean from the reconstruction (includes conditioning)
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x_recons
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
            + extract(self.post_coeff_3, t_, x_t.shape) * cond
        )
        
        # Extract posterior variance
        p_variance, p_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            "fixedlarge": (
                torch.cat(
                    [
                        torch.tensor([self.post_variance[1]], device=x_t.device),
                        self.betas[1:],
                    ]
                ),
                torch.log(
                    torch.cat(
                        [
                            torch.tensor([self.post_variance[1]], device=x_t.device),
                            self.betas[1:],
                        ]
                    )
                ),
            ),
            "fixedsmall": (
                self.post_variance,
                self.post_log_variance_clipped,
            ),
        }[self.var_type]
        post_variance = extract(p_variance, t_, x_t.shape)
        post_log_variance = extract(p_log_variance, t_, x_t.shape)
        return post_mean, post_variance, post_log_variance
    
    def compute_noisy_input(self, x_start, eps, t, low_res):
        """
        Computes noisy input for a given timestep in the forward process
        
        Modified from standard DDPM to include conditioning:
        x_t = x_start * sqrt_alpha_bar + low_res + eps * minus_sqrt_alpha_bar
        
        In standard DDPM this would be:
        x_t = x_start * sqrt_alpha_bar + eps * minus_sqrt_alpha_bar
        """
        assert eps.shape == x_start.shape
        assert low_res is not None, "Conditioning (low_res) must be provided"
        assert low_res.shape == x_start.shape
        # Samples the noisy input x_t ~ N(x_t|x_0) in the forward process
        return (
            x_start * extract(self.sqrt_alpha_bar, t, x_start.shape)
            + low_res
            + eps * extract(self.minus_sqrt_alpha_bar, t, x_start.shape)
        )
    
    @torch.no_grad()
    def sample(
        self,
        x_t,
        cond,
        z_vae,
        n_steps=None,
        guidance_weight=0.0,
        checkpoints=[],
        ddpm_latents=None,
        clip_denoised=False
    ):
        """
        Generate samples using the trained model
        
        Args:
            x_t: Starting noise tensor
            cond: Conditioning signal (low-res)
            z_vae: Z vector from VAE
            n_steps: Number of denoising steps (default: self.T)
            guidance_weight: Weight for classifier-free guidance
            checkpoints: Timesteps to save intermediate results
            ddpm_latents: Pre-defined random noise for each step (for reproducibility)
            clip_denoised: Whether to clip values to [-1, 1] during denoising
            
        Returns:
            Dictionary of generated samples at specified checkpoints
        """
        x = x_t
        B, *_ = x_t.shape
        sample_dict = {}
        
        # Assert conditioning and z_vae are provided
        assert cond is not None, "Conditioning must be provided"
        assert z_vae is not None, "Z vector must be provided"
        assert cond.shape[0] == B, f"Conditioning batch size {cond.shape[0]} doesn't match input batch size {B}"
        assert z_vae.shape[0] == B, f"Z vector batch size {z_vae.shape[0]} doesn't match input batch size {B}"
        
        if ddpm_latents is not None:
            ddpm_latents = ddpm_latents.to(x_t.device)
        
        num_steps = self.T if n_steps is None else n_steps
        checkpoints = [num_steps] if checkpoints == [] else checkpoints
        
        # Loop through timesteps from T down to 1
        for idx, t in enumerate(reversed(range(0, num_steps))):
            z = (
                torch.randn_like(x_t)
                if ddpm_latents is None
                else torch.stack([ddpm_latents[idx]] * B)
            )
            
            # Get posterior mean and variance
            (
                post_mean,
                post_variance,
                post_log_variance,
            ) = self.get_posterior_mean_covariance(
                x,
                t,
                cond=cond,
                z_vae=z_vae,
                guidance_weight=guidance_weight,
                clip_denoised=clip_denoised
            )
            
            # No noise when t == 0
            nonzero_mask = (
                torch.tensor(t != 0, device=x.device)
                .float()
                .view(-1, *([1] * (len(x_t.shape) - 1)))
            )
            
            # Langevin step!
            x = post_mean + nonzero_mask * torch.exp(0.5 * post_log_variance) * z
            
            if t == 0:
                # Add results to checkpoints
                x = (x + 1) / 2 # Rescale to [0, 1]
            if idx + 1 in checkpoints:
                sample_dict[str(idx + 1)] = x

        return sample_dict
    
    def forward(self, x, eps, t, low_res, z):
        """Forward pass of the model for training"""
        # Assert inputs are available
        assert low_res is not None, "Conditioning (low_res) must be provided"
        assert z is not None, "Z vector must be provided"
        
        # Predict noise
        x_t = self.compute_noisy_input(x, eps, t, low_res=low_res)
        return self.decoder(x_t, t, low_res=low_res, z=z)
    
    def train_step(self, x_0, optimizer, cond, z):
        """Single training step to predict noise in a noisy image"""
        # Assert inputs are available
        assert cond is not None, "Conditioning must be provided"
        assert z is not None, "Z vector must be provided"
        
        optimizer.zero_grad()
        
        # Sample random timesteps
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Predict noise
        eps_pred = self(x_0, eps, t, low_res=cond, z=z)
        # pad eps to match eps in the last dimension
        # Calculate loss (mean squared error between actual and predicted noise), we have to take care of the shape
        # as eps_pred may have an even number of channels in the last dimension, if eps has an odd number of channels
        # we need to pad eps_pred to match eps
        loss= F.mse_loss(eps_pred[:, :, :eps.shape[-1]], eps)

        # Backprop
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, epochs=10, lr=2e-4, device="cuda", vae=None):
        """Training loop for the LGP Voxel Diffusion"""
        optimizer = Adam(self.decoder.parameters(), lr=lr)
        self.to(device)
        wandb.init(project="lgp-vdiffusion", config={
            "epochs": epochs,
            "learning_rate": lr
        })

        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move data to device
                if isinstance(batch, list) or isinstance(batch, tuple):
                    x = batch[0].to(device)
                    coord = batch[1].to(device) if len(batch) > 1 else None
                else:
                    x = batch.to(device)
                    coord = None
                if x.dim() < 3:
                    x = x.unsqueeze(1)
                # Generate conditioning
                assert vae is not None, "VAE must be provided for conditioning"
                with torch.no_grad():
                    # mu, logvar = vae.encode(x * 0.5 + 0.5)
                    mu, logvar = vae.encode(coord if coord is not None else x*0.5 + 0.5)
                    z = vae.reparametrize(mu, logvar)
                    cond = vae.decode(z)
                    cond = cond.unsqueeze(1)  # Ensure shape is [B, C, L]
                    #cond = 2 * cond - 1
                
                # Normalize to [-1, 1] if in [0, 1]
                if x.min() >= 0 and x.max() <= 1:
                    x = 2 * x - 1
                    
                # Training step
                loss = self.train_step(x, optimizer, cond=cond, z=z)
                wandb.log({"loss": loss, "epoch": epoch + 1})
                epoch_loss += loss
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss})
            
            avg_loss = epoch_loss / len(dataloader)
            wandb.log({"avg_loss": avg_loss, "epoch": epoch + 1})
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path, device="cuda"):
        """Load model weights"""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
    
    @torch.no_grad()
    def sample_and_save(self, n_samples, seq_length, device, save_path, channels=1, cond=None, z_vae=None, guidance_weight=0.0):
        """Generate samples using the trained model and save them as a plot"""
        # Start with pure noise
        x_t = torch.randn(n_samples, channels, seq_length, device=device)
        
        # Assert conditioning and z_vae are provided
        assert cond is not None, "Conditioning must be provided"
        assert z_vae is not None, "Z vector must be provided"
        assert cond.shape[0] == n_samples, f"Conditioning batch size {cond.shape[0]} doesn't match n_samples {n_samples}"
        assert z_vae.shape[0] == n_samples, f"Z vector batch size {z_vae.shape[0]} doesn't match n_samples {n_samples}"
        
        # Generate samples
        samples_dict = self.sample(
            x_t, 
            cond=cond,
            z_vae=z_vae,
            guidance_weight=guidance_weight
        )
        samples = samples_dict[max(samples_dict.keys(), key=lambda x: int(x))]
        
        # Convert to numpy and rescale to [0, 1]
        #samples = (samples + 1) / 2
        #samples = samples.clamp(0, 1).cpu().squeeze().numpy()


# Example usage:
if __name__ == "__main__":
    # This is just an example and won't run without a proper dataset
    
    # Create a simple UNet model with z dimension
    # Assuming sequence length is 128 and z_dim is 32
    unet = LGPVoxelUNet(in_channels=1, model_channels=64, out_channels=1, z_dim=32)
    
    # Create DDPM model
    ddpm = LGPVoxelDiffusion(unet, T=1000)
    
    # Training would look like:
    # from torch.utils.data import DataLoader
    # import numpy as np
    #
    # # Create a simple sine wave dataset
    # class SineWaveDataset(torch.utils.data.Dataset):
    #     def __init__(self, size=1000, seq_length=128):
    #         self.size = size
    #         self.seq_length = seq_length
    #         self.data = []
    #         for _ in range(size):
    #             freq = np.random.uniform(1, 5)
    #             phase = np.random.uniform(0, 2*np.pi)
    #             x = np.linspace(0, 2*np.pi, seq_length)
    #             y = np.sin(freq * x + phase)
    #             self.data.append(torch.FloatTensor(y).unsqueeze(0))  # Shape: [1, seq_length]
    #     
    #     def __len__(self):
    #         return self.size
    #         
    #     def __getitem__(self, idx):
    #         return self.data[idx]
    #
    # # Create dataset and dataloader
    # dataset = SineWaveDataset(size=1000, seq_length=128)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #
    # # Define a simple 1D VAE for conditioning
    # class SimpleVAE1D(nn.Module):
    #     def __init__(self, seq_length=128, z_dim=32):
    #         super().__init__()
    #         # Encoder
    #         self.encoder = nn.Sequential(
    #             nn.Conv1d(1, 16, 3, padding=1),
    #             nn.ReLU(),
    #             nn.MaxPool1d(2),
    #             nn.Conv1d(16, 32, 3, padding=1),
    #             nn.ReLU(),
    #             nn.MaxPool1d(2),
    #             nn.Flatten(),
    #             nn.Linear(32 * (seq_length // 4), 128),
    #             nn.ReLU(),
    #         )
    #         self.fc_mu = nn.Linear(128, z_dim)
    #         self.fc_var = nn.Linear(128, z_dim)
    #         
    #         # Decoder
    #         self.decoder_input = nn.Linear(z_dim, 32 * (seq_length // 4))
    #         self.decoder = nn.Sequential(
    #             nn.Unflatten(1, (32, seq_length // 4)),
    #             nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
    #             nn.ReLU(),
    #             nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
    #             nn.Sigmoid(),
    #         )
    #         
    #     def encode(self, x):
    #         h = self.encoder(x)
    #         return self.fc_mu(h), self.fc_var(h)
    #         
    #     def reparameterize(self, mu, logvar):
    #         std = torch.exp(0.5 * logvar)
    #         eps = torch.randn_like(std)
    #         return mu + eps * std
    #         
    #     def decode(self, z):
    #         h = self.decoder_input(z)
    #         return self.decoder(h)
    #         
    #     def forward(self, x):
    #         mu, logvar = self.encode(x)
    #         z = self.reparameterize(mu, logvar)
    #         return self.decode(z), mu, logvar
    #
    # # Create VAE
    # vae = SimpleVAE1D(seq_length=128, z_dim=32)
    #
    # # Train
    # ddpm.train(dataloader, epochs=50, lr=2e-4, vae=vae)
    #
    # # Save model
    # ddpm.save_model('ddpm_sine_waves.pth')
    #
    # # Generate samples
    # z_samples = torch.randn(4, 32).to('cuda')  # 4 random z vectors
    # cond_samples = vae.decode(z_samples)  # Generate conditioning from z
    # ddpm.sample_and_save(4, 128, 'cuda', 'sine_samples.png', channels=1, 
    #                       cond=cond_samples, z_vae=z_samples)
