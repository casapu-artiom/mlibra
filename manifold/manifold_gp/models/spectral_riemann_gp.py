import torch
import gpytorch
import math
from gpytorch.mlls import MarginalLogLikelihood

class FastSpectralRiemannGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.train_y = train_y
        self.weights = None

    def compute_spectral_weights(self):
        """
        The 'Trick': Pre-calculate weights in the spectral domain.
        w = [Phi / (Phi + sigma^2)] * (U^T * y)
        """
        # 1. Ensure kernel has computed eigenvectors/values (U and Lambda)
        if not hasattr(self.covar_module, 'eigval'):
            self.covar_module.eval()

        # 2. Get spectral density (Phi) and manifold basis (U)
        # Note: RiemannMaternKernel provides spectral_density()
        phi = self.covar_module.spectral_density() # Shape: (m,)
        U = self.covar_module.eigvec                # Shape: (N, m)
        noise_var = self.likelihood.noise           # sigma^2
        
        # 3. Project labels into spectral space: U^T * y
        # We subtract the mean for a zero-centered GP
        y_centered = self.train_y - self.mean_module.constant
        spectral_y = torch.matmul(U.T, y_centered)
        
        # 4. Apply the Wiener-like filter: Phi / (Phi + noise)
        # This is the 'trick' from the brain reconstruction script
        gp_filter = phi / (phi + noise_var)
        self.weights = (gp_filter * spectral_y).unsqueeze(-1) # Shape: (m, 1)

    def predict_fast(self, x_new):
        """
        Fast prediction using the spectral weights.
        mu = interpolated_U * weights
        """
        if self.weights is None:
            self.compute_spectral_weights()

        with torch.no_grad():
            # Use the existing out-of-sample logic from RiemannKernel
            # We need the raw interpolated eigenvectors, not the scaled 'features'
            edge_value, edge_index = self.covar_module.knn.search(x_new, self.covar_module.nearest_neighbors)
            
            # Interpolate eigenvectors to the new locations (U_star)
            u_star = self.covar_module.laplacian_operator.out_of_sample(
                self.covar_module.eigvec, edge_value, edge_index
            )
            
            # Apply the bump function for boundary awareness
            from ..utils import bump_function
            modulation = bump_function(
                edge_value[:, 0].sqrt(), 
                self.covar_module.bump_scale * self.covar_module.graphbandwidth.squeeze(), 
                self.covar_module.bump_decay
            ).unsqueeze(-1)
            
            # Final Prediction: mu = (U_star * modulation) @ weights + mean
            predictions = torch.matmul(u_star * modulation, self.weights)
            return predictions.squeeze() + self.mean_module.constant

    def forward(self, x):
        # Keep standard forward for training compatibility
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class SpectralMarginalLogLikelihood(MarginalLogLikelihood):
    """
    Computes the Marginal Log Likelihood using the spectral decomposition:
    K = U * Phi * U^T
    
    This turns the O(N^3) Cholesky into an O(m) summation.
    """
    def __init__(self, likelihood, model):
        super().__init__(likelihood, model)

    def forward(self, output, target, *params):
        # 1. Extract necessary components from the kernel and likelihood
        # We assume the kernel is a RiemannMaternKernel or similar
        kernel = self.model.covar_module.base_kernel if hasattr(self.model.covar_module, 'base_kernel') else self.model.covar_module
        
        # Ensure eigenvalues and eigenvectors are computed
        if not hasattr(kernel, 'eigval') or kernel.eigval is None:
            kernel.eval()
            
        U = kernel.eigvec           # Shape: (N, m)
        Phi = kernel.spectral_density() # Shape: (m,)
        noise = self.likelihood.noise   # sigma^2
        
        N = target.size(-1)
        m = Phi.size(-1)
        
        # 2. Project target onto the manifold basis: U^T * y
        # y_centered handles the mean module
        y_centered = target - output.mean
        u_t_y = torch.matmul(U.T, y_centered.unsqueeze(-1)).squeeze(-1) # (m,)
        
        # 3. Compute the Data Fit term: y^T (K + sigma^2 I)^-1 y
        # We split this into the 'Manifold Space' and the 'Noise Space'
        # Manifold: sum_i [ (u_i^T y)^2 / (phi_i + sigma^2) ]
        manifold_fit = torch.sum(u_t_y.pow(2) / (Phi + noise))
        
        # Noise: [ ||y||^2 - ||U^T y||^2 ] / sigma^2
        # This accounts for signal perpendicular to our chosen m eigenvectors
        y_norm_sq = torch.sum(y_centered.pow(2))
        proj_norm_sq = torch.sum(u_t_y.pow(2))
        noise_fit = (y_norm_sq - proj_norm_sq) / noise
        
        data_fit = manifold_fit + noise_fit
        
        # 4. Compute the Complexity term (Log Determinant): log |K + sigma^2 I|
        # Log-det = sum_i log(phi_i + sigma^2) + (N - m) * log(sigma^2)
        manifold_logdet = torch.sum(torch.log(Phi + noise))
        noise_logdet = (N - m) * torch.log(noise)
        
        log_det = manifold_logdet + noise_logdet
        
        # 5. Combine into Total MLL
        res = -0.5 * (data_fit + log_det + N * math.log(2 * math.pi))
        
        # GPyTorch expects the MLL to be normalized by the number of data points
        return res.div(N)