import torch
import pytest
import gpytorch
from l3di.lgp import Custom3DKernel

def test_kernel_initialization():
    """Test that the kernel can be initialized with default parameters."""
    kernel = Custom3DKernel()
    assert isinstance(kernel, gpytorch.kernels.Kernel)
    # Check default nu value based on class definition summary
    assert hasattr(kernel, 'nu')
    assert kernel.nu == 2.5

def test_kernel_forward_shape(mock_3d_coords):
    """Test the output shape of the kernel forward pass."""
    kernel = Custom3DKernel()
    x = mock_3d_coords  # Shape: (Batch, 3)
    
    # gpytorch kernels return a LazyTensor; evaluate() returns the actual tensor
    covar = kernel(x, x).evaluate()
    
    # Output should be a square matrix (Batch, Batch)
    assert covar.shape == (x.shape[0], x.shape[0])
    
    # Covariance matrix must be symmetric
    assert torch.allclose(covar, covar.t(), atol=1e-4)
    
    # Diagonal elements should be positive (variance)
    assert torch.all(torch.diag(covar) > 0)

def test_kernel_batch_processing():
    """Test that the kernel can handle batched inputs if required, 
    though typically GP kernels in this context handle (N, D) inputs."""
    kernel = Custom3DKernel()
    
    # Create a batch of inputs: (Batch, Points, Dims) -> (2, 5, 3)
    batch_x = torch.randn(2, 5, 3)
    
    # If the kernel supports batch_shape in forward or via __call__
    covar = kernel(batch_x, batch_x).evaluate()
    
    # Expected output: (2, 5, 5)
    assert covar.shape == (2, 5, 5)

def test_kernel_values_logic():
    """
    Test basic logic of the kernel values:
    - Self-similarity should be high.
    - Distant points should have lower covariance.
    """
    kernel = Custom3DKernel()
    
    p1 = torch.tensor([[0.0, 0.0, 0.0]])
    p2 = torch.tensor([[0.0, 0.0, 0.0]])
    p3 = torch.tensor([[10.0, 10.0, 10.0]])
    
    cov_self = kernel(p1, p2).evaluate().item()
    cov_dist = kernel(p1, p3).evaluate().item()
    
    assert cov_self > cov_dist
    assert cov_self > 0
