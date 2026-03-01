import torch
import pytest
import gpytorch
from l3di.lgp import IndependentMultitaskGPModel

def test_gpmodel_initialization():
    """Test initialization of the IndependentMultitaskGPModel."""
    num_inducing = 20
    input_dim = 3
    num_tasks = 5
    inducing_points = torch.randn(num_inducing, input_dim)
    
    model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=num_tasks,
        kernel_type="rbf"
    )
    
    assert isinstance(model, gpytorch.models.ApproximateGP)
    # Check if variational strategy is initialized
    assert hasattr(model, 'variational_strategy')
    # Check if mean_module and covar_module exist (standard in gpytorch models)
    assert hasattr(model, 'mean_module')
    assert hasattr(model, 'covar_module')

def test_gpmodel_forward_shape(mock_3d_coords):
    """Test the forward pass output shape."""
    num_inducing = 15
    input_dim = 3
    num_tasks = 4
    inducing_points = torch.randn(num_inducing, input_dim)
    
    model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=num_tasks
    )
    
    # Set to eval mode for deterministic output (distribution)
    model.eval()
    
    # Input: (Batch, 3)
    x = mock_3d_coords
    batch_size = x.shape[0]
    
    with torch.no_grad():
        output = model(x)
        
    # Output should be a MultitaskMultivariateNormal
    assert isinstance(output, gpytorch.distributions.MultitaskMultivariateNormal)
    
    # Mean shape should be (Batch, Num_Tasks)
    assert output.mean.shape == (batch_size, num_tasks)
    
    # Variance/Covariance checks
    # variance shape: (Batch, Num_Tasks)
    assert output.variance.shape == (batch_size, num_tasks)

def test_gpmodel_device_transfer(device):
    """Test moving the model to a device."""
    if device.type == 'cpu':
        pytest.skip("Skipping device transfer test on CPU")
        
    num_inducing = 10
    input_dim = 3
    num_tasks = 2
    inducing_points = torch.randn(num_inducing, input_dim)
    
    model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=num_tasks
    )
    
    model.to(device)
    
    # Check a parameter
    param = next(model.parameters())
    assert param.device.type == device.type
