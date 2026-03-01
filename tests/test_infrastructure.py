import torch
import pytest

def test_device_fixture(device):
    """Verify device fixture returns a torch device."""
    assert isinstance(device, torch.device)

def test_mock_data_fixtures(mock_1d_data, mock_3d_coords, mock_latent_z):
    """Verify shapes of mock data fixtures."""
    # Check 1D data
    assert mock_1d_data.ndim == 3
    assert mock_1d_data.shape[0] == 4  # batch
    assert mock_1d_data.shape[1] == 1  # channels
    
    # Check 3D coords
    assert mock_3d_coords.ndim == 2
    assert mock_3d_coords.shape[1] == 3
    
    # Check latent z
    assert mock_latent_z.ndim == 2
    assert mock_latent_z.shape[1] == 32
