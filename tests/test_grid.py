"""
test_grid.py: Unit tests for the grid system
"""

import pytest
import numpy as np
from pyember.core.grid import OneDimGrid, GridConfig


@pytest.fixture
def basic_grid():
    """Create a basic grid for testing"""
    config = GridConfig(
        x_min=0.0,
        x_max=1.0,
        n_points=5
    )
    return OneDimGrid(config)


def test_grid_initialization():
    """Test basic grid initialization"""
    grid = OneDimGrid()
    assert grid.x is not None
    assert len(grid.x) == grid.config.n_points
    assert grid.x[0] == grid.config.x_min
    assert grid.x[-1] == grid.config.x_max


def test_grid_metrics(basic_grid):
    """Test computation of grid metrics"""
    grid = basic_grid
    
    # Check grid spacing
    assert len(grid.dx) == len(grid.x)
    assert np.all(grid.dx[:-1] > 0)
    
    # Check finite difference coefficients
    assert len(grid.cf) == len(grid.x)
    assert len(grid.cfm) == len(grid.x)
    assert len(grid.cfp) == len(grid.x)
    
    # Interior points should sum to zero
    for j in range(1, len(grid.x)-1):
        assert abs(grid.cfm[j] + grid.cf[j] + grid.cfp[j]) < 1e-14


def test_grid_adaptation():
    """Test grid adaptation to solution features"""
    grid = OneDimGrid(GridConfig(
        x_min=0.0,
        x_max=1.0,
        n_points=20,
        vtol=0.1,
        dvtol=0.2,
        grid_min=1e-4,
        grid_max=0.1
    ))
    
    # Initial number of points
    n_initial = len(grid.x)
    
    # Create solution with sharp gradient
    y = np.tanh((grid.x - 0.5) * 10)
    
    # Adapt grid
    modified = grid.adapt_grid(y)
    
    # Grid should be modified
    assert modified
    
    # Points should be within bounds
    assert np.all(grid.x >= grid.config.x_min)
    assert np.all(grid.x <= grid.config.x_max)
    
    # Grid spacing should respect limits
    dx = np.diff(grid.x)
    assert np.all(dx >= grid.config.grid_min)
    assert np.all(dx <= grid.config.grid_max)
    
    # Should have more points near gradient
    mid_points = np.logical_and(grid.x > 0.4, grid.x < 0.6)
    n_mid_points = np.sum(mid_points)
    assert n_mid_points > n_initial / 4  # At least 25% of points near gradient
    
    # Grid should be monotonic
    assert np.all(np.diff(grid.x) > 0)


def test_cylindrical_grid():
    """Test cylindrical coordinate transformation"""
    grid = OneDimGrid(GridConfig(
        cylindrical=True,
        alpha=1
    ))
    
    assert grid.r is not None
    assert len(grid.r) == len(grid.x)
    # For alpha=1, r should equal x
    np.testing.assert_array_almost_equal(grid.r, grid.x)


def test_grid_validation():
    """Test grid configuration validation"""
    # Invalid grid extent
    config = GridConfig(x_min=1.0, x_max=0.0)
    grid = OneDimGrid(config)
    assert not grid.validate()
    
    # Invalid number of points
    config = GridConfig(n_points=1)
    grid = OneDimGrid(config)
    assert not grid.validate()
    
    # Invalid grid spacing bounds
    config = GridConfig(grid_min=-1.0)
    grid = OneDimGrid(config)
    assert not grid.validate()
    
    # Valid configuration
    config = GridConfig()
    grid = OneDimGrid(config)
    assert grid.validate()


def test_grid_neighbors(basic_grid):
    """Test neighbor point identification"""
    grid = basic_grid
    
    # Left boundary
    left, right = grid.get_neighbors(0)
    assert left == 0
    assert right == 1
    
    # Interior point
    left, right = grid.get_neighbors(2)
    assert left == 1
    assert right == 3
    
    # Right boundary
    left, right = grid.get_neighbors(4)
    assert left == 3
    assert right == 4
    
def test_grid_quality_metrics():
    """Test grid quality metrics"""
    grid = OneDimGrid(GridConfig(
        x_min=0.0,
        x_max=1.0,
        n_points=5,
        uniformityTol=2.5
    ))
    
    # Test uniform grid
    metric = grid.uniformity_metric()
    assert metric < 1.1  # Should be close to 1.0 for uniform grid
    
    # Test weight function
    assert grid.grid_weight_function(0.5) > grid.grid_weight_function(0.1)
    
    # Test spacing check
    assert not grid.check_spacing()  # Initial grid should be okay

def test_boundary_extension():
    """Test grid boundary extension"""
    grid = OneDimGrid(GridConfig(
        x_min=-2.0,
        x_max=2.0,
        n_points=20,
        fixed_left_loc=False,
        boundaryTol=0.1,
        grid_max=0.2
    ))

    # Create solution with stronger gradient at boundary
    y = 10 * np.exp(-(grid.x/0.5)**2)  # Narrower Gaussian
    
    # Extend boundaries
    modified = grid.extend_boundaries(y)
    assert modified  # Grid should be modified
    
    
def test_error_estimation():
    """Test error estimation"""
    grid = OneDimGrid(GridConfig(
        x_min=0.0,
        x_max=1.0,
        n_points=20
    ))
    
    # Create solution with known features
    x = grid.x
    y = np.sin(2 * np.pi * x)  # Simple periodic function
    
    # Compute error weights
    weights = grid.compute_error_weights(y)
    
    # Error should be larger where curvature is larger
    assert np.max(weights) > 0
    assert weights[0] == 0  # No error at boundaries
    assert weights[-1] == 0
