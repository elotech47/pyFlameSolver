"""
test_grid.py: Unit tests for the grid system matching C++ implementation
"""

import pytest
import numpy as np
from pyember.core.grid import OneDimGrid, GridConfig, BoundaryCondition


@pytest.fixture
def basic_grid():
    """Create a basic grid for testing"""
    grid = OneDimGrid()
    grid.setOptions(GridConfig(
        vtol=0.12,
        dvtol=0.2,
        gridMin=5e-7,
        gridMax=2e-4,
        uniformityTol=2.5,
        boundaryTol=5e-5,
        boundaryTolRm=1e-5,
        addPointCount=3,
        cylindricalFlame=False
    ))
    
    # Initialize with 5 points
    grid.x = np.linspace(0, 1, 5)
    grid.setSize(5)
    grid.dampVal = np.ones_like(grid.x)  # Initialize damping
    grid.updateValues()
    return grid


def test_grid_initialization():
    """Test grid initialization matches C++ behavior"""
    grid = OneDimGrid()
    grid.setOptions(GridConfig())
    
    # Check default flags
    assert grid.updated is True
    assert grid.leftBC == BoundaryCondition.FixedValue
    assert grid.rightBC == BoundaryCondition.FixedValue
    
    # Check coordinate system defaults
    assert grid.alpha == 0  # Planar by default
    assert grid.beta == 1.0  # Standard strain metric
    
    # Check indices
    assert grid.ju == 0  # Unburned index
    assert grid.jb == 0  # Burned index


def test_grid_metrics(basic_grid):
    """Test computation of grid metrics matches C++"""
    grid = basic_grid
    
    # Check main grid arrays exist
    assert grid.hh is not None  # Grid spacing
    assert grid.cfm is not None  # Left coefficients
    assert grid.cf is not None  # Center coefficients 
    assert grid.cfp is not None  # Right coefficients
    assert grid.rphalf is not None  # Radial coordinates at midpoints
    
    # Check sizes
    assert len(grid.hh) == grid.jj  # One less than points
    assert len(grid.rphalf) == grid.jj
    assert len(grid.r) == grid.nPoints
    
    # Check grid spacing calculation
    np.testing.assert_allclose(
        grid.hh,
        np.diff(grid.x)
    )
    
    # For planar grid (alpha=0), rphalf should be ones
    np.testing.assert_allclose(grid.rphalf, np.ones_like(grid.rphalf))


def test_cylindrical_coordinates():
    """Test cylindrical coordinate handling"""
    grid = OneDimGrid()
    config = GridConfig()
    config.cylindricalFlame = True
    grid.setOptions(config)
    
    # Initialize grid
    grid.x = np.linspace(0, 1, 5)
    grid.setSize(5)
    grid.dampVal = np.ones_like(grid.x)
    grid.updateValues()
    
    # Check coordinate transformation
    assert grid.alpha == 1  # Cylindrical
    np.testing.assert_allclose(
        grid.r,
        grid.x  # For alpha=1, r should equal x
    )
    np.testing.assert_allclose(
        grid.rphalf,
        0.5*(grid.x[1:] + grid.x[:-1])  # Midpoint values
    )


def test_adaptation():
    """Test grid adaptation with numpy arrays"""
    grid = OneDimGrid()
    grid.setOptions(GridConfig())
    
    # Initialize grid
    grid.x = np.linspace(0, 1, 20)
    grid.setSize(20)
    grid.dampVal = np.ones_like(grid.x)
    grid.updateValues()
    
    # Create test solution with sharp gradient
    y1 = np.tanh((grid.x - 0.5) * 10)
    y2 = 1 - y1
    y = [y1.copy(), y2.copy()]  # Make copies to allow modification
    
    # Set adaptation parameters
    grid.nAdapt = 2
    grid.vtol = [0.12, 0.12]
    grid.dvtol = [0.2, 0.2]
    
    # Adapt grid
    modified = grid.adapt(y)
    
    assert modified
    assert np.all(np.diff(grid.x) > 0)  # Check monotonicity
    assert len(grid.x) > 20  # Should add points near gradient


def test_boundary_handling():
    """Test boundary condition handling"""
    grid = OneDimGrid()
    grid.setOptions(GridConfig())
    
    # Initialize grid
    grid.x = np.linspace(0, 1, 10)
    grid.setSize(10)
    grid.dampVal = np.ones_like(grid.x)
    grid.updateValues()
    
    # Test fixed value conditions
    grid.leftBC = BoundaryCondition.FixedValue
    grid.rightBC = BoundaryCondition.FixedValue
    grid.fixedLeftLoc = True
    
    y = [np.sin(2*np.pi*grid.x)]  # Test solution
    grid.nAdapt = 1
    grid.vtol = [0.12]
    grid.dvtol = [0.2]
    
    # Add points at right boundary
    added = grid.addRight(y)
    assert added
    assert grid.x[-1] == 1.0  # Right boundary should stay fixed
    
    # Left boundary shouldn't move
    added = grid.addLeft(y)
    assert not added  # Should not add points due to fixedLeftLoc


def test_point_removal():
    """Test point removal logic with proper criteria"""
    grid = OneDimGrid()
    grid.setOptions(GridConfig())
    
    # Initialize with extra points
    grid.x = np.linspace(0, 1, 30)
    grid.setSize(30)
    grid.dampVal = np.ones_like(grid.x)
    grid.updateValues()
    
    # Create a solution that's very smooth near the boundary
    x = grid.x
    # Use a function that's nearly constant near x=1
    y = [np.sin(np.pi*x/2) * np.exp(-(x-0.5)**2/0.1)]
    
    grid.nAdapt = 1
    grid.vtol = [0.12]
    grid.dvtol = [0.2]
    grid.boundaryTolRm = 0.05  # Set removal tolerance
    
    # Verify initial conditions
    assert len(grid.x) == 30
    
    # Remove points
    removed = grid.removeRight(y)
    assert removed  # Should remove point from smooth region
    assert len(grid.x) == 29  # Should have one less point
    
    # Check that remaining points maintain proper spacing
    dx = np.diff(grid.x)
    assert np.all(dx > 0)  # Still monotonic
    assert np.all(dx >= grid.gridMin)  # Respect minimum spacing

def test_grid_based():
    """Test grid metrics inheritance"""
    grid = OneDimGrid()
    grid.setOptions(GridConfig())
    
    # Initialize
    grid.x = np.linspace(0, 1, 5)
    grid.setSize(5)
    grid.dampVal = np.ones_like(grid.x)
    grid.updateValues()
    
    # Check metric arrays
    assert grid.hh is not None
    assert grid.cfm is not None
    assert grid.cf is not None
    assert grid.cfp is not None
    assert grid.rphalf is not None
    
    # Check coordinate parameters
    assert hasattr(grid, 'alpha')
    assert hasattr(grid, 'beta')