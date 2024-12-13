
"""
Tests for diffusion system with proper boundary conditions
"""
import pytest
import numpy as np
from pyember.transport.diffusion import DiffusionSystem, BoundaryCondition
from pyember.core.grid import OneDimGrid, GridConfig
from pyember.solvers.integrator import TridiagonalIntegrator

n_points = 100

@pytest.fixture
def simple_diffusion_system():
    """Create a simple diffusion system for testing"""
    grid = OneDimGrid(GridConfig(
        x_min=0.0,
        x_max=1.0,
        n_points=n_points
    ))
    
    system = DiffusionSystem(grid)
    # Set default boundary conditions
    system.set_boundary_conditions(
        leftBC=BoundaryCondition.FixedValue,
        rightBC=BoundaryCondition.FixedValue
    )
    return system

def test_diffusion_initialization(simple_diffusion_system):
    """Test diffusion system initialization"""
    system = simple_diffusion_system
    
    # Check arrays initialized correctly
    assert len(system.D) == n_points
    assert len(system.B) == n_points
    assert len(system.c1) == n_points
    assert len(system.c2) == n_points
    assert len(system.splitConst) == n_points
    
    # Check default values
    assert np.all(system.D == 1.0)
    assert np.all(system.B == 1.0)
    assert np.all(system.splitConst == 0.0)
    
    # Check validation
    assert system.validate()

def test_boundary_conditions():
    """Test different boundary condition types"""
    grid = OneDimGrid(GridConfig(x_min=0.0, x_max=1.0, n_points=n_points))
    system = DiffusionSystem(grid)
    
    # Test fixed value boundaries
    system.set_boundary_conditions(
        leftBC=BoundaryCondition.FixedValue,
        rightBC=BoundaryCondition.FixedValue
    )
    a, b, c = system.get_coefficients()
    assert b[0] == 1.0 and c[0] == 0.0  # Left fixed
    assert b[-1] == 1.0 and a[-1] == 0.0  # Right fixed
    
    # Test zero gradient boundaries
    system.set_boundary_conditions(
        leftBC=BoundaryCondition.ZeroGradient,
        rightBC=BoundaryCondition.ZeroGradient
    )
    a, b, c = system.get_coefficients()
    # Check that boundary coefficients enforce zero gradient
    assert abs(b[1] + c[1]) < 1e-10  # Left zero gradient
    assert abs(a[-2] + b[-2]) < 1e-10  # Right zero gradient
    
    # Test wall flux boundary
    system.set_boundary_conditions(
        leftBC=BoundaryCondition.WallFlux,
        rightBC=BoundaryCondition.ZeroGradient
    )
    system.set_wall_flux(yInf=300.0, wallConst=100.0)
    a, b, c = system.get_coefficients()
    # Wall flux coefficients should include conductance
    assert b[0] != 1.0  # Modified by wall flux
    assert c[0] != 0.0  # Modified by wall flux

def test_diffusion_coefficients(simple_diffusion_system):
    """Test computation of system coefficients"""
    system = simple_diffusion_system
    
    # Set non-uniform properties
    D = np.linspace(0.5, 1.5, n_points)  # Varying diffusion
    rho = np.linspace(0.8, 1.2, n_points)  # Varying density
    B = np.ones(n_points)  # Unit prefactor
    system.set_properties(D, rho, B)
    
    # Get coefficients
    a, b, c = system.get_coefficients()
    
    # Check sizes
    assert len(a) == n_points
    assert len(b) == n_points
    assert len(c) == n_points
    
    # Check fixed value boundary conditions
    assert a[0] == 0.0
    assert c[-1] == 0.0
    assert b[0] == 1.0 and b[-1] == 1.0
    
    # Check conservation (row sum = 0) for interior points
    for j in range(1, n_points-1):
        assert abs(a[j] + b[j] + c[j]) < 1e-10
        
    # # Check coefficient signs
    # assert np.all(a[1:] <= 0)  # Lower diagonal negative
    # assert np.all(c[:-1] <= 0)  # Upper diagonal negative
    # assert np.all(b[1:-1] >= 0)  # Main diagonal positive


def test_diffusion_integration():
    """Test integration of diffusion equation"""
    # Create grid
    L = 1.0
    N = 101
    grid = OneDimGrid(GridConfig(
        x_min=0.0,
        x_max=L,
        n_points=N
    ))
    
    # Create system
    system = DiffusionSystem(grid)
    system.set_boundary_conditions(
        leftBC=BoundaryCondition.ZeroGradient,
        rightBC=BoundaryCondition.FixedValue
    )
    
    # Set properties
    D = np.ones(N)
    rho = np.ones(N)
    system.set_properties(D, rho)
    
    # Create integrator
    integrator = TridiagonalIntegrator(system)
    
    # Initial condition (Gaussian)
    x = grid.x
    sigma = 0.1
    y0 = np.exp(-(x - L/2)**2 / (2*sigma**2))
    integrator.set_y0(y0)
    
    # Integrate with smaller timesteps
    t0 = 0.0
    dt = 0.0001  # Timestep
    tf = 0.1
    
    integrator.initialize(t0, dt)
    assert not np.any(np.isnan(integrator.y))  # Check after initialization
    
    nsteps = int(tf/dt)
    for _ in range(nsteps):
        integrator.step()
        # Check solution remains valid
        assert not np.any(np.isnan(integrator.y))
        assert not np.any(np.isinf(integrator.y))
        assert np.all(integrator.y >= 0)  # Physical constraint
        
    y = integrator.get_y()
    
    # Check solution properties
    # 1. Zero gradient at right boundary
    right_flux = (y[-1] - y[-2]) / (x[-1] - x[-2])
    assert abs(right_flux) < 1e-3
    
    # 2. Fixed value at left boundary
    assert abs(y[0] - y0[0]) < 1e-10
    
    # 3. Solution remains bounded
    assert np.all(y <= np.max(y0))
    
def test_split_constants():
    """Test operator splitting terms"""
    grid = OneDimGrid(GridConfig(x_min=0.0, x_max=1.0, n_points=5))
    system = DiffusionSystem(grid)
    
    # Set non-zero split constants
    split = np.linspace(-1, 1, 5)
    system.splitConst = split.copy()
    
    # Check RHS includes split constants
    rhs = system.get_rhs(0.0, np.zeros(5))
    np.testing.assert_allclose(rhs, split)
    
    # Check reset works
    system.reset_split_constants()
    assert np.all(system.splitConst == 0.0)

def test_wall_flux():
    """Test wall flux boundary condition"""
    grid = OneDimGrid(GridConfig(x_min=0.0, x_max=1.0, n_points=5))
    system = DiffusionSystem(grid)
    
    # Set wall flux condition
    system.set_boundary_conditions(
        leftBC=BoundaryCondition.WallFlux,
        rightBC=BoundaryCondition.ZeroGradient
    )
    
    Tinf = 300.0
    kwall = 100.0
    system.set_wall_flux(yInf=Tinf, wallConst=kwall)
    
    # Check RHS includes wall flux term
    rhs = system.get_rhs(0.0, np.zeros(5))
    assert rhs[0] != 0.0  # Wall flux contribution
    assert np.all(rhs[1:] == 0.0)  # No other contributions
    
def test_cylindrical_diffusion():
    """Test diffusion in cylindrical coordinates"""
    # Create cylindrical grid
    grid = OneDimGrid(GridConfig(
        x_min=0.1,  # Avoid r=0 singularity
        x_max=1.0,
        n_points=51,
        cylindrical=True,
        alpha=1
    ))
    
    # Create system
    system = DiffusionSystem(grid)
    
    # Set constant properties
    D = np.ones_like(grid.x)
    rho = np.ones_like(grid.x)
    system.set_properties(D, rho)
    
    # Check coefficients include geometric factors
    a, b, c = system.get_coefficients()
    
    # Coefficients should vary with radius
    assert not np.allclose(a[1:-1], a[2:])
    assert not np.allclose(c[:-2], c[1:-1])