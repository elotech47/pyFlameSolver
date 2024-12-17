
# """
# Tests for diffusion system with proper boundary conditions
# """
# import pytest
# import numpy as np
# from pyember.transport.diffusion import DiffusionSystem, BoundaryCondition
# from pyember.core.grid import OneDimGrid, GridConfig
# from pyember.solvers.integrator import TridiagonalIntegrator

# n_points = 100

# @pytest.fixture
# def simple_diffusion_system():
#     """Create a simple diffusion system for testing"""
#     grid = OneDimGrid(GridConfig(
#         gridMax=1.0,
#         gridMin=0.0,
#     ))
    
#     system = DiffusionSystem(grid)
#     # Set default boundary conditions
#     system.set_boundary_conditions(
#         leftBC=BoundaryCondition.FixedValue,
#         rightBC=BoundaryCondition.FixedValue
#     )
#     return system

# def test_diffusion_initialization(simple_diffusion_system):
#     """Test diffusion system initialization"""
#     system = simple_diffusion_system
    
#     # Check arrays initialized correctly
#     assert len(system.D) == n_points
#     assert len(system.B) == n_points
#     assert len(system.c1) == n_points
#     assert len(system.c2) == n_points
#     assert len(system.splitConst) == n_points
    
#     # Check default values
#     assert np.all(system.D == 1.0)
#     assert np.all(system.B == 1.0)
#     assert np.all(system.splitConst == 0.0)
    
#     # Check validation
#     assert system.validate()

# def test_boundary_conditions():
#     """Test different boundary condition types"""
#     grid = OneDimGrid(GridConfig())
#     system = DiffusionSystem(grid)
    
#     # Test fixed value boundaries
#     system.set_boundary_conditions(
#         leftBC=BoundaryCondition.FixedValue,
#         rightBC=BoundaryCondition.FixedValue
#     )
#     a, b, c = system.get_coefficients()
#     assert b[0] == 1.0 and c[0] == 0.0  # Left fixed
#     assert b[-1] == 1.0 and a[-1] == 0.0  # Right fixed
    
#     # Test zero gradient boundaries
#     system.set_boundary_conditions(
#         leftBC=BoundaryCondition.ZeroGradient,
#         rightBC=BoundaryCondition.ZeroGradient
#     )
#     a, b, c = system.get_coefficients()
#     # Check that boundary coefficients enforce zero gradient
#     assert abs(b[1] + c[1]) < 1e-10  # Left zero gradient
#     assert abs(a[-2] + b[-2]) < 1e-10  # Right zero gradient
    
#     # Test wall flux boundary
#     system.set_boundary_conditions(
#         leftBC=BoundaryCondition.WallFlux,
#         rightBC=BoundaryCondition.ZeroGradient
#     )
#     system.set_wall_flux(yInf=300.0, wallConst=100.0)
#     a, b, c = system.get_coefficients()
#     # Wall flux coefficients should include conductance
#     assert b[0] != 1.0  # Modified by wall flux
#     assert c[0] != 0.0  # Modified by wall flux

# def test_diffusion_coefficients(simple_diffusion_system):
#     """Test computation of system coefficients"""
#     system = simple_diffusion_system
    
#     # Set non-uniform properties
#     D = np.linspace(0.5, 1.5, n_points)  # Varying diffusion
#     rho = np.linspace(0.8, 1.2, n_points)  # Varying density
#     B = np.ones(n_points)  # Unit prefactor
#     system.set_properties(D, rho, B)
    
#     # Get coefficients
#     a, b, c = system.get_coefficients()
    
#     # Check sizes
#     assert len(a) == n_points
#     assert len(b) == n_points
#     assert len(c) == n_points
    
#     # Check fixed value boundary conditions
#     assert a[0] == 0.0
#     assert c[-1] == 0.0
#     assert b[0] == 1.0 and b[-1] == 1.0
    
#     # Check conservation (row sum = 0) for interior points
#     for j in range(1, n_points-1):
#         assert abs(a[j] + b[j] + c[j]) < 1e-10
        
#     # # Check coefficient signs
#     # assert np.all(a[1:] <= 0)  # Lower diagonal negative
#     # assert np.all(c[:-1] <= 0)  # Upper diagonal negative
#     # assert np.all(b[1:-1] >= 0)  # Main diagonal positive


# def test_diffusion_integration():
#     """Test integration of diffusion equation"""
#     # Create grid
#     L = 1.0
#     N = 101
#     grid = OneDimGrid(GridConfig(
#     ))
    
#     # Create system
#     system = DiffusionSystem(grid)
#     system.set_boundary_conditions(
#         leftBC=BoundaryCondition.ZeroGradient,
#         rightBC=BoundaryCondition.FixedValue
#     )
    
#     # Set properties
#     D = np.ones(N)
#     rho = np.ones(N)
#     system.set_properties(D, rho)
    
#     # Create integrator
#     integrator = TridiagonalIntegrator(system)
    
#     # Initial condition (Gaussian)
#     x = grid.x
#     sigma = 0.1
#     y0 = np.exp(-(x - L/2)**2 / (2*sigma**2))
#     integrator.set_y0(y0)
    
#     # Integrate with smaller timesteps
#     t0 = 0.0
#     dt = 0.0001  # Timestep
#     tf = 0.1
    
#     integrator.initialize(t0, dt)
#     assert not np.any(np.isnan(integrator.y))  # Check after initialization
    
#     nsteps = int(tf/dt)
#     for _ in range(nsteps):
#         integrator.step()
#         # Check solution remains valid
#         assert not np.any(np.isnan(integrator.y))
#         assert not np.any(np.isinf(integrator.y))
#         assert np.all(integrator.y >= 0)  # Physical constraint
        
#     y = integrator.get_y()
    
#     # Check solution properties
#     # 1. Zero gradient at right boundary
#     right_flux = (y[-1] - y[-2]) / (x[-1] - x[-2])
#     assert abs(right_flux) < 1e-3
    
#     # 2. Fixed value at left boundary
#     assert abs(y[0] - y0[0]) < 1e-10
    
#     # 3. Solution remains bounded
#     assert np.all(y <= np.max(y0))
    
# def test_split_constants():
#     """Test operator splitting terms"""
#     grid = OneDimGrid(GridConfig())
#     system = DiffusionSystem(grid)
    
#     # Set non-zero split constants
#     split = np.linspace(-1, 1, 5)
#     system.splitConst = split.copy()
    
#     # Check RHS includes split constants
#     rhs = system.get_rhs(0.0, np.zeros(5))
#     np.testing.assert_allclose(rhs, split)
    
#     # Check reset works
#     system.reset_split_constants()
#     assert np.all(system.splitConst == 0.0)

# def test_wall_flux():
#     """Test wall flux boundary condition"""
#     grid = OneDimGrid(GridConfig(x_min=0.0, x_max=1.0, n_points=5))
#     system = DiffusionSystem(grid)
    
#     # Set wall flux condition
#     system.set_boundary_conditions(
#         leftBC=BoundaryCondition.WallFlux,
#         rightBC=BoundaryCondition.ZeroGradient
#     )
    
#     Tinf = 300.0
#     kwall = 100.0
#     system.set_wall_flux(yInf=Tinf, wallConst=kwall)
    
#     # Check RHS includes wall flux term
#     rhs = system.get_rhs(0.0, np.zeros(5))
#     assert rhs[0] != 0.0  # Wall flux contribution
#     assert np.all(rhs[1:] == 0.0)  # No other contributions
    
# def test_cylindrical_diffusion():
#     """Test diffusion in cylindrical coordinates"""
#     # Create cylindrical grid
#     grid = OneDimGrid(GridConfig(
#     ))
    
#     # Create system
#     system = DiffusionSystem(grid)
    
#     # Set constant properties
#     D = np.ones_like(grid.x)
#     rho = np.ones_like(grid.x)
#     system.set_properties(D, rho)
    
#     # Check coefficients include geometric factors
#     a, b, c = system.get_coefficients()
    
#     # Coefficients should vary with radius
#     assert not np.allclose(a[1:-1], a[2:])
#     assert not np.allclose(c[:-2], c[1:-1])



"""
Tests for diffusion system matching C++ implementation
"""
import pytest
import numpy as np
from pyember.transport.diffusion import DiffusionSystem, BoundaryCondition
from pyember.core.grid import OneDimGrid, GridConfig
from pyember.solvers.integrator import TridiagonalIntegrator

@pytest.fixture
def basic_grid():
    """Create a basic grid for testing"""
    grid = OneDimGrid()
    grid.setOptions(GridConfig())
    grid.x = np.linspace(0, 1, 5)
    grid.setSize(5)
    grid.dampVal = np.ones_like(grid.x)
    grid.updateValues()
    return grid

@pytest.fixture
def simple_diffusion_system(basic_grid):
    """Create a simple diffusion system for testing"""
    system = DiffusionSystem(basic_grid)
    # Initialize with unit properties
    system.D = np.ones(len(basic_grid.x))
    system.B = np.ones(len(basic_grid.x))
    system.splitConst = np.zeros(len(basic_grid.x))
    return system

def test_diffusion_initialization(simple_diffusion_system):
    """Test diffusion system initialization"""
    system = simple_diffusion_system
    
    # Check arrays initialized correctly
    N = len(system.x)
    assert len(system.D) == N
    assert len(system.B) == N
    assert len(system.c1) == N
    assert len(system.c2) == N
    assert len(system.splitConst) == N
    
    # Check initial values match initialization
    np.testing.assert_array_equal(system.D, np.ones(N))
    np.testing.assert_array_equal(system.B, np.ones(N))
    np.testing.assert_array_equal(system.splitConst, np.zeros(N))

def test_boundary_conditions(basic_grid):
    """Test different boundary condition types"""
    system = DiffusionSystem(basic_grid)
    
    # Set unit properties
    N = len(basic_grid.x)
    system.D = np.ones(N)
    system.B = np.ones(N)
    
    # Test fixed value boundaries
    system.leftBC = BoundaryCondition.FixedValue
    system.rightBC = BoundaryCondition.FixedValue
    
    a, b, c = system.get_coefficients()
    # assert b[0] == 1.0  # Left fixed
    # assert c[0] == 0.0
    # assert a[0] == 0.0
    # assert b[-1] == 1.0  # Right fixed
    # assert a[-1] == 0.0
    # assert c[-1] == 0.0
    
    # Test zero gradient boundaries
    system.leftBC = BoundaryCondition.ZeroGradient
    system.rightBC = BoundaryCondition.ZeroGradient
    
    a, b, c = system.get_coefficients()
    # Check coefficients enforce zero gradient
    assert np.isclose(b[1], -system.c1[1] * system.c2[1])  # Left
    assert np.isclose(c[1], system.c1[1] * system.c2[1])
    assert np.isclose(a[-2], system.c1[-2] * system.c2[-3])  # Right
    assert np.isclose(b[-2], -system.c1[-2] * system.c2[-3])

def test_wall_flux(basic_grid):
    """Test wall flux boundary condition"""
    system = DiffusionSystem(basic_grid)
    N = len(basic_grid.x)
    
    # Set unit properties
    system.D = np.ones(N)
    system.B = np.ones(N)
    
    # Set wall flux condition
    system.leftBC = BoundaryCondition.WallFlux
    system.rightBC = BoundaryCondition.FixedValue
    system.yInf = 300.0
    system.wallConst = 100.0
    
    # Check coefficients
    a, b, c = system.get_coefficients()
    
    # Wall flux coefficients should match C++
    d = 0.5 * (system.D[0] + system.D[1])
    c0 = system.B[0] * (system.grid.alpha + 1) / system.hh[0]
    expected_b0 = -c0 * (d / system.hh[0] + system.wallConst)
    expected_c0 = d * c0 / system.hh[0]
    
    np.testing.assert_allclose(b[0], expected_b0)
    np.testing.assert_allclose(c[0], expected_c0)
    
    # Check RHS includes wall flux term
    rhs = system.get_rhs(0.0, np.zeros(N))
    wall_term = (system.B[0] * (system.grid.alpha + 1) / system.hh[0] * 
                system.wallConst * system.yInf)
    assert np.isclose(rhs[0], wall_term)
    assert np.all(rhs[1:] == 0.0)

def test_cylindrical_coordinates(basic_grid):
    """Test cylindrical coordinate handling"""
    # Set cylindrical coordinates
    basic_grid.alpha = 1  # Cylindrical
    system = DiffusionSystem(basic_grid)
    
    N = len(basic_grid.x)
    system.D = np.ones(N)
    system.B = np.ones(N)
    
    # Get coefficients
    a, b, c = system.get_coefficients()
    
    # Check c1 includes r terms
    for j in range(1, N-1):
        expected_c1 = 0.5 * system.B[j] / (system.dlj[j] * system.r[j])
        np.testing.assert_allclose(system.c1[j], expected_c1)
        
    # Check c2 includes rphalf terms
    for j in range(1, N-1):
        expected_c2 = (system.rphalf[j] * (system.D[j] + system.D[j+1]) / 
                      system.hh[j])
        np.testing.assert_allclose(system.c2[j], expected_c2)

def test_integration(basic_grid):
    """Test basic integration with fixed boundaries"""
    system = DiffusionSystem(basic_grid)
    N = len(basic_grid.x)
    
    # Set properties
    system.D = np.ones(N)
    system.B = np.ones(N)
    system.leftBC = BoundaryCondition.FixedValue
    system.rightBC = BoundaryCondition.FixedValue
    
    # Create integrator
    integrator = TridiagonalIntegrator(system)
    
    # Initial condition - simple sine wave
    y0 = np.sin(np.pi * basic_grid.x)
    integrator.set_y0(y0)
    
    # Short integration
    integrator.initialize(0.0, 0.01)
    integrator.step()
    
    y = integrator.get_y()
    assert not np.any(np.isnan(y))  # No invalid values
    assert y[0] == y0[0]  # Left BC maintained
    assert y[-1] == y0[-1]  # Right BC maintained