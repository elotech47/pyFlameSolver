"""
Tests for convection transport system
"""
import pytest
import numpy as np
import cantera as ct
from pyember.transport.convection import (
    ConvectionSystemUTW, ConvectionSystemY, ConvectionSystemSplit,
    BoundaryCondition, ContinuityBoundaryCondition, ConvectionConfig
)
from pyember.core.grid import OneDimGrid, GridConfig

@pytest.fixture
def simple_grid():
    """Create a simple grid for testing"""
    config = GridConfig(
        x_min=0.0,
        x_max=1.0,
        n_points=5,
        cylindrical=False,
        alpha=0,
        beta=1.0  # Add beta for strain rate
    )
    return OneDimGrid(config)

@pytest.fixture
def simple_gas():
    """Create a simple gas mixture for testing"""
    gas = ct.Solution('gri30.yaml')
    gas.TPX = 300.0, ct.one_atm, 'H2:1, O2:1, N2:3.76'
    return gas

class TestUTWSystem:
    """Tests for the UTW (Velocity-Temperature-Weight) system"""

    @pytest.fixture
    def utw_system(self, simple_grid, simple_gas):
        """Create basic UTW system"""
        system = ConvectionSystemUTW()
        system.gas = simple_gas
        system.set_grid(simple_grid)
        system.resize(simple_grid.config.n_points)
        system.set_boundary_conditions(
            left_bc=BoundaryCondition.FixedValue,
            right_bc=BoundaryCondition.FixedValue,
            continuity_bc=ContinuityBoundaryCondition.Left
        )
        return system

    def test_initialization(self, utw_system):
        """Test basic initialization"""
        n = utw_system.T.shape[0]
        assert n == 5  # Check grid size
        assert utw_system.T is not None
        assert utw_system.U is not None
        assert utw_system.Wmx is not None
        assert utw_system.rho is not None
        assert utw_system.V is not None
        assert utw_system.rV is not None
        assert np.all(utw_system.split_const_T == 0)
        assert np.all(utw_system.split_const_U == 0)
        assert np.all(utw_system.split_const_W == 0)

    def test_boundary_conditions(self, utw_system):
        """Test boundary condition handling"""
        # Set up test conditions
        utw_system.T = np.array([300., 400., 500., 600., 700.])
        utw_system.U = np.zeros_like(utw_system.T)
        utw_system.Wmx = np.ones_like(utw_system.T) * utw_system.gas.mean_molecular_weight
        utw_system.r_vzero = 0.1

        # Test left boundary continuity
        utw_system.continuity_bc = ContinuityBoundaryCondition.Left
        utw_system._calculate_V()
        assert utw_system.rV[0] == 0.1  # Should match r_vzero

        # Test zero continuity
        utw_system.continuity_bc = ContinuityBoundaryCondition.Zero
        utw_system.j_cont_bc = 2
        utw_system._calculate_V()
        # Velocity should change sign around stagnation point
        signs = np.sign(utw_system.V[1:]) * np.sign(utw_system.V[:-1])
        assert np.any(signs <= 0)  # At least one sign change

    def test_rV_to_V_conversion(self, utw_system):
        """Test conversion between rV and V"""
        # Test planar case (alpha = 0)
        utw_system.alpha = 0
        utw_system.rV = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        utw_system.rV_to_V()
        np.testing.assert_array_equal(utw_system.V, utw_system.rV)

        # Test cylindrical case (alpha = 1)
        utw_system.alpha = 1
        utw_system.rV = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        utw_system.rV_to_V()
        # In cylindrical coordinates, V = rV/r
        expected_V = utw_system.rV[1:] / utw_system.x[1:]
        np.testing.assert_array_almost_equal(utw_system.V[1:], expected_V)

    def test_time_derivatives(self, utw_system):
        """Test computation of time derivatives"""
        # Set up test state
        utw_system.T = np.linspace(300, 700, 5)
        utw_system.U = np.zeros_like(utw_system.T)
        utw_system.Wmx = np.ones_like(utw_system.T) * utw_system.gas.mean_molecular_weight
        utw_system.r_vzero = 0.1
        utw_system.strain_rate = 100.0
        utw_system.strain_rate_deriv = 0.0
        
        # Compute derivatives
        y = utw_system.roll_y()
        ydot = utw_system.f(0.0, y)

        # Basic checks
        assert not np.any(np.isnan(ydot))
        assert len(ydot) == len(y)
        assert np.all(np.isfinite(ydot))

class TestSpeciesSystem:
    """Tests for the species transport system"""

    @pytest.fixture
    def species_system(self, simple_grid):
        """Create basic species transport system"""
        system = ConvectionSystemY()
        system.grid = simple_grid
        system.resize(simple_grid.config.n_points)
        return system

    def test_initialization(self, species_system):
        """Test basic initialization"""
        assert species_system.v is not None
        assert len(species_system.v) == 5
        assert species_system.split_const is not None
        assert np.all(species_system.split_const == 0)

    def test_velocity_interpolation(self, species_system):
        """Test velocity field interpolation"""
        # Set up interpolation data
        species_system.v_interp = {
            0.0: np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            1.0: np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        }

        # Test interpolation at midpoint
        species_system.update_v(0.5)
        expected_v = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        np.testing.assert_array_almost_equal(species_system.v, expected_v)

    def test_rhs_computation(self, species_system):
        """Test right-hand side computation"""
        # Set up test conditions
        y = np.array([0.2, 0.3, 0.4, 0.3, 0.2])  # Mass fraction profile
        species_system.v = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        species_system.Y_left = 0.1

        # Compute RHS
        ydot = species_system.f(0.0, y)

        # Check results
        assert not np.any(np.isnan(ydot))
        assert len(ydot) == len(y)
        assert np.all(np.isfinite(ydot))

class TestConvectionSplit:
    """Tests for the main convection system"""

    @pytest.fixture
    def split_system(self, simple_grid, simple_gas):
        """Create split convection system"""
        system = ConvectionSystemSplit()
        system.set_gas(simple_gas)
        n_spec = simple_gas.n_species
        state = np.zeros((3 + n_spec) * simple_grid.config.n_points)
        system.resize(simple_grid.config.n_points, n_spec, state)
        return system

    def test_initialization(self, split_system, simple_grid):
        """Test system initialization"""
        assert split_system.utw_system is not None
        assert len(split_system.species_systems) == split_system.n_spec
        assert split_system.T is not None
        assert split_system.U is not None
        assert split_system.Y is not None
        assert split_system.Y.shape[0] == split_system.n_spec
        assert split_system.Y.shape[1] == simple_grid.config.n_points

    def test_integration(self, split_system):
        """Test time integration"""
        # Set up initial conditions
        T0 = 300.0 * np.ones(split_system.n_points)
        U0 = np.zeros_like(T0)
        Y0 = np.zeros((split_system.n_spec, split_system.n_points))
        Y0[0] = 1.0  # Set first species to 1.0

        # Set initial state
        split_system.T = T0
        split_system.U = U0
        split_system.Y = Y0

        # Integrate
        t_final = 1e-5
        split_system.integrate_to_time(t_final)

        # Check results
        assert not np.any(np.isnan(split_system.T))
        assert not np.any(np.isnan(split_system.U))
        assert not np.any(np.isnan(split_system.Y))
        assert np.all(np.isfinite(split_system.T))
        assert np.all(np.isfinite(split_system.U))
        assert np.all(np.isfinite(split_system.Y))
        
        # Check mass fraction normalization
        Y_sum = np.sum(split_system.Y, axis=0)
        np.testing.assert_array_almost_equal(Y_sum, np.ones_like(Y_sum))

    def test_tolerances(self, split_system):
        """Test tolerance settings"""
        config = ConvectionConfig(
            integrator_rel_tol=1e-6,
            integrator_abs_tol_species=1e-8,
            integrator_abs_tol_momentum=1e-7,
            integrator_abs_tol_energy=1e-7
        )
        split_system.set_tolerances(config)

        assert split_system.rel_tol == 1e-6
        assert split_system.abs_tol_Y == 1e-8
        assert split_system.abs_tol_U == 1e-7
        assert split_system.abs_tol_T == 1e-7
        assert split_system.abs_tol_W == 2e-7  # 20 * species tolerance



"""
Tests for convection transport system
"""
import pytest
import numpy as np
import cantera as ct
from pyember.transport.convection import (
    ConvectionSystemUTW, ConvectionSystemY, ConvectionSystemSplit,
    BoundaryCondition, ContinuityBoundaryCondition, ConvectionConfig
)
from pyember.core.grid import OneDimGrid, GridConfig

@pytest.fixture
def simple_grid():
    """Create a simple grid for testing"""
    config = GridConfig(
        x_min=0.0,
        x_max=1.0,
        n_points=5,
        cylindrical=False,
        alpha=0,
        beta=1.0  # Add beta for strain rate
    )
    grid = OneDimGrid(config)
    grid.r_phalf = np.ones_like(grid.x)  # Add r_phalf for continuity
    return grid

@pytest.fixture
def simple_gas():
    """Create a simple gas mixture for testing"""
    gas = ct.Solution('gri30.yaml')
    gas.TPX = 300.0, ct.one_atm, 'H2:1, O2:1, N2:3.76'
    return gas

class TestUTWSystem:
    """Tests for the UTW (Velocity-Temperature-Weight) system"""

    @pytest.fixture
    def utw_system(self, simple_grid, simple_gas):
        """Create basic UTW system"""
        system = ConvectionSystemUTW()
        system.gas = simple_gas
        system.set_grid(simple_grid)
        system.resize(simple_grid.config.n_points)
        system.set_boundary_conditions(
            left_bc=BoundaryCondition.FixedValue,
            right_bc=BoundaryCondition.FixedValue,
            continuity_bc=ContinuityBoundaryCondition.Left
        )
        # Initialize required properties
        system.strain_rate = 100.0
        system.strain_rate_deriv = 0.0
        system.x_vzero = 0.5  # Set stagnation point location
        system.drho_dt = np.zeros_like(system.T)
        return system

    def test_initialization(self, utw_system):
        """Test basic initialization"""
        n = utw_system.T.shape[0]
        assert n == 5  # Check grid size
        assert utw_system.T is not None
        assert utw_system.U is not None
        assert utw_system.Wmx is not None
        assert utw_system.rho is not None
        assert utw_system.V is not None
        assert utw_system.rV is not None
        assert np.all(utw_system.split_const_T == 0)
        assert np.all(utw_system.split_const_U == 0)
        assert np.all(utw_system.split_const_W == 0)

    def test_boundary_conditions(self, utw_system):
        """Test boundary condition handling"""
        # Set up test conditions
        utw_system.T = np.array([300., 400., 500., 600., 700.])
        utw_system.U = np.zeros_like(utw_system.T)
        utw_system.Wmx = np.ones_like(utw_system.T) * utw_system.gas.mean_molecular_weight
        utw_system.r_vzero = 0.1

        # Test left boundary continuity
        utw_system.continuity_bc = ContinuityBoundaryCondition.Left
        utw_system._calculate_V()
        assert utw_system.rV[0] == 0.1  # Should match r_vzero

        # Test zero continuity
        utw_system.continuity_bc = ContinuityBoundaryCondition.Zero
        utw_system.j_cont_bc = 2
        utw_system._calculate_V()
        # Velocity should change sign around stagnation point
        signs = np.sign(utw_system.V[1:]) * np.sign(utw_system.V[:-1])
        assert np.any(signs <= 0)  # At least one sign change

    def test_time_derivatives(self, utw_system):
        """Test computation of time derivatives"""
        # Set up test state
        utw_system.T = np.linspace(300, 700, 5)
        utw_system.U = np.zeros_like(utw_system.T)
        utw_system.Wmx = np.ones_like(utw_system.T) * utw_system.gas.mean_molecular_weight
        utw_system.r_vzero = 0.1
        utw_system.strain_rate = 100.0
        utw_system.strain_rate_deriv = 0.0
        
        # Compute derivatives
        y = utw_system.roll_y()
        ydot = utw_system.f(0.0, y)

        # Basic checks
        assert not np.any(np.isnan(ydot))
        assert len(ydot) == len(y)
        assert np.all(np.isfinite(ydot))

class TestSpeciesSystem:
    """Tests for the species transport system"""

    @pytest.fixture
    def species_system(self, simple_grid):
        """Create basic species transport system"""
        system = ConvectionSystemY()
        system.grid = simple_grid
        system.resize(simple_grid.config.n_points)
        # Initialize velocity interpolation
        system.v_interp = {0.0: np.ones(simple_grid.config.n_points)}
        return system

    def test_initialization(self, species_system):
        """Test basic initialization"""
        assert species_system.v is not None
        assert len(species_system.v) == 5
        assert species_system.split_const is not None
        assert np.all(species_system.split_const == 0)

    def test_velocity_interpolation(self, species_system):
        """Test velocity field interpolation"""
        # Set up interpolation data
        species_system.v_interp = {
            0.0: np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            1.0: np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        }

        # Test interpolation at midpoint
        species_system.update_v(0.5)
        expected_v = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        np.testing.assert_array_almost_equal(species_system.v, expected_v)

class TestConvectionSplit:
    """Tests for the main convection system"""

    @pytest.fixture
    def split_system(self, simple_grid, simple_gas):
        """Create split convection system"""
        system = ConvectionSystemSplit()
        system.set_gas(simple_gas)
        # Initialize with correct state array shape
        n_points = simple_grid.config.n_points
        n_spec = simple_gas.n_species
        state = np.zeros(3 * n_points)  # Only UTW variables initially
        system.resize(n_points, n_spec, state)
        system.utw_system.t = 0.0  # Initialize time
        system.utw_system.x_vzero = 0.5  # Set stagnation point
        return system

    def test_initialization(self, split_system, simple_grid):
        """Test system initialization"""
        assert split_system.utw_system is not None
        assert len(split_system.species_systems) == split_system.n_spec
        assert split_system.T is not None
        assert split_system.U is not None
        assert split_system.Y is not None
        assert split_system.Y.shape[0] == split_system.n_spec
        assert split_system.Y.shape[1] == simple_grid.config.n_points

    def test_tolerances(self, split_system):
        """Test tolerance settings"""
        config = ConvectionConfig(
            integrator_rel_tol=1e-6,
            integrator_abs_tol_species=1e-8,
            integrator_abs_tol_momentum=1e-7,
            integrator_abs_tol_energy=1e-7
        )
        split_system.set_tolerances(config)

        assert split_system.rel_tol == 1e-6
        assert split_system.abs_tol_Y == 1e-8
        assert split_system.abs_tol_U == 1e-7
        assert split_system.abs_tol_T == 1e-7
        assert split_system.abs_tol_W == 2e-7  # 20 * species tolerance
        
        
    def test_physical_consistency(split_system, simple_grid):
        """Test physical consistency of the solution"""
        # Set up initial conditions
        center = 0.5
        width = 0.1
        x = np.linspace(0, 1, simple_grid.config.n_points)
        
        # Temperature profile (smooth transition)
        T0 = 300 + 1000 * np.exp(-(x - center)**2 / width**2)
        U0 = np.zeros_like(T0)
        Y0 = np.zeros((split_system.n_spec, simple_grid.config.n_points))
        Y0[0] = 1.0  # First species
        
        split_system.T = T0
        split_system.U = U0
        split_system.Y = Y0
        
        # Integrate for a short time
        split_system.integrate_to_time(1e-5)
        
        # Check physical constraints
        assert np.all(split_system.T > 0)  # Temperature must be positive
        assert np.all(split_system.Y >= 0)  # Mass fractions must be non-negative
        assert np.all(split_system.Y <= 1)  # Mass fractions must be <= 1
        np.testing.assert_array_almost_equal(
            np.sum(split_system.Y, axis=0),
            np.ones(simple_grid.config.n_points)
        )  # Mass fractions must sum to 1