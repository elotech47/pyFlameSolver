
"""
Complete diffusion flame solver implementation focusing on diffusion-convection and grid adaptation.
Matches C++ implementation while excluding chemistry.
"""
import numpy as np
import cantera as ct
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..core.grid import OneDimGrid, GridConfig
from ..transport.diffusion import DiffusionSystem, BoundaryCondition
from ..transport.convection import ConvectionSystemSplit, ContinuityBoundaryCondition
from ..transport.chemistry import SourceSystem
from .integrator import TridiagonalIntegrator
from .cross_system import CrossTermSystem
from .split_manager import SplitConstantsManager
from scipy.integrate import solve_ivp


@dataclass
class FlameConfig:
    """Configuration for flame solver"""
    mechanism: str
    fuel: str
    oxidizer: str
    pressure: float  # [Pa]
    T_fuel: float  # [K]
    T_oxidizer: float  # [K]
    strain_rate: float = 100.0  # [1/s]
    grid_points: int = 101
    x_min: float = -0.02  # [m]
    x_max: float = 0.02   # [m]
    regrid_time_interval: float = 1e-4
    regrid_step_interval: int = 10
    center_width: float = 0.001
    slope_width: float = 0.0005
    smooth_count: int = 4
    equilibrateCounterflow: Optional[bool] = False
    fuelLeft: Optional[bool] = True
    
class DiffusionFlame:
    """
    Solver for diffusion flame configurations.
    Implements core functionality with diffusion, convection and grid adaptation.
    """
    def __init__(self, config: FlameConfig):
        self.config = config
        
        # Initialize Cantera object
        self.gas = ct.Solution(config.mechanism)
        self.n_species = self.gas.n_species
        self.n_vars = self.n_species + 2  # T, U, Y1...YK
        # Initialize grid
        self.grid = OneDimGrid()
        self.grid.setOptions(GridConfig())
        self.grid.x = np.linspace(config.x_min, config.x_max, config.grid_points)
        self.grid.setSize(len(self.grid.x))
        self.grid.dampVal = np.ones_like(self.grid.x)
        self.grid.updateValues()
        
        # Initialize solution arrays
        self.T = np.zeros(self.grid.x.shape)
        self.Y = np.zeros((self.n_species, len(self.grid.x)))
        self.U = np.zeros_like(self.T)
        self.V = np.zeros_like(self.T)
        
        
        # Properties
        self.rho = np.zeros_like(self.T)
        self.cp = np.zeros_like(self.T)
        self.k = np.zeros_like(self.T)  # Thermal conductivity
        self.D = np.zeros((self.n_species, len(self.grid.x)))  # Species diffusion
        self.mu = np.zeros_like(self.T)  # Viscosity
        
        # Add new components
        self.source_systems = []
        self.cross_terms = CrossTermSystem(self.grid, self.n_species)
        self.split_constants = SplitConstantsManager(
            len(self.grid.x), self.n_species)
        
        # Initialize source systems
        for j in range(len(self.grid.x)):
            source = SourceSystem(self.gas)
            source.set_position(j, self.grid.x[j])
            source.P = self.config.pressure
            self.source_systems.append(source)
        
        self.U_system = DiffusionSystem(self.grid)  # Momentum
        self.T_system = DiffusionSystem(self.grid)  # Energy
        self.Y_systems = []  # Species
        for k in range(self.n_species):
            self.Y_systems.append(DiffusionSystem(self.grid))
            
        # Initialize integrators
        self.U_integrator = TridiagonalIntegrator(self.U_system)  # Momentum
        self.T_integrator = TridiagonalIntegrator(self.T_system)  # Energy
        self.Y_integrators = []  # Species
        for system in self.Y_systems:
            self.Y_integrators.append(TridiagonalIntegrator(system))

        # Strain rate parameters
        self.strain_rate = config.strain_rate
        self.strain_rate_deriv = 0.0
        
        # Initialize convection system
        self.convection = ConvectionSystemSplit(n_spec = self.n_species)
        self.convection.set_gas(self.gas)
        
        # Time tracking
        self.t = 0.0
        self.dt = 0.0
        self.t_regrid = 0.0
        self.n_regrid = 0
        
        # Initialize solution
        self.initialize()

    
    def setBoundaryValues(self, T, Y, V=None):
        
        x = self.grid.x
        n_points = len(x)
        
        jm = (n_points-1) // 2
        gas = self.gas
        gas.TPX = self.config.T_fuel, self.config.pressure, self.config.fuel
        Yfuel = gas.Y

        # Oxidizer
        gas.TPX = self.config.T_oxidizer, self.config.pressure, self.config.oxidizer
        if self.config.equilibrateCounterflow:
            gas.equilibrate(self.config.equilibrateCounterflow)
        rhou = gas.density  # use oxidizer value for diffusion flame

        Yoxidizer = gas.Y
        Toxidizer = gas.T

        if self.config.fuelLeft:
            T[0] = self.config.T_fuel
            Y[:,0] = Yfuel
            T[-1] = Toxidizer
            Y[:,-1] = Yoxidizer
        else:
            T[0] = Toxidizer
            Y[:,0] = Yoxidizer
            T[-1] = self.config.T_fuel
            Y[:,-1] = Yfuel
        print('Boundary values set.')
        return rhou
    
    def initialize(self):
        """Initialize flame structure exactly matching C++ implementation"""
        x = self.grid.x
        n_points = len(x)
        center = 0.5 * (x[0] + x[-1])
        beta = 1.0
        # Scale the profile parameters to fit domain
        scale = 0.8 * (x[-1] - x[0]) / (self.config.center_width + 
                                       2 * self.config.slope_width)
        if scale < 1.0:
            width = self.config.slope_width * scale 
            center_width = self.config.center_width  * scale
        else:
            width = self.config.slope_width
            center_width = self.config.center_width 

        # Calculate grid points for different regions
        dx = x[1] - x[0]
        center_points = int(0.5 + 0.5 * center_width / dx)
        slope_points = int(0.5 + width / dx)
        
        j_mid = n_points // 2
        j_left2 = j_mid - center_points
        j_left1 = j_left2 - slope_points
        j_right1 = j_mid + center_points
        j_right2 = j_right1 + slope_points

        # Initialize arrays
        self.T = np.zeros_like(x)
        self.Y = np.zeros((self.n_species, n_points))
        
        rhou = self.setBoundaryValues(self.T, self.Y)
        # Set up pure fuel and oxidizer states
        self.gas.TPX = self.config.T_fuel, self.config.pressure, self.config.fuel
        Y_fuel = self.gas.Y
        rho_fuel = self.gas.density
        
        self.gas.TPX = self.config.T_oxidizer, self.config.pressure, self.config.oxidizer
        Y_oxidizer = self.gas.Y
        rho_oxidizer = self.gas.density
        
        self.Y_fuel = Y_fuel
        self.Y_oxidizer = Y_oxidizer
        
        # Set stoichiometric mixture at center
        self.gas.set_equivalence_ratio(1.0, self.config.fuel, self.config.oxidizer)
        T_center = 0.5 * (self.config.T_fuel + self.config.T_oxidizer)
        self.gas.TP = T_center, self.config.pressure
        self.gas.equilibrate('HP')  # Equilibrate center mixture
        T_center = self.gas.T
        Y_center = self.gas.Y
        
        # Initialize profiles region by region
        # Left boundary to first transition
        self.T[:j_left1] = self.config.T_fuel
        self.Y[:,:j_left1] = Y_fuel[:,np.newaxis]
        
        # First slope
        ramp = np.linspace(0, 1, j_left2 - j_left1)
        self.T[j_left1:j_left2] = self.config.T_fuel + (T_center - self.config.T_fuel) * ramp
        for k in range(self.n_species):
            self.Y[k,j_left1:j_left2] = Y_fuel[k] + (Y_center[k] - Y_fuel[k]) * ramp
            
        # Center plateau
        self.T[j_left2:j_right1] = T_center
        self.Y[:,j_left2:j_right1] = Y_center[:,np.newaxis]
        
        # Second slope
        ramp = np.linspace(0, 1, j_right2 - j_right1)
        self.T[j_right1:j_right2] = T_center + (self.config.T_oxidizer - T_center) * ramp
        for k in range(self.n_species):
            self.Y[k,j_right1:j_right2] = Y_center[k] + (Y_oxidizer[k] - Y_center[k]) * ramp
            
        # Right boundary region
        self.T[j_right2:] = self.config.T_oxidizer
        self.Y[:,j_right2:] = Y_oxidizer[:,np.newaxis]
        
        # Apply smoothing - matches C++ implementation
        for _ in range(self.config.smooth_count):
            self.T = self._smooth_profile(self.T)
            self.Y = self._smooth_profile(self.Y)
        
        # Initialize velocity and density
        for j in range(n_points):
            self.gas.TPY = self.T[j], self.config.pressure, self.Y[:,j]
            self.rho[j] = self.gas.density
            self.U[j] = self.config.strain_rate / beta * np.sqrt(rhou / self.rho[j])
        
        self.U = self._smooth_profile(self.U)
        
        # Set up stagnation point flow
        j_stag = n_points // 4
        self.V = np.zeros_like(x)
        self.V[j_stag] = 0.0
        
        # Forward/backward integration of continuity equation
        for j in range(j_stag + 1, n_points):
            self.V[j] = self.V[j-1] - self.rho[j] * self.U[j] * (x[j] - x[j-1])
        for j in range(j_stag - 1, -1, -1):
            self.V[j] = self.V[j+1] + self.rho[j] * self.U[j] * (x[j+1] - x[j])
        
        # Configure convection system
        
        self.convection.set_grid(self.grid)
        self.convection.resize(n_points, self.n_species)
        
        # Set UTW system properties 
        self.convection.utw_system.r_phalf = np.ones(n_points)
        self.convection.utw_system.drho_dt = np.zeros(n_points)
        self.convection.utw_system.strain_rate = self.strain_rate
        self.convection.utw_system.strain_rate_deriv = self.strain_rate_deriv
        self.convection.utw_system.j_cont_bc = j_stag
        self.convection.utw_system.x_vzero = center
        self.convection.utw_system.P = self.config.pressure
        
        # Set boundary conditions
        self.convection.utw_system.set_boundary_conditions(
            left_bc=BoundaryCondition.FixedValue,
            right_bc=BoundaryCondition.FixedValue,
            continuity_bc=ContinuityBoundaryCondition.Zero,
            j_cont_bc=j_stag
        )
        
        # Initialize convection state
        self.convection.T = self.T.copy()
        self.convection.U = self.U.copy() 
        self.convection.Y = self.Y.copy()
        self.convection.V = self.V.copy()   
        self.convection.Wmx = np.ones(n_points) * self.gas.mean_molecular_weight
        
        self.convection.initialize_utw_system()
        
        # Initialize integrators
        self.T_integrator.set_y0(self.T)
        self.T_integrator.initialize(self.t, self.dt)
        
        self.U_integrator.set_y0(self.U)
        self.U_integrator.initialize(self.t, self.dt)
        
        for k, integrator in enumerate(self.Y_integrators):
            integrator.set_y0(self.Y[k])
            integrator.initialize(self.t, self.dt)

        # Initialize convection time
        self.convection.utw_system.t = self.t
        self.convection.utw_system.dt = self.dt
        
        # In your main solver
        self.convection.initialize_split_constants()
        
        # Update transport properties
        self.update_properties()
        
    def step(self, dt: float):
        """Step implementation matching C++ SplitSolver exactly"""
        self.dt = min(dt, 1e-3)
        
        # Setup step
        self._setup_step()
        
        # Store initial state
        state_0 = np.vstack([self.T, self.U, self.Y])
        
        self.split_constants.calculate_split_constants(
            self.dt, use_balanced=True)
        
        # First quarter diffusion
        self._setup_cross_terms()
        print("First quarter diffusion")
        self._apply_diffusion(0.25 * self.dt)  # 1/4 step
        delta_diff = np.vstack([self.T, self.U, self.Y]) - state_0
        
        # First half convection
        state_pre_conv = np.vstack([self.T, self.U, self.Y])
        print("First half convection")
        self._apply_convection(0.5 * self.dt)  # 1/2 step
        delta_conv = np.vstack([self.T, self.U, self.Y]) - state_pre_conv
        
        # Second quarter diffusion
        state_pre_diff = np.vstack([self.T, self.U, self.Y])
        self._setup_cross_terms()
        print("Second quarter diffusion")
        self._apply_diffusion(0.25 * self.dt)  # 2/4 step
        delta_diff += np.vstack([self.T, self.U, self.Y]) - state_pre_diff
        
        # Full production step
        state_pre_prod = np.vstack([self.T, self.U, self.Y])
        print("Full production step")
        self._apply_production(self.dt)  # Full step
        delta_prod = np.vstack([self.T, self.U, self.Y]) - state_pre_prod
        
        # Third quarter diffusion
        state_pre_diff = np.vstack([self.T, self.U, self.Y])
        self._setup_cross_terms()
        print("Third quarter diffusion")
        self._apply_diffusion(0.25 * self.dt)  # 3/4 step
        delta_diff += np.vstack([self.T, self.U, self.Y]) - state_pre_diff
        
        # Second half convection
        state_pre_conv = np.vstack([self.T, self.U, self.Y])
        print("Second half convection")
        self._apply_convection(0.5 * self.dt)  # 2/2 step
        delta_conv += np.vstack([self.T, self.U, self.Y]) - state_pre_conv
        
        # Final quarter diffusion
        state_pre_diff = np.vstack([self.T, self.U, self.Y])
        self._setup_cross_terms()
        print("Final quarter diffusion")
        self._apply_diffusion(0.25 * self.dt)  # 4/4 step
        delta_diff += np.vstack([self.T, self.U, self.Y]) - state_pre_diff
        
        # Update split constants
        self.split_constants.update_derivatives(
            delta_conv, delta_diff, delta_prod, self.dt)
        
            
        # Complete step
        self._finish_step()
        
    def _setup_step(self):
        """Setup before integration"""
        # Reset boundary conditions
        if self.grid.leftBC == BoundaryCondition.FixedValue:
            self.T[0] = self.config.T_fuel
            self.Y[:,0] = self.Y_fuel
            
        if self.grid.rightBC == BoundaryCondition.FixedValue:
            self.T[-1] = self.config.T_oxidizer
            self.Y[:,-1] = self.Y_oxidizer
            
        # Update properties and boundary conditions
        self.update_properties()
        self._update_boundary_conditions()
        
    def _prepare_integrators(self):
        """Setup integrators with proper time steps"""
        dx_min = np.min(self.grid.hh)
        
        # Momentum diffusion time step
        nu_max = np.max(self.mu / self.rho)  # kinematic viscosity
        dt_mom = 0.5 * dx_min**2 / nu_max
        self.U_integrator.dt = min(dt_mom, self.dt)
        
        # Temperature diffusion time step
        alpha_max = np.max(self.k / (self.rho * self.cp))  # thermal diffusivity
        dt_temp = 0.5 * dx_min**2 / alpha_max
        self.T_integrator.dt = min(dt_temp, self.dt)
        
        # Species diffusion time steps
        for k in range(self.n_species):
            D_max = np.max(self.D[k] / self.rho)  # mass diffusivity
            dt_species = 0.5 * dx_min**2 / D_max
            self.Y_integrators[k].dt = min(dt_species, self.dt)
            
    def set_convection_solver_state(self, t: float):
        """Set convection solver state matching C++"""
        self.convection.t = t
        self.convection.utw_system.T = self.T
        self.convection.utw_system.U = self.U
        self.convection.Y = self.Y
        for j in range(len(self.grid.x)):
            self.gas.TPY = self.T[j], self.config.pressure, self.Y[:,j]
            self.convection.utw_system.Wmx[j] = self.gas.mean_molecular_weight
            self.convection.utw_system.rho[j] = self.gas.density
        
    def _apply_diffusion(self, dt: float):
        """Apply diffusion step exactly matching C++ implementation"""
        # Set diffusion solver state
        self._set_diffusion_solver_state(self.t)
        
        # Set split constants from manager
        split_diff = self.split_constants.split_diff
        
        # Update diffusion terms with split constants
        for i in range(self.n_vars):
            # Energy equation
            if i == 0:
                self.T_system.split_const = split_diff[i]
                self.T_integrator.dt = dt
                self.T_integrator.step()
                self.T = self.T_integrator.get_y()
                
            # Momentum equation    
            elif i == 1:
                self.U_system.split_const = split_diff[i]
                self.U_integrator.dt = dt
                self.U_integrator.step()
                self.U = self.U_integrator.get_y()
                
            # Species equations
            else:
                k = i - 2
                self.Y_systems[k].split_const = split_diff[i]
                self.Y_integrators[k].dt = dt
                self.Y_integrators[k].step()
                self.Y[k] = self.Y_integrators[k].get_y()
                
        # Normalize mass fractions
        Y_sum = np.sum(self.Y, axis=0)
        self.Y /= Y_sum[np.newaxis, :]
        
    def _set_diffusion_solver_state(self, t: float):
        """Set diffusion solver state matching C++"""
        # Initialize each integrator
        self.T_integrator.initialize(t, self.dt)
        self.U_integrator.initialize(t, self.dt)
        
        for integrator in self.Y_integrators:
            integrator.initialize(t, self.dt)
            
        # Set current solution
        self.T_integrator.set_y0(self.T)
        self.U_integrator.set_y0(self.U)
        
        for k, integrator in enumerate(self.Y_integrators):
            integrator.set_y0(self.Y[k])
        
    def _apply_convection(self, dt: float):
        """Apply convection step exactly matching C++"""
        # Set initial state
        self.set_convection_solver_state(self.t)
        
        # Set split constants
        split_const = np.vstack([
            self.split_constants.split_conv[0],  # Temperature
            self.split_constants.split_conv[1],  # Momentum
            self.split_constants.split_conv[2:]  # Species
        ])
        self.convection.set_split_constants(split_const)
        
        # Enforce boundary conditions
        self._enforce_boundary_conditions()
        
        # Integrate
        self.convection.integrate_to_time(self.t + dt)
        
        # Update solution
        self.T = self.convection.T
        self.U = self.convection.utw_system.U
        self.Y = self.convection.Y
        self.V = self.convection.utw_system.V
        
    def _finish_step(self):
        """Complete time step with grid adaptation"""
        # Advance time
        self.t += self.dt
        
        # #Check for regridding
        # if (self.t > self.t_regrid or 
        #     self.n_regrid >= self.config.regrid_step_interval and 2==1):
            
        #     self._adapt_grid()
        #     self.t_regrid = self.t + self.config.regrid_time_interval
        #     self.n_regrid = 0
            
        self.n_regrid += 1

    
    def _adapt_grid(self):
        """Adapt grid to solution"""
        # Store current state
        x_prev = self.grid.x.copy()
        T_prev = self.T.copy()
        U_prev = self.U.copy()
        Y_prev = self.Y.copy()
        
        # Update damping values
        self._update_grid_damping()
        
        # Prepare solution for adaptation
        current_solution = [T_prev, U_prev] + [Y_prev[k] for k in range(self.n_species)]
        self.grid.nAdapt = len(current_solution)
        
        # Perform adaptation
        self.grid.adapt(current_solution)
        
        if self.grid.updated:
            # First resize all arrays to new grid size
            n_points = len(self.grid.x)
            self.T = np.zeros(n_points)
            self.U = np.zeros(n_points)
            self.Y = np.zeros((self.n_species, n_points))
            
            # Then interpolate to new grid points
            self.T[:] = np.interp(self.grid.x, x_prev, T_prev)
            self.U[:] = np.interp(self.grid.x, x_prev, U_prev)
            for k in range(self.n_species):
                self.Y[k,:] = np.interp(self.grid.x, x_prev, Y_prev[k])
                
            # Update grid metrics
            self.grid.updateValues()
            
            # Resize auxiliary arrays and systems
            self._resize_arrays()
            
            # Update properties on new grid
            self.update_properties()
            
            # Re-initialize integrators with new solution
            self.T_integrator.set_y0(self.T)
            self.T_integrator.initialize(self.t, self.dt)
            
            for k, integrator in enumerate(self.Y_integrators):
                integrator.set_y0(self.Y[k])
                integrator.initialize(self.t, self.dt)
            
    
    def _update_grid_damping(self):
        """Update grid damping values"""
        # Resize dampVal to match grid size
        self.grid.dampVal = np.zeros_like(self.grid.x)
        
        nPoints = len(self.grid.x)
        nSpec = self.D.shape[0]  # Assuming D is a 2D array with shape (nSpec, nPoints)
        
        for j in range(nPoints):
            # Calculate thermal diffusivity as k/(ρ*cp)
            num = self.k[j]/(self.rho[j] * self.cp[j])
            
            # Check each species diffusion coefficient
            for k in range(nSpec):
                if self.D[k,j] > 0:
                    num = min(num, self.D[k,j])
            
            # Calculate denominator using strain function
            den = max(abs(self.rho[j] * self.strain_rate), 1e-100)
            
            # Calculate damping value
            self.grid.dampVal[j] = np.sqrt(num/den)
        
        # Store previous damping values
        self.dampVal_prev = self.grid.dampVal.copy()
        
    def _setup_cross_terms(self):
        """Setup cross terms before diffusion"""
        # Update transport properties if needed
        self.update_properties()
        
        # Set cross term properties
        self.cross_terms.set_properties(
            rho=self.rho,
            cp=self.cp,
            cp_spec=self.cp_spec,
            W=self.gas.molecular_weights,
            lambda_=self.k
        )
        
        # Calculate cross terms
        self.cross_terms.calculate_cross_terms(
            self.T, self.Y, self.D, self.Dkt)
            
        # Update split constants for diffusion systems
        split_diff = self.split_constants.get_split_constants('diffusion')
        
        # Add cross terms to split constants
        split_diff[0] += self.cross_terms.dTdt_cross  # Temperature
        split_diff[1] += self.cross_terms.dUdt_cross  # Momentum
        split_diff[2:] += self.cross_terms.dYdt_cross  # Species
        
    def _apply_production(self, dt: float):
        """Apply chemical production step"""
        # Set source system states
        for j, system in enumerate(self.source_systems):
            system.initialize(self.T[j], self.U[j], self.Y[:,j])
            
        # Set split constants
        split_prod = self.split_constants.get_split_constants('production')
        for j, system in enumerate(self.source_systems):
            system.split_const_T = split_prod[0,j]
            system.split_const_U = split_prod[1,j]
            system.split_const_Y = split_prod[2:,j]
            
        # Integrate each point
        for j, system in enumerate(self.source_systems):
            # Create state vector
            y0 = np.concatenate([[self.U[j], self.T[j]], self.Y[:,j]])
            
            # Integrate
            sol = solve_ivp(
                system.get_rhs,
                (0, dt),
                y0,
                method='LSODA',
                rtol=1e-6,
                atol=1e-8
            )
            
            if sol.success:
                # Update solution
                self.U[j] = sol.y[0,-1]
                self.T[j] = sol.y[1,-1]
                self.Y[:,j] = sol.y[2:,-1]
            else:
                print(f"Production integration failed at point {j}")
                
    def _update_auxiliary_properties(self):
        """Update auxiliary properties needed for cross terms"""
        n_points = len(self.grid.x)
        
        # Initialize arrays if needed
        if not hasattr(self, 'cp_spec'):
            self.cp_spec = np.zeros((self.n_species, n_points))
        if not hasattr(self, 'Dkt'):
            self.Dkt = np.zeros((self.n_species, n_points))
            
        # Calculate properties at each point
        for j in range(n_points):
            self.gas.TPY = self.T[j], self.config.pressure, self.Y[:,j]
            
            # Species specific heats
            self.cp_spec[:,j] = self.gas.partial_molar_cp / self.gas.molecular_weights
            
            # Thermal diffusion coefficients
            self.Dkt[:,j] = self.gas.thermal_diff_coeffs
            
        
    def _resize_arrays(self):
        """Resize arrays after grid adaptation"""
        super()._resize_arrays()
        
        # Resize new components
        n_points = len(self.grid.x)
        
        # Resize cross terms
        self.cross_terms.resize(n_points, self.n_species)
        
        # Resize split constants
        self.split_constants.resize(n_points, self.n_species)
        
        # Update source systems
        self.source_systems = []
        for j in range(n_points):
            source = SourceSystem(self.gas)
            source.set_position(j, self.grid.x[j])
            source.P = self.config.pressure
            self.source_systems.append(source)
            
    def update_properties(self):
        """Update transport and thermodynamic properties"""
        for j in range(len(self.grid.x)):
            # Set state
            self.gas.TPY = self.T[j], self.config.pressure, self.Y[:,j]
            
            # Get properties
            self.rho[j] = self.gas.density
            self.cp[j] = self.gas.cp_mass
            self.k[j] = self.gas.thermal_conductivity
            self.mu[j] = self.gas.viscosity  # Get viscosity
            self.D[:,j] = self.gas.mix_diff_coeffs_mass * self.rho[j]
            
        # Update momentum diffusion system
        self.U_system.set_properties(
            D=self.mu,           # D = μ (viscosity)
            rho=self.rho,
            B=1.0/self.rho      # B = 1/ρ
        )
        
        # Update energy (temperature) diffusion system
        self.T_system.set_properties(
            D=self.k,            # D = k (thermal conductivity)
            rho=self.rho,
            B=1.0/(self.rho * self.cp)  # B = 1/(ρcp)
        )
        
        # Update species diffusion systems
        for k, system in enumerate(self.Y_systems):
            system.set_properties(
                D=self.D[k],     # D = ρD (mass diffusion coefficient)
                rho=self.rho,
                B=1.0/self.rho   # B = 1/ρ
            )
            
        # Additional properties for cross terms
        self._update_auxiliary_properties()
            
    def _smooth_profile(self, y: np.ndarray) -> np.ndarray:
        """Apply smoothing to profile"""
        y_smooth = y.copy()
        if y.ndim == 1:
            # Continuing from _smooth_profile method:
            for i in range(1, len(y)-1):
                y_smooth[i] = 0.25 * y[i-1] + 0.5 * y[i] + 0.25 * y[i+1]
        else:
            for i in range(1, y.shape[1]-1):
                y_smooth[:,i] = 0.25 * y[:,i-1] + 0.5 * y[:,i] + 0.25 * y[:,i+1]
        return y_smooth
        
    def _update_boundary_conditions(self):
        """Update boundary conditions for all systems"""
        # Left boundary
        self.grid.leftBC = BoundaryCondition.FixedValue
        
        # Set boundary conditions for all diffusion systems
        self.U_system.set_boundary_conditions(
            self.grid.leftBC,
            BoundaryCondition.FixedValue
        )
        
        self.T_system.set_boundary_conditions(
            self.grid.leftBC,
            BoundaryCondition.FixedValue
        )
        
        for system in self.Y_systems:
            system.set_boundary_conditions(
                self.grid.leftBC,
                BoundaryCondition.FixedValue
            )
            
    def _enforce_boundary_conditions(self):
        """Enforce boundary values"""
        # Temperature boundaries
        self.T[0] = self.config.T_fuel
        self.T[-1] = self.config.T_oxidizer
        
        # Species boundaries
        self.Y[:,0] = self.Y_fuel
        self.Y[:,-1] = self.Y_oxidizer
        
        # Update convection system
        self.convection.T = self.T
        self.convection.Y = self.Y
        
    def _resize_arrays(self):
        """Resize arrays after grid adaptation"""
        n_points = len(self.grid.x)
        
        # Resize property arrays
        self.rho = np.zeros(n_points)
        self.cp = np.zeros(n_points)
        self.k = np.zeros(n_points)
        self.D = np.zeros((self.n_species, n_points))
        
        # Resize diffusion systems
        self.T_system.resize(n_points)
        for system in self.Y_systems:
            system.resize(n_points)
        
        # Update integrator sizes
        self.T_integrator.resize(n_points)
        for integrator in self.Y_integrators:
            integrator.resize(n_points)
            
        # Update convection system
        self.convection.resize(n_points, self.n_species)
        self.convection.set_grid(self.grid)  # Important: update grid reference
    
        
    def _enforce_physical_constraints(self):
        """Enforce physical constraints on solution"""
        # Mass fraction constraints
        self.Y = np.maximum(self.Y, 0)  # Non-negative
        Y_sum = np.sum(self.Y, axis=0)
        self.Y /= Y_sum[np.newaxis, :]  # Normalize
        
        # Temperature constraints
        T_min = min(self.config.T_fuel, self.config.T_oxidizer)
        T_max = max(self.config.T_fuel, self.config.T_oxidizer) * 2.0
        self.T = np.clip(self.T, T_min, T_max)
        
        # Enforce monotonicity at stagnation point
        j_stag = self.convection.utw_system.j_cont_bc
        
        # Fuel side
        for j in range(j_stag):
            for k in range(self.n_species):
                if self.Y_fuel[k] > self.Y_oxidizer[k]:
                    self.Y[k,j] = np.clip(self.Y[k,j], self.Y[k,j+1], self.Y_fuel[k])
                else:
                    self.Y[k,j] = np.clip(self.Y[k,j], 0, self.Y[k,j+1])
                    
        # Oxidizer side
        for j in range(j_stag+1, len(self.grid.x)):
            for k in range(self.n_species):
                if self.Y_oxidizer[k] > self.Y_fuel[k]:
                    self.Y[k,j] = np.clip(self.Y[k,j], self.Y[k,j-1], self.Y_oxidizer[k])
                else:
                    self.Y[k,j] = np.clip(self.Y[k,j], 0, self.Y[k,j-1])
                    
    def get_solution(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current solution state"""
        return self.T, self.Y, self.U