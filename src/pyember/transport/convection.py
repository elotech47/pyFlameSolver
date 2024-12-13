"""
ConvectionSystem: Handles convective transport in flame solver.
"""
import numpy as np
from ..core.base import TransportComponent
from ..core.grid import OneDimGrid
import cantera as ct
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import enum
from concurrent.futures import ThreadPoolExecutor
import time

class BoundaryCondition(enum.Enum):
    """Boundary condition types"""
    FixedValue = "fixed_value"
    ZeroGradient = "zero_gradient"
    WallFlux = "wall_flux"
    ControlVolume = "control_volume"

class ContinuityBoundaryCondition(enum.Enum):
    """Boundary conditions for continuity equation"""
    Left = "left"
    Right = "right"
    Zero = "zero"
    Temp = "temp"
    Qdot = "qdot"

@dataclass
class ConvectionConfig:
    """Configuration for convection system"""
    integrator_rel_tol: float = 1e-8
    integrator_abs_tol_species: float = 1e-8
    integrator_abs_tol_momentum: float = 1e-8 
    integrator_abs_tol_energy: float = 1e-8

class ConvectionSystemUTW:
    """
    Velocity-Temperature-Weight (UTW) system solver.
    Handles the coupled equations for velocity, temperature and molecular weight.
    """
    def __init__(self):
        self.gas = None
        self.continuity_bc = ContinuityBoundaryCondition.Left
        self.j_cont_bc = 0
        self.n_vars = 3
        
        # State variables
        self.T = None  # Temperature
        self.U = None  # Velocity
        self.Wmx = None  # Mean molecular weight
        
        # Grid properties
        self.x = None
        self.beta = None
        self.alpha = None
        self.hh = None  # Grid spacing
        self.r_phalf = None
        
        # Boundary conditions
        self.T_left = None
        self.W_left = None
        self.r_vzero = None
        self.x_vzero = None
        
        # Derived quantities
        self.rho = None  # Density
        self.V = None   # Mass flux
        self.rV = None  # r*V (cylindrical/spherical coordinates)
        
        # Time derivatives
        self.drho_dt = None
        self.dT_dt = None
        self.dU_dt = None
        self.dW_dt = None
        
        # Split terms
        self.split_const_T = None
        self.split_const_U = None
        self.split_const_W = None

    def resize(self, n_points: int):
        """Initialize arrays with new size"""
        self.T = np.zeros(n_points)
        self.U = np.zeros(n_points)
        self.Wmx = np.zeros(n_points)
        self.rho = np.zeros(n_points)
        self.V = np.zeros(n_points)
        self.rV = np.zeros(n_points)
        self.dT_dt = np.zeros(n_points)
        self.dU_dt = np.zeros(n_points)
        self.dW_dt = np.zeros(n_points)
        self.drho_dt = np.zeros(n_points)
        
        # Initialize split constants
        self.reset_split_constants()

    def reset_split_constants(self):
        """Reset split constants to zero"""
        if self.T is not None:
            n_points = len(self.T)
            self.split_const_T = np.zeros(n_points)
            self.split_const_U = np.zeros(n_points)
            self.split_const_W = np.zeros(n_points)

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the UTW system ODEs.
        
        Args:
            t: Current time
            y: State vector [T, U, Wmx]
            
        Returns:
            Time derivatives [dT/dt, dU/dt, dWmx/dt]
        """
        # Unroll state vector
        self.unroll_y(y)
        
        # Update density
        self.rho = self.gas.pressure * self.Wmx / (self.gas.gas_constant * self.T)
        
        # Calculate V based on continuity equation
        self._calculate_V()
        
        # Calculate upwinded derivatives
        dT_dx = np.zeros_like(self.T)
        dU_dx = np.zeros_like(self.U)
        dW_dx = np.zeros_like(self.Wmx)
        
        for j in range(self.n_points - 1):
            if self.rV[j] < 0 or j == 0:
                dT_dx[j] = (self.T[j+1] - self.T[j]) / self.hh[j]
                dU_dx[j] = (self.U[j+1] - self.U[j]) / self.hh[j]
                dW_dx[j] = (self.Wmx[j+1] - self.Wmx[j]) / self.hh[j]
            else:
                dT_dx[j] = (self.T[j] - self.T[j-1]) / self.hh[j-1]
                dU_dx[j] = (self.U[j] - self.U[j-1]) / self.hh[j-1]
                dW_dx[j] = (self.Wmx[j] - self.Wmx[j-1]) / self.hh[j-1]

        # Calculate time derivatives
        self._calculate_time_derivatives(dT_dx, dU_dx, dW_dx)
        
        return self.roll_ydot()

    def _calculate_V(self):
        """Calculate velocity field based on continuity equation"""
        if self.continuity_bc == ContinuityBoundaryCondition.Left:
            self.rV[0] = self.r_vzero
            for j in range(self.n_points - 1):
                self.rV[j+1] = (self.rV[j] - self.hh[j] * self.r_phalf[j] * 
                               (self.drho_dt[j] + self.rho[j] * self.beta * self.U[j]))
                               
        elif self.continuity_bc == ContinuityBoundaryCondition.Zero:
            # Implementation for stagnation point handling
            j = self.j_cont_bc
            d_vdx0 = -self.r_phalf[j] * (self.drho_dt[j] - self.rho[j] * self.beta * self.U[j])
            
            if j != 0:
                d_vdx0 = (0.5 * d_vdx0 - 0.5 * self.r_phalf[j-1] * 
                         (self.drho_dt[j-1] + self.rho[j-1] * self.beta * self.U[j-1]))
            
            self.rV[j] = (self.x[j] - self.x_vzero) * d_vdx0
            
            # Forward integration
            for j in range(self.j_cont_bc, self.n_points - 1):
                self.rV[j+1] = (self.rV[j] - self.hh[j] * self.r_phalf[j] * 
                               (self.drho_dt[j] + self.rho[j] * self.beta * self.U[j]))
            
            # Backward integration if needed
            if self.j_cont_bc != 0:
                for j in range(self.j_cont_bc - 1, 0, -1):
                    self.rV[j-1] = (self.rV[j] + self.hh[j-1] * self.r_phalf[j-1] * 
                                   (self.drho_dt[j-1] + self.rho[j-1] * self.beta * self.U[j-1]))
        
        self.rV_to_V()

    def _calculate_time_derivatives(self, dT_dx, dU_dx, dW_dx):
        """Calculate time derivatives for UTW system"""
        # Left boundary conditions
        self.dU_dt[0] = (self.split_const_U[0] - self.U[0]**2 + 
                        self.rho[0]/self.rho[0] * (self.strain_rate_deriv/self.beta + 
                        self.strain_rate**2/self.beta**2))
        
        if self.left_bc in [BoundaryCondition.ControlVolume, BoundaryCondition.WallFlux]:
            center_vol = self.x[1]**(self.alpha + 1) / (self.alpha + 1)
            r_vzero_mod = max(self.rV[0], 0.0)
            
            self.dT_dt[0] = (-r_vzero_mod * (self.T[0] - self.T_left) / 
                            (self.rho[0] * center_vol) + self.split_const_T[0])
            self.dW_dt[0] = (-r_vzero_mod * (self.Wmx[0] - self.W_left) / 
                            (self.rho[0] * center_vol) - self.Wmx[0]**2 * self.split_const_W[0])
        else:
            self.dT_dt[0] = self.split_const_T[0]
            self.dW_dt[0] = -self.Wmx[0]**2 * self.split_const_W[0]
        
        # Interior points
        for j in range(1, self.n_points - 1):
            self.dU_dt[j] = (-self.V[j] * dU_dx[j] / self.rho[j] - self.U[j]**2 +
                            self.rho[0]/self.rho[j] * (self.strain_rate_deriv/self.beta + 
                            self.strain_rate**2/self.beta**2) + self.split_const_U[j])
            self.dT_dt[j] = -self.V[j] * dT_dx[j] / self.rho[j] + self.split_const_T[j]
            self.dW_dt[j] = (-self.V[j] * dW_dx[j] / self.rho[j] - 
                            self.Wmx[j]**2 * self.split_const_W[j])

        # Right boundary
        j = self.n_points - 1
        if self.rV[j] < 0 or self.right_bc == BoundaryCondition.FixedValue:
            self.dU_dt[j] = (self.split_const_U[j] - self.U[j]**2 +
                            self.rho[0]/self.rho[j] * (self.strain_rate_deriv/self.beta + 
                            self.strain_rate**2/self.beta**2))
            self.dT_dt[j] = self.split_const_T[j]
            self.dW_dt[j] = -self.Wmx[j]**2 * self.split_const_W[j]
        else:
            self.dU_dt[j] = (self.split_const_U[j] - 
                            self.V[j] * (self.U[j] - self.U[j-1])/self.hh[j-1]/self.rho[j] -
                            self.U[j]**2 + self.rho[0]/self.rho[j] * 
                            (self.strain_rate_deriv/self.beta + self.strain_rate**2/self.beta**2))
            self.dT_dt[j] = (self.split_const_T[j] - 
                            self.V[j] * (self.T[j] - self.T[j-1])/self.hh[j-1]/self.rho[j])
            self.dW_dt[j] = (-self.Wmx[j]**2 * self.split_const_W[j] - 
                            self.V[j] * (self.Wmx[j] - self.Wmx[j-1])/self.hh[j-1]/self.rho[j])

    def unroll_y(self, y: np.ndarray):
        """Extract state variables from solution vector"""
        n = len(y) // 3
        self.T = y[:n]
        self.U = y[n:2*n]
        self.Wmx = y[2*n:]

    def roll_y(self) -> np.ndarray:
        """Combine state variables into solution vector"""
        return np.concatenate([self.T, self.U, self.Wmx])

    def roll_ydot(self) -> np.ndarray:
        """Combine time derivatives into solution vector"""
        return np.concatenate([self.dT_dt, self.dU_dt, self.dW_dt])

    def rV_to_V(self):
        """Convert rV to V accounting for coordinate system"""
        self.V[0] = self.rV[0]
        if self.alpha == 0:
            self.V = self.rV.copy()
        else:
            self.V[1:] = self.rV[1:] / self.x[1:]
            
    def set_gas(self, gas: ct.Solution):
        """Set gas object for thermodynamic properties"""
        self.gas = gas
        
    def set_grid(self, grid: OneDimGrid):
        """Set grid properties"""
        self.x = grid.x
        self.hh = grid.hh
        self.n_points = grid.n_points
        self.r_phalf = grid.r_phalf
        self.alpha = grid.alpha
        self.beta = grid.beta
        
    def set_boundary_conditions(self, left_bc: BoundaryCondition, right_bc: BoundaryCondition,
                                continuity_bc: ContinuityBoundaryCondition = ContinuityBoundaryCondition.Left,
                                j_cont_bc: int = 0):
        """Set boundary conditions"""
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.continuity_bc = continuity_bc
        self.j_cont_bc = j_cont_bc
        

class ConvectionSystemY:
    """
    Species transport system solver.
    Handles the convective transport of species mass fractions.
    """
    def __init__(self):
        self.v = None  # Velocity field
        self.v_interp = {}  # Time-interpolated velocity fields
        self.split_const = None
        self.Y_left = None
        self.quasi2d = False
        self.vz_interp = None  # For quasi-2D simulations
        self.vr_interp = None  # For quasi-2D simulations
        self.grid = None
        self.k = None  # Species index

    def resize(self, n_points: int):
        """Initialize arrays with new size"""
        self.v = np.zeros(n_points)
        self.split_const = np.zeros(n_points)

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the species transport equation.
        
        Args:
            t: Current time
            y: Species mass fraction
            
        Returns:
            Time derivative dY/dt
        """
        y_dot = np.zeros_like(y)
        n_points = len(y)
        
        # Get interpolated velocity field
        if not self.quasi2d:
            self.update_v(t)
        
        # Left boundary conditions
        if self.grid.left_bc in [BoundaryCondition.ControlVolume, BoundaryCondition.WallFlux]:
            center_vol = self.grid.x[1]**(self.grid.alpha + 1) / (self.grid.alpha + 1)
            v_zero_mod = max(self.v[0], 0.0)
            y_dot[0] = -v_zero_mod * (y[0] - self.Y_left) / center_vol + self.split_const[0]
        else:
            y_dot[0] = self.split_const[0]

        # Interior points
        for j in range(1, n_points - 1):
            if self.v[j] < 0:
                dy_dx = (y[j+1] - y[j]) / self.grid.hh[j]
            else:
                dy_dx = (y[j] - y[j-1]) / self.grid.hh[j-1]
                
            if self.quasi2d:
                y_dot[j] = (-self.vr_interp(self.grid.x[j], t) * dy_dx / 
                           self.vz_interp(self.grid.x[j], t) + self.split_const[j])
            else:
                y_dot[j] = -self.v[j] * dy_dx + self.split_const[j]

        # Right boundary
        j = n_points - 1
        if self.v[j] < 0 or self.grid.right_bc == BoundaryCondition.FixedValue:
            y_dot[j] = self.split_const[j]
        else:
            if self.quasi2d:
                y_dot[j] = (self.split_const[j] - 
                           self.vr_interp(self.grid.x[j], t) * 
                           (y[j] - y[j-1]) / self.grid.hh[j-1] / 
                           self.vz_interp(self.grid.x[j], t))
            else:
                y_dot[j] = (self.split_const[j] - 
                           self.v[j] * (y[j] - y[j-1]) / self.grid.hh[j-1])

        return y_dot

    def update_v(self, t: float):
        """Update velocity field using time interpolation"""
        if len(self.v_interp) == 1:
            # Only one time point available
            self.v = next(iter(self.v_interp.values()))
            return

        # Find neighboring time points
        times = sorted(self.v_interp.keys())
        idx = np.searchsorted(times, t)
        
        if idx == 0:
            # Extrapolate before first point
            t1, t2 = times[0], times[1]
            v1, v2 = self.v_interp[t1], self.v_interp[t2]
        elif idx == len(times):
            # Extrapolate after last point
            t1, t2 = times[-2], times[-1]
            v1, v2 = self.v_interp[t1], self.v_interp[t2]
        else:
            # Interpolate between points
            t1, t2 = times[idx-1], times[idx]
            v1, v2 = self.v_interp[t1], self.v_interp[t2]

        # Linear interpolation
        s = (t - t1) / (t2 - t1)
        self.v = v1 * (1 - s) + v2 * s


class ConvectionSystemSplit:
    """
    Main convection system that couples UTW and species transport.
    Uses operator splitting to solve the systems separately.
    """
    def __init__(self):
        self.U = None  # Velocity
        self.T = None  # Temperature
        self.Y = None  # Species mass fractions
        self.Wmx = None  # Mean molecular weight
        self.dWdt = None  # Time derivative of mean molecular weight
        
        self.utw_system = ConvectionSystemUTW()
        self.species_systems = []
        self.n_spec = 0
        self.n_vars = 3
        self.gas = None
        self.quasi2d = False
        
        # Integration parameters
        self.rel_tol = 1e-8
        self.abs_tol_U = 1e-8
        self.abs_tol_T = 1e-8
        self.abs_tol_W = 1e-7
        self.abs_tol_Y = 1e-8

        # Timing
        self.utw_timer = time.time
        self.species_timer = time.time

    def set_gas(self, gas: Any):
        """Set the gas object and update molecular weights"""
        self.gas = gas
        self.utw_system.gas = gas
        if self.n_spec > 0:
            self.W = np.array(gas.molecular_weights)

    def resize(self, n_points: int, n_spec: int, state: np.ndarray):
        """
        Resize the system for new grid size or number of species.
        
        Args:
            n_points: Number of grid points
            n_spec: Number of species
            state: Current state matrix
        """
        self.n_points = n_points
        
        # Update species systems if needed
        if self.n_spec != n_spec:
            self.species_systems = [ConvectionSystemY() for _ in range(n_spec)]
            self.n_spec = n_spec
            if self.gas:
                self.W = np.array(self.gas.molecular_weights)

        # Reshape state arrays
        self.T = state[self.n_vars::self.n_vars]
        self.U = state[self.n_vars+1::self.n_vars]
        self.Y = state[self.n_vars+2:].reshape(n_spec, n_points)
        
        # Initialize systems
        self.utw_system.resize(n_points)
        for system in self.species_systems:
            system.resize(n_points)
            
        self.Wmx = np.zeros(n_points)
        self.dWdt = np.zeros(n_points)

    def set_tolerances(self, config: ConvectionConfig):
        """Set integration tolerances"""
        self.rel_tol = config.integrator_rel_tol
        self.abs_tol_U = config.integrator_abs_tol_momentum
        self.abs_tol_T = config.integrator_abs_tol_energy
        self.abs_tol_W = config.integrator_abs_tol_species * 20
        self.abs_tol_Y = config.integrator_abs_tol_species

    def set_state(self, t_initial: float):
        """Initialize state vectors for integration"""
        # Set UTW system state
        utw_y0 = np.zeros(3 * self.n_points)
        for j in range(self.n_points):
            utw_y0[3*j] = self.T[j]
            utw_y0[3*j+1] = self.U[j]
            self.gas.TPY = self.T[j], self.gas.P, self.Y[:,j]
            utw_y0[3*j+2] = self.Wmx[j] = self.gas.mean_molecular_weight
            
        # Set species system states
        species_y0 = [self.Y[k,:] for k in range(self.n_spec)]
        
        return t_initial, utw_y0, species_y0

    def integrate_to_time(self, tf: float):
        """
        Integrate the split system to final time tf.
        
        Args:
            tf: Final time
        """
        t0, utw_y0, species_y0 = self.set_state(self.utw_system.t)
        
        # Integrate UTW system
        self.utw_timer = time.time()
        utw_sol = solve_ivp(
            self.utw_system.f,
            (t0, tf),
            utw_y0,
            method='BDF',
            rtol=self.rel_tol,
            atol=[self.abs_tol_T]*self.n_points + 
                 [self.abs_tol_U]*self.n_points + 
                 [self.abs_tol_W]*self.n_points
        )
        
        # Store velocity field for species integration
        times = utw_sol.t
        for i, t in enumerate(times):
            y = utw_sol.y[:,i]
            self.utw_system.unroll_y(y)
            self.species_systems[0].v_interp[t] = self.utw_system.V/self.utw_system.rho
            
        # Integrate species equations in parallel
        self.species_timer = time.time()
        with ThreadPoolExecutor() as executor:
            futures = []
            for k, (system, y0) in enumerate(zip(self.species_systems, species_y0)):
                system.v_interp = self.species_systems[0].v_interp
                future = executor.submit(
                    solve_ivp,
                    system.f,
                    (t0, tf),
                    y0,
                    method='BDF',
                    rtol=self.rel_tol,
                    atol=[self.abs_tol_Y]*self.n_points
                )
                futures.append(future)
                
            # Get results
            species_sols = [f.result() for f in futures]
            
        # Update state
        self.utw_system.unroll_y(utw_sol.y[:,-1])
        for k, sol in enumerate(species_sols):
            self.Y[k,:] = sol.y[:,-1]

    def update_continuity_boundary_condition(self, qdot: np.ndarray, new_bc: ContinuityBoundaryCondition):
        """Update boundary condition for continuity equation"""
        self.utw_system.update_continuity_boundary_condition(qdot, new_bc)

    def set_density_derivative(self, drho_dt: np.ndarray):
        """Set density time derivative"""
        self.utw_system.drho_dt = drho_dt

    def reset_split_constants(self):
        """Reset all split constants to zero"""
        self.utw_system.reset_split_constants()
        for system in self.species_systems:
            system.reset_split_constants()

    def set_split_constants(self, split_const: np.ndarray):
        """Set split constants for all equations"""
        self.utw_system.split_const_T = split_const[self.n_vars,:]
        self.utw_system.split_const_U = split_const[self.n_vars+1,:]
        
        # Convert species split constants to molar form
        self.utw_system.split_const_W[:] = 0
        for j in range(self.n_points):
            for k in range(self.n_spec):
                self.utw_system.split_const_W[j] += split_const[self.n_vars+2+k,j] / self.W[k]
                
        # Set species split constants
        for k in range(self.n_spec):
            self.species_systems[k].split_const = split_const[self.n_vars+2+k,:]