# """
# ConvectionSystem: Handles convective transport in flame solver.
# """
# import numpy as np
# from ..core.base import TransportComponent
# from ..core.grid import OneDimGrid
# import cantera as ct
# from scipy.integrate import solve_ivp
# from dataclasses import dataclass
# from typing import Optional, Dict, List, Tuple, Any
# import enum
# from concurrent.futures import ThreadPoolExecutor
# import time
# from scipy.integrate import ode
# import multiprocessing as mp
# from functools import partial

# class BoundaryCondition(enum.Enum):
#     """Boundary condition types"""
#     FixedValue = "fixed_value" 
#     ZeroGradient = "zero_gradient"
#     WallFlux = "wall_flux"
#     ControlVolume = "control_volume"

# class ContinuityBoundaryCondition(enum.Enum):
#     """Boundary conditions for continuity equation"""
#     Left = "left"
#     Right = "right"
#     Zero = "zero"
#     Temp = "temp"
#     Qdot = "qdot"

# @dataclass
# class ConvectionConfig:
#     """Configuration for convection system"""
#     integrator_rel_tol: float = 1e-8
#     integrator_abs_tol_species: float = 1e-8
#     integrator_abs_tol_momentum: float = 1e-8 
#     integrator_abs_tol_energy: float = 1e-8

# class ConvectionSystemUTW:
#     """
#     Velocity-Temperature-Weight (UTW) system solver.
#     Handles the coupled equations for velocity, temperature and molecular weight.
#     """
#     def __init__(self):
#         self.gas = None
#         self.continuity_bc = ContinuityBoundaryCondition.Left
#         self.j_cont_bc = 0
#         self.n_vars = 3
        
#         # State variables
#         self.T = None  # Temperature
#         self.U = None  # Velocity
#         self.Wmx = None  # Mean molecular weight
        
#         # Grid properties
#         self.grid = None
#         self.x = None
#         self.beta = None
#         self.alpha = None
#         self.hh = None  # Grid spacing
#         self.r_phalf = None
#         self.n_points = None
        
#         # Boundary conditions
#         self.T_left = None
#         self.W_left = None
#         self.r_vzero = None
#         self.x_vzero = None
        
#         # Derived quantities
#         self.rho = None  # Density
#         self.V = None   # Mass flux
#         self.rV = None  # r*V (cylindrical/spherical coordinates)
        
#         # Time derivatives
#         self.drho_dt = None
#         self.dT_dt = None
#         self.dU_dt = None
#         self.dW_dt = None
        
#         # Split terms
#         self.split_const_T = None
#         self.split_const_U = None
#         self.split_const_W = None
        
#         # Integration time
#         self.t = 0.0
        
#         # Strain rate properties
#         self.strain_rate = 0.0
#         self.strain_rate_deriv = 0.0
    
#     def set_grid(self, grid: OneDimGrid):
#         """Set grid properties"""
#         self.grid = grid  # Store grid object
#         self.x = grid.x
#         self.hh = grid.hh
#         self.n_points = len(grid.x)
#         if hasattr(grid, 'r_phalf'):
#             self.r_phalf = grid.r_phalf
#         else:
#             self.r_phalf = np.ones_like(grid.x)  # Default for planar grid
#         self.alpha = grid.alpha if hasattr(grid, 'alpha') else 0
#         self.beta = grid.beta if hasattr(grid, 'beta') else 1.0

#     def resize(self, n_points: int):
#         """Initialize arrays with new size"""
#         self.T = np.zeros(n_points)
#         self.U = np.zeros(n_points)
#         self.Wmx = np.zeros(n_points)
#         self.rho = np.zeros(n_points)
#         self.V = np.zeros(n_points)
#         self.rV = np.zeros(n_points)
#         self.dT_dt = np.zeros(n_points)
#         self.dU_dt = np.zeros(n_points)
#         self.dW_dt = np.zeros(n_points)
#         self.drho_dt = np.zeros(n_points)
        
#         # Initialize split constants
#         self.reset_split_constants()

#     def reset_split_constants(self):
#         """Reset split constants to zero"""
#         if self.T is not None:
#             n_points = len(self.T)
#             self.split_const_T = np.zeros(n_points)
#             self.split_const_U = np.zeros(n_points)
#             self.split_const_W = np.zeros(n_points)

#     def f(self, t: float, y: np.ndarray) -> np.ndarray:
#         """
#         Right-hand side of the UTW system ODEs.
        
#         Args:
#             t: Current time
#             y: State vector [T, U, Wmx]
            
#         Returns:
#             Time derivatives [dT/dt, dU/dt, dWmx/dt]
#         """
#         # Unroll state vector
#         self.unroll_y(y)
        
#         # Update density
#         self.rho = self.gas.P * self.Wmx / (ct.gas_constant * self.T)
        
#         # Calculate V based on continuity equation
#         self._calculate_V()
        
#         # Calculate upwinded derivatives
#         dT_dx = np.zeros_like(self.T)
#         dU_dx = np.zeros_like(self.U)
#         dW_dx = np.zeros_like(self.Wmx)
        
#         for j in range(self.n_points - 1):
#             if self.rV[j] < 0 or j == 0:
#                 dT_dx[j] = (self.T[j+1] - self.T[j]) / self.hh[j]
#                 dU_dx[j] = (self.U[j+1] - self.U[j]) / self.hh[j]
#                 dW_dx[j] = (self.Wmx[j+1] - self.Wmx[j]) / self.hh[j]
#             else:
#                 dT_dx[j] = (self.T[j] - self.T[j-1]) / self.hh[j-1]
#                 dU_dx[j] = (self.U[j] - self.U[j-1]) / self.hh[j-1]
#                 dW_dx[j] = (self.Wmx[j] - self.Wmx[j-1]) / self.hh[j-1]

#         # Calculate time derivatives
#         self._calculate_time_derivatives(dT_dx, dU_dx, dW_dx)
        
#         return self.roll_ydot()
    
#     def _calculate_V(self):
#         """
#         Calculate velocity field based on continuity equation with complete boundary condition handling.
#         Matches C++ implementation including temperature-based BCs.
#         """
#         if self.continuity_bc == ContinuityBoundaryCondition.Zero:
#             # Find stagnation point location 
#             j = self.j_cont_bc
            
#             # Calculate velocity gradient at stagnation point
#             d_vdx0 = -self.r_phalf[j] * (self.drho_dt[j] - self.rho[j] * self.beta * self.U[j])
            
#             if j != 0:
#                 # Average with previous point if not at boundary
#                 d_vdx0 = (0.5 * d_vdx0 - 0.5 * self.r_phalf[j-1] * 
#                          (self.drho_dt[j-1] + self.rho[j-1] * self.beta * self.U[j-1]))

#             # Set stagnation point velocity
#             self.rV[j] = (self.x[j] - self.x_vzero) * d_vdx0

#             # Forward integration (right side of stagnation point)
#             for j in range(self.j_cont_bc, self.n_points - 1):
#                 self.rV[j+1] = (self.rV[j] - self.hh[j] * self.r_phalf[j] * 
#                                (self.drho_dt[j] + self.rho[j] * self.beta * self.U[j]))

#             # Backward integration (left side of stagnation point)
#             if self.j_cont_bc != 0:
#                 j = self.j_cont_bc - 1
#                 self.rV[j] = (self.x[j] - self.x_vzero) * d_vdx0
#                 for j in range(self.j_cont_bc - 1, 0, -1):
#                     self.rV[j-1] = (self.rV[j] + self.hh[j-1] * self.r_phalf[j-1] * 
#                                   (self.drho_dt[j-1] + self.rho[j-1] * self.beta * self.U[j-1]))
        
#         elif self.continuity_bc == ContinuityBoundaryCondition.Temp:
#             # Temperature-based boundary condition (missing from original)
#             j = self.j_cont_bc
            
#             # Find rV[j] that makes dTdt[j] = 0
#             if j == 0 or self.split_const_T[j] / (self.T[j+1] - self.T[j]) < 0:
#                 self.rV[j] = (self.r_phalf[j] * self.rho[j] * self.split_const_T[j] * 
#                              self.hh[j] / (self.T[j+1] - self.T[j]))
#             else:
#                 self.rV[j] = (self.r_phalf[j-1] * self.rho[j] * self.split_const_T[j] * 
#                              self.hh[j-1] / (self.T[j] - self.T[j-1]))
            
#             # Forward integration
#             for j in range(self.j_cont_bc, self.n_points - 1):
#                 self.rV[j+1] = (self.rV[j] - self.hh[j] * self.r_phalf[j] * 
#                                (self.drho_dt[j] + self.rho[j] * self.beta * self.U[j]))
            
#             # Backward integration
#             for j in range(self.j_cont_bc, 0, -1):
#                 self.rV[j-1] = (self.rV[j] + self.hh[j-1] * self.r_phalf[j-1] * 
#                                (self.drho_dt[j-1] + self.rho[j-1] * self.beta * self.U[j-1]))
                
#         else:  # Left or Right boundary conditions
#             self.rV[0] = self.r_vzero
#             for j in range(self.n_points - 1):
#                 self.rV[j+1] = (self.rV[j] - self.hh[j] * self.r_phalf[j] * 
#                                (self.drho_dt[j] + self.rho[j] * self.beta * self.U[j]))

#         # Convert rV to V accounting for coordinate system
#         self.rV_to_V()

#         # Additional stagnation point enforcement
#         if self.continuity_bc == ContinuityBoundaryCondition.Zero:
#             j = self.j_cont_bc
#             if j > 0:
#                 self.V[j-1] = min(self.V[j-1], 0.0)  # Force negative velocity on fuel side
#             if j < self.n_points - 1:
#                 self.V[j+1] = max(self.V[j+1], 0.0)  # Force positive velocity on oxidizer side
#             self.V[j] = 0.0  # Enforce zero velocity at stagnation point


#     def update_continuity_boundary_condition(self, q_dot: np.ndarray, new_bc: ContinuityBoundaryCondition):
#         """
#         Complete implementation of boundary condition updates matching C++.
#         """
#         assert np.all(np.isfinite(self.V)), "Velocity field contains invalid values"
#         self.continuity_bc = new_bc

#         if new_bc == ContinuityBoundaryCondition.Zero:
#             # Find stagnation point location
#             if self.x[-1] > self.x_vzero:
#                 j_start = np.where(self.x > self.x_vzero)[0][0]
#             else:
#                 j_start = self.n_points - 1

#             # Find where velocity changes sign
#             self.j_cont_bc = j_start
#             found_stagnation = False
#             for i in range(1, self.n_points):
#                 if (j_start + i < self.n_points and 
#                     np.sign(self.V[j_start + i]) != np.sign(self.V[j_start])):
#                     self.j_cont_bc = j_start + i
#                     found_stagnation = True
#                     break
#                 elif (j_start >= i and 
#                       np.sign(self.V[j_start - i]) != np.sign(self.V[j_start])):
#                     self.j_cont_bc = j_start - i + 1
#                     found_stagnation = True
#                     break

#             # Update stagnation point location
#             if self.j_cont_bc == 0:
#                 assert self.V[self.j_cont_bc] <= 0, "Invalid velocity at left boundary"
#                 self.x_vzero = (self.x[0] - self.V[0] * self.hh[0] / 
#                                (self.V[1] - self.V[0]))
#             elif self.j_cont_bc == self.n_points - 1:
#                 assert self.V[self.j_cont_bc] >= 0, "Invalid velocity at right boundary"
#                 self.x_vzero = (self.x[-1] - self.V[-1] * self.hh[-1] / 
#                                (self.V[-1] - self.V[-2]))
#             else:
#                 j = self.j_cont_bc
#                 assert self.V[j] * self.V[j-1] <= 0, "No velocity sign change at stagnation point"
#                 self.x_vzero = (self.x[j] - self.V[j] * self.hh[j-1] / 
#                                (self.V[j] - self.V[j-1]))

#         elif new_bc == ContinuityBoundaryCondition.Temp:
#             # Find temperature crossing point
#             T_mid = 0.5 * (np.max(self.T) + np.min(self.T))
#             self.j_cont_bc = 0
            
#             # Find leftmost location where T crosses T_mid
#             for j in range(1, self.n_points):
#                 if (self.T[j] - T_mid) * (self.T[j-1] - T_mid) <= 0:
#                     self.j_cont_bc = j
#                     break

#         elif new_bc == ContinuityBoundaryCondition.Qdot:
#             # Find maximum heat release point
#             self.j_cont_bc = np.argmax(q_dot)
            
#         elif new_bc == ContinuityBoundaryCondition.Left:
#             self.j_cont_bc = 0
            
#         elif new_bc == ContinuityBoundaryCondition.Right:
#             self.j_cont_bc = self.n_points - 1
            
#         else:
#             raise ValueError(f"Invalid boundary condition: {new_bc}")
        

#     def _calculate_time_derivatives(self, dT_dx, dU_dx, dW_dx):
#         """Calculate time derivatives for UTW system"""
#         # Left boundary conditions
#         self.dU_dt[0] = (self.split_const_U[0] - self.U[0]**2 + 
#                         self.rho[0]/self.rho[0] * (self.strain_rate_deriv/self.beta + 
#                         self.strain_rate**2/self.beta**2))
        
#         if self.left_bc in [BoundaryCondition.ControlVolume, BoundaryCondition.WallFlux]:
#             center_vol = self.x[1]**(self.alpha + 1) / (self.alpha + 1)
#             r_vzero_mod = max(self.rV[0], 0.0)
            
#             self.dT_dt[0] = (-r_vzero_mod * (self.T[0] - self.T_left) / 
#                             (self.rho[0] * center_vol) + self.split_const_T[0])
#             self.dW_dt[0] = (-r_vzero_mod * (self.Wmx[0] - self.W_left) / 
#                             (self.rho[0] * center_vol) - self.Wmx[0]**2 * self.split_const_W[0])
#         else:
#             self.dT_dt[0] = self.split_const_T[0]
#             self.dW_dt[0] = -self.Wmx[0]**2 * self.split_const_W[0]
        
#         # Interior points
#         for j in range(1, self.n_points - 1):
#             self.dU_dt[j] = (-self.V[j] * dU_dx[j] / self.rho[j] - self.U[j]**2 +
#                             self.rho[0]/self.rho[j] * (self.strain_rate_deriv/self.beta + 
#                             self.strain_rate**2/self.beta**2) + self.split_const_U[j])
#             self.dT_dt[j] = -self.V[j] * dT_dx[j] / self.rho[j] + self.split_const_T[j]
#             self.dW_dt[j] = (-self.V[j] * dW_dx[j] / self.rho[j] - 
#                             self.Wmx[j]**2 * self.split_const_W[j])

#         # Right boundary
#         j = self.n_points - 1
#         if self.rV[j] < 0 or self.right_bc == BoundaryCondition.FixedValue:
#             self.dU_dt[j] = (self.split_const_U[j] - self.U[j]**2 +
#                             self.rho[0]/self.rho[j] * (self.strain_rate_deriv/self.beta + 
#                             self.strain_rate**2/self.beta**2))
#             self.dT_dt[j] = self.split_const_T[j]
#             self.dW_dt[j] = -self.Wmx[j]**2 * self.split_const_W[j]
#         else:
#             self.dU_dt[j] = (self.split_const_U[j] - 
#                             self.V[j] * (self.U[j] - self.U[j-1])/self.hh[j-1]/self.rho[j] -
#                             self.U[j]**2 + self.rho[0]/self.rho[j] * 
#                             (self.strain_rate_deriv/self.beta + self.strain_rate**2/self.beta**2))
#             self.dT_dt[j] = (self.split_const_T[j] - 
#                             self.V[j] * (self.T[j] - self.T[j-1])/self.hh[j-1]/self.rho[j])
#             self.dW_dt[j] = (-self.Wmx[j]**2 * self.split_const_W[j] - 
#                             self.V[j] * (self.Wmx[j] - self.Wmx[j-1])/self.hh[j-1]/self.rho[j])

#     def unroll_y(self, y: np.ndarray):
#         """Extract state variables from solution vector"""
#         n = len(y) // 3
#         self.T = y[:n]
#         self.U = y[n:2*n]
#         self.Wmx = y[2*n:]

#     def roll_y(self) -> np.ndarray:
#         """Combine state variables into solution vector"""
#         return np.concatenate([self.T, self.U, self.Wmx])

#     def roll_ydot(self) -> np.ndarray:
#         """Combine time derivatives into solution vector"""
#         return np.concatenate([self.dT_dt, self.dU_dt, self.dW_dt])

#     def rV_to_V(self):
#         """Convert rV to V accounting for coordinate system"""
#         self.V[0] = self.rV[0]
#         if self.alpha == 0:
#             self.V = self.rV.copy()
#         else:
#             self.V[1:] = self.rV[1:] / self.x[1:]
            
#     def set_gas(self, gas: ct.Solution):
#         """Set gas object for thermodynamic properties"""
#         self.gas = gas
        
#     def set_grid(self, grid: OneDimGrid):
#         """Set grid properties"""
#         self.x = grid.x
#         self.hh = grid.hh
#         self.n_points = len(grid.x)
#         self.r_phalf = grid.rphalf
#         self.alpha = grid.alpha
#         self.beta = grid.beta
        
#     def set_boundary_conditions(self, left_bc: BoundaryCondition, right_bc: BoundaryCondition,
#                                 continuity_bc: ContinuityBoundaryCondition = ContinuityBoundaryCondition.Left,
#                                 j_cont_bc: int = 0):
#         """Set boundary conditions"""
#         self.left_bc = left_bc
#         self.right_bc = right_bc
#         self.continuity_bc = continuity_bc
#         self.j_cont_bc = j_cont_bc
        

# class ConvectionSystemY:
#     """
#     Species transport system solver.
#     Handles the convective transport of species mass fractions.
#     """
#     def __init__(self):
#         self.v = None  # Velocity field
#         self.v_interp = {}  # Time-interpolated velocity fields
#         self.split_const = None
#         self.Y_left = None
#         self.quasi2d = False
#         self.vz_interp = None  # For quasi-2D simulations
#         self.vr_interp = None  # For quasi-2D simulations
#         self.grid = None
#         self.k = None  # Species index
        
#         # Boundary conditions
#         self.left_bc = None
#         self.right_bc = None

#     def resize(self, n_points: int):
#         """Initialize arrays with new size"""
#         self.v = np.zeros(n_points)
#         self.split_const = np.zeros(n_points)
#         self.v_interp = {0.0: np.zeros(n_points)}  # Initialize with zeros

    
#     def f(self, t: float, y: np.ndarray) -> np.ndarray:
#         """Optimized right-hand side calculation"""
#         y_dot = np.zeros_like(y)
#         n_points = len(y)
        
#         # Get interpolated velocity field (pre-computed)
#         if not self.quasi2d:
#             self.update_v(t)
        
#         # Vectorized operations for interior points
#         j_range = np.arange(1, n_points-1)
#         neg_v_mask = self.v[j_range] < 0
        
#         # Forward differences
#         dy_dx_forward = np.zeros_like(j_range, dtype=float)
#         mask = neg_v_mask
#         dy_dx_forward[mask] = ((y[j_range[mask] + 1] - y[j_range[mask]]) / 
#                               self.grid.hh[j_range[mask]])
        
#         # Backward differences
#         dy_dx_backward = np.zeros_like(j_range, dtype=float)
#         mask = ~neg_v_mask
#         dy_dx_backward[mask] = ((y[j_range[mask]] - y[j_range[mask] - 1]) / 
#                                self.grid.hh[j_range[mask] - 1])
        
#         # Combine derivatives
#         dy_dx = np.where(neg_v_mask, dy_dx_forward, dy_dx_backward)
        
#         # Update interior points
#         if self.quasi2d:
#             vr = self.vr_interp(self.grid.x[j_range], t)
#             vz = self.vz_interp(self.grid.x[j_range], t)
#             y_dot[j_range] = -vr * dy_dx / vz + self.split_const[j_range]
#         else:
#             y_dot[j_range] = -self.v[j_range] * dy_dx + self.split_const[j_range]
        
#         # Boundary conditions
#         if self.left_bc in [BoundaryCondition.ControlVolume, BoundaryCondition.WallFlux]:
#             center_vol = self.grid.x[1]**(self.grid.alpha + 1) / (self.grid.alpha + 1)
#             v_zero_mod = max(self.v[0], 0.0)
#             y_dot[0] = -v_zero_mod * (y[0] - self.Y_left) / center_vol + self.split_const[0]
#         else:
#             y_dot[0] = self.split_const[0]
            
#         # Right boundary
#         j = n_points - 1
#         if self.v[j] < 0 or self.right_bc == BoundaryCondition.FixedValue:
#             y_dot[j] = self.split_const[j]
#         else:
#             if self.quasi2d:
#                 y_dot[j] = (self.split_const[j] - 
#                            self.vr_interp(self.grid.x[j], t) * 
#                            (y[j] - y[j-1]) / self.grid.hh[j-1] / 
#                            self.vz_interp(self.grid.x[j], t))
#             else:
#                 y_dot[j] = (self.split_const[j] - 
#                            self.v[j] * (y[j] - y[j-1]) / self.grid.hh[j-1])

#         return y_dot

#     def update_v(self, t: float):
#         """Update velocity field using time interpolation"""
#         if len(self.v_interp) == 1:
#             # Only one time point available
#             self.v = next(iter(self.v_interp.values()))
#             return

#         # Find neighboring time points
#         times = sorted(self.v_interp.keys())
#         idx = np.searchsorted(times, t)
        
#         if idx == 0:
#             # Extrapolate before first point
#             t1, t2 = times[0], times[1]
#             v1, v2 = self.v_interp[t1], self.v_interp[t2]
#         elif idx == len(times):
#             # Extrapolate after last point
#             t1, t2 = times[-2], times[-1]
#             v1, v2 = self.v_interp[t1], self.v_interp[t2]
#         else:
#             # Interpolate between points
#             t1, t2 = times[idx-1], times[idx]
#             v1, v2 = self.v_interp[t1], self.v_interp[t2]

#         # Linear interpolation
#         s = (t - t1) / (t2 - t1)
#         self.v = v1 * (1 - s) + v2 * s
        
#     def set_boundary_conditions(self, left_bc: BoundaryCondition, right_bc: BoundaryCondition,
#                                 Y_left: float):
#         """Set boundary conditions"""
#         self.left_bc = left_bc
#         self.right_bc = right_bc
#         self.Y_left = Y_left



# class ConvectionSystemSplit:
#     """
#     Main convection system that couples UTW and species transport.
#     Uses operator splitting to solve the systems separately.
#     """
#     def __init__(self):
#         self.U = None  # Velocity
#         self.T = None  # Temperature
#         self.Y = None  # Species mass fractions
#         self.Wmx = None  # Mean molecular weight
#         self.dWdt = None  # Time derivative of mean molecular weight
        
#         self.utw_system = ConvectionSystemUTW()
#         self.species_systems = []
#         self.n_spec = 0
#         self.n_vars = 3 # UTW system variables (T, U, Wmx)
#         self.gas = None
#         self.quasi2d = False
        
#         self.strain_rate = 0.0
#         self.strain_rate_deriv = 0.0
        
#         # Integration parameters
#         self.rel_tol = 1e-8
#         self.abs_tol_U = 1e-8
#         self.abs_tol_T = 1e-8
#         self.abs_tol_W = 1e-7
#         self.abs_tol_Y = 1e-8

#         # Timing
#         self.utw_timer = time.time
#         self.species_timer = time.time

#     def set_gas(self, gas: Any):
#         """Set the gas object and update molecular weights"""
#         self.gas = gas
#         self.utw_system.gas = gas
#         if self.n_spec > 0:
#             self.W = np.array(gas.molecular_weights)

#     def resize(self, n_points: int, n_spec: int, state: np.ndarray):
#         """
#         Resize the system for new grid size or number of species.
#         """
#         self.n_points = n_points
        
#         # Update species systems if needed
#         if self.n_spec != n_spec:
#             self.species_systems = [ConvectionSystemY() for _ in range(n_spec)]
#             self.n_spec = n_spec
#             if self.gas:
#                 self.W = np.array(self.gas.molecular_weights)

#         # Reshape state arrays
#         self.T = state[0]
#         self.U = state[1]
#         self.Y = state[self.n_vars:]
        
#         # Initialize systems
#         self.utw_system.resize(n_points)
        
#         # Propagate grid to species systems
#         if hasattr(self, 'grid'):
#             for system in self.species_systems:
#                 system.resize(n_points)
#                 system.grid = self.grid  # Use stored grid
                
#         self.Wmx = np.zeros(n_points)
#         self.dWdt = np.zeros(n_points)

#     def set_grid(self, grid: OneDimGrid):
#         """Set grid for all systems"""
#         self.grid = grid  # Store grid object
#         self.utw_system.set_grid(grid)
#         for system in self.species_systems:
#             system.grid = grid

#     def set_gas(self, gas: Any):
#         """Set the gas object and update molecular weights"""
#         self.gas = gas
#         self.utw_system.gas = gas
#         if self.n_spec > 0:
#             self.W = np.array(gas.molecular_weights)

#     def set_tolerances(self, config: ConvectionConfig):
#         """Set integration tolerances"""
#         self.rel_tol = config.integrator_rel_tol
#         self.abs_tol_U = config.integrator_abs_tol_momentum
#         self.abs_tol_T = config.integrator_abs_tol_energy
#         self.abs_tol_W = config.integrator_abs_tol_species * 20
#         self.abs_tol_Y = config.integrator_abs_tol_species

#     def set_state(self, t_initial: float):
#         """Initialize state vectors for integration"""
#         # Set UTW system state
#         utw_y0 = np.zeros(3 * self.n_points)
#         for j in range(self.n_points):
#             utw_y0[3*j] = self.T[j]
#             utw_y0[3*j+1] = self.U[j]
#             self.gas.TPY = self.T[j], self.gas.P, self.Y[:,j]
#             utw_y0[3*j+2] = self.Wmx[j] = self.gas.mean_molecular_weight
            
#         # Set species system states
#         species_y0 = [self.Y[k,:] for k in range(self.n_spec)]
        
#         return t_initial, utw_y0, species_y0

    
#     def _integrate_species(self, args) -> Tuple[int, np.ndarray]:
#         """
#         Integrate a single species system.
        
#         Args:
#             args: Tuple of (species_index, initial_state, v_interp, t0, tf)
            
#         Returns:
#             Tuple of (species_index, final_state)
#         """
#         k, (y0, v_interp, t0, tf) = args
#         system = self.species_systems[k]
#         system.v_interp = v_interp
        
#         solver = ode(system.f)
#         solver.set_integrator(
#             'vode',
#             method='adams',
#             with_jacobian=False,
#             rtol=1e-6,
#             atol=1e-8,
#             nsteps=30000
#         )
#         solver.set_initial_value(y0, t0)
        
#         while solver.successful() and solver.t < tf:
#             solver.integrate(tf)
#             if solver.successful():
#                 print(f"[SUCCESS] Species {k} integrated to {solver.t:.2f} s")
#             else:
#                 print(f"[ERROR] Species {k} integration failed at {solver.t:.2f} s")
        
#         return k, solver.y

#     def integrate_to_time(self, tf: float):
#         """
#         Optimized integration to final time tf using ode and parallel processing.
#         """
#         t0, utw_y0, species_y0 = self.set_state(self.utw_system.t)
        
#         # Set up UTW integrator
#         self.utw_timer = time.time()
#         utw_solver = ode(self.utw_system.f)
#         utw_solver.set_integrator(
#             'vode', 
#             method='adams',
#             with_jacobian=False,
#             # rtol=self.rel_tol,
#             # atol=[self.abs_tol_T]*self.n_points + 
#             #      [self.abs_tol_U]*self.n_points + 
#             #      [self.abs_tol_W]*self.n_points,
#             rtol=1e-6,
#             atol=1e-8,
#             nsteps=30000
#         )
#         utw_solver.set_initial_value(utw_y0, t0)
        
#         # Storage for velocity field interpolation
#         v_interp_times = []
#         v_interp_values = []
        
#         # Integrate UTW system with fixed timesteps for better interpolation
#         dt = (tf - t0) / 10  # 10 intermediate points
#         while utw_solver.successful() and utw_solver.t < tf:
#             next_t = min(utw_solver.t + dt, tf)
#             utw_solver.integrate(next_t)
            
#             # Store velocity field
#             self.utw_system.unroll_y(utw_solver.y)
#             v_interp_times.append(utw_solver.t)
#             v_interp_values.append(self.utw_system.V/self.utw_system.rho)
#             if utw_solver.successful():
#                 print(f"[SUCCESS] UTW system integrated to {utw_solver.t:.2f} s")
#             else:
#                 print(f"[ERROR] UTW system integration failed at {utw_solver.t:.2f} s")
#         # Create interpolation data for species systems
#         v_interp = dict(zip(v_interp_times, v_interp_values))
        
#         # Parallel species integration using process pool
#         self.species_timer = time.time()
        
#         # Prepare arguments for parallel processing
#         integration_args = [
#             (k, (y0, v_interp, t0, tf)) 
#             for k, y0 in enumerate(species_y0)
#         ]
        
#         # Create process pool
#         n_cores = mp.cpu_count() - 1  # Leave one core free
#         with mp.Pool(processes=n_cores) as pool:
#             results = pool.map(self._integrate_species, integration_args)
        
#         # Update states
#         self.utw_system.unroll_y(utw_solver.y)
#         for k, y in sorted(results):  # Sort to maintain species order
#             self.Y[k,:] = y


#     def update_continuity_boundary_condition(self, qdot: np.ndarray, new_bc: ContinuityBoundaryCondition):
#         """Update boundary condition for continuity equation"""
#         self.utw_system.update_continuity_boundary_condition(qdot, new_bc)

#     def set_density_derivative(self, drho_dt: np.ndarray):
#         """Set density time derivative"""
#         self.utw_system.drho_dt = drho_dt

#     def reset_split_constants(self):
#         """Reset all split constants to zero"""
#         self.utw_system.reset_split_constants()
#         for system in self.species_systems:
#             system.reset_split_constants()

#     def set_split_constants(self, split_const: np.ndarray):
#         """Set split constants for all equations"""
#         self.utw_system.split_const_T = split_const[self.n_vars,:]
#         self.utw_system.split_const_U = split_const[self.n_vars+1,:]
        
#         # Convert species split constants to molar form
#         self.utw_system.split_const_W[:] = 0
#         for j in range(self.n_points):
#             for k in range(self.n_spec):
#                 self.utw_system.split_const_W[j] += split_const[self.n_vars+2+k,j] / self.W[k]
                
#         # Set species split constants
#         for k in range(self.n_spec):
#             self.species_systems[k].split_const = split_const[self.n_vars+2+k,:]

"""
Corrected convection system implementation matching C++ exactly.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import cantera as ct
from enum import Enum
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import time

class BoundaryCondition(Enum):
    """Boundary condition types matching C++"""
    FixedValue = "fixed_value"
    ZeroGradient = "zero_gradient"
    WallFlux = "wall_flux"
    ControlVolume = "control_volume"

class ContinuityBoundaryCondition(Enum):
    """Boundary conditions for continuity equation"""
    Left = "left"
    Right = "right"
    Zero = "zero"
    Temp = "temp"
    Qdot = "qdot"

class ConvectionSystemUTW:
    """
    Velocity-Temperature-Weight (UTW) system solver.
    Matches C++ implementation exactly.
    """
    def __init__(self):
        # Gas properties
        self.gas = None
        self.P = None
        
        # Grid properties
        self.x = None
        self.hh = None
        self.r_phalf = None
        self.alpha = 0
        self.beta = 1.0
        self.n_points = 0
        
        # Solution arrays
        self.T = None
        self.U = None
        self.Wmx = None
        self.rho = None
        
        # Boundary conditions
        self.T_left = None
        self.W_left = None
        self.r_vzero = None
        self.x_vzero = None
        
        # Convection
        self.rV = None
        self.V = None
        self.drho_dt = None
        
        # Time derivatives 
        self.dT_dt = None
        self.dU_dt = None
        self.dW_dt = None
        
        # Split terms
        self.split_const_T = None
        self.split_const_U = None  
        self.split_const_W = None
        
        # Boundary conditions
        self.left_bc = BoundaryCondition.FixedValue
        self.right_bc = BoundaryCondition.FixedValue
        self.continuity_bc = ContinuityBoundaryCondition.Left
        self.j_cont_bc = 0
        
        # Strain rate
        self.strain_rate = 0.0
        self.strain_rate_deriv = 0.0
        
        # Time
        self.t = 0.0

    def resize(self, n_points: int):
        """Initialize arrays with new size"""
        self.n_points = n_points
        
        # Solution arrays
        self.T = np.zeros(n_points)
        self.U = np.zeros(n_points)
        self.Wmx = np.zeros(n_points)
        self.rho = np.zeros(n_points)
        
        # Velocity arrays
        self.rV = np.zeros(n_points)
        self.V = np.zeros(n_points)
        
        # Derivatives
        self.dT_dt = np.zeros(n_points)
        self.dU_dt = np.zeros(n_points)
        self.dW_dt = np.zeros(n_points)
        self.drho_dt = np.zeros(n_points)
        
        # Split terms
        self.split_const_T = np.zeros(n_points)
        self.split_const_U = np.zeros(n_points)
        self.split_const_W = np.zeros(n_points)
    

    def set_grid(self, grid):
        """Set grid properties"""
        self.x = grid.x
        self.hh = grid.hh
        self.r_phalf = grid.rphalf
        self.alpha = grid.alpha
        self.beta = grid.beta
        self.n_points = len(grid.x)

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side calculation for UTW system"""
        # Unroll state vector
        self.unroll_y(y)
        
        # Update density
        self.rho = self.gas.P * self.Wmx / (ct.gas_constant * self.T)
        # Calculate V field
        self._calculate_V()
        
        # Calculate upwinded derivatives
        dT_dx = np.zeros(self.n_points)
        dU_dx = np.zeros(self.n_points)
        dW_dx = np.zeros(self.n_points)
        
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
        self._calculate_time_derivatives(t, dT_dx, dU_dx, dW_dx)
        
        return self.roll_ydot()

    def _calculate_V(self):
        """Calculate velocity field with proper boundary conditions"""
        if self.continuity_bc == ContinuityBoundaryCondition.Left:
            # Left boundary condition
            self.rV[0] = self.r_vzero
            for j in range(self.n_points - 1):
                self.rV[j+1] = (self.rV[j] - self.hh[j] * self.r_phalf[j] * 
                               (self.drho_dt[j] + self.rho[j] * self.beta * self.U[j]))
                
        elif self.continuity_bc == ContinuityBoundaryCondition.Zero:
            # Stagnation point
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
                
            # Backward integration
            if self.j_cont_bc != 0:
                j = self.j_cont_bc - 1
                self.rV[j] = (self.x[j] - self.x_vzero) * d_vdx0
                for j in range(self.j_cont_bc - 1, 0, -1):
                    self.rV[j-1] = (self.rV[j] + self.hh[j-1] * self.r_phalf[j-1] * 
                                  (self.drho_dt[j-1] + self.rho[j-1] * self.beta * self.U[j-1]))
                    
        elif self.continuity_bc == ContinuityBoundaryCondition.Temp:
            j = self.j_cont_bc
            if j == 0 or self.split_const_T[j] / (self.T[j+1] - self.T[j]) < 0:
                self.rV[j] = (self.r_phalf[j] * self.rho[j] * self.split_const_T[j] * 
                             self.hh[j] / (self.T[j+1] - self.T[j]))
            else:
                self.rV[j] = (self.r_phalf[j-1] * self.rho[j] * self.split_const_T[j] * 
                             self.hh[j-1] / (self.T[j] - self.T[j-1]))
                
            # Forward/backward integration
            for j in range(self.j_cont_bc, self.n_points - 1):
                self.rV[j+1] = (self.rV[j] - self.hh[j] * self.r_phalf[j] * 
                               (self.drho_dt[j] + self.rho[j] * self.beta * self.U[j]))
                
            for j in range(self.j_cont_bc, 0, -1):
                self.rV[j-1] = (self.rV[j] + self.hh[j-1] * self.r_phalf[j-1] * 
                               (self.drho_dt[j-1] + self.rho[j-1] * self.beta * self.U[j-1]))
        
        # Convert rV to V
        self.rV_to_V()
        
        # Enforce stagnation point constraints
        if self.continuity_bc == ContinuityBoundaryCondition.Zero:
            j = self.j_cont_bc
            if j > 0:
                self.V[j-1] = min(self.V[j-1], 0.0)
            if j < self.n_points - 1:
                self.V[j+1] = max(self.V[j+1], 0.0)
            self.V[j] = 0.0

    def _calculate_time_derivatives(self, t: float, dT_dx, dU_dx, dW_dx):
        """Calculate time derivatives of UTW system"""
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
                            (self.rho[0] * center_vol) - 
                            self.Wmx[0]**2 * self.split_const_W[0])
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

    def rV_to_V(self):
        """Convert rV to V accounting for coordinate system"""
        self.V[0] = self.rV[0]
        if self.alpha == 0:
            self.V = self.rV.copy()
        else:
            self.V[1:] = self.rV[1:] / self.x[1:]

    def unroll_y(self, y: np.ndarray):
        """Extract state variables from solution vector"""
        n = len(y) // 3
        self.T = y[:n]
        self.U = y[n:2*n] 
        self.Wmx = y[2*n:]

    def roll_y(self) -> np.ndarray:
        """Get state vector ensuring no NaN values"""
        if self.T is None or self.U is None or self.Wmx is None:
            raise ValueError("State variables not initialized")
        return np.concatenate([self.T, self.U, self.Wmx])

    def roll_ydot(self) -> np.ndarray:
        """Combine time derivatives into solution vector"""
        return np.concatenate([self.dT_dt, self.dU_dt, self.dW_dt])

    def update_continuity_boundary_condition(self, qdot: np.ndarray, new_bc: ContinuityBoundaryCondition):
        """Update boundary condition for continuity equation"""
        self.continuity_bc = new_bc
        
        if new_bc == ContinuityBoundaryCondition.Zero:
            # Find stagnation point location
            if self.x[-1] > self.x_vzero:
                j_start = np.where(self.x > self.x_vzero)[0][0]
            else:
                j_start = self.n_points - 1
                
            # Find where velocity changes sign
            self.j_cont_bc = j_start
            found_stagnation = False
            
            for i in range(1, self.n_points):
                if (j_start + i < self.n_points and 
                    np.sign(self.V[j_start + i]) != np.sign(self.V[j_start])):
                    self.j_cont_bc = j_start + i
                    found_stagnation = True
                    break
                elif (j_start >= i and 
                    np.sign(self.V[j_start - i]) != np.sign(self.V[j_start])):
                    self.j_cont_bc = j_start - i + 1
                    found_stagnation = True
                    break
                    
            # Update stagnation point location
            if self.j_cont_bc == 0:
                self.x_vzero = (self.x[0] - self.V[0] * self.hh[0] / 
                              (self.V[1] - self.V[0]))
            elif self.j_cont_bc == self.n_points - 1:
                self.x_vzero = (self.x[-1] - self.V[-1] * self.hh[-1] / 
                              (self.V[-1] - self.V[-2]))
            else:
                j = self.j_cont_bc
                self.x_vzero = (self.x[j] - self.V[j] * self.hh[j-1] / 
                              (self.V[j] - self.V[j-1]))
                              
        elif new_bc == ContinuityBoundaryCondition.Temp:
            # Find temperature crossing point
            T_mid = 0.5 * (np.max(self.T) + np.min(self.T))
            self.j_cont_bc = 0
            
            # Find leftmost location where T crosses T_mid
            for j in range(1, self.n_points):
                if (self.T[j] - T_mid) * (self.T[j-1] - T_mid) <= 0:
                    self.j_cont_bc = j
                    break
                    
        elif new_bc == ContinuityBoundaryCondition.Qdot:
            # Find maximum heat release point
            self.j_cont_bc = np.argmax(qdot)
            
    def set_boundary_conditions(self, left_bc: BoundaryCondition, right_bc: BoundaryCondition,
                                continuity_bc: ContinuityBoundaryCondition, j_cont_bc: int):
        """Set boundary conditions"""
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.continuity_bc = continuity_bc
        self.j_cont_bc = j_cont_bc
        
    def reset_split_constants(self):
        """Reset split constants to zero"""
        self.split_const_T = np.zeros(self.n_points)
        self.split_const_U = np.zeros(self.n_points)
        self.split_const_W = np.zeros(self.n_points)
        
    


class ConvectionSystemY:
    """Species transport system solver"""
    def __init__(self):
        self.v = None  # Velocity field
        self.v_interp = {}  # Time-interpolated velocity fields
        self.split_const = None
        self.Y_left = None
        
        # Grid properties
        self.grid = None
        self.x = None
        self.hh = None
        self.alpha = None
        self.n_points = None
        self.r = None
        
        # Boundary conditions
        self.left_bc = None
        self.right_bc = None
        
        # Initialize solver arrays
        self.T = None
        self.U = None
        self.Y = None
        
        # Boundary conditions
        self.left_bc = BoundaryCondition.FixedValue
        self.right_bc = BoundaryCondition.FixedValue
        
        # Species index
        self.k = None
        
    def resize(self, n_points: int):
        """Initialize arrays with new size"""
        self.n_points = n_points
        self.v = np.zeros(n_points)
        self.split_const = np.zeros(n_points)
        
        # Initialize grid arrays
        if self.hh is None:
            self.hh = np.zeros(n_points-1)
        if self.x is None:
            self.x = np.zeros(n_points)
        if self.r is None:
            self.r = np.zeros(n_points)
        
    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side for species transport equation"""
        y_dot = np.zeros_like(y)
        
        # Get interpolated velocity field
        self.update_v(t)
        
        # Left boundary conditions
        if self.left_bc in [BoundaryCondition.ControlVolume, BoundaryCondition.WallFlux]:
            center_vol = self.x[1]**(self.alpha + 1) / (self.alpha + 1)
            v_zero_mod = max(self.v[0], 0.0)
            y_dot[0] = -v_zero_mod * (y[0] - self.Y_left) / center_vol + self.split_const[0]
        else:
            y_dot[0] = self.split_const[0]
            
        # Interior points with upwinding
        for j in range(1, self.n_points - 1):
            if self.v[j] < 0:
                dy_dx = (y[j+1] - y[j]) / self.hh[j]
            else:
                dy_dx = (y[j] - y[j-1]) / self.hh[j-1]
            y_dot[j] = -self.v[j] * dy_dx + self.split_const[j]
            
        # Right boundary
        j = self.n_points - 1
        if self.v[j] < 0 or self.right_bc == BoundaryCondition.FixedValue:
            y_dot[j] = self.split_const[j]
        else:
            y_dot[j] = (self.split_const[j] - 
                       self.v[j] * (y[j] - y[j-1]) / self.hh[j-1])
            
        return y_dot

    def update_v(self, t: float):
        """Update velocity field using time interpolation"""
        if len(self.v_interp) == 1:
            self.v = next(iter(self.v_interp.values()))
            return
            
        # Find neighboring time points
        times = sorted(self.v_interp.keys())
        idx = np.searchsorted(times, t)
        
        if idx == 0:
            t1, t2 = times[0], times[1]
            v1, v2 = self.v_interp[t1], self.v_interp[t2]
        elif idx == len(times):
            t1, t2 = times[-2], times[-1]
            v1, v2 = self.v_interp[t1], self.v_interp[t2]
        else:
            t1, t2 = times[idx-1], times[idx]
            v1, v2 = self.v_interp[t1], self.v_interp[t2]
            
        # Linear interpolation
        s = (t - t1) / (t2 - t1)
        self.v = v1 * (1 - s) + v2 * s

    def reset_split_constants(self):
        """Reset split constants to zero"""
        self.split_const = np.zeros(self.n_points)


class ConvectionSystemSplit:
    """Main convection system coupling UTW and species transport"""
    def __init__(self, n_spec: int = 53):
        # Solution arrays
        self.U = None
        self.T = None
        self.Y = None
        self.V = None
        self.Wmx = None
        self.grid = None
        self.quasi2d = False
        self.v_interp = {}
        # Systems
        self.utw_system = ConvectionSystemUTW()
        self.species_systems = [ConvectionSystemY() for _ in range(n_spec)]
        
        # Properties
        self.n_spec = n_spec
        self.n_vars = 3  # UTW variables
        self.gas = None
        
        # Integration parameters
        self.rel_tol = 1e-6
        self.abs_tol_U = 1e-8
        self.abs_tol_T = 1e-8
        self.abs_tol_W = 1e-7
        self.abs_tol_Y = 1e-8
        
        # Timing
        self.t = 0.0
        self.utw_timer = time.time
        self.species_timer = time.time
        
    def resize(self, n_points: int, n_spec: int):
        """Resize all systems"""
        self.n_points = n_points
        
        # Update species systems if needed
        if self.n_spec != n_spec:
            self.species_systems = [ConvectionSystemY() for _ in range(n_spec)]
            self.n_spec = n_spec
            
        # Initialize arrays
        self.T = np.zeros(n_points)
        self.U = np.zeros(n_points)
        self.V = np.zeros(n_points)
        self.Y = np.zeros((n_spec, n_points))
        self.Wmx = np.zeros(n_points)
        
        self.v_interp.clear()
        
        # Resize systems
        # Set up UTW system
        self.utw_system.resize(n_points)
        self.utw_system.set_grid(self.grid)
        
        # Set up species systems
        for system in self.species_systems:
            system.resize(n_points)
            # Copy grid properties
            if self.grid is not None:
                system.x = self.grid.x
                system.hh = self.grid.hh
                system.r = self.grid.r
                system.alpha = self.grid.alpha
                
    def initialize_utw_system(self):
        """Initialize UTW system with proper initial state"""
        # Ensure gas object exists
        if self.gas is None:
            raise ValueError("Gas object must be set before initialization")
            
        # Initialize UTW arrays if not already done
        if self.Wmx is None:
            self.Wmx = np.zeros(self.n_points)
            
        # # Calculate molecular weight for each point
        # for j in range(self.n_points):
        #     # Set gas state and get properties
        #     self.gas.TPY = self.T[j], self.gas.P, self.Y[:,j]
            
            
        # Initialize UTW system 
        self.utw_system.T = self.T.copy()
        self.utw_system.U = self.U.copy()
        self.utw_system.V = self.V.copy()
        self.utw_system.Wmx = self.Wmx.copy()
        self.utw_system.P = self.gas.P
        
        # Calculate initial density
        self.utw_system.rho = np.zeros(self.n_points)
        for j in range(self.n_points):
            self.gas.TPY = self.T[j], self.gas.P, self.Y[:,j]
            self.utw_system.rho[j] = self.gas.density
            self.Wmx[j] = self.gas.mean_molecular_weight
            
    def set_split_constants(self, split_const: np.ndarray):
        """Set split constants exactly matching C++ implementation"""
        # Extract indices
        k_energy = 0  # Temperature
        k_momentum = 1  # Velocity 
        k_species = 2  # Start of species indices
        
        # Set UTW system split constants
        self.utw_system.split_const_T = split_const[k_energy]
        self.utw_system.split_const_U = split_const[k_momentum]
        
        # Calculate weighted species contribution for molecular weight
        self.utw_system.split_const_W = np.zeros(self.n_points)
        for j in range(self.n_points):
            for k in range(self.n_spec):
                self.utw_system.split_const_W[j] += (
                    split_const[k_species + k, j] / self.W[k]
                )
                
        # Set species system split constants
        for k in range(self.n_spec):
            self.species_systems[k].split_const = split_const[k_species + k]
            
    def initialize_split_constants(self):
        """Reset all split constants to zero"""
        self.utw_system.reset_split_constants()
        for system in self.species_systems:
            system.reset_split_constants()
            
    def set_grid(self, grid):
        """Set grid for all systems"""
        self.grid = grid
        self.n_points = len(grid.x)
        self.utw_system.set_grid(grid)
        for system in self.species_systems:
            system.grid = grid
            system.x = grid.x
            system.hh = grid.hh
            system.alpha = grid.alpha
            
    def set_gas(self, gas):
        """Set gas object"""
        self.gas = gas
        self.utw_system.gas = gas
        if self.n_spec > 0:
            self.W = np.array(gas.molecular_weights)
        
    def update_state(self):
        self.T = self.utw_system.T
        self.U = self.utw_system.U
        self.V = self.utw_system.V
        self.Wmx = self.utw_system.Wmx
        
            
    def integrate_to_time(self, tf: float):
        """Integrate convection system matching C++ implementation"""
        self.t_stage_stop = tf
        
        if not self.quasi2d:
            # UTW system integration
            self.utw_timer = time.time()
            self.v_interp.clear()
            
            # Initial RHS evaluation
            y_utw = self.utw_system.roll_y()
            ydot_utw = self.utw_system.f(self.t, y_utw)
            
            # Store initial velocity field
            self.v_interp[self.t] = self.utw_system.V / self.utw_system.rho
            
            # Integrate UTW system to final time
            solver = solve_ivp(
                self.utw_system.f,
                (self.t, tf),
                y_utw,
                method='BDF',  # Similar to CVODE Adams
                rtol=self.rel_tol,
                atol=[self.abs_tol_T]*self.n_points + 
                     [self.abs_tol_U]*self.n_points + 
                     [self.abs_tol_W]*self.n_points,
                max_step=tf-self.t,
                dense_output=True  # For interpolation
            )
            
            # Store velocity field history
            for t_eval in solver.t:
                y = solver.sol(t_eval)
                self.utw_system.unroll_y(y)
                self.v_interp[t_eval] = self.utw_system.V / self.utw_system.rho
                
            # Update final state
            self.utw_system.unroll_y(solver.y[:,-1])
            # Update final state
            self.update_state()
            
            
        # Species integration (can be parallel)
        self.species_timer = time.time()
        
        for k in range(self.n_spec):
            system = self.species_systems[k]
            system.v_interp = self.v_interp
            
            # Integrate species equation
            sol = solve_ivp(
                system.f,
                (self.t, tf),
                self.Y[k],
                method='LSODA',
                rtol=self.rel_tol,
                atol=[self.abs_tol_Y]*self.n_points
            )
            
            if sol.success:
                self.Y[k] = sol.y[:,-1]
            else:
                print(f"Species {k} integration failed")
        
        
        
        self.t = tf