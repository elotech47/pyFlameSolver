from typing import Optional, Dict, Any, Tuple
import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from ..core.base import TransportComponent

class SourceSystem(TransportComponent):
    def __init__(self, gas: ct.Solution):
        """Initialize source system with improved error handling"""
        self.gas = gas
        self.n_spec = gas.n_species
        
        # Solution state
        self.T: float = None
        self.U: float = None 
        self.Y: np.ndarray = None
        
        # Properties
        self.W = gas.molecular_weights
        self.h = np.zeros(self.n_spec)
        self.cp = np.zeros(self.n_spec)
        self.wdot = np.zeros(self.n_spec)
        
        # Initial conditions for integration
        self.rho_initial: float = None
        self.cp_initial: float = None
        
        # Parameters
        self.P: float = None
        self.x_pos: float = None
        self.j: int = None
        self.rhou: float = None
        self.heat_loss = None
        self.rate_mult = None
        
        # Split terms
        self.split_const_T = 0.0
        self.split_const_U = 0.0
        self.split_const_Y = np.zeros(self.n_spec)
        
        # Integration state
        self._t = 0.0
        self.debug = False
    
    def initialize(self, T: float, U: float, Y: np.ndarray):
            """Set initial state and pre-compute quantities for integration"""
            self.T = T
            self.U = U
            self.Y = Y.copy()
            
            # Pre-compute quantities that don't change during integration
            self.gas.TPY = self.T, self.P, self.Y
            self.rho_initial = self.gas.density
            self.cp_initial = self.gas.cp_mass
            
            self._t = 0.0  # Reset integration time
            
    def set_position(self, j: int, x: float):
        """Set grid position"""
        self.j = j
        self.x_pos = x
        
    def set_debug(self, debug: bool):
        """Enable/disable debug mode"""
        self.debug = debug
        
    def time(self) -> float:
        """Get current integration time"""
        return self._t
        
    def update_properties(self):
        """Update thermodynamic properties"""
        try:
            self.gas.TPY = self.T, self.P, self.Y
            self.cp = self.gas.partial_molar_cp / self.W
            self.h = self.gas.partial_molar_enthalpies / self.W
            
            if self.rate_mult is not None:
                self.gas.set_multiplier(self.rate_mult(self.x_pos))
            self.wdot = self.gas.net_production_rates
            
        except Exception as e:
            raise RuntimeError(f"Error updating properties at x={self.x_pos}, T={self.T}: {str(e)}")
        
    def normalize_mass_fractions(self, Y: np.ndarray) -> np.ndarray:
        """Normalize mass fractions with bounds checking"""
        if np.any(Y < -1e-10):
            Y = np.maximum(Y, 0)  # Clip small negative values
            
        Y_sum = Y.sum()
        if abs(Y_sum - 1.0) > 1e-8:  # Only normalize if significantly different from 1
            if Y_sum > 0:
                Y = Y / Y_sum
            else:
                raise ValueError(f"Invalid mass fraction sum: {Y_sum}")
        return Y
        
    def get_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """RHS function with improved mass fraction handling"""
        try:
            # Unpack state
            U, T = state[0], state[1]
            Y = state[2:]
            
            # Temperature bounds checking
            if T < 200 or T > 6000:
                raise ValueError(f"Temperature {T}K out of valid range [200, 6000]")
            
            # Mass fraction normalization and bounds checking
            try:
                Y = self.normalize_mass_fractions(Y)
            except ValueError as e:
                print(f"State at error:")
                print(f"T = {T:.1f} K")
                print(f"Y = {Y}")
                print(f"Sum(Y) = {Y.sum():.3e}")
                raise ValueError(f"Mass fraction normalization failed: {str(e)}")
            
            # Update gas state
            self.gas.TPY = T, self.P, Y
            rho = self.gas.density
            
            # Get reaction rates with rate multiplier
            if self.rate_mult is not None:
                mult = self.rate_mult(self.x_pos)
                self.gas.set_multiplier(mult)
                
            wdot = self.gas.net_production_rates
            
            # Species equations with split terms
            dYdt = wdot * self.W / rho + self.split_const_Y
            
            # Check for NaN/inf in species rates
            if not np.all(np.isfinite(dYdt)):
                bad_species = np.where(~np.isfinite(dYdt))[0]
                raise ValueError(f"Non-finite species rates for: {bad_species}")
            
            # Energy equation
            h = self.gas.partial_molar_enthalpies / self.W
            q_dot = -np.sum(h * wdot)
            dTdt = q_dot / (rho * self.gas.cp_mass)
            
            if self.heat_loss is not None:
                q_loss = self.heat_loss(self.x_pos, t, U, T, Y)
                dTdt -= q_loss / (rho * self.gas.cp_mass)
                
            # Add split terms
            dTdt += self.split_const_T
            dUdt = self.split_const_U
            
            if self.debug:
                print(f"t={t:.3e}, T={T:.1f}, max|dT/dt|={abs(dTdt):.1e}")
                print(f"max|dY/dt|={np.max(np.abs(dYdt)):.1e}")
                
            return np.concatenate([[dUdt, dTdt], dYdt])
            
        except Exception as e:
            print(f"\nError in RHS at x={self.x_pos:.6f}, t={t:.3e}:")
            print(f"T = {T:.1f} K")
            print(f"Y = {Y}")
            print(f"Sum(Y) = {Y.sum():.3e}")
            if hasattr(self.gas, 'species_names'):
                print("\nSpecies composition:")
                for k, (name, yk) in enumerate(zip(self.gas.species_names, Y)):
                    if yk > 1e-6:
                        print(f"{k:3d} {name:10s}: {yk:.4e}")
            raise RuntimeError(f"RHS error at x={self.x_pos}, t={t}: {str(e)}")
            
    def integrate_to_time(self, tf: float) -> Tuple[bool, str]:
        """Integrate with improved error handling"""
        try:
            # Initial state
            y0 = np.concatenate([[self.U, self.T], self.Y])
            
            # First try with standard BDF
            sol = solve_ivp(
                self.get_rhs,
                (0, tf),
                y0,
                method='BDF',
                rtol=1e-8,
                atol=1e-10,
                max_step=tf/20,
                first_step=tf/100
            )
            
            if not sol.success:
                # If BDF fails, try Radau with looser tolerances
                sol = solve_ivp(
                    self.get_rhs,
                    (0, tf),
                    y0,
                    method='Radau',
                    rtol=1e-6,
                    atol=1e-8,
                    max_step=tf/10
                )
            
            if sol.success:
                # Get final state
                self.U = sol.y[0,-1]
                self.T = sol.y[1,-1]
                self.Y = sol.y[2:,-1]
                
                # Normalize final mass fractions
                self.Y = self.normalize_mass_fractions(self.Y)
                
                self._t = tf
                return True, f"steps={sol.nfev}"
            else:
                return False, f"Integration failed: {sol.message}"
                
        except Exception as e:
            return False, f"Error: {str(e)}"
            
    def get_stats(self) -> str:
        """Get integration statistics"""
        return f"t={self._t:.3e}"

    def write_state(self, out, init: bool = False):
        """Write current state for debugging"""
        if init:
            out.write("T = []\nU = []\nY = []\nt = []\n")
        
        out.write(f"T.append({self.T})\n")
        out.write(f"U.append({self.U})\n")
        Y_str = ", ".join(f"{y:.6e}" for y in self.Y)
        out.write(f"Y.append([{Y_str}])\n")
        out.write(f"t.append({self._t})\n")
    
    def compute_properties(self) -> None:
        return super().compute_properties()