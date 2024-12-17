from typing import Optional, Dict, Any
import numpy as np
import cantera as ct
from ..core.base import TransportComponent

class SourceSystem(TransportComponent):
    """
    Chemical source term system handling reactions and heat release.
    Matches C++ implementation.
    """
    def __init__(self, gas: ct.Solution):
        """Initialize source system"""
        self.gas = gas
        self.n_spec = gas.n_species
        
        # Solution state
        self.T: float = None
        self.U: float = None 
        self.Y: np.ndarray = None
        
        # Properties
        self.W = gas.molecular_weights
        self.h = np.zeros(self.n_spec)  # Species enthalpies
        self.cp = np.zeros(self.n_spec)  # Species heat capacities
        self.wdot = np.zeros(self.n_spec)  # Production rates
        
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
        
    def initialize(self, T: float, U: float, Y: np.ndarray):
        """Set initial state"""
        self.T = T
        self.U = U 
        self.Y = Y.copy()
        self.update_properties()
        
    def set_position(self, j: int, x: float):
        """Set grid position"""
        self.j = j
        self.x_pos = x
        
    def update_properties(self):
        """Update thermodynamic properties"""
        self.gas.TPY = self.T, self.P, self.Y
        self.cp = self.gas.partial_molar_cp / self.W
        self.h = self.gas.partial_molar_enthalpies / self.W
        
        # Get reaction rates with multiplier if set
        if self.rate_mult is not None:
            self.gas.set_multiplier(self.rate_mult(self.x_pos))
        self.wdot = self.gas.net_production_rates
        
    def get_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Calculate chemical source terms.
        
        Returns:
            ndarray: [dU/dt, dT/dt, dY1/dt, ..., dYK/dt]
        """
        # Unpack state
        U = state[0]
        T = state[1] 
        Y = state[2:]
        
        # Update state and properties
        self.T = T
        self.U = U
        self.Y = Y
        self.update_properties()
        
        # Calculate source terms
        dYdt = self.wdot * self.W / self.gas.density
        
        # Heat release
        q_dot = -np.sum(self.h * self.wdot)
        
        # Temperature
        dTdt = q_dot / (self.gas.density * self.gas.cp_mass)
        
        # Add heat loss if specified
        if self.heat_loss is not None:
            q_loss = self.heat_loss(self.x_pos, T)
            dTdt -= q_loss / (self.gas.density * self.gas.cp_mass)
            
        # Add split terms
        dUdt = self.split_const_U
        dTdt += self.split_const_T
        dYdt += self.split_const_Y
        
        return np.concatenate([[dUdt, dTdt], dYdt])
        
    def set_heat_loss(self, heat_loss):
        """Set heat loss function"""
        self.heat_loss = heat_loss
        
    def set_rate_multiplier(self, rate_mult):
        """Set reaction rate multiplier function"""
        self.rate_mult = rate_mult
        
    def compute_properties(self) -> None:
        return super().compute_properties()