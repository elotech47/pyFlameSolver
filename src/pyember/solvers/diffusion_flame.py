"""
Diffusion flame solver implementation
"""
from typing import Optional, Dict, Any
import numpy as np
import cantera as ct

from ..core.grid import OneDimGrid, GridConfig
from ..transport.diffusion import DiffusionSystem
from .integrator import TridiagonalIntegrator
from ..transport.convection import ConvectionSystem

class DiffusionFlame:
    """
    Solver for diffusion flame configurations.
    Implements core functionality from Ember's flamesolver.cpp
    for diffusion flame cases.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize diffusion flame solver
        
        Args:
            config: Configuration dictionary containing:
                - mechanism: Path to Cantera mechanism file
                - fuel: Fuel composition string
                - oxidizer: Oxidizer composition string
                - pressure: Operating pressure [Pa]
                - T_fuel: Fuel temperature [K]
                - T_oxidizer: Oxidizer temperature [K]
                - grid: Grid configuration parameters
        """
        self.config = config
        
        # Initialize Cantera objects
        self.gas = ct.Solution(config['mechanism'])
        self.n_species = self.gas.n_species
        
        # Initialize grid
        grid_config = GridConfig(**config.get('grid', {}))
        self.grid = OneDimGrid(grid_config)
        
        # Initialize solution arrays
        self.T = np.zeros(self.grid.x.shape)  # Temperature
        self.Y = np.zeros((self.n_species, len(self.grid.x)))  # Mass fractions
        self.rho = np.zeros_like(self.T)  # Density
        self.cp = np.zeros_like(self.T)  # Specific heat
        self.D = np.zeros((self.n_species, len(self.grid.x)))  # Diffusion coefficients
        self.k = np.zeros_like(self.T)  # Thermal conductivity
        
        # Initialize diffusion systems
        self.T_system = DiffusionSystem(self.grid)  # Temperature
        self.Y_systems = []  # Species mass fractions
        for k in range(self.n_species):
            self.Y_systems.append(DiffusionSystem(self.grid))
            
        # Initialize integrators
        self.T_integrator = TridiagonalIntegrator(self.T_system)
        self.Y_integrators = []
        for system in self.Y_systems:
            self.Y_integrators.append(TridiagonalIntegrator(system))
            
        # Add convection system
        self.convection = ConvectionSystem(self.grid, config.get('convection', {}))
        
        # Strain rate parameters
        self.strain_rate = config.get('strain_rate', 100.0)  # Default strain rate
            
        # Set initial conditions
        self.initialize()

    def initialize(self):
        """Initialize the flame structure"""
        # Set up initial temperature profile
        T_fuel = self.config['T_fuel']
        T_oxidizer = self.config['T_oxidizer']
        x = self.grid.x
        
        # Create temperature profile with smooth transition
        center = 0.5 * (x[0] + x[-1])
        width = 0.2 * (x[-1] - x[0])
        self.T = T_fuel + 0.5*(T_oxidizer - T_fuel)*(1 + np.tanh((x - center)/width))
        
        # Set up species profiles
        self.gas.TPX = T_fuel, self.config['pressure'], self.config['fuel']
        Y_fuel = self.gas.Y
        
        self.gas.TPX = T_oxidizer, self.config['pressure'], self.config['oxidizer']
        Y_oxidizer = self.gas.Y
        
        # Initialize mass fractions with smooth transition
        for k in range(self.n_species):
            self.Y[k] = Y_fuel[k] + 0.5*(Y_oxidizer[k] - Y_fuel[k])*(1 + np.tanh((x - center)/width))
            
        # Update properties
        self.update_properties()
        
        # Initialize integrators
        t0 = 0.0
        dt = 1e-5  # Initial timestep
        self.t = t0
        self.T_integrator.set_y0(self.T)
        self.T_integrator.initialize(t0, dt)
        
        for k, integrator in enumerate(self.Y_integrators):
            integrator.set_y0(self.Y[k])
            integrator.initialize(t0, dt)

    def update_properties(self):
        """Update thermodynamic and transport properties"""
        for j in range(len(self.grid.x)):
            # Set gas state
            self.gas.TPY = self.T[j], self.config['pressure'], self.Y[:,j]
            
            # Get properties
            self.rho[j] = self.gas.density
            self.cp[j] = self.gas.cp_mass
            self.k[j] = self.gas.thermal_conductivity
            
            # Species diffusion coefficients
            self.D[:,j] = self.gas.mix_diff_coeffs
            
        # Update diffusion systems
        # Temperature diffusion
        self.T_system.set_properties(
            D=self.k/(self.rho * self.cp),  # Thermal diffusivity
            rho=self.rho,
            B=self.rho * self.cp
        )
        
        # Species diffusion
        for k, system in enumerate(self.Y_systems):
            system.set_properties(
                D=self.D[k],
                rho=self.rho
            )

    # def step(self, dt: float):
    #     """
    #     Advance solution by one timestep
        
    #     Args:
    #         dt: Timestep size [s]
    #     """
    #     # Update properties
    #     self.update_properties()
        
    #     # Step temperature
    #     self.T_integrator.dt = dt
    #     self.T_integrator.step()
    #     self.T = self.T_integrator.get_y()
    #     self.t += dt
    #     # Step species
    #     for k, integrator in enumerate(self.Y_integrators):
    #         integrator.dt = dt
    #         integrator.step()
    #         self.Y[k] = integrator.get_y()
            
    #     # Normalize mass fractions
    #     Y_sum = np.sum(self.Y, axis=0)
    #     for k in range(self.n_species):
    #         self.Y[k] /= Y_sum
    def step(self, dt: float):
        """
        Advance solution using operator splitting
        """
        # 1. Update properties
        self.update_properties()
        
        # 2. Convection step (first half)
        self.convection.utw_system.strain_rate = self.strain_rate
        self.convection.compute_convection_terms()
        self._apply_convection(dt/2)
        
        # 3. Diffusion step
        self._apply_diffusion(dt)
        
        # 4. Convection step (second half)
        self.update_properties()  # Need updated properties
        self.convection.compute_convection_terms()
        self._apply_convection(dt/2)
        
        # 5. Update properties again
        self.update_properties()

    def _apply_convection(self, dt: float):
        """Apply convection step"""
        # Temperature convection
        dTdt_conv = self.convection.get_convective_flux(self.T)
        self.T += dt * dTdt_conv
        
        # Species convection
        for k in range(self.n_species):
            dYdt_conv = self.convection.get_convective_flux(self.Y[k])
            self.Y[k] += dt * dYdt_conv
            
        # Normalize mass fractions
        Y_sum = np.sum(self.Y, axis=0)
        self.Y /= Y_sum[np.newaxis, :]
        
    def _apply_diffusion(self, dt: float):
        """Apply diffusion step"""
        # Temperature diffusion
        self.T_integrator.dt = dt
        self.T_integrator.step()
        self.T = self.T_integrator.get_y()
        
        self.t += dt
        
        # Species diffusion
        for k, integrator in enumerate(self.Y_integrators):
            integrator.dt = dt
            integrator.step()
            self.Y[k] = integrator.get_y()
            
        # Normalize mass fractions
        Y_sum = np.sum(self.Y, axis=0)
        self.Y /= Y_sum[np.newaxis, :]
        
    def get_solution(self):
        """Return current solution state"""
        return self.T, self.Y