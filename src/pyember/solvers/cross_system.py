import numpy as np
from typing import Optional
from ..core.grid import OneDimGrid

class CrossTermSystem:
    """
    System handling cross-coupling terms between different physical processes.
    Matches C++ implementation exactly.
    """
    def __init__(self, grid: OneDimGrid, n_spec: int = 53):
        self.n_spec = n_spec
        self.grid = grid
        self.initialize()
        
    def initialize(self):
        """Initialize arrays"""
        n_points = len(self.grid.x)
        
        # Cross term contributions
        self.dYdt_cross = np.zeros((self.n_spec, n_points))
        self.dTdt_cross = np.zeros(n_points)  # Temperature cross terms
        self.dUdt_cross = np.zeros(n_points)  # Velocity cross terms
        
        # Transport properties
        self.rho = None
        self.cp = None
        self.cp_spec = None
        self.lambda_ = None  # Thermal conductivity
        
        # Mass transport
        self.j_corr = np.zeros(n_points)  # Correction flux
        self.j_fick = np.zeros((self.n_spec, n_points))  # Fickian diffusion flux
        self.j_soret = np.zeros((self.n_spec, n_points))
        self.sum_cpj = np.zeros(n_points)
        
    def resize(self, n_points: int, n_spec: int):
        """Resize system arrays"""
        self.dYdt_cross = np.zeros((n_spec, n_points))
        self.dTdt_cross = np.zeros(n_points)
        self.dUdt_cross = np.zeros(n_points)
        
        self.j_fick = np.zeros((n_spec, n_points-1))
        self.j_soret = np.zeros((n_spec, n_points-1))
        
    def calculate_cross_terms(self, T: np.ndarray, Y: np.ndarray, 
                            rhoD: np.ndarray, Dkt: np.ndarray):
        """
        Calculate cross-coupling terms between species and temperature.
        
        Args:
            T: Temperature profile
            Y: Species mass fractions
            rhoD: Species diffusion coefficients (ρD)
            Dkt: Thermal diffusion coefficients
        """
        n_points = len(self.grid.x)
        n_spec = Y.shape[0]
        
        # Reset arrays
        self.j_corr.fill(0)
        self.sum_cpj.fill(0)
        
        # Calculate diffusive fluxes at cell faces
        for j in range(n_points-1):
            # Fickian diffusion
            for k in range(n_spec):
                self.j_fick[k,j] = -0.5 * (rhoD[k,j] + rhoD[k,j+1]) * \
                                  (Y[k,j+1] - Y[k,j]) / self.grid.hh[j]
                                  
            # Soret diffusion  
            for k in range(n_spec):
                self.j_soret[k,j] = -0.5 * (Dkt[k,j]/T[j] + Dkt[k,j+1]/T[j+1]) * \
                                   (T[j+1] - T[j]) / self.grid.hh[j]
                                   
            # Correction flux
            self.j_corr[j] = -np.sum(self.j_fick[:,j] + self.j_soret[:,j])
            
        # Calculate species cross terms
        for j in range(1, n_points-1):
            # Sum of specific heat * flux terms
            for k in range(n_spec):
                self.sum_cpj[j] += (0.5 * (self.cp_spec[k,j] + self.cp_spec[k,j+1]) / 
                                  self.W[k] * (self.j_fick[k,j] + self.j_soret[k,j] + 
                                  0.5 * (Y[k,j] + Y[k,j+1]) * self.j_corr[j]))
                
            # Species cross terms                  
            for k in range(n_spec):
                self.dYdt_cross[k,j] = (-0.5 / (self.grid.r[j] * self.rho[j] * 
                                       self.grid.dlj[j]) * 
                    (self.grid.rphalf[j] * (Y[k,j] + Y[k,j+1]) * self.j_corr[j] -
                     self.grid.rphalf[j-1] * (Y[k,j-1] + Y[k,j]) * self.j_corr[j-1]))
                     
                self.dYdt_cross[k,j] -= (1.0 / (self.grid.r[j] * self.rho[j] * 
                                        self.grid.dlj[j]) *
                    (self.grid.rphalf[j] * self.j_soret[k,j] - 
                     self.grid.rphalf[j-1] * self.j_soret[k,j-1]))
                     
            # Temperature cross term
            dTdx = (self.grid.cf[j] * T[j-1] + 
                   self.grid.cfm[j] * T[j] + 
                   self.grid.cfp[j] * T[j+1])
            self.dTdt_cross[j] = (-0.5 * (self.sum_cpj[j] + self.sum_cpj[j-1]) * 
                                 dTdx / (self.cp[j] * self.rho[j]))
                                 
    def set_properties(self, rho: np.ndarray, cp: np.ndarray, 
                      cp_spec: np.ndarray, W: np.ndarray,
                      lambda_: np.ndarray):
        """Set transport properties"""
        self.rho = rho
        self.cp = cp
        self.cp_spec = cp_spec
        self.W = W
        self.lambda_ = lambda_