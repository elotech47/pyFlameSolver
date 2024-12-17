import numpy as np
from typing import Optional, Dict, List

class SplitConstantsManager:
    """
    Manages operator splitting terms and coupling between different processes.
    Matches C++ implementation exactly.
    """
    def __init__(self, n_points: int, n_spec: int):
        self.n_points = n_points
        self.n_spec = n_spec
        self.n_vars = n_spec + 2  # T, U, Y1...YK
        
        # Split constants for each process
        self.split_conv = np.zeros((self.n_vars, n_points))
        self.split_diff = np.zeros((self.n_vars, n_points))
        self.split_prod = np.zeros((self.n_vars, n_points))
        
        # Time derivatives
        self.ddt_conv = np.zeros((self.n_vars, n_points))
        self.ddt_diff = np.zeros((self.n_vars, n_points))
        self.ddt_prod = np.zeros((self.n_vars, n_points))
        self.ddt_cross = np.zeros((self.n_vars, n_points))
        
    def calculate_split_constants(self, dt: float, use_balanced: bool = True):
        """
        Calculate split constants for each process.
        
        Args:
            dt: Timestep
            use_balanced: Use balanced splitting method
        """
        if use_balanced:
            # Balanced splitting approach (Strang-like)
            
            # Diffusion split constants
            self.split_diff = 0.25 * (self.ddt_prod + self.ddt_conv + 
                                    self.ddt_cross - 3 * self.ddt_diff)
            
            # Convection split constants                        
            self.split_conv = 0.25 * (self.ddt_prod + self.ddt_diff + 
                                    self.ddt_cross - 3 * self.ddt_conv)
            
            # Production split constants
            self.split_prod = 0.5 * (self.ddt_conv + self.ddt_diff + 
                                   self.ddt_cross - self.ddt_prod)
        else:
            # Simple splitting - cross terms in diffusion
            self.split_diff = self.ddt_cross
            self.split_conv.fill(0)
            self.split_prod.fill(0)
        
            
    def update_derivatives(self, delta_conv: np.ndarray, delta_diff: np.ndarray,
                         delta_prod: np.ndarray, dt: float):
        """Update time derivatives from changes over timestep"""
        self.ddt_conv = delta_conv / dt - self.split_conv
        self.ddt_diff = delta_diff / dt - self.split_diff
        self.ddt_prod = delta_prod / dt - self.split_prod
        
    def get_split_constants(self, process: str) -> np.ndarray:
        """Get split constants for specific process"""
        if process == 'convection':
            return self.split_conv
        elif process == 'diffusion':
            return self.split_diff
        elif process == 'production':
            return self.split_prod
        else:
            raise ValueError(f"Unknown process: {process}")
            
    def reset(self):
        """Reset all split constants"""
        self.split_conv.fill(0)
        self.split_diff.fill(0)
        self.split_prod.fill(0)
        
    def resize(self, n_points: int, n_spec: Optional[int] = None):
        """Resize arrays for new grid"""
        if n_spec is not None:
            self.n_spec = n_spec
            self.n_vars = n_spec + 2
            
        self.n_points = n_points
        
        # Resize all arrays
        shape = (self.n_vars, n_points)
        self.split_conv = np.zeros(shape)
        self.split_diff = np.zeros(shape)
        self.split_prod = np.zeros(shape)
        
        self.ddt_conv = np.zeros(shape)
        self.ddt_diff = np.zeros(shape)
        self.ddt_prod = np.zeros(shape)
        self.ddt_cross = np.zeros(shape)