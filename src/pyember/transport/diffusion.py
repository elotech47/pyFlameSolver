"""
Diffusion system implementation for PyEmber flame solver.
"""
from typing import Optional, Tuple
import numpy as np
from ..core.base import TransportComponent
from ..solvers.integrator import TridiagonalSystem
from ..core.grid import OneDimGrid

from enum import Enum

class BoundaryCondition(Enum):
    """Boundary conditions for diffusion system"""
    FixedValue = "fixed"
    ZeroGradient = "zeroGradient" 
    ControlVolume = "controlVolume"
    WallFlux = "wallFlux"

class DiffusionSystem(TransportComponent, TridiagonalSystem):
    """
    System representing species and heat diffusion in the flame with proper boundary conditions.
    """
    def __init__(self, grid: OneDimGrid, config: Optional[dict] = None):
        super().__init__(config)
        self.grid = grid
        self.N = len(grid.x)
        self.n_points = self.N
    
        # Set boundary conditions
        self.leftBC = BoundaryCondition.FixedValue
        self.rightBC = BoundaryCondition.FixedValue
        
        # Initialize system arrays
        self.initialize()
        
    def initialize(self) -> None:
        """Initialize system arrays"""
        n = self.n_points
        self.B = np.ones(n)  # Default unit prefactor
        self.D = np.ones(n)  # Default unit diffusion
        self.rho = np.ones(n)  # Default unit density
        self.splitConst = np.zeros(self.N)  # Split constants
        self.c1 = np.zeros(self.N)
        self.c2 = np.zeros(self.N)
        
        # Wall flux boundary condition parameters
        self.yInf = 0.0  # Value at infinity
        self.wallConst = 0.0  # Wall conductance
        
    def validate(self) -> bool:
        """Validate system state"""
        if self.B is None or self.D is None or self.rho is None:
            return False
        if len(self.B) != self.n_points:
            return False
        if np.any(self.D <= 0):  # Diffusion must be positive
            return False
        return True
    
    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get coefficients for the tridiagonal system,
        following the C++ approach exactly for both fixed value and zero gradient BCs.
        """
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        c = np.zeros(self.N)

        # Compute c1, c2 coefficients (same as C++ logic)
        for j in range(1, self.N-1):
            r_j = self.grid.r[j] if self.grid.r is not None else 1.0
            rphalf_j = (0.5 * (self.grid.r[j] + self.grid.r[j+1])
                        if self.grid.r is not None else 1.0)
            dlj = 0.5 * (self.grid.x[j+1] - self.grid.x[j-1])
            
            # As per C++ code:
            self.c1[j] = 0.5 * self.B[j] / (dlj * r_j)
            self.c2[j] = rphalf_j * (self.D[j] + self.D[j+1]) / self.grid.hh[j]

        # Left boundary conditions
        if self.leftBC == BoundaryCondition.FixedValue:
            # As in C++:
            # jStart = 1
            # b[0] = 1.0; a[0] = 0.0; c[0] = 0.0
            jStart = 1
            b[0] = 1.0
            a[0] = 0.0
            c[0] = 0.0

        elif self.leftBC == BoundaryCondition.ControlVolume:
            # As in C++:
            jStart = 1
            c0 = self.B[0]*(self.grid.alpha+1)*(self.D[0]+self.D[1])/(2*self.grid.hh[0]*self.grid.hh[0])
            b[0] = -c0
            c[0] = c0

        elif self.leftBC == BoundaryCondition.WallFlux:
            # As in C++:
            jStart = 1
            c0 = self.B[0]*(self.grid.alpha+1)/self.grid.hh[0]
            d = 0.5*(self.D[0]+self.D[1])
            b[0] = -c0*(d/self.grid.hh[0] + self.wallConst)
            c[0] = d*c0/self.grid.hh[0]

        elif self.leftBC == BoundaryCondition.ZeroGradient:
            # As in C++:
            # jStart = 2
            # b[1] = -c1[1]*c2[1]
            # c[1] = c1[1]*c2[1]
            # a[1] = 0.0 (not explicitly set in C++, assumed zero)
            jStart = 2
            b[1] = -self.c1[1] * self.c2[1]
            c[1] = self.c1[1] * self.c2[1]
            a[1] = 0.0

        else:
            raise ValueError(f"Unsupported left boundary condition: {self.leftBC}")

        # Right boundary conditions
        if self.rightBC == BoundaryCondition.FixedValue:
            # As in C++:
            # jStop = N-1
            # b[-1] = 1.0; a[-1] = 0.0; c[-1] = 0.0
            jStop = self.N - 1
            b[-1] = 1.0
            a[-1] = 0.0
            c[-1] = 0.0

        elif self.rightBC == BoundaryCondition.ZeroGradient:
            # As in C++:
            # jStop = N-2
            # a[N-2] = c1[N-2]*c2[N-3]
            # b[N-2] = -c1[N-2]*c2[N-3]
            # c[N-2] = 0.0 (assume)
            jStop = self.N - 2
            a[self.N-2] = self.c1[self.N-2]*self.c2[self.N-3]
            b[self.N-2] = -self.c1[self.N-2]*(self.c2[self.N-3])
            c[self.N-2] = 0.0

        else:
            raise ValueError(f"Unsupported right boundary condition: {self.rightBC}")

        # Interior points (as in C++)
        for j in range(jStart, jStop):
            # The C++ code sets interior points as:
            # a[j] = c1[j]*c2[j-1]
            # b[j] = -c1[j]*(c2[j-1] + c2[j])
            # c[j] = c1[j]*c2[j]
            # Note: We must not overwrite the points where we set boundary conditions (j=1 and j=N-2)
            if j not in [1, self.N-2]:
                a[j] = self.c1[j]*self.c2[j-1]
                b[j] = -self.c1[j]*(self.c2[j-1] + self.c2[j])
                c[j] = self.c1[j]*self.c2[j]

        return a, b, c


    
    # def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Get coefficients for the tridiagonal system with corrected zero gradient implementation.
    #     """
    #     # Initialize coefficient arrays
    #     a = np.zeros(self.N)  # Lower diagonal
    #     b = np.zeros(self.N)  # Main diagonal
    #     c = np.zeros(self.N)  # Upper diagonal

    #     # Compute intermediate coefficients (c1, c2)
    #     for j in range(1, self.N-1):
    #         r_j = self.grid.r[j] if self.grid.r is not None else 1.0
    #         rphalf_j = (0.5 * (self.grid.r[j] + self.grid.r[j+1])
    #                    if self.grid.r is not None else 1.0)
            
    #         dlj = 0.5 * (self.grid.x[j+1] - self.grid.x[j-1])
    #         self.c1[j] = 0.5 * self.B[j] / (dlj * r_j)
    #         self.c2[j] = rphalf_j * (self.D[j] + self.D[j+1]) / self.grid.hh[j]

    #     # Left boundary
    #     if self.leftBC == BoundaryCondition.FixedValue:
    #         print('Using fixed value boundary condition')
    #         jStart = 1
    #         b[0] = 1.0
    #         a[0] = 0.0
    #         c[0] = 0.0

    #     elif self.leftBC == BoundaryCondition.ControlVolume:
    #         jStart = 1
    #         c0 = self.B[0] * (self.grid.alpha + 1) * (self.D[0] + self.D[1]) / (2 * self.grid.hh[0] * self.grid.hh[0])
    #         b[0] = -c0
    #         c[0] = c0

    #     elif self.leftBC == BoundaryCondition.WallFlux:
    #         jStart = 1
    #         c0 = self.B[0] * (self.grid.alpha + 1) / self.grid.hh[0]
    #         d = 0.5 * (self.D[0] + self.D[1])
    #         b[0] = -c0 * (d / self.grid.hh[0] + self.wallConst)
    #         c[0] = d * c0 / self.grid.hh[0]

    #     elif self.leftBC == BoundaryCondition.ZeroGradient:
    #         print('Using zero gradient boundary condition')
    #         jStart = 2
    #         # Match the C++ condition:
    #         # b[1] = -c1[1]*c2[1]
    #         # c[1] = c1[1]*c2[1]
    #         # a[1] is not explicitly set in C++, assume 0.0 is fine.
    #         b[1] = -self.c1[1] * self.c2[1]
    #         c[1] = self.c1[1] * self.c2[1]
    #         a[1] = 0.0
    #     else:
    #         raise ValueError(f"Unsupported left boundary condition - {self.leftBC}")

    #     # Right boundary
    #     if self.rightBC == BoundaryCondition.FixedValue:
    #         print('Using fixed value boundary condition')
    #         jStop = self.N - 1
    #         b[-1] = 1.0
    #         a[-1] = 0.0
    #         c[-1] = 0.0

    #     elif self.rightBC == BoundaryCondition.ZeroGradient:
    #         jStop = self.N - 2
    #         print('Using zero gradient boundary condition')
    #         # Match the C++ condition:
    #         # a[N-2] = c1[N-2]*c2[N-3]
    #         # b[N-2] = -c1[N-2]*c2[N-3]
    #         # c[N-2] not set in C++ snippet, assume c[N-2] = 0.0
    #         a[self.N-2] = self.c1[self.N-2] * self.c2[self.N-3]
    #         b[self.N-2] = -self.c1[self.N-2] * self.c2[self.N-3]
    #         c[self.N-2] = 0.0
        
    #     else:
    #         raise ValueError(f"Unsupported right boundary condition - {self.rightBC}")

    #     # Interior points
    #     for j in range(jStart, jStop):
    #         if j not in [1, self.N-2]:  # Skip boundary adjacent points handled above
    #             a[j] = self.c1[j] * self.c2[j-1]
    #             b[j] = -self.c1[j] * (self.c2[j-1] + self.c2[j])
    #             c[j] = self.c1[j] * self.c2[j]

    #     return a, b, c

    def get_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute right-hand side including boundary conditions.
        Matching C++ implementation exactly.
        """
        k = self.splitConst.copy()
        
        if self.leftBC == BoundaryCondition.WallFlux:
            k[0] += (self.B[0] * (self.grid.alpha + 1) / self.grid.hh[0] * 
                    self.wallConst * self.yInf)
        
        return k
        
    def set_wall_flux(self, yInf: float, wallConst: float):
        """Set wall flux boundary condition parameters"""
        self.yInf = yInf
        self.wallConst = wallConst
        
    def set_boundary_conditions(self, 
                              leftBC: BoundaryCondition,
                              rightBC: BoundaryCondition):
        """Set boundary conditions for both ends"""
        self.leftBC = leftBC
        self.rightBC = rightBC
        
    def set_properties(self, D: np.ndarray, rho: np.ndarray, B: Optional[np.ndarray] = None):
        """
        Set physical properties for the system.
        
        Args:
            D: Diffusion coefficients
            rho: Density
            B: Optional prefactor terms (defaults to ones)
        """
        if len(D) != self.n_points or len(rho) != self.n_points:
            raise ValueError("Property arrays must match grid size")
            
        self.D = D
        self.rho = rho
        if B is not None:
            if len(B) != self.n_points:
                raise ValueError("Prefactor array must match grid size")
            self.B = B
        
    def reset_split_constants(self):
        """Reset operator splitting terms"""
        self.splitConst.fill(0.0)
            
    def compute_properties(self) -> None:
        return super().compute_properties()