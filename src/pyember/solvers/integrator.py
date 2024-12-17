from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Callable
from ..core.base import IntegratorComponent

class ODESystem(ABC):
    """Abstract base class for ODE systems to be integrated"""
    
    @abstractmethod
    def get_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Evaluate right-hand side of the ODE system dy/dt = f(t,y)
        
        Args:
            t: Current time
            y: Current state vector
            
        Returns:
            np.ndarray: Right-hand side evaluation f(t,y)
        """
        pass

class BaseIntegrator(IntegratorComponent):
    """Base class for numerical integrators"""
    
    def __init__(self, system: ODESystem, config: Optional[dict] = None):
        super().__init__(config)
        self.system = system
        self.t: float = 0.0  # Current time
        self.dt: float = 0.0  # Current timestep
        self.y: np.ndarray = None  # Current solution
        self.ydot: np.ndarray = None  # Current time derivative
        
    @abstractmethod
    def initialize(self, t0: float, dt: float) -> None:
        """
        Initialize the integrator
        
        Args:
            t0: Initial time
            dt: Initial timestep
        """
        self.t = t0
        self.dt = dt
        
    @abstractmethod
    def step(self) -> None:
        """Advance solution by one timestep"""
        pass
        
    def set_y0(self, y0: np.ndarray) -> None:
        """Set initial condition"""
        self.y = y0.copy()
        
    def get_y(self) -> np.ndarray:
        """Get current solution"""
        return self.y
        
    def get_ydot(self) -> np.ndarray:
        """Get current time derivative"""
        return self.ydot

class TridiagonalSystem(ODESystem):
    """System with tridiagonal Jacobian structure"""
    
    def __init__(self):
        """Initialize base system"""
        pass
    
    @abstractmethod
    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get tridiagonal system coefficients
        
        Returns:
            Tuple containing:
            - Lower diagonal coefficients
            - Main diagonal coefficients  
            - Upper diagonal coefficients
        """
        pass

class TridiagonalIntegrator(BaseIntegrator):
    """
    Integrator specialized for tridiagonal systems using BDF methods
    Matches C++ implementation with optimized Python code
    """
    def __init__(self, system: TridiagonalSystem, config: Optional[dict] = None):
        super().__init__(system, config)
        self.system = system
        self.step_count = 0
        
        # Arrays for solution history
        self.y_prev = None  # y_(n-1)
        
        # Arrays for Thomas algorithm
        self.lu_b = None  # Modified diagonal elements
        self.lu_c = None  # Modified upper diagonal
        self.lu_d = None  # Modified RHS
        self.inv_denom = None  # 1/(b[i] - c[i-1]*a[i])
        
    def initialize(self, t0: float, dt: float) -> None:
        """Initialize the integrator"""
        super().initialize(t0, dt)
        self.step_count = 0
        
        if self.y is not None:
            N = len(self.y)
            # Initialize arrays for Thomas algorithm
            self.lu_b = np.zeros(N)
            self.lu_c = np.zeros(N)
            self.lu_d = np.zeros(N)
            self.inv_denom = np.zeros(N)
            self.y_prev = self.y.copy()
            
        self._initialized = True
        
    def step(self) -> None:
        """Take one step using BDF methods"""
        if not self.is_initialized():
            raise RuntimeError("Integrator must be initialized before stepping")
            
        if self.step_count == 0:
            # First timestep: Take 8 substeps using BDF1
            self.y_prev = self.y.copy()  # Save current y as y_(n-1)
            
            # Get ODE coefficients
            a, b, c = self.system.get_coefficients()
            k = self.system.get_rhs(self.t, self.y)
            
            # Modify diagonal for BDF1
            sub_dt = self.dt / 8.0
            self.lu_b = b - 1.0/sub_dt
            
            # Precompute Thomas algorithm coefficients
            N = len(self.y)
            self.lu_c[0] = c[0] / self.lu_b[0]
            for i in range(1, N):
                self.inv_denom[i] = 1.0 / (self.lu_b[i] - self.lu_c[i-1] * a[i])
                self.lu_c[i] = c[i] * self.inv_denom[i]
            
            # Take 8 substeps
            for _ in range(8):
                # RHS for BDF1
                self.y = -self.y/sub_dt - k
                
                # Forward substitution
                self.lu_d[0] = self.y[0] / self.lu_b[0]
                for i in range(1, N):
                    self.lu_d[i] = (self.y[i] - self.lu_d[i-1]*a[i]) * self.inv_denom[i]
                
                # Back substitution
                self.y[N-1] = self.lu_d[N-1]
                for i in range(N-2, -1, -1):
                    self.y[i] = self.lu_d[i] - self.lu_c[i] * self.y[i+1]
                
        else:
            # Get ODE coefficients - needed for all steps
            a, b, c = self.system.get_coefficients()
            k = self.system.get_rhs(self.t, self.y)
            
            if self.step_count == 1:
                # Setup for BDF2 - only done once
                # Modify diagonal elements for BDF2
                self.lu_b = b - 3.0/(2.0*self.dt)
                
                # Precompute Thomas algorithm coefficients
                N = len(self.y)
                self.lu_c[0] = c[0] / self.lu_b[0]
                for i in range(1, N):
                    self.inv_denom[i] = 1.0 / (self.lu_b[i] - self.lu_c[i-1] * a[i])
                    self.lu_c[i] = c[i] * self.inv_denom[i]
            
            # Store current y as y_(n-1) for next step
            y_nm1 = self.y.copy()
            
            # RHS for BDF2
            self.y = -2.0 * y_nm1 / self.dt + self.y_prev / (2.0*self.dt) - k
            self.y_prev = y_nm1  # Current y becomes previous y
            
            # Forward substitution
            N = len(self.y)
            self.lu_d[0] = self.y[0] / self.lu_b[0]
            for i in range(1, N):
                self.lu_d[i] = (self.y[i] - self.lu_d[i-1]*a[i]) * self.inv_denom[i]
            
            # Back substitution
            self.y[N-1] = self.lu_d[N-1]
            for i in range(N-2, -1, -1):
                self.y[i] = self.lu_d[i] - self.lu_c[i] * self.y[i+1]
        
        self.step_count += 1
        self.t += self.dt if self.step_count > 1 else self.dt/8.0
        
    def get_ydot(self) -> np.ndarray:
        """Get current time derivative matching C++ implementation"""
        if self.y is None:
            return None
            
        # Get coefficients
        a, b, c = self.system.get_coefficients()
        k = self.system.get_rhs(self.t, self.y)
        
        N = len(self.y)
        ydot = np.zeros(N)
        
        # Calculate ydot using tridiagonal system
        ydot[0] = b[0]*self.y[0] + c[0]*self.y[1] + k[0]
        for i in range(1, N-1):
            ydot[i] = a[i]*self.y[i-1] + b[i]*self.y[i] + c[i]*self.y[i+1] + k[i]
        ydot[N-1] = a[N-1]*self.y[N-2] + b[N-1]*self.y[N-1] + k[N-1]
        
        return ydot
        
    def resize(self, n: int) -> None:
        """Resize the system to a new size"""
        self.y = np.zeros(n)
        self.y_prev = np.zeros(n)
        self.lu_b = np.zeros(n)
        self.lu_c = np.zeros(n)
        self.lu_d = np.zeros(n)
        self.inv_denom = np.zeros(n)
        self.system.resize(n)
        
    def _build_matrix(self, a: np.ndarray, b: np.ndarray, 
                     c: np.ndarray, dt: float) -> np.ndarray:
        """Build the tridiagonal system matrix"""
        N = len(self.y)
        M = np.zeros((N, N))
        
        for i in range(N):
            if i > 0:
                M[i,i-1] = -dt * a[i]
            M[i,i] = 1.0 - dt * b[i] 
            if i < N-1:
                M[i,i+1] = -dt * c[i]
                
        return M
    