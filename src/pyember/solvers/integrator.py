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
    """
    def __init__(self, system: TridiagonalSystem, config: Optional[dict] = None):
        super().__init__(system, config)
        self.system = system
        self.step_count = 0
        self.y_prev = None
        self.y_prev2 = None
        
    def initialize(self, t0: float, dt: float) -> None:
        """Initialize the integrator"""
        super().initialize(t0, dt)
        self.step_count = 0
        if self.y is not None:
            self.y_prev = self.y.copy()
            self.ydot = self.system.get_rhs(self.t, self.y)
        self._initialized = True
        
    def step(self) -> None:
        """Take one step using BDF methods"""
        if not self.is_initialized():
            raise RuntimeError("Integrator must be initialized before stepping")
            
        if self.step_count == 0:
            # First step: Use backward Euler (BDF1)
            # dy/dt = f(t,y)
            # (y(n+1) - y(n))/dt = f(t(n+1), y(n+1))
            a, b, c = self.system.get_coefficients()
            
            # Build system matrix: (I - dt*J)
            M = np.eye(len(self.y))
            for i in range(len(self.y)):
                if i > 0:
                    M[i,i-1] = -self.dt * a[i]
                M[i,i] = 1.0 - self.dt * b[i]
                if i < len(self.y)-1:
                    M[i,i+1] = -self.dt * c[i]
            
            self.y_prev = self.y.copy()
            self.y = np.linalg.solve(M, self.y)
            
        else:
            # Subsequent steps: Use BDF2
            # (3y(n+1) - 4y(n) + y(n-1))/(2dt) = f(t(n+1), y(n+1))
            self.y_prev2 = self.y_prev.copy()
            self.y_prev = self.y.copy()
            
            a, b, c = self.system.get_coefficients()
            
            # Build system matrix: (3I - 2dt*J)
            M = 3.0 * np.eye(len(self.y))
            for i in range(len(self.y)):
                if i > 0:
                    M[i,i-1] = -2.0 * self.dt * a[i]
                M[i,i] = 3.0 - 2.0 * self.dt * b[i]
                if i < len(self.y)-1:
                    M[i,i+1] = -2.0 * self.dt * c[i]
            
            # RHS: 4y(n) - y(n-1)
            rhs = 4.0 * self.y_prev - self.y_prev2
            self.y = np.linalg.solve(M, rhs)
            
        self.step_count += 1
        self.t += self.dt
        self.ydot = self.system.get_rhs(self.t, self.y)
        
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