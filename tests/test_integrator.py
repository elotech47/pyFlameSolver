"""
Tests for numerical integrators
"""
import pytest
import numpy as np
from pyember.solvers.integrator import (
    ODESystem, BaseIntegrator, TridiagonalSystem, TridiagonalIntegrator
)
from typing import Tuple

class DecaySystem(TridiagonalSystem):
    """Test system with known analytical solution"""
    
    def get_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Simple decay equation dy/dt = k*y"""
        k = -2.0  # Decay rate
        return k * y
        
    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return coefficients for tridiagonal system"""
        N = len(self.y) if hasattr(self, 'y') else 5
        k = -2.0
        a = np.zeros(N)  # Lower diagonal
        b = k * np.ones(N)  # Main diagonal (contains decay rate)
        c = np.zeros(N)  # Upper diagonal
        return a, b, c

def test_tridiagonal_integrator():
    """Test integration of simple tridiagonal system"""
    system = DecaySystem()
    t0 = 0.0
    dt = 0.01
    config = {'t0': t0, 'dt': dt}
    
    integrator = TridiagonalIntegrator(system, config)
    
    # Initial condition
    y0 = np.array([0.0, 0.5, 2.0, 1.0, 0.0])
    integrator.set_y0(y0)
    
    # Initialize and integrate
    tf = 1.0
    integrator.initialize(t0, dt)
    
    while integrator.t < tf:
        integrator.step()
        
    # Check final time
    np.testing.assert_allclose(integrator.t, tf, rtol=1e-10)
    
    # Solution should decay exponentially
    y_final = integrator.get_y()
    y_exact = y0 * np.exp(-2.0 * tf)  # k = -2.0
    np.testing.assert_allclose(y_final, y_exact, rtol=1e-3)

def test_initialization():
    """Test integrator initialization"""
    system = DecaySystem()
    t0 = 0.0
    dt = 0.01
    config = {'t0': t0, 'dt': dt}
    
    integrator = TridiagonalIntegrator(system, config)
    
    y0 = np.ones(5)
    integrator.set_y0(y0)
    
    # Check initial condition
    np.testing.assert_array_equal(integrator.get_y(), y0)
    
    # Initialize with timestep
    integrator.initialize(t0, dt)
    
    assert integrator.t == t0
    assert integrator.dt == dt
    assert integrator.is_initialized()