"""
PyTest configuration and fixtures
"""
import pytest
import numpy as np

@pytest.fixture
def simple_grid():
    """Return a simple uniform grid for testing."""
    return np.linspace(0, 1, 100)

@pytest.fixture
def simple_solution():
    """Return a simple Cantera Solution for testing."""
    import cantera as ct
    return ct.Solution('gri30.yaml')