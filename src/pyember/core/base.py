"""
Base classes and interfaces for PyEmber components.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class EmberComponent(ABC):
    """
    Base class for all PyEmber components providing common functionality
    and enforcing interface requirements.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component with current configuration."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate component state and configuration."""
        pass

class GridComponent(EmberComponent):
    """Base class for grid-related components."""
    @abstractmethod
    def update_grid_metrics(self) -> None:
        """Update grid metrics including spacing and finite difference coefficients."""
        pass

class TransportComponent(EmberComponent):
    """Base class for transport-related components."""
    @abstractmethod
    def compute_properties(self) -> None:
        """Compute transport properties."""
        pass

class ChemistryComponent(EmberComponent):
    """Base class for chemistry-related components."""
    @abstractmethod
    def compute_rates(self) -> None:
        """Compute reaction rates."""
        pass

class IntegratorComponent(EmberComponent):
    """Base class for integrator components."""
    @abstractmethod
    def step(self) -> None:
        """Advance solution by one timestep."""
        pass