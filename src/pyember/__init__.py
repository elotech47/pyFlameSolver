"""
PyEmber: Python implementation of Ember flame solver
"""
from importlib.metadata import version

__version__ = version("pyember")

from .core.base import (
    EmberComponent,
    GridComponent,
    TransportComponent,
    ChemistryComponent,
    IntegratorComponent
)