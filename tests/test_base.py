"""
Tests for base components
"""
import pytest
from pyember.core.base import EmberComponent

def test_ember_component_requires_implementation():
    """Test that EmberComponent cannot be instantiated without implementation."""
    with pytest.raises(TypeError):
        EmberComponent()

def test_ember_component_configuration():
    """Test component configuration handling."""
    class TestComponent(EmberComponent):
        def initialize(self):
            pass
        def validate(self):
            return True
    
    config = {'test': 'value'}
    component = TestComponent(config)
    assert component._config == config