#!/usr/bin/env python3
"""
TOPCART - Topological Cartesian Database System

This module ensures the multi-cube orchestrator architecture is used by default
and provides a unified interface for all TOPCART functionality.
"""

# Force multi-cube architecture by default
from .topcart_config import force_multi_cube_architecture, create_topcart_system

# Automatically configure for multi-cube mode
force_multi_cube_architecture()

# Main TOPCART interface
from .multi_cube_orchestrator import MultiCubeOrchestrator
from .proper_cartesian_engine import ProperCartesianEngine, CartesianPosition
from .coordinate_engine import EnhancedCoordinateEngine
from .ollama_integration import OllamaLLMIntegrator

# Configuration and validation
from .topcart_config import (
    get_topcart_config, 
    enable_benchmark_mode,
    validate_topcart_architecture,
    print_topcart_status
)

# Default TOPCART system factory
def create_default_topcart():
    """
    Create default TOPCART system (always multi-cube orchestrator).
    
    This ensures consistent architecture usage across all applications.
    """
    return create_topcart_system()

# Convenience aliases
TOPCART = create_default_topcart
MultiCubeTOPCART = MultiCubeOrchestrator

# Version and metadata
__version__ = "1.0.0"
__author__ = "TOPCART Development Team"
__description__ = "Topological Cartesian Database with Multi-Cube Orchestrator"

# Export main components
__all__ = [
    # Main system
    'create_default_topcart',
    'TOPCART',
    'MultiCubeTOPCART',
    
    # Core components
    'MultiCubeOrchestrator',
    'ProperCartesianEngine', 
    'CartesianPosition',
    'EnhancedCoordinateEngine',
    'OllamaLLMIntegrator',
    
    # Configuration
    'get_topcart_config',
    'enable_benchmark_mode',
    'validate_topcart_architecture',
    'print_topcart_status',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]

# Print initialization status
print("🚀 TOPCART initialized with multi-cube orchestrator architecture")
print("   Use topcart.create_default_topcart() for consistent multi-cube system")
