#!/usr/bin/env python3
"""
TOPCART Configuration System

Ensures the full multi-cube orchestrator architecture is used consistently
across all applications and benchmarks.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TOPCARTMode(Enum):
    """TOPCART system modes"""
    SINGLE_CUBE = "single_cube"           # Simple single coordinate space
    MULTI_CUBE = "multi_cube"             # Full orchestrator with domain experts
    HYBRID = "hybrid"                     # Adaptive between single/multi
    

@dataclass
class TOPCARTConfig:
    """Configuration for TOPCART system"""
    
    # Core architecture
    mode: TOPCARTMode = TOPCARTMode.MULTI_CUBE
    force_orchestrator: bool = True
    enable_dnn_optimization: bool = True
    
    # Cube configuration
    enable_code_cube: bool = True
    enable_data_cube: bool = True
    enable_user_cube: bool = True
    enable_system_cube: bool = True
    enable_temporal_cube: bool = True
    
    # Performance settings
    cube_capacity: int = 1000
    orchestration_strategy: str = "adaptive"
    cross_cube_search: bool = True
    
    # Optimization settings
    swarm_particles: int = 15
    swarm_iterations: int = 30
    equalization_threshold: float = 0.1
    coordination_target: float = 0.85
    
    # Benchmark settings
    benchmark_mode: bool = False
    log_performance: bool = True
    validate_results: bool = True


class TOPCARTConfigManager:
    """Manages TOPCART configuration and ensures consistent architecture usage"""
    
    def __init__(self):
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment or defaults"""
        
        # Check environment variables
        mode_str = os.getenv('TOPCART_MODE', 'multi_cube').lower()
        force_orchestrator = os.getenv('TOPCART_FORCE_ORCHESTRATOR', 'true').lower() == 'true'
        enable_dnn = os.getenv('TOPCART_ENABLE_DNN', 'true').lower() == 'true'
        
        # Map mode string to enum
        mode_map = {
            'single': TOPCARTMode.SINGLE_CUBE,
            'single_cube': TOPCARTMode.SINGLE_CUBE,
            'multi': TOPCARTMode.MULTI_CUBE,
            'multi_cube': TOPCARTMode.MULTI_CUBE,
            'hybrid': TOPCARTMode.HYBRID
        }
        
        mode = mode_map.get(mode_str, TOPCARTMode.MULTI_CUBE)
        
        self._config = TOPCARTConfig(
            mode=mode,
            force_orchestrator=force_orchestrator,
            enable_dnn_optimization=enable_dnn,
            benchmark_mode=os.getenv('TOPCART_BENCHMARK', 'false').lower() == 'true'
        )
    
    def get_config(self) -> TOPCARTConfig:
        """Get current configuration"""
        return self._config
    
    def set_config(self, config: TOPCARTConfig):
        """Set new configuration"""
        self._config = config
    
    def force_multi_cube_mode(self):
        """Force multi-cube orchestrator mode"""
        self._config.mode = TOPCARTMode.MULTI_CUBE
        self._config.force_orchestrator = True
        print("🚀 TOPCART: Forced multi-cube orchestrator mode")
    
    def enable_benchmark_mode(self):
        """Enable benchmark mode with full validation"""
        self._config.benchmark_mode = True
        self._config.log_performance = True
        self._config.validate_results = True
        self._config.force_orchestrator = True
        self._config.mode = TOPCARTMode.MULTI_CUBE
        print("📊 TOPCART: Benchmark mode enabled with full validation")
    
    def validate_architecture(self) -> Dict[str, Any]:
        """Validate that the correct architecture is being used"""
        
        validation = {
            'mode': self._config.mode.value,
            'orchestrator_forced': self._config.force_orchestrator,
            'dnn_enabled': self._config.enable_dnn_optimization,
            'cubes_enabled': {
                'code_cube': self._config.enable_code_cube,
                'data_cube': self._config.enable_data_cube,
                'user_cube': self._config.enable_user_cube,
                'system_cube': self._config.enable_system_cube,
                'temporal_cube': self._config.enable_temporal_cube
            },
            'cross_cube_search': self._config.cross_cube_search,
            'benchmark_mode': self._config.benchmark_mode
        }
        
        # Check if we're in the correct mode
        if self._config.mode != TOPCARTMode.MULTI_CUBE:
            validation['warning'] = f"Not using multi-cube mode: {self._config.mode.value}"
        
        if not self._config.force_orchestrator:
            validation['warning'] = "Orchestrator not forced - may fall back to single cube"
        
        return validation


# Global configuration manager
_config_manager = TOPCARTConfigManager()


def get_topcart_config() -> TOPCARTConfig:
    """Get global TOPCART configuration"""
    return _config_manager.get_config()


def force_multi_cube_architecture():
    """Force multi-cube orchestrator architecture globally"""
    _config_manager.force_multi_cube_mode()


def enable_benchmark_mode():
    """Enable benchmark mode with full validation"""
    _config_manager.enable_benchmark_mode()


def validate_topcart_architecture() -> Dict[str, Any]:
    """Validate current TOPCART architecture"""
    return _config_manager.validate_architecture()


def create_topcart_system():
    """Create TOPCART system using current configuration"""
    
    config = get_topcart_config()
    
    if config.mode == TOPCARTMode.MULTI_CUBE or config.force_orchestrator:
        # Use multi-cube orchestrator
        from .multi_cube_orchestrator import MultiCubeOrchestrator
        
        system = MultiCubeOrchestrator(
            enable_dnn_optimization=config.enable_dnn_optimization
        )
        
        print(f"✅ Created multi-cube TOPCART orchestrator")
        print(f"   🧊 {len(system.cubes)} specialized domain expert cubes")
        print(f"   🚀 DNN optimization: {'✅ Enabled' if config.enable_dnn_optimization else '❌ Disabled'}")
        
        return system
        
    elif config.mode == TOPCARTMode.SINGLE_CUBE:
        # Use single cube system (for comparison)
        from .proper_cartesian_engine import ProperCartesianEngine
        
        system = ProperCartesianEngine()
        print(f"⚠️ Created single-cube TOPCART system (not recommended for production)")
        
        return system
        
    else:
        raise ValueError(f"Unsupported TOPCART mode: {config.mode}")


def print_topcart_status():
    """Print current TOPCART configuration status"""
    
    config = get_topcart_config()
    validation = validate_topcart_architecture()
    
    print("🎯 TOPCART SYSTEM STATUS")
    print("=" * 40)
    print(f"Mode: {config.mode.value}")
    print(f"Orchestrator Forced: {config.force_orchestrator}")
    print(f"DNN Optimization: {config.enable_dnn_optimization}")
    print(f"Cross-Cube Search: {config.cross_cube_search}")
    print(f"Benchmark Mode: {config.benchmark_mode}")
    
    print(f"\nEnabled Cubes:")
    for cube_name, enabled in validation['cubes_enabled'].items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {cube_name}")
    
    if 'warning' in validation:
        print(f"\n⚠️ WARNING: {validation['warning']}")
    else:
        print(f"\n✅ Architecture validation passed")


if __name__ == "__main__":
    print_topcart_status()