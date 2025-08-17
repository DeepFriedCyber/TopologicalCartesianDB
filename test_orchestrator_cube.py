#!/usr/bin/env python3
"""
Test the new Orchestrator Cube architecture
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.topological_cartesian.orchestrator_cube import OrchestratorCube
from src.topological_cartesian.multi_cube_orchestrator import CartesianCube, CubeType
from src.topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

def test_orchestrator_cube():
    """Test the orchestrator cube functionality"""
    
    print("🎯 Testing Orchestrator Cube Architecture...")
    
    # Create orchestrator cube
    orchestrator = OrchestratorCube()
    
    # Create some mock worker cubes
    worker_cubes = {
        'code_cube': CartesianCube(
            name='code_cube',
            cube_type=CubeType.CODE,
            dimensions=['complexity', 'maintainability'],
            coordinate_ranges={'complexity': (0.0, 1.0), 'maintainability': (0.0, 1.0)},
            specialization='code_analysis',
            coordinate_engine=EnhancedCoordinateEngine(),
            expertise_domains=['programming']
        ),
        'data_cube': CartesianCube(
            name='data_cube',
            cube_type=CubeType.DATA,
            dimensions=['volume', 'complexity'],
            coordinate_ranges={'volume': (0.0, 1.0), 'complexity': (0.0, 1.0)},
            specialization='data_processing',
            coordinate_engine=EnhancedCoordinateEngine(),
            expertise_domains=['data_analysis']
        )
    }
    
    # Register worker cubes
    for cube in worker_cubes.values():
        orchestrator.register_worker_cube(cube)
    
    # Test orchestration with different query types
    test_queries = [
        "Find code functions with high complexity",
        "Analyze data processing performance over time",
        "Compare user behavior patterns with system performance metrics"
    ]
    
    print(f"\n🔍 Testing {len(test_queries)} orchestration scenarios:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: '{query}'")
        
        # Analyze orchestration requirements
        query_coords = orchestrator._analyze_query_orchestration_requirements(query)
        print(f"   🎯 Query coordinates: {query_coords}")
        
        # Execute orchestration
        result = orchestrator.orchestrate_query(query)
        
        print(f"   ✅ Strategy used: {result.strategy_used}")
        print(f"   📊 Cross-cube coherence: {result.cross_cube_coherence:.3f}")
        print(f"   🎯 Accuracy estimate: {result.accuracy_estimate:.3f}")
        print(f"   ⚡ Processing time: {result.total_processing_time:.3f}s")
        print(f"   🧊 Cube results: {len(result.cube_results)} cubes responded")
    
    # Get analytics
    analytics = orchestrator.get_orchestration_analytics()
    print(f"\n📈 Orchestration Analytics:")
    print(f"   📊 Total orchestrations: {analytics['total_orchestrations']}")
    print(f"   🎯 Strategy performance:")
    
    for strategy, perf in analytics['strategy_performance'].items():
        print(f"      • {strategy}: {perf['usage_count']} uses, "
              f"effectiveness: {perf['effectiveness_score']:.3f}")
    
    print(f"   🧊 Worker cubes: {analytics['worker_cubes_registered']}")
    
    print(f"\n🎉 Orchestrator Cube architecture test completed!")
    print(f"✅ Successfully treats orchestration as a specialized coordinate domain")

if __name__ == "__main__":
    test_orchestrator_cube()
