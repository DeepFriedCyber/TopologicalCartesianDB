"""
Demo script for the Hybrid Cube Architecture of TopologicalCartesianDB.

This script demonstrates how to use the core TopologicalCartesianDB with
different feature cubes like ParsevalCube, ProvenanceCube, and OptimizationCube.
"""
import numpy as np
from core.spatial_db import TopologicalCartesianDB
from cubes.parseval_cube import ParsevalCube
from cubes.provenance_cube import ProvenanceCube
from cubes.optimization_cube import OptimizationCube
import time


def demo_basic_tcdb():
    """Demonstrate basic TopologicalCartesianDB functionality."""
    print("\n===== Basic TopologicalCartesianDB Demo =====")
    
    # Create a basic database instance
    db = TopologicalCartesianDB(cell_size=1.0)
    
    # Insert some points with associated data
    db.insert(1.0, 2.0, {"name": "Point A"})
    db.insert(2.0, 3.0, {"name": "Point B"})
    db.insert(0.5, 1.5, {"name": "Point C"})
    
    print(f"Database has {len(db.points)} points")
    
    # Query for points
    results = db.query_with_data(1.0, 2.0, 1.0)
    print(f"Query results within radius 1.0 of [1.0, 2.0]:")
    for i, (point, data) in enumerate(results):
        # Calculate distance
        distance = ((point[0] - 1.0)**2 + (point[1] - 2.0)**2)**0.5
        print(f"  Result {i+1}: {data['name']} at {point}, distance: {distance:.4f}")
    
    return db


def demo_parseval_cube():
    """Demonstrate ParsevalCube functionality."""
    print("\n===== ParsevalCube Demo =====")
    
    # Create a database with ParsevalCube
    db = TopologicalCartesianDB(cell_size=1.0)
    parseval_cube = ParsevalCube(db, dimensions=2)
    
    # Insert vectors with ParsevalCube
    vector1 = [1.0, 2.0]
    vector2 = [2.0, 3.0]
    
    # Create vector IDs
    vec_id1 = "vector_a"
    vec_id2 = "vector_b"
    
    parseval_cube.insert_vector(vec_id1, vector1)
    parseval_cube.insert_vector(vec_id2, vector2)
    
    # Verify Parseval's theorem using verify_parseval_equality
    is_preserved1 = parseval_cube.verify_parseval_equality(vector1)
    is_preserved2 = parseval_cube.verify_parseval_equality(vector2)
    print(f"Vector 1 preserves energy (Parseval): {is_preserved1}")
    print(f"Vector 2 preserves energy (Parseval): {is_preserved2}")
    
    # Project vector onto basis
    standard_basis = parseval_cube._create_standard_basis()
    vector_array = np.array(vector1, dtype=float)
    coefficients = parseval_cube._project_vector(vector_array, standard_basis)
    print(f"Original vector: {vector1}")
    print(f"Projected coefficients: {coefficients}")
    
    # Get stats about vectors
    stats = parseval_cube.get_stats()
    print(f"ParsevalCube stats: {stats}")
    
    return parseval_cube


def demo_provenance_cube():
    """Demonstrate ProvenanceCube functionality."""
    print("\n===== ProvenanceCube Demo =====")
    
    # Create a database with ProvenanceCube
    db = TopologicalCartesianDB(cell_size=1.0)
    provenance_cube = ProvenanceCube(db)
    
    # Insert some points with provenance information
    provenance_cube.insert(1.0, 2.0, {"name": "Point A", "source": "Sensor 1"})
    provenance_cube.insert(2.0, 3.0, {"name": "Point B", "source": "Sensor 2"})
    
    # Query with provenance tracking
    query_x, query_y = 1.5, 2.5
    query_result = provenance_cube.query_with_provenance(query_x, query_y, radius=2.0)
    
    # Extract results and provenance information
    results = query_result['results']
    provenance_info = query_result['provenance']
    
    print(f"Query with provenance for point [{query_x}, {query_y}], radius 2.0:")
    for i, point_data in enumerate(results):
        if isinstance(point_data, tuple) and len(point_data) == 2:
            point, data = point_data
            if isinstance(data, dict):
                print(f"  Result {i+1}: {data.get('name')} from {data.get('source')}")
                # Calculate distance
                distance = ((point[0] - query_x)**2 + (point[1] - query_y)**2)**0.5
                print(f"    Distance: {distance:.4f}")
    
    print("\nProvenance Information:")
    print(f"  Query ID: {provenance_info.get('query_id')}")
    print(f"  Execution Time: {provenance_info.get('execution_time', 0):.6f} seconds")
    print(f"  Result Count: {provenance_info.get('result_count', 0)}")
    
    # Get query history
    history = provenance_cube.get_query_history()
    print("\nQuery History:")
    for i, entry in enumerate(history[:3]):  # Show just the first 3 entries
        print(f"  Query {i+1}: {entry.get('query_id')} - {entry.get('query_type')}")
    
    return provenance_cube


def demo_optimization_cube():
    """Demonstrate OptimizationCube functionality."""
    print("\n===== OptimizationCube Demo =====")
    
    # Create a database with OptimizationCube
    db = TopologicalCartesianDB(cell_size=1.0)
    optimization_cube = OptimizationCube(db, enable_cache=True, enable_early_termination=True)
    
    # Insert many points to test performance
    print("Inserting 1000 points...")
    for i in range(1000):
        x, y = np.random.random(), np.random.random()
        optimization_cube.insert(x, y, {"id": i})
    
    # Measure query performance directly
    print("Measuring query performance...")
    start_time = time.time()
    optimization_cube.query(0.5, 0.5, 0.2)
    query_time = time.time() - start_time
    print(f"Query time: {query_time:.6f} seconds")
    
    # Test the optimized vector query if available
    if hasattr(optimization_cube, 'query_vector_optimized'):
        print("\nTesting optimized vector query...")
        start_time = time.time()
        result = optimization_cube.query_vector_optimized([0.5, 0.5], 0.2)
        opt_query_time = time.time() - start_time
        print(f"Optimized query time: {opt_query_time:.6f} seconds")
        print(f"Results found: {len(result['results'])}")
        print(f"Optimization info: {result['optimization']}")
    
    # Get optimization metrics
    metrics = optimization_cube.get_optimization_metrics()
    print("\nOptimization Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Try to optimize
    print("\nOptimizing database...")
    opt_results = optimization_cube.optimize()
    print(f"Optimization results: {opt_results}")
    
    # Measure query performance after optimization
    start_time = time.time()
    optimization_cube.query(0.5, 0.5, 0.2)
    optimized_query_time = time.time() - start_time
    print(f"Query time after optimization: {optimized_query_time:.6f} seconds")
    if query_time > 0:
        print(f"Improvement: {(1 - optimized_query_time/query_time) * 100:.2f}%")
    
    return optimization_cube


def demo_combined_cubes():
    """Demonstrate using multiple cubes together."""
    print("\n===== Combined Cubes Demo =====")
    
    # To avoid complications with the adapter structure, we'll create a simpler demo
    # We'll use a fresh base database with each cube separately
    
    # Create a base database
    db = TopologicalCartesianDB(cell_size=1.0)
    
    # Create individual cubes (not chained together)
    parseval_cube = ParsevalCube(db, dimensions=2)
    provenance_cube = ProvenanceCube(db)  # Connect directly to base db
    optimization_cube = OptimizationCube(db)  # Connect directly to base db
    
    print("Testing multi-cube functionality:")
    
    # Insert data with parseval
    vec_id = "combined_test_vector"
    vector = [1.0, 2.0]
    parseval_cube.insert_vector(vec_id, vector)
    
    # Insert with base db (all cubes will see it)
    db.insert(1.5, 2.5, {"name": "Combined Test", "source": "Demo"})
    
    # Query with each cube
    print("\nQuerying with each cube type:")
    
    # Base DB query
    base_results = db.query_with_data(1.0, 2.0, radius=1.0)
    print(f"Base DB found {len(base_results)} results")
    
    # Provenance cube query
    prov_result = provenance_cube.query_with_provenance(1.0, 2.0, radius=1.0)
    prov_results = prov_result.get('results', [])
    print(f"Provenance cube found {len(prov_results)} results with tracking")
    
    # Optimization metrics
    opt_metrics = optimization_cube.get_optimization_metrics()
    print(f"Optimization metrics available: {list(opt_metrics.keys())}")
    
    # Show adapter type from each cube
    print("\nAdapter types:")
    print(f"  Parseval: {parseval_cube.__class__.__name__}")
    print(f"  Provenance: {provenance_cube.__class__.__name__}")
    print(f"  Optimization: {optimization_cube.__class__.__name__}")
    
    return db, parseval_cube, provenance_cube, optimization_cube


if __name__ == "__main__":
    print("=== TopologicalCartesianDB with Hybrid Cube Architecture Demo ===\n")
    
    # Run each demo
    basic_db = demo_basic_tcdb()
    parseval = demo_parseval_cube()
    provenance = demo_provenance_cube()
    optimization = demo_optimization_cube()
    
    # Demonstrate combined functionality
    combined = demo_combined_cubes()
    
    print("\n=== Demo Complete ===")
