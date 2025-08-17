"""
Benchmark script for the Hybrid Cube Architecture of TopologicalCartesianDB.
Tests performance of the core database and various cube combinations.
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from core.spatial_db import TopologicalCartesianDB
from cubes.cube_adapter import CubeAdapter
from cubes.parseval_cube import ParsevalCube
from cubes.provenance_cube import ProvenanceCube
from cubes.optimization_cube import OptimizationCube


def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1_000:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def benchmark_core_db(point_counts=[100, 1000, 10000]):
    """Benchmark the core database operations."""
    print("=" * 50)
    print("BENCHMARK: Core Database Operations")
    print("=" * 50)
    
    results = {'insert': [], 'query': []}
    
    for count in point_counts:
        print(f"\nDataset size: {count} points")
        
        # Create database
        db = TopologicalCartesianDB(cell_size=1.0)
        
        # Benchmark insertion
        start_time = time.time()
        for i in range(count):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            db.insert(x, y, {"id": i})
        insert_time = time.time() - start_time
        
        results['insert'].append(insert_time)
        print(f"Insert {count} points: {format_time(insert_time)} " + 
              f"({count/insert_time:.2f} points/s)")
        
        # Benchmark querying
        query_count = 100
        start_time = time.time()
        for _ in range(query_count):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            db.query(x, y, radius=10.0)
        query_time = time.time() - start_time
        
        results['query'].append(query_time / query_count)
        print(f"Average query time ({query_count} queries): " + 
              f"{format_time(query_time / query_count)}")
    
    return point_counts, results


def benchmark_parseval_cube(vector_counts=[100, 1000, 5000], dimensions=2):
    """Benchmark the ParsevalCube operations."""
    print("=" * 50)
    print(f"BENCHMARK: ParsevalCube Operations ({dimensions}D)")
    print("=" * 50)
    
    results = {'insert': [], 'verify': []}
    
    for count in vector_counts:
        print(f"\nDataset size: {count} vectors")
        
        # Create database with ParsevalCube
        db = TopologicalCartesianDB(cell_size=1.0)
        parseval_cube = ParsevalCube(db, dimensions=dimensions)
        
        # Generate random vectors
        vectors = []
        for i in range(count):
            vector = [random.uniform(-10, 10) for _ in range(dimensions)]
            vectors.append((f"vec_{i}", vector))
        
        # Benchmark insertion
        start_time = time.time()
        for vec_id, vector in vectors:
            parseval_cube.insert_vector(vec_id, vector)
        insert_time = time.time() - start_time
        
        results['insert'].append(insert_time)
        print(f"Insert {count} vectors: {format_time(insert_time)} " + 
              f"({count/insert_time:.2f} vectors/s)")
        
        # Benchmark Parseval verification
        verify_count = min(100, count)
        start_time = time.time()
        for _, vector in vectors[:verify_count]:
            parseval_cube.verify_parseval_equality(vector)
        verify_time = time.time() - start_time
        
        results['verify'].append(verify_time / verify_count)
        print(f"Average verification time ({verify_count} vectors): " + 
              f"{format_time(verify_time / verify_count)}")
    
    return vector_counts, results


def benchmark_provenance_cube(point_counts=[100, 1000, 10000]):
    """Benchmark the ProvenanceCube operations."""
    print("=" * 50)
    print("BENCHMARK: ProvenanceCube Operations")
    print("=" * 50)
    
    results = {'query': []}
    
    for count in point_counts:
        print(f"\nDataset size: {count} points")
        
        # Create database with ProvenanceCube
        db = TopologicalCartesianDB(cell_size=1.0)
        provenance_cube = ProvenanceCube(db)
        
        # Insert points
        for i in range(count):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            provenance_cube.insert(x, y, {"id": i})
        
        # Benchmark querying with provenance
        query_count = 100
        start_time = time.time()
        for _ in range(query_count):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            provenance_cube.query_with_provenance(x, y, radius=10.0)
        query_time = time.time() - start_time
        
        results['query'].append(query_time / query_count)
        print(f"Average query with provenance time ({query_count} queries): " + 
              f"{format_time(query_time / query_count)}")
    
    return point_counts, results


def benchmark_optimization_cube(point_counts=[100, 1000, 10000]):
    """Benchmark the OptimizationCube operations."""
    print("=" * 50)
    print("BENCHMARK: OptimizationCube Operations")
    print("=" * 50)
    
    results = {'standard_query': [], 'optimized_query': [], 'speedup': []}
    
    for count in point_counts:
        print(f"\nDataset size: {count} points")
        
        # Create database with OptimizationCube
        db = TopologicalCartesianDB(cell_size=1.0)
        optimization_cube = OptimizationCube(db)
        
        # Insert points
        for i in range(count):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            optimization_cube.insert(x, y, {"id": i})
        
        # Benchmark standard querying
        query_count = 100
        start_time = time.time()
        for _ in range(query_count):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            optimization_cube.query(x, y, radius=10.0)
        standard_query_time = time.time() - start_time
        
        results['standard_query'].append(standard_query_time / query_count)
        print(f"Average standard query time ({query_count} queries): " + 
              f"{format_time(standard_query_time / query_count)}")
        
        # Benchmark optimized vector querying if available
        if hasattr(optimization_cube, 'query_vector_optimized'):
            start_time = time.time()
            for _ in range(query_count):
                optimization_cube.query_vector_optimized([random.uniform(-100, 100), 
                                                        random.uniform(-100, 100)], 10.0)
            optimized_query_time = time.time() - start_time
            
            speedup = standard_query_time / max(optimized_query_time, 1e-10)
            results['optimized_query'].append(optimized_query_time / query_count)
            results['speedup'].append(speedup)
            
            print(f"Average optimized query time: {format_time(optimized_query_time / query_count)}")
            print(f"Speedup: {speedup:.2f}x")
        else:
            results['optimized_query'].append(None)
            results['speedup'].append(None)
            print("Optimized query not available")
    
    return point_counts, results


def benchmark_cube_composition(point_count=5000):
    """Benchmark different cube compositions."""
    print("=" * 50)
    print("BENCHMARK: Cube Composition Performance")
    print("=" * 50)
    
    # Create test dataset
    points = [(random.uniform(-100, 100), random.uniform(-100, 100)) 
             for _ in range(point_count)]
    query_points = [(random.uniform(-100, 100), random.uniform(-100, 100)) 
                   for _ in range(100)]
    
    # Test configurations
    configurations = [
        ("Core DB", lambda: TopologicalCartesianDB(cell_size=1.0)),
        ("ParsevalCube", lambda: ParsevalCube(TopologicalCartesianDB(cell_size=1.0), dimensions=2)),
        ("ProvenanceCube", lambda: ProvenanceCube(TopologicalCartesianDB(cell_size=1.0))),
        ("OptimizationCube", lambda: OptimizationCube(TopologicalCartesianDB(cell_size=1.0))),
        ("P → O", lambda: OptimizationCube(ParsevalCube(TopologicalCartesianDB(cell_size=1.0), 
                                                       dimensions=2))),
        ("Pr → O", lambda: OptimizationCube(ProvenanceCube(TopologicalCartesianDB(cell_size=1.0)))),
        ("P → Pr → O", lambda: OptimizationCube(ProvenanceCube(
                       ParsevalCube(TopologicalCartesianDB(cell_size=1.0), dimensions=2))))
    ]
    
    results = {}
    
    for name, create_fn in configurations:
        print(f"\nTesting: {name}")
        db = create_fn()
        
        # Insert points
        start_time = time.time()
        for x, y in points:
            db.insert(x, y, {"config": name})
        insert_time = time.time() - start_time
        
        # Query points
        start_time = time.time()
        for x, y in query_points:
            db.query(x, y, radius=10.0)
        query_time = time.time() - start_time
        
        results[name] = {"insert": insert_time, "query": query_time}
        print(f"Insert time: {format_time(insert_time)} ({point_count/insert_time:.2f} points/s)")
        print(f"Query time: {format_time(query_time/100)} per query")
    
    return configurations, results


def plot_benchmark_results(point_counts, core_results, parseval_counts, parseval_results,
                          provenance_results, optimization_results):
    """Plot benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Core DB performance
    plt.subplot(2, 2, 1)
    plt.plot(point_counts, core_results['insert'], 'o-', label='Insert')
    plt.plot(point_counts, core_results['query'], 's-', label='Query')
    plt.title('Core Database Performance')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # ParsevalCube performance
    plt.subplot(2, 2, 2)
    plt.plot(parseval_counts, parseval_results['insert'], 'o-', label='Insert Vector')
    plt.plot(parseval_counts, parseval_results['verify'], 's-', label='Verify Parseval')
    plt.title('ParsevalCube Performance')
    plt.xlabel('Number of Vectors')
    plt.ylabel('Time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # ProvenanceCube performance
    plt.subplot(2, 2, 3)
    plt.plot(point_counts, provenance_results['query'], 'o-', label='Query with Provenance')
    plt.title('ProvenanceCube Performance')
    plt.xlabel('Number of Points')
    plt.ylabel('Time per Query (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # OptimizationCube performance
    plt.subplot(2, 2, 4)
    plt.plot(point_counts, optimization_results['standard_query'], 'o-', label='Standard Query')
    # Only plot optimized results where available
    valid_indices = [i for i, v in enumerate(optimization_results['optimized_query']) if v is not None]
    if valid_indices:
        valid_counts = [point_counts[i] for i in valid_indices]
        valid_times = [optimization_results['optimized_query'][i] for i in valid_indices]
        plt.plot(valid_counts, valid_times, 's-', label='Optimized Query')
    plt.title('OptimizationCube Performance')
    plt.xlabel('Number of Points')
    plt.ylabel('Time per Query (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hybrid_cube_benchmark_results.png')
    print("Results saved to 'hybrid_cube_benchmark_results.png'")
    

def plot_composition_results(configurations, comp_results):
    """Plot composition benchmark results."""
    names = [name for name, _ in configurations]
    insert_times = [comp_results[name]["insert"] for name in names]
    query_times = [comp_results[name]["query"]/100 for name in names]
    
    plt.figure(figsize=(12, 6))
    
    # Bar chart of insert and query times
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, insert_times, width, label='Insert Time')
    plt.bar(x + width/2, query_times, width, label='Query Time (per query)')
    
    plt.ylabel('Time (s)')
    plt.title('Performance by Cube Composition')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hybrid_cube_composition_results.png')
    print("Composition results saved to 'hybrid_cube_composition_results.png'")


def run_benchmarks():
    """Run all benchmarks."""
    print(" HYBRID CUBE ARCHITECTURE BENCHMARK SUITE ".center(70, "="))
    print()
    
    # Define dataset sizes
    point_counts = [100, 1000, 10000]
    vector_counts = [100, 1000, 5000]  # Smaller for vector operations
    
    # Run core database benchmark
    point_counts, core_results = benchmark_core_db(point_counts)
    print()
    
    # Run ParsevalCube benchmark
    vector_counts, parseval_results = benchmark_parseval_cube(vector_counts)
    print()
    
    # Run ProvenanceCube benchmark
    _, provenance_results = benchmark_provenance_cube(point_counts)
    print()
    
    # Run OptimizationCube benchmark
    _, optimization_results = benchmark_optimization_cube(point_counts)
    print()
    
    # Run cube composition benchmark
    configurations, comp_results = benchmark_cube_composition()
    print()
    
    # Plot results
    plot_benchmark_results(point_counts, core_results, vector_counts, 
                          parseval_results, provenance_results, optimization_results)
    
    plot_composition_results(configurations, comp_results)
    
    print("\nBenchmarks completed!")


if __name__ == "__main__":
    run_benchmarks()
