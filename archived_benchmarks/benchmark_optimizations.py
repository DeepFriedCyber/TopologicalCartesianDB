"""
Benchmark script for TopologicalCartesianDB performance optimizations.
This script measures the performance gains from the various optimization techniques.
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from src.enhanced_tcdb import EnhancedTopologicalCartesianDB

def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1_000:.2f} ms"
    else:
        return f"{seconds:.4f} s"

def run_grid_index_benchmark(dimensions=2, point_counts=[100, 1000, 10000]):
    """Benchmark grid indexing vs. linear scan."""
    print("=" * 50)
    print(f"BENCHMARK: Grid Indexing vs. Linear Scan ({dimensions}D)")
    print("=" * 50)
    
    results = {'grid': [], 'linear': [], 'speedup': []}
    
    for count in point_counts:
        print(f"\nDataset size: {count} points")
        
        # Create database
        db = EnhancedTopologicalCartesianDB(dimensions=dimensions)
        
        # Generate random vectors
        vectors = []
        for i in range(count):
            vector = [random.uniform(-100, 100) for _ in range(dimensions)]
            vectors.append((f"vec_{i}", vector))
            db.insert_vector(f"vec_{i}", vector)
        
        # Create query vector
        query_vector = [0.0] * dimensions
        radius = 10.0
        
        # Run query with grid index (for 2D)
        if dimensions == 2:
            start = time.time()
            results_grid = db.query_vector(query_vector, radius)
            grid_time = time.time() - start
            
            print(f"Grid index:  {format_time(grid_time)} - found {len(results_grid)} points")
        else:
            # No grid index for higher dimensions in current implementation
            grid_time = float('inf')
            print("Grid index:  Not available for dimensions > 2")
        
        # Run query with early termination
        start = time.time()
        results_opt = db.query_vector_optimized(query_vector, radius, use_early_termination=True)
        opt_time = time.time() - start
        
        early_terminations = results_opt['provenance'].get('early_terminations', 0)
        print(f"Early term: {format_time(opt_time)} - found {len(results_opt['results'])} points")
        print(f"Early terminations: {early_terminations}")
        
        # Run query with full calculation
        start = time.time()
        results_full = db.query_vector_optimized(query_vector, radius, use_early_termination=False)
        linear_time = time.time() - start
        
        print(f"Linear scan: {format_time(linear_time)} - found {len(results_full['results'])} points")
        
        # Calculate speedup
        if grid_time != float('inf'):
            grid_speedup = linear_time / grid_time
            print(f"Grid index speedup: {grid_speedup:.2f}x")
            results['grid'].append(grid_time)
        else:
            grid_speedup = 0
            results['grid'].append(None)
            
        early_speedup = linear_time / opt_time
        print(f"Early termination speedup: {early_speedup:.2f}x")
        
        results['linear'].append(linear_time)
        results['speedup'].append(early_speedup)
    
    return point_counts, results

def run_memory_optimization_benchmark(dimensions=2, point_count=10000, cell_size=1.0):
    """Benchmark memory optimization."""
    print("=" * 50)
    print(f"BENCHMARK: Memory Optimization ({dimensions}D)")
    print("=" * 50)
    
    # Create database
    db = EnhancedTopologicalCartesianDB(dimensions=dimensions, cell_size=cell_size)
    
    # Generate random vectors with clustered distribution
    print(f"Generating {point_count} vectors with clustered distribution...")
    
    # Create clusters
    cluster_centers = []
    for _ in range(5):
        center = [random.uniform(-100, 100) for _ in range(dimensions)]
        cluster_centers.append(center)
    
    # Generate points around clusters
    for i in range(point_count):
        # Select a random cluster
        center = random.choice(cluster_centers)
        
        # Generate point around center (Gaussian distribution)
        vector = [np.random.normal(c, 10) for c in center]
        db.insert_vector(f"vec_{i}", vector)
    
    # Get initial statistics
    print("\nInitial index statistics:")
    stats_before = db.get_index_statistics()
    print(f"Total cells: {stats_before['cell_count']}")
    print(f"Empty cells: {stats_before['empty_cells']} ({stats_before['empty_cells'] / stats_before['cell_count'] * 100:.1f}%)")
    print(f"Memory usage: {stats_before['estimated_memory_bytes'] / (1024*1024):.2f} MB")
    
    # Optimize memory
    print("\nRunning memory optimization...")
    start = time.time()
    result = db.optimize_memory()
    opt_time = time.time() - start
    
    print(f"Optimization time: {format_time(opt_time)}")
    print(f"Cells removed: {result['cells_removed']}")
    print(f"Memory saved: {result['memory_saved'] / 1024:.2f} KB")
    
    # Get updated statistics
    stats_after = db.get_index_statistics()
    print(f"\nAfter optimization:")
    print(f"Total cells: {stats_after['cell_count']}")
    print(f"Empty cells: {stats_after['empty_cells']}")
    print(f"Memory usage: {stats_after['estimated_memory_bytes'] / (1024*1024):.2f} MB")
    
    return {
        'cells_before': stats_before['cell_count'],
        'cells_after': stats_after['cell_count'],
        'cells_removed': result['cells_removed'],
        'memory_saved': result['memory_saved'],
        'optimization_time': opt_time
    }

def visualize_results(point_counts, results, memory_results):
    """Visualize benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Query performance comparison
    plt.subplot(2, 2, 1)
    if any(results['grid']):
        plt.plot(point_counts, results['grid'], 'b-o', label='Grid Index')
    plt.plot(point_counts, results['linear'], 'r-o', label='Linear Scan')
    plt.xlabel('Number of Points')
    plt.ylabel('Query Time (s)')
    plt.title('Query Performance Comparison')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Speedup factors
    plt.subplot(2, 2, 2)
    plt.plot(point_counts, results['speedup'], 'm-o')
    plt.xlabel('Number of Points')
    plt.ylabel('Speedup Factor')
    plt.title('Early Termination Speedup vs. Linear Scan')
    plt.xscale('log')
    plt.grid(True)
    
    # Plot 3: Memory optimization results
    plt.subplot(2, 2, 3)
    bars = ['Before', 'After']
    values = [memory_results['cells_before'], memory_results['cells_after']]
    plt.bar(bars, values, color=['blue', 'green'])
    plt.xlabel('Optimization Stage')
    plt.ylabel('Number of Cells')
    plt.title('Memory Optimization Results')
    
    # Plot 4: Memory saved
    plt.subplot(2, 2, 4)
    saved_kb = memory_results['memory_saved'] / 1024
    labels = ['Memory Saved']
    plt.bar(labels, [saved_kb], color='orange')
    plt.ylabel('Memory (KB)')
    plt.title(f'Memory Saved: {saved_kb:.2f} KB')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nResults visualized and saved to 'benchmark_results.png'")

def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print(" TOPOLOGICAL CARTESIAN DB PERFORMANCE BENCHMARK SUITE ".center(70, "="))
    print("=" * 70 + "\n")
    
    # Run grid indexing benchmark
    point_counts = [100, 1000, 10000]
    point_counts, grid_results = run_grid_index_benchmark(dimensions=2, point_counts=point_counts)
    
    # Run memory optimization benchmark
    memory_results = run_memory_optimization_benchmark(dimensions=2, point_count=10000, cell_size=2.0)
    
    # Visualize results
    visualize_results(point_counts, grid_results, memory_results)
    
    print("\nBenchmarks completed!")

if __name__ == "__main__":
    main()
