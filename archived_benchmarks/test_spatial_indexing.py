import unittest
import random
import time
from src.enhanced_tcdb import EnhancedTopologicalCartesianDB

class TestSpatialGridIndexing(unittest.TestCase):
    """
    Tests for the spatial grid indexing performance in EnhancedTopologicalCartesianDB.
    """
    
    def test_grid_performance(self):
        """Test that spatial grid indexing provides significant performance benefits."""
        # Create database with 1.0 cell size
        db = EnhancedTopologicalCartesianDB(dimensions=2, cell_size=1.0)
        
        # Insert 10,000 random points
        print("Inserting 10,000 random points...")
        for i in range(10000):
            x, y = random.uniform(-100, 100), random.uniform(-100, 100)
            db.insert_vector(f"vec_{i}", [x, y])
        
        # Query using the grid index
        print("Querying with grid index...")
        start = time.time()
        results = db.query_vector([0, 0], 10.0)
        grid_time = time.time() - start
        print(f"Grid index query time: {grid_time:.6f} seconds")
        
        # Force a linear scan by using the internal method with a backup approach
        print("Querying with linear scan...")
        start = time.time()
        # Simulate linear scan by checking all vectors manually
        linear_results = []
        query_vec = [0, 0]
        radius_sq = 10.0 ** 2
        for vec_id, vector in db.vectors.items():
            dist_sq = sum((vector[i] - query_vec[i])**2 for i in range(2))
            if dist_sq <= radius_sq:
                linear_results.append((vec_id, vector.tolist()))
        linear_time = time.time() - start
        print(f"Linear scan query time: {linear_time:.6f} seconds")
        
        # Assert grid is faster (typically at least 10x faster for large datasets)
        self.assertLess(grid_time, linear_time)
        print(f"Speed improvement: {linear_time / grid_time:.2f}x faster with grid")
        
        # Verify both methods returned the same number of results
        self.assertEqual(len(results), len(linear_results))
        print(f"Both methods found {len(results)} points within radius 10.0")
        
        # Performance benchmark (should be very fast with grid)
        self.assertLess(grid_time, 0.1, "Grid index query should complete in under 100ms")
    
    def test_grid_size_optimization(self):
        """Test performance with different grid cell sizes."""
        print("\nTesting different grid cell sizes...")
        points = [(random.uniform(-100, 100), random.uniform(-100, 100)) 
                 for _ in range(5000)]
        
        cell_sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        query_times = {}
        
        for cell_size in cell_sizes:
            db = EnhancedTopologicalCartesianDB(dimensions=2, cell_size=cell_size)
            
            # Insert all points
            for i, (x, y) in enumerate(points):
                db.insert_vector(f"vec_{i}", [x, y])
            
            # Measure query time
            start = time.time()
            for _ in range(10):  # Run multiple queries to get better average
                results = db.query_vector([0, 0], 10.0)
            query_time = (time.time() - start) / 10
            
            query_times[cell_size] = query_time
            print(f"Cell size {cell_size}: {query_time:.6f} seconds per query")
        
        # Determine optimal cell size
        optimal_cell_size = min(query_times, key=query_times.get)
        print(f"\nOptimal cell size: {optimal_cell_size} (query time: {query_times[optimal_cell_size]:.6f}s)")
        
        # Print improvement over worst case
        worst_cell_size = max(query_times, key=query_times.get)
        improvement = query_times[worst_cell_size] / query_times[optimal_cell_size]
        print(f"Improvement over worst case: {improvement:.2f}x faster")
        
        # This test doesn't have a strict assertion, as optimal cell size depends on data
        # but we verify that we've found the optimal size for this dataset

if __name__ == '__main__':
    unittest.main()
