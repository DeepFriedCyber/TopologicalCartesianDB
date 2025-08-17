"""
Test module for early termination optimization in the EnhancedTopologicalCartesianDB.
"""
import unittest
import time
import random
import numpy as np
from src.enhanced_tcdb import EnhancedTopologicalCartesianDB

class TestEarlyTermination(unittest.TestCase):
    """
    Tests for the early termination feature in vector queries.
    """
    
    def test_early_termination_performance(self):
        """Test that early termination improves query performance."""
        # Create two databases with the same settings
        dimensions = 100  # High dimensions to show the advantage of early termination
        db_with_early = EnhancedTopologicalCartesianDB(dimensions=dimensions)
        
        # Generate random vectors
        print(f"Inserting 1,000 {dimensions}-dimensional vectors...")
        vectors = []
        for i in range(1000):
            # Create high-dimensional vectors
            vector = [random.uniform(-10, 10) for _ in range(dimensions)]
            vectors.append((f"vec_{i}", vector))
            db_with_early.insert_vector(f"vec_{i}", vector)
        
        # Run queries with early termination
        print("Running queries with early termination...")
        start = time.time()
        query_vector = [0] * dimensions
        result_early = db_with_early.query_vector_optimized(query_vector, 5.0, use_early_termination=True)
        early_time = time.time() - start
        early_results = len(result_early['results'])
        early_terminations = result_early['provenance']['early_terminations']
        
        print(f"Early termination query time: {early_time:.6f} seconds")
        print(f"Found {early_results} vectors within radius")
        print(f"Early terminations: {early_terminations}")
        
        # Run queries without early termination
        print("Running queries without early termination...")
        start = time.time()
        result_full = db_with_early.query_vector_optimized(query_vector, 5.0, use_early_termination=False)
        full_time = time.time() - start
        full_results = len(result_full['results'])
        
        print(f"Full calculation query time: {full_time:.6f} seconds")
        print(f"Found {full_results} vectors within radius")
        
        # Verify results are the same
        self.assertEqual(early_results, full_results, 
                        "Early termination should return the same number of results")
        
        # Verify early termination is faster
        self.assertLess(early_time, full_time, 
                       "Early termination should be faster")
        
        # Calculate speedup
        speedup = full_time / early_time
        print(f"Speedup with early termination: {speedup:.2f}x")
        
        # Calculate average dimensions checked before termination
        if 'dimensions_checked' in result_early['provenance'] and result_early['provenance']['dimensions_checked']:
            avg_dims_checked = sum(result_early['provenance']['dimensions_checked']) / len(result_early['provenance']['dimensions_checked'])
            print(f"Average dimensions checked before termination: {avg_dims_checked:.2f} out of {dimensions}")
            
            # Verify early termination happens well before checking all dimensions
            self.assertLess(avg_dims_checked, dimensions * 0.5, 
                           "On average, should terminate before checking half the dimensions")
        
    def test_energy_tracking(self):
        """Test that energy tracking is accurate with early termination."""
        db = EnhancedTopologicalCartesianDB(dimensions=4)
        
        # Insert test vectors
        vectors = [
            ([1, 2, 3, 4], "vec1"),  # Distance = sqrt(30) from origin
            ([5, 0, 0, 0], "vec2"),  # Distance = 5 from origin
            ([10, 10, 10, 10], "vec3")  # Distance = sqrt(400) = 20 from origin
        ]
        
        for vector, vec_id in vectors:
            db.insert_vector(vec_id, vector)
        
        # Query with early termination
        query_vector = [0, 0, 0, 0]
        radius = 10.0
        result = db.query_vector_optimized(query_vector, radius, use_early_termination=True)
        
        # Verify correct results
        self.assertEqual(len(result['results']), 2)  # Should find vec1 and vec2
        
        # Verify energy calculations
        if 'energy_tracking' in result['provenance']:
            energy_tracking = result['provenance']['energy_tracking']
            
            # Check vec1
            vec1_energy = next((e for e in energy_tracking if e['vector_id'] == 'vec1'), None)
            self.assertIsNotNone(vec1_energy)
            self.assertAlmostEqual(vec1_energy['total_energy'], 30, places=5)
            
            # Check vec2
            vec2_energy = next((e for e in energy_tracking if e['vector_id'] == 'vec2'), None)
            self.assertIsNotNone(vec2_energy)
            self.assertAlmostEqual(vec2_energy['total_energy'], 25, places=5)
            
            # Check vec3 - should have been terminated early
            vec3_energy = next((e for e in energy_tracking if e['vector_id'] == 'vec3'), None)
            if vec3_energy:
                self.assertGreater(vec3_energy['total_energy'], radius**2)

if __name__ == '__main__':
    unittest.main()
