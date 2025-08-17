"""
Test the Hybrid Cube Architecture functionality.
"""
import unittest
import numpy as np

from core.spatial_db import TopologicalCartesianDB
from cubes.cube_adapter import CubeAdapter
from cubes.parseval_cube import ParsevalCube
from cubes.provenance_cube import ProvenanceCube
from cubes.optimization_cube import OptimizationCube


class TestHybridCubeArchitecture(unittest.TestCase):
    """Test cases for the Hybrid Cube Architecture."""
    
    def setUp(self):
        """Set up test environment."""
        self.db = TopologicalCartesianDB(cell_size=1.0)
        self.add_test_points()
    
    def add_test_points(self):
        """Add test points to the database."""
        self.db.insert(1.0, 2.0, {"name": "A"})
        self.db.insert(2.0, 3.0, {"name": "B"})
        self.db.insert(3.0, 4.0, {"name": "C"})
    
    def test_core_database(self):
        """Test core database functionality."""
        # Test points were added
        self.assertEqual(len(self.db.points), 3)
        
        # Test query - only finds exact point
        results = self.db.query(2.0, 3.0, 1.0)
        self.assertEqual(len(results), 1)  # Should find point B
        
        # Test query with data
        results = self.db.query_with_data(2.0, 3.0, 1.0)
        self.assertEqual(len(results), 1)
        
        # Test optimization
        opt_results = self.db.optimize()
        self.assertIn('total_points', opt_results)
        self.assertEqual(opt_results['total_points'], 3)
    
    def test_cube_adapter(self):
        """Test the base CubeAdapter functionality."""
        # Create adapter
        adapter = CubeAdapter(self.db)
        
        # Test pass-through methods
        adapter.insert(4.0, 5.0, {"name": "D"})
        results = adapter.query(4.0, 5.0, 1.0)
        self.assertEqual(len(results), 1)
        
        # Test class hierarchy
        hierarchy = adapter.__class_hierarchy__
        self.assertEqual(len(hierarchy), 2)
        self.assertEqual(hierarchy[0], "CubeAdapter")
        self.assertEqual(hierarchy[1], "TopologicalCartesianDB")
    
    def test_parseval_cube(self):
        """Test ParsevalCube functionality."""
        parseval = ParsevalCube(self.db, dimensions=2)
        
        # Test vector insertion
        parseval.insert_vector("vector1", [1.0, 1.0])
        parseval.insert_vector("vector2", [0.0, 2.0])
        
        # Test Parseval's theorem verification
        self.assertTrue(parseval.verify_parseval_equality([1.0, 1.0]))
        self.assertTrue(parseval.verify_parseval_equality([0.0, 2.0]))
        
        # Test vector count in stats
        stats = parseval.get_stats()
        self.assertEqual(stats['vector_count'], 2)
    
    def test_provenance_cube(self):
        """Test ProvenanceCube functionality."""
        provenance = ProvenanceCube(self.db)
        
        # Test query with provenance
        query_result = provenance.query_with_provenance(2.0, 3.0, 1.0)
        self.assertIn('results', query_result)
        self.assertIn('provenance', query_result)
        
        # Test query history
        history = provenance.get_query_history()
        self.assertEqual(len(history), 1)
    
    def test_optimization_cube(self):
        """Test OptimizationCube functionality."""
        optimization = OptimizationCube(self.db)
        
        # Test normal query
        optimization.query(2.0, 3.0, 1.0)
        
        # Test optimized vector query if available
        if hasattr(optimization, 'query_vector_optimized'):
            result = optimization.query_vector_optimized([2.0, 3.0], 1.0)
            self.assertIn('results', result)
            self.assertIn('optimization', result)
        
        # Test metrics
        metrics = optimization.get_optimization_metrics()
        self.assertIn('cache_hits', metrics)
        self.assertIn('cache_misses', metrics)
        
        # Test cache clearing
        cache_info = optimization.clear_cache()
        self.assertIn('entries_removed', cache_info)
    
    def test_cube_composition(self):
        """Test composing multiple cubes together."""
        parseval = ParsevalCube(self.db, dimensions=2)
        provenance = ProvenanceCube(parseval)
        optimization = OptimizationCube(provenance)
        
        # Test the complete stack
        hierarchy = optimization.__class_hierarchy__
        self.assertEqual(len(hierarchy), 4)
        self.assertEqual(hierarchy[0], "OptimizationCube")
        self.assertEqual(hierarchy[1], "ProvenanceCube")
        self.assertEqual(hierarchy[2], "ParsevalCube")
        self.assertEqual(hierarchy[3], "TopologicalCartesianDB")
        
        # Test that operations flow through the stack
        optimization.insert(5.0, 6.0, {"name": "E"})
        
        # Query through the stack
        query_result = optimization.query(5.0, 6.0, 1.0)
        self.assertTrue(len(query_result) > 0)


if __name__ == "__main__":
    unittest.main()
