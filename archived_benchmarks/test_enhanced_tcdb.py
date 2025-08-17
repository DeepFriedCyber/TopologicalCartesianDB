import unittest
import numpy as np
from src.enhanced_tcdb import EnhancedTopologicalCartesianDB

class TestEnhancedTopologicalCartesianDB(unittest.TestCase):
    def test_vector_insertion(self):
        """Test insertion of vectors with various dimensions."""
        # 2D vectors
        db = EnhancedTopologicalCartesianDB(dimensions=2)
        db.insert_vector("vec1", [3.0, 4.0])
        
        # Verify vector is stored correctly
        self.assertTrue("vec1" in db.vectors)
        self.assertEqual(db.vectors["vec1"].tolist(), [3.0, 4.0])
        
        # 4D vectors
        db_4d = EnhancedTopologicalCartesianDB(dimensions=4)
        db_4d.insert_vector("vec1", [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(db_4d.vectors["vec1"].tolist(), [1.0, 2.0, 3.0, 4.0])
        
        # Test dimension mismatch
        with self.assertRaises(ValueError):
            db.insert_vector("vec2", [1.0, 2.0, 3.0])  # 3D vector in 2D database
    
    def test_parseval_energy_conservation(self):
        """Test Parseval's theorem for energy conservation."""
        db = EnhancedTopologicalCartesianDB(dimensions=2)
        
        # Insert a vector (3-4-5 triangle)
        db.insert_vector("vec1", [3.0, 4.0])
        
        # Query within radius 5 from origin
        results = db.query_vector([0.0, 0.0], 5.0)
        
        # Should find our vector
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "vec1")
        
        # Verify Parseval: ||x||² = 3² + 4² = 25
        vector = results[0][1]
        self.assertAlmostEqual(vector[0]**2 + vector[1]**2, 25.0)
        
        # Verify using the built-in Parseval verification
        self.assertTrue(db.verify_parseval_equality([3.0, 4.0]))
    
    def test_higher_dimensional_parseval(self):
        """Test Parseval's theorem in higher dimensions."""
        db = EnhancedTopologicalCartesianDB(dimensions=4)
        
        # Insert a 4D vector
        db.insert_vector("vec1", [1.0, 2.0, 3.0, 4.0])
        
        # Verify Parseval: ||x||² = 1² + 2² + 3² + 4² = 30
        self.assertTrue(db.verify_parseval_equality([1.0, 2.0, 3.0, 4.0]))
        
        # Calculate manually
        vector = [1.0, 2.0, 3.0, 4.0]
        vector_norm_sq = sum(v**2 for v in vector)
        self.assertEqual(vector_norm_sq, 30.0)
    
    def test_query_with_provenance(self):
        """Test query with provenance information."""
        db = EnhancedTopologicalCartesianDB(dimensions=2)
        
        # Insert a point
        db.insert(3.0, 4.0)
        
        # Query with provenance
        result = db.query_with_provenance(0.0, 0.0, 5.0)
        
        # Check result structure
        self.assertIn('results', result)
        self.assertIn('provenance', result)
        
        # Check provenance details
        provenance = result['provenance']
        self.assertIn('energy_breakdown', provenance)
        self.assertIn('parseval_compliance', provenance)
        self.assertEqual(provenance['query_point'], (0.0, 0.0))
        self.assertEqual(provenance['radius'], 5.0)
        self.assertGreaterEqual(provenance['points_examined'], 1)
        self.assertEqual(provenance['points_found'], 1)
        
        # Check parseval compliance
        parseval_data = provenance['parseval_compliance']
        self.assertIsNotNone(parseval_data)
        self.assertTrue(parseval_data['verified'])
    
    def test_2d_compatibility_methods(self):
        """Test 2D compatibility methods."""
        db = EnhancedTopologicalCartesianDB(dimensions=2)
        
        # Test insert and query methods for 2D compatibility
        db.insert(3.0, 4.0)
        
        # Query should return points as tuples
        results = db.query(0.0, 0.0, 5.0)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][0], 3.0)
        self.assertAlmostEqual(results[0][1], 4.0)


if __name__ == '__main__':
    unittest.main()
