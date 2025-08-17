import unittest
from tcdb import Simplex, TopologicalCartesianDB
from mpmath import mp

class TestSimplex(unittest.TestCase):
    def test_tetrahedron_volume(self):
        # Create a regular tetrahedron with side length 1
        vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, mp.sqrt(3)/2, 0],
            [0.5, mp.sqrt(3)/6, mp.sqrt(6)/3]
        ]
        
        simplex = Simplex(vertices)
        volume = simplex.volume()
        
        # The volume of a regular tetrahedron with side length 1 is sqrt(2)/12
        expected = mp.sqrt(2) / 12
        # Use a small epsilon for floating point comparison
        self.assertTrue(abs(volume - expected) < 1e-10, 
                        f"Volume {float(volume)} differs from expected {float(expected)}")
    
    def test_db_operations(self):
        db = TopologicalCartesianDB()
        
        # Add two tetrahedrons
        vertices1 = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        
        vertices2 = [
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2]
        ]
        
        id1 = db.add_simplex(vertices1, {"name": "tetra1"})
        id2 = db.add_simplex(vertices2, {"name": "tetra2"})
        
        # Verify point query works
        self.assertIn(id1, db.query_point([0, 0, 0]))
        self.assertIn(id2, db.query_point([1, 1, 1]))
        
        # Check total volume
        total_volume = db.get_total_volume()
        self.assertEqual(len(db.simplices), 2)
        
        # Each tetrahedron has volume 1/6, total should be 1/3
        # Use a small epsilon for floating point comparison
        expected = mp.mpf(1/3)
        self.assertTrue(abs(total_volume - expected) < 1e-10, 
                        f"Total volume {float(total_volume)} differs from expected {float(expected)}")

if __name__ == "__main__":
    unittest.main()
