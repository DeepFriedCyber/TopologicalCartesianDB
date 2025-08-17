import unittest
import numpy as np
from mpmath import mp
from src.cube import Cube
from src.topological_cartesian_db import TopologicalCartesianDB

class TestCube(unittest.TestCase):
    def test_cube_creation(self):
        # Test basic cube creation
        cube = Cube([0, 0, 0], [1, 1, 1])
        self.assertEqual(cube.dimensions, 3)
        self.assertEqual(cube.volume(), 1.0)
        
        # Test creation with different dimensions
        cube2d = Cube([0, 0], [2, 3])
        self.assertEqual(cube2d.dimensions, 2)
        self.assertEqual(cube2d.volume(), 6.0)
        
        # Test creation from center and size
        cube_center = Cube.from_center_and_size([0, 0, 0], 2.0)
        self.assertEqual(cube_center.volume(), 8.0)
        
        # Test with different sizes per dimension
        cube_rect = Cube.from_center_and_size([1, 1, 1], [2, 4, 6])
        self.assertEqual(cube_rect.volume(), 48.0)
        
    def test_contains_point(self):
        cube = Cube([0, 0, 0], [1, 1, 1])
        
        # Test points inside
        self.assertTrue(cube.contains_point([0.5, 0.5, 0.5]))
        self.assertTrue(cube.contains_point([0, 0, 0]))
        self.assertTrue(cube.contains_point([1, 1, 1]))
        
        # Test points outside
        self.assertFalse(cube.contains_point([1.5, 0.5, 0.5]))
        self.assertFalse(cube.contains_point([-0.1, 0.5, 0.5]))
        
    def test_cube_intersection(self):
        cube1 = Cube([0, 0, 0], [1, 1, 1])
        
        # Test intersecting cubes
        cube2 = Cube([0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
        self.assertTrue(cube1.intersects(cube2))
        
        # Test non-intersecting cubes
        cube3 = Cube([2, 2, 2], [3, 3, 3])
        self.assertFalse(cube1.intersects(cube3))
        
    def test_subdivision(self):
        cube = Cube([0, 0, 0], [1, 1, 1])
        subcubes = cube.subdivide()
        
        # Should get 2^3 = 8 subcubes
        self.assertEqual(len(subcubes), 8)
        
        # Check total volume
        total_volume = sum(subcube.volume() for subcube in subcubes)
        self.assertAlmostEqual(total_volume, 1.0)
        
        # Check each subcube is 1/8 of the original volume
        for subcube in subcubes:
            self.assertAlmostEqual(subcube.volume(), 0.125)
            
    def test_parseval_equality(self):
        # Create a unit cube
        cube = Cube([0, 0, 0], [1, 1, 1])
        
        # Standard basis
        basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        # Test vector (3-4-5 triangle in 3D)
        vector = [3, 4, 0]
        
        # Verify Parseval's equality
        self.assertTrue(cube.verify_parseval_equality(vector, basis))
        
        # Test with the classic 3-4-5 triangle
        # ||(3,4,0)||^2 = 3^2 + 4^2 + 0^2 = 25
        # Projection coefficients: [3, 4, 0]
        # Sum of squares of coefficients: 3^2 + 4^2 + 0^2 = 25
        vector_norm_sq = sum(v**2 for v in vector)
        coefficients = cube.project_vector_onto_basis(vector, basis)
        coeff_sum_sq = sum(c**2 for c in coefficients)
        
        self.assertEqual(vector_norm_sq, 25)
        self.assertEqual(coeff_sum_sq, 25)


class TestTopologicalCartesianDB(unittest.TestCase):
    def test_db_creation_and_basic_operations(self):
        db = TopologicalCartesianDB()
        
        # Add cubes
        id1 = db.add_cube([0, 0, 0], [1, 1, 1], {"name": "unit_cube"})
        id2 = db.add_cube([2, 2, 2], [3, 3, 3], {"name": "another_cube"})
        
        # Check cube count
        self.assertEqual(len(db.cubes), 2)
        
        # Check data association
        self.assertEqual(db.get_cube_data(id1)["name"], "unit_cube")
        self.assertEqual(db.get_cube_data(id2)["name"], "another_cube")
        
    def test_point_queries(self):
        db = TopologicalCartesianDB()
        
        # Add overlapping cubes
        id1 = db.add_cube([0, 0, 0], [2, 2, 2], {"name": "large_cube"})
        id2 = db.add_cube([1, 1, 1], [3, 3, 3], {"name": "overlapping_cube"})
        
        # Test point in first cube only
        point1 = [0.5, 0.5, 0.5]
        cubes1 = db.query_point(point1)
        self.assertEqual(len(cubes1), 1)
        self.assertEqual(cubes1[0], id1)
        
        # Test point in second cube only
        point2 = [2.5, 2.5, 2.5]
        cubes2 = db.query_point(point2)
        self.assertEqual(len(cubes2), 1)
        self.assertEqual(cubes2[0], id2)
        
        # Test point in both cubes
        point3 = [1.5, 1.5, 1.5]
        cubes3 = db.query_point(point3)
        self.assertEqual(len(cubes3), 2)
        self.assertIn(id1, cubes3)
        self.assertIn(id2, cubes3)
        
        # Test point outside all cubes
        point4 = [4, 4, 4]
        cubes4 = db.query_point(point4)
        self.assertEqual(len(cubes4), 0)
        
    def test_region_queries(self):
        db = TopologicalCartesianDB()
        
        # Add multiple cubes
        id1 = db.add_cube([0, 0, 0], [1, 1, 1])
        id2 = db.add_cube([2, 2, 2], [3, 3, 3])
        id3 = db.add_cube([4, 4, 4], [5, 5, 5])
        
        # Query region that intersects first cube
        region1 = db.query_region([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        self.assertEqual(len(region1), 1)
        self.assertEqual(region1[0], id1)
        
        # Query region that intersects all cubes
        region2 = db.query_region([0.5, 0.5, 0.5], [4.5, 4.5, 4.5])
        self.assertEqual(len(region2), 3)
        
    def test_parseval_verification(self):
        db = TopologicalCartesianDB()
        
        # Test with 3-4-5 triangle
        vector = [3, 4, 0]
        self.assertTrue(db.verify_parseval_equality(vector))
        
        # Get projection coefficients
        basis = db.create_orthonormal_basis(3)
        coefficients = db.project_vector(vector, basis)
        
        # Verify sum of squares = ||vector||^2
        vector_norm_sq = sum(v**2 for v in vector)
        coeff_sum_sq = sum(c**2 for c in coefficients)
        self.assertEqual(vector_norm_sq, 25)
        self.assertEqual(coeff_sum_sq, 25)


if __name__ == '__main__':
    unittest.main()
