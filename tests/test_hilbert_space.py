"""
Test-driven development tests for the Hilbert Space extension to TCDB.
These tests define the expected behavior and interface for the extension.
"""

import unittest
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Any
import numpy.typing as npt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockTCDB:
    """Mock TCDB class for testing."""
    def __init__(self):
        self.points = {}
        self._vector_storage = {}
        self._storage = {}

class TestHilbertSpace(unittest.TestCase):
    """
    Tests for HilbertSpace which provides a mathematical framework for
    working with vectors in TCDB.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.base_db = MockTCDB()
        
        # Define a simple orthonormal basis for testing
        self.test_dimension = 3
        
        # Identity basis (standard basis vectors)
        self.identity_basis = np.eye(self.test_dimension)

    def test_inner_product(self):
        """Test inner product calculation."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace
        hilbert = HilbertSpace(basis=self.identity_basis)
        
        # Define test vectors
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])
        
        # Act
        ip = hilbert.inner_product(vec1, vec2)
        expected_ip = np.dot(vec1, vec2)
        
        # Assert
        self.assertAlmostEqual(ip, expected_ip)

    def test_distance(self):
        """Test distance calculation."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace
        hilbert = HilbertSpace(basis=self.identity_basis)
        
        # Define test vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        
        # Act
        dist = hilbert.distance(vec1, vec2)
        expected_dist = np.sqrt(2.0)  # Euclidean distance between (1,0,0) and (0,1,0)
        
        # Assert
        self.assertAlmostEqual(dist, expected_dist)

    def test_project(self):
        """Test vector projection onto basis."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace
        hilbert = HilbertSpace(basis=self.identity_basis)
        
        # Define test vector
        vec = np.array([1.0, 2.0, 3.0])
        
        # Act
        coeffs = hilbert.project(vec)
        
        # Assert
        # With identity basis, coefficients should equal the vector
        np.testing.assert_array_almost_equal(coeffs, vec)

    def test_reconstruct(self):
        """Test vector reconstruction from coefficients."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace
        hilbert = HilbertSpace(basis=self.identity_basis)
        
        # Define test vector and get its projection
        original_vec = np.array([1.0, 2.0, 3.0])
        coeffs = hilbert.project(original_vec)
        
        # Act
        reconstructed_vec = hilbert.reconstruct(coeffs)
        
        # Assert
        np.testing.assert_array_almost_equal(reconstructed_vec, original_vec)

    def test_verify_orthonormality(self):
        """Test verification of orthonormal basis."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace
        hilbert = HilbertSpace(basis=self.identity_basis)
        
        # Act
        is_orthonormal = hilbert.verify_orthonormality()
        
        # Assert
        self.assertTrue(is_orthonormal)
        
        # Test with non-orthonormal basis
        non_orthonormal_basis = np.array([
            [1.0, 0.5, 0.0],  # Not orthogonal to second vector
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        non_ortho_hilbert = HilbertSpace(basis=non_orthonormal_basis)
        self.assertFalse(non_ortho_hilbert.verify_orthonormality())

    def test_verify_parseval(self):
        """Test verification of Parseval's identity."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace
        hilbert = HilbertSpace(basis=self.identity_basis)
        
        # Define test vector
        vec = np.array([1.0, 2.0, 3.0])
        
        # Act
        satisfies_parseval = hilbert.verify_parseval(vec)
        
        # Assert
        self.assertTrue(satisfies_parseval)


class TestProductHilbertSpace(unittest.TestCase):
    """Tests for ProductHilbertSpace which combines multiple Hilbert spaces."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_db = MockTCDB()
        
        # Create two simple spaces
        self.dim1 = 2
        self.dim2 = 3
        self.basis1 = np.eye(self.dim1)
        self.basis2 = np.eye(self.dim2)

    def test_initialization(self):
        """Test initialization of product space."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, ProductHilbertSpace
        space1 = HilbertSpace(basis=self.basis1)
        space2 = HilbertSpace(basis=self.basis2)
        
        # Act
        product_space = ProductHilbertSpace([space1, space2])
        
        # Assert
        self.assertEqual(len(product_space.component_spaces), 2)
        self.assertEqual(product_space.total_dimension, self.dim1 + self.dim2)

    def test_inner_product(self):
        """Test inner product in product space."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, ProductHilbertSpace
        space1 = HilbertSpace(basis=self.basis1)
        space2 = HilbertSpace(basis=self.basis2)
        product_space = ProductHilbertSpace([space1, space2])
        
        # Define test vectors spanning both spaces
        vec1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vec2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Act
        ip = product_space.inner_product(vec1, vec2)
        expected_ip = np.dot(vec1, vec2)
        
        # Assert
        self.assertAlmostEqual(ip, expected_ip)

    def test_project_and_reconstruct(self):
        """Test projection and reconstruction in product space."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, ProductHilbertSpace
        space1 = HilbertSpace(basis=self.basis1)
        space2 = HilbertSpace(basis=self.basis2)
        product_space = ProductHilbertSpace([space1, space2])
        
        # Define test vector
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Act
        coeffs = product_space.project(vec)
        reconstructed = product_space.reconstruct(coeffs)
        
        # Assert
        np.testing.assert_array_almost_equal(reconstructed, vec)

    def test_verify_parseval(self):
        """Test verification of Parseval's identity in product space."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, ProductHilbertSpace
        space1 = HilbertSpace(basis=self.basis1)
        space2 = HilbertSpace(basis=self.basis2)
        product_space = ProductHilbertSpace([space1, space2])
        
        # Define test vector
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Act
        satisfies_parseval = product_space.verify_parseval(vec)
        
        # Assert
        self.assertTrue(satisfies_parseval)


class TestHilbertSpaceTCDB(unittest.TestCase):
    """Tests for HilbertSpaceTCDB extension class."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_db = MockTCDB()
        
        # Add some test vectors to the database
        self.base_db._vector_storage = {
            'vec1': {
                'vector': np.array([1.0, 2.0, 3.0]),
                'energy': 14.0  # 1² + 2² + 3²
            },
            'vec2': {
                'vector': np.array([4.0, 5.0, 6.0]),
                'energy': 77.0  # 4² + 5² + 6²
            }
        }
        
        # Define basis
        self.basis = np.eye(3)

    def test_initialization(self):
        """Test initialization of HilbertSpaceTCDB."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, HilbertSpaceTCDB
        
        # Act
        hilbert_db = HilbertSpaceTCDB(self.base_db, basis=self.basis)
        
        # Assert
        self.assertIs(hilbert_db.base_db, self.base_db)
        self.assertEqual(hilbert_db.space.basis.shape, self.basis.shape)

    def test_store_vector(self):
        """Test storing a vector."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, HilbertSpaceTCDB
        hilbert_db = HilbertSpaceTCDB(self.base_db, basis=self.basis)
        vector = np.array([7.0, 8.0, 9.0])
        
        # Act
        result = hilbert_db.store_vector('vec3', vector)
        
        # Assert
        self.assertTrue(result)
        self.assertIn('vec3', self.base_db._vector_storage)
        np.testing.assert_array_equal(
            self.base_db._vector_storage['vec3']['vector'],
            vector
        )
        self.assertAlmostEqual(
            self.base_db._vector_storage['vec3']['energy'],
            np.sum(vector**2)
        )

    def test_get_vector(self):
        """Test retrieving a vector."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, HilbertSpaceTCDB
        hilbert_db = HilbertSpaceTCDB(self.base_db, basis=self.basis)
        
        # Act
        vector = hilbert_db.get_vector('vec1')
        
        # Assert
        self.assertIsNotNone(vector)
        np.testing.assert_array_equal(
            vector,
            self.base_db._vector_storage['vec1']['vector']
        )
        
        # Test nonexistent vector
        self.assertIsNone(hilbert_db.get_vector('nonexistent'))

    def test_get_vector_energy(self):
        """Test retrieving vector energy."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, HilbertSpaceTCDB
        hilbert_db = HilbertSpaceTCDB(self.base_db, basis=self.basis)
        
        # Act
        energy = hilbert_db.get_vector_energy('vec1')
        
        # Assert
        self.assertIsNotNone(energy)
        self.assertAlmostEqual(energy, 14.0)
        
        # Test nonexistent vector
        self.assertIsNone(hilbert_db.get_vector_energy('nonexistent'))

    def test_compute_inner_product(self):
        """Test computing inner product between stored vectors."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, HilbertSpaceTCDB
        hilbert_db = HilbertSpaceTCDB(self.base_db, basis=self.basis)
        
        # Act
        ip = hilbert_db.compute_inner_product('vec1', 'vec2')
        expected_ip = np.dot(
            self.base_db._vector_storage['vec1']['vector'],
            self.base_db._vector_storage['vec2']['vector']
        )
        
        # Assert
        self.assertIsNotNone(ip)
        self.assertAlmostEqual(ip, expected_ip)
        
        # Test with nonexistent vector
        self.assertIsNone(hilbert_db.compute_inner_product('vec1', 'nonexistent'))

    def test_compute_distance(self):
        """Test computing distance between stored vectors."""
        # Arrange
        from extensions.hilbert_space import HilbertSpace, HilbertSpaceTCDB
        hilbert_db = HilbertSpaceTCDB(self.base_db, basis=self.basis)
        
        # Act
        distance = hilbert_db.compute_distance('vec1', 'vec2')
        vec1 = self.base_db._vector_storage['vec1']['vector']
        vec2 = self.base_db._vector_storage['vec2']['vector']
        expected_distance = np.sqrt(np.sum((vec1 - vec2)**2))
        
        # Assert
        self.assertIsNotNone(distance)
        self.assertAlmostEqual(distance, expected_distance)
        
        # Test with nonexistent vector
        self.assertIsNone(hilbert_db.compute_distance('vec1', 'nonexistent'))


if __name__ == '__main__':
    unittest.main()