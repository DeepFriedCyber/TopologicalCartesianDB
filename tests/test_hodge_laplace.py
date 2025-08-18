"""
Test-driven development tests for the Hodge Laplace extension to TCDB.
These tests define the expected behavior and interface for the extension.
"""

import unittest
import numpy as np
import os
import sys
import pytest
import warnings
import time
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
import scipy.sparse as sparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from extensions.hodge_laplace import HodgeLaplaceTCDB, HodgeAnomalyTCDB


class MockTCDB:
    """Mock TCDB class for testing."""
    def __init__(self):
        self.points = {}
        self._storage = {}
        self.simplicial_complex = {}
        self.dimension = 2


class TestHodgeLaplaceTCDB(unittest.TestCase):
    """
    Tests for HodgeLaplaceTCDB which implements Hodge Laplacian operators
    for topological analysis on the database.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.base_db = MockTCDB()
        
        # Create a small triangular mesh for testing
        # Points in a grid pattern
        points = [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
            (0.0, 2.0), (1.0, 2.0), (2.0, 2.0)
        ]
        
        # Store points in the database
        for i, (x, y) in enumerate(points):
            point_id = f"p_{i}"
            self.base_db.points[point_id] = np.array([float(x), float(y)])
        
        # Define simplices (triangles)
        triangles = [
            ("p_0", "p_1", "p_4"),
            ("p_0", "p_3", "p_4"),
            ("p_1", "p_2", "p_4"),
            ("p_2", "p_4", "p_5"),
            ("p_3", "p_4", "p_7"),
            ("p_3", "p_6", "p_7"),
            ("p_4", "p_5", "p_8"),
            ("p_4", "p_7", "p_8")
        ]
        
        # Store simplices in the database
        self.base_db.simplicial_complex = {
            0: {p: {} for p in self.base_db.points.keys()},
            1: {},
            2: {}
        }
        
        # Add edges (1-simplices)
        edge_id = 0
        for t in triangles:
            for i in range(3):
                edge = tuple(sorted([t[i], t[(i+1)%3]]))
                edge_key = f"e_{edge_id}"
                self.base_db.simplicial_complex[1][edge_key] = {"vertices": edge}
                edge_id += 1
        
        # Add triangles (2-simplices)
        for i, t in enumerate(triangles):
            tri_key = f"t_{i}"
            self.base_db.simplicial_complex[2][tri_key] = {"vertices": t}

    def test_initialization(self):
        """Test initialization of HodgeLaplaceTCDB."""
        # Arrange & Act
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        
        # Assert
        self.assertIs(hodge_db.base_db, self.base_db)
        self.assertFalse(hodge_db.is_built)
        self.assertEqual(hodge_db.max_dimension, self.base_db.dimension)
        self.assertEqual(len(hodge_db.boundary_operators), 0)
        self.assertEqual(len(hodge_db.laplacian_operators), 0)

    def test_build_boundary_operators(self):
        """Test building boundary operators."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        
        # Act
        hodge_db._build_boundary_operators()
        
        # Assert
        self.assertEqual(len(hodge_db.boundary_operators), hodge_db.max_dimension)
        
        # Check shapes of boundary matrices
        # B_1 maps from 1-simplices to 0-simplices
        self.assertEqual(hodge_db.boundary_operators[0].shape, 
                        (len(self.base_db.simplicial_complex[0]), 
                         len(self.base_db.simplicial_complex[1])))
        
        # B_2 maps from 2-simplices to 1-simplices
        self.assertEqual(hodge_db.boundary_operators[1].shape, 
                        (len(self.base_db.simplicial_complex[1]), 
                         len(self.base_db.simplicial_complex[2])))

    def test_build_laplacian_operators(self):
        """Test building Hodge Laplacian operators."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        
        # Act
        hodge_db.build_laplacian_operators()
        
        # Assert
        self.assertTrue(hodge_db.is_built)
        self.assertEqual(len(hodge_db.laplacian_operators), hodge_db.max_dimension + 1)
        
        # Check each Laplacian is square
        for k, L_k in enumerate(hodge_db.laplacian_operators):
            num_k_simplices = len(self.base_db.simplicial_complex[k])
            self.assertEqual(L_k.shape, (num_k_simplices, num_k_simplices))
        
        # Check properties of Laplacians (symmetric and positive semi-definite)
        for L_k in hodge_db.laplacian_operators:
            # Check symmetry
            np.testing.assert_allclose(L_k, L_k.T, rtol=1e-10, atol=1e-10)
            
            # Check eigenvalues are non-negative
            eigvals = np.linalg.eigvalsh(L_k)
            self.assertTrue(np.all(eigvals >= -1e-10))  # Allow small numerical errors

    def test_get_hodge_decomposition(self):
        """Test Hodge decomposition of edge flows."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Create a test edge flow (assign random values to each edge)
        edge_flow = np.random.rand(len(self.base_db.simplicial_complex[1]))
        
        # Act
        gradient, harmonic, curl = hodge_db.get_hodge_decomposition(edge_flow)
        
        # Assert
        # Check dimensions
        self.assertEqual(gradient.shape, edge_flow.shape)
        self.assertEqual(harmonic.shape, edge_flow.shape)
        self.assertEqual(curl.shape, edge_flow.shape)
        
        # Verify decomposition adds up to original flow
        np.testing.assert_allclose(gradient + harmonic + curl, edge_flow, rtol=1e-10, atol=1e-10)
        
        # Verify orthogonality (in theory, these dot products should be zero)
        self.assertAlmostEqual(np.dot(gradient, harmonic), 0, delta=1e-10)
        self.assertAlmostEqual(np.dot(gradient, curl), 0, delta=1e-10)
        self.assertAlmostEqual(np.dot(harmonic, curl), 0, delta=1e-10)

    def test_get_betti_numbers(self):
        """Test computation of Betti numbers."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Act
        betti_numbers = hodge_db.get_betti_numbers()
        
        # Assert
        # Check we have correct number of Betti numbers
        self.assertEqual(len(betti_numbers), hodge_db.max_dimension + 1)
        
        # Our test complex is a planar grid with no holes, so:
        # β₀ should be 1 (one connected component)
        # β₁ should be 0 (no 1D holes/cycles)
        # β₂ should be 0 (no 2D voids)
        self.assertEqual(betti_numbers[0], 1)
        self.assertEqual(betti_numbers[1], 0)
        self.assertEqual(betti_numbers[2], 0)

    def test_get_harmonic_basis(self):
        """Test computation of harmonic basis."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Act
        harmonic_bases = hodge_db.get_harmonic_basis()
        
        # Assert
        # Check dimensions of returned bases
        self.assertEqual(len(harmonic_bases), hodge_db.max_dimension + 1)
        
        for k, basis in enumerate(harmonic_bases):
            # Basis dimension should equal Betti number
            betti_k = hodge_db.get_betti_numbers()[k]
            
            if betti_k > 0:
                self.assertEqual(basis.shape[1], betti_k)
                self.assertEqual(basis.shape[0], len(self.base_db.simplicial_complex[k]))
            else:
                self.assertIsNone(basis)  # No basis when Betti number is 0

    def test_analyze_edge_flow(self):
        """Test edge flow analysis."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Create a test edge flow (assign random values to each edge)
        edge_flow = np.random.rand(len(self.base_db.simplicial_complex[1]))
        
        # Act
        analysis = hodge_db.analyze_edge_flow(edge_flow)
        
        # Assert
        self.assertIn('curl_component_magnitude', analysis)
        self.assertIn('gradient_component_magnitude', analysis)
        self.assertIn('harmonic_component_magnitude', analysis)
        self.assertIn('curl_percentage', analysis)
        self.assertIn('gradient_percentage', analysis)
        self.assertIn('harmonic_percentage', analysis)
        
        # Check percentages sum to approximately 100%
        total_percentage = (analysis['curl_percentage'] + 
                           analysis['gradient_percentage'] + 
                           analysis['harmonic_percentage'])
        self.assertAlmostEqual(total_percentage, 100.0, delta=0.1)

    def test_spectral_clustering(self):
        """Test spectral clustering based on Hodge Laplacians."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Act
        # Cluster vertices (0-simplices) into 2 clusters
        dimension = 0
        n_clusters = 2
        clusters = hodge_db.spectral_clustering(dimension, n_clusters)
        
        # Assert
        # Check we have cluster assignments for all k-simplices
        self.assertEqual(len(clusters), len(self.base_db.simplicial_complex[dimension]))
        
        # Check cluster labels are within range
        unique_clusters = set(clusters)
        self.assertLessEqual(len(unique_clusters), n_clusters)
        for c in unique_clusters:
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, n_clusters)

    def test_persistent_laplacian(self):
        """Test computation of persistent Laplacian."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Define a filtration parameter (e.g., distance threshold)
        filtration_values = [0.5, 1.0, 1.5, 2.0]
        
        # Act
        persistent_laplacians = hodge_db.compute_persistent_laplacian(
            dimension=1, filtration_values=filtration_values
        )
        
        # Assert
        self.assertEqual(len(persistent_laplacians), len(filtration_values))
        
        # Check properties of persistent Laplacians
        for L in persistent_laplacians:
            # Should be square matrices
            self.assertEqual(L.shape[0], L.shape[1])
            
            # Should be symmetric
            np.testing.assert_allclose(L, L.T, rtol=1e-10, atol=1e-10)
            
            # Should be positive semi-definite (all eigenvalues ≥ 0)
            eigvals = np.linalg.eigvalsh(L)
            self.assertTrue(np.all(eigvals >= -1e-10))


class TestSimplicialComplex(unittest.TestCase):
    """Tests for building simplicial complexes from point clouds."""
    
    def test_build_2d_complex(self):
        """Test building a 2D simplicial complex from points."""
        # Arrange: Create simple point set forming a triangle
        points = np.array([[0,0], [1,0], [0,1], [1,1]])
        
        # Act
        detector = HodgeAnomalyTCDB()
        complex = detector._build_simplicial_complex(points, k_neighbors=2, max_dim=2)
        
        # Assert: Verify structure
        self.assertEqual(len(complex[0]), 4)  # 4 vertices
        self.assertGreater(len(complex[1]), 0)  # Should have edges
        self.assertGreater(len(complex[2]), 0)  # Should have triangles


class TestHodgeLaplacianCalculator(unittest.TestCase):
    """Tests for Hodge Laplacian computation."""
    
    def test_hodge_laplacian_computation(self):
        """Test Hodge Laplacian computation for simple complex."""
        # Arrange: Create simple triangle complex
        complex = {
            0: [[0], [1], [2]],
            1: [[0,1], [1,2], [2,0]],
            2: [[0,1,2]]
        }
        
        # Act
        detector = HodgeAnomalyTCDB()
        laplacians = detector._compute_hodge_laplacians(complex)
        
        # Assert: Verify structure and properties
        self.assertIn(0, laplacians)
        self.assertIn(1, laplacians)
        self.assertIn(2, laplacians)
        
        # Verify Laplacian properties (symmetric, positive semi-definite)
        for dim, lap in laplacians.items():
            self.assertEqual(lap.shape[0], lap.shape[1])  # Square
            np.testing.assert_array_almost_equal(lap.toarray(), lap.toarray().T)  # Symmetric


class TestAnomalyDetection(unittest.TestCase):
    """Tests for anomaly detection using Hodge Laplacians."""
    
    def test_anomaly_detection(self):
        """Test anomaly detection with known outliers."""
        # Arrange: Create points with clear outlier
        points = np.array([[0,0], [1,0], [0,1], [10,10]])  # Last point is outlier
        
        # Act
        detector = HodgeAnomalyTCDB()
        results = detector.detect_spatial_anomalies(points=points, k_neighbors=2)
        
        # Assert: Verify outlier detection
        self.assertIn('anomalous_points', results)
        self.assertIn('anomaly_indices', results)
        self.assertGreater(len(results['anomalous_points']), 0)
        
        # Verify outlier (index 3) is detected
        self.assertIn(3, results['anomaly_indices'])
    
    def test_anomaly_explanation_index_bug(self):
        """Test that anomaly explanation handles index correctly."""
        # Arrange: Create points and detect anomalies
        points = np.array([[0,0], [1,0], [0,1], [10,10]])
        
        # Act
        detector = HodgeAnomalyTCDB()
        results = detector.detect_spatial_anomalies(points=points, k_neighbors=2)
        
        # Try to explain anomaly
        explanation = detector.explain_anomaly(3, points=points)
        
        # Assert: Should not raise IndexError and should be valid explanation
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        self.assertNotIn("IndexError", explanation)


class TestHodgeLaplaceEdgeCases(unittest.TestCase):
    """Tests for edge cases in the Hodge Laplace implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_db = MockTCDB()
        
        # Create a simple triangle
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        
        # Store points in the database
        for i, (x, y) in enumerate(points):
            point_id = f"p_{i}"
            self.base_db.points[point_id] = np.array([float(x), float(y)])
        
        # Create a simplicial complex with just the triangle
        self.base_db.simplicial_complex = {
            0: {f"p_{i}": {} for i in range(len(points))},
            1: {},
            2: {}
        }
        
        # Add edges
        edges = [("p_0", "p_1"), ("p_1", "p_2"), ("p_2", "p_0")]
        for i, edge in enumerate(edges):
            edge_key = f"e_{i}"
            self.base_db.simplicial_complex[1][edge_key] = {"vertices": edge}
        
        # Add triangle
        self.base_db.simplicial_complex[2]["t_0"] = {"vertices": ("p_0", "p_1", "p_2")}

    def test_empty_complex_handling(self):
        """Test handling of empty simplicial complex."""
        # Arrange
        empty_db = MockTCDB()
        empty_db.simplicial_complex = {0: {}, 1: {}, 2: {}}
        
        # Act
        hodge_db = HodgeLaplaceTCDB(empty_db)
        
        # Should not raise an exception
        hodge_db.build_laplacian_operators()
        
        # Assert
        self.assertTrue(hodge_db.is_built)
        # Should have empty laplacian operators
        self.assertEqual(len(hodge_db.laplacian_operators), empty_db.dimension + 1)

    def test_handle_zero_edge_flow(self):
        """Test Hodge decomposition with zero edge flow."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Create a zero edge flow
        edge_flow = np.zeros(len(self.base_db.simplicial_complex[1]))
        
        # Act
        gradient, harmonic, curl = hodge_db.get_hodge_decomposition(edge_flow)
        
        # Assert
        np.testing.assert_array_equal(gradient, np.zeros_like(edge_flow))
        np.testing.assert_array_equal(harmonic, np.zeros_like(edge_flow))
        np.testing.assert_array_equal(curl, np.zeros_like(edge_flow))

    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatch in edge flow."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Create an edge flow with wrong dimension
        wrong_edge_flow = np.zeros(len(self.base_db.simplicial_complex[1]) + 1)
        
        # Act & Assert
        with pytest.raises(Exception):
            gradient, harmonic, curl = hodge_db.get_hodge_decomposition(wrong_edge_flow)

    def test_analyze_zero_edge_flow(self):
        """Test analysis of zero edge flow."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        hodge_db.build_laplacian_operators()
        
        # Create a zero edge flow
        edge_flow = np.zeros(len(self.base_db.simplicial_complex[1]))
        
        # Act
        analysis = hodge_db.analyze_edge_flow(edge_flow)
        
        # Assert - all percentages should be 0
        self.assertEqual(analysis['gradient_percentage'], 0.0)
        self.assertEqual(analysis['harmonic_percentage'], 0.0)
        self.assertEqual(analysis['curl_percentage'], 0.0)


class TestHodgeAnomalyAdvanced(unittest.TestCase):
    """Advanced tests for the HodgeAnomalyTCDB class."""
    
    def test_multiple_outliers(self):
        """Test detection of multiple outliers."""
        # Arrange: Create points with multiple outliers
        points = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],  # normal cluster
            [10, 10], [11, 10], [10, 11]      # outlier cluster
        ])
        
        # Act
        detector = HodgeAnomalyTCDB()
        results = detector.detect_spatial_anomalies(points=points, k_neighbors=3)
        
        # Assert: Should detect all outliers
        self.assertGreaterEqual(len(results['anomaly_indices']), 1)
        # At least one of the outliers should be detected
        self.assertTrue(any(idx >= 4 for idx in results['anomaly_indices']), 
                        "At least one outlier should be detected")

    def test_no_outliers(self):
        """Test behavior when there are no obvious outliers."""
        # Arrange: Create uniformly distributed points
        points = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ])
        
        # Act
        detector = HodgeAnomalyTCDB()
        results = detector.detect_spatial_anomalies(points=points, k_neighbors=4)
        
        # Assert: Might still return some points as anomalous, but should be consistent
        self.assertIn('anomaly_indices', results)

    def test_different_neighbor_counts(self):
        """Test effect of different k_neighbors values."""
        # Arrange
        points = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],  # normal cluster
            [10, 10]                          # outlier
        ])
        
        # Act with different k values
        detector = HodgeAnomalyTCDB()
        results_k2 = detector.detect_spatial_anomalies(points=points, k_neighbors=2)
        results_k4 = detector.detect_spatial_anomalies(points=points, k_neighbors=4)
        
        # Assert: Outlier should be detected in both cases
        self.assertIn(4, results_k2['anomaly_indices'])
        self.assertIn(4, results_k4['anomaly_indices'])


class TestRobustness(unittest.TestCase):
    """Tests for robustness of the Hodge Laplace implementation."""
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Arrange
        base_db = MockTCDB()
        
        # Create a triangle with very large coordinates
        points = [(0.0, 0.0), (1e6, 0.0), (0.0, 1e6)]
        
        # Store points in the database
        for i, (x, y) in enumerate(points):
            point_id = f"p_{i}"
            base_db.points[point_id] = np.array([float(x), float(y)])
        
        # Create a simplicial complex
        base_db.simplicial_complex = {
            0: {f"p_{i}": {} for i in range(len(points))},
            1: {},
            2: {}
        }
        
        # Add edges
        edges = [("p_0", "p_1"), ("p_1", "p_2"), ("p_2", "p_0")]
        for i, edge in enumerate(edges):
            edge_key = f"e_{i}"
            base_db.simplicial_complex[1][edge_key] = {"vertices": edge}
        
        # Add triangle
        base_db.simplicial_complex[2]["t_0"] = {"vertices": ("p_0", "p_1", "p_2")}
        
        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hodge_db = HodgeLaplaceTCDB(base_db)
            hodge_db.build_laplacian_operators()
            
            # Should not raise numerical stability errors
            betti_numbers = hodge_db.get_betti_numbers()
        
        # Assert
        self.assertEqual(len(betti_numbers), hodge_db.max_dimension + 1)

    def test_large_complex(self):
        """Test performance with a large simplicial complex."""
        # Skip this test if it would take too long
        pytest.skip("Skipping performance test for regular runs")
        
        # Arrange: Create a large grid of points
        n = 20  # 20x20 grid = 400 points
        points = []
        for i in range(n):
            for j in range(n):
                points.append((float(i), float(j)))
        
        # Create a mock database with these points
        base_db = MockTCDB()
        for i, (x, y) in enumerate(points):
            point_id = f"p_{i}"
            base_db.points[point_id] = np.array([x, y])
        
        # Act & Assert: Should complete in reasonable time
        start_time = time.time()
        
        detector = HodgeAnomalyTCDB()
        complex = detector._build_simplicial_complex(
            np.array(list(base_db.points.values())), 
            k_neighbors=5, 
            max_dim=2
        )
        
        duration = time.time() - start_time
        
        # Assert: Complex should be built and have reasonable size
        self.assertEqual(len(complex[0]), len(points))
        self.assertGreater(len(complex[1]), 0)
        self.assertLess(duration, 30)  # Should complete in under 30 seconds


class TestHodgeAnomalyIntegration(unittest.TestCase):
    """Tests for HodgeAnomalyTCDB integration with base TCDB."""
    
    def setUp(self):
        """Set up a database with points in specific patterns."""
        self.base_db = MockTCDB()
        
        # Create points forming multiple clusters with an outlier
        cluster1 = [(i, j) for i in range(3) for j in range(3)]  # 3x3 grid
        cluster2 = [(i+5, j+5) for i in range(3) for j in range(3)]  # Another 3x3 grid
        outlier = [(10, 0)]  # Outlier point
        
        all_points = cluster1 + cluster2 + outlier
        
        # Store points in database
        for i, (x, y) in enumerate(all_points):
            point_id = f"p_{i}"
            self.base_db.points[point_id] = np.array([float(x), float(y)])
        
        # Initialize simplicial complex
        self.base_db.simplicial_complex = {
            0: {p: {} for p in self.base_db.points.keys()},
            1: {},
            2: {}
        }
        
        # Add some edges (1-simplices) connecting nearby points
        edge_id = 0
        # Connect points in the grid (excluding outlier)
        for i in range(9):
            for j in range(i+1, 9):
                # Connect only if Manhattan distance is 1
                p1 = self.base_db.points[f"p_{i}"]
                p2 = self.base_db.points[f"p_{j}"]
                if np.sum(np.abs(p1 - p2)) <= 1.5:
                    edge_key = f"e_{edge_id}"
                    self.base_db.simplicial_complex[1][edge_key] = {"vertices": (f"p_{i}", f"p_{j}")}
                    edge_id += 1
        
        # Add some triangles (2-simplices)
        triangles = [
            ("p_0", "p_1", "p_4"),
            ("p_0", "p_3", "p_4"),
            ("p_1", "p_2", "p_5"),
            ("p_3", "p_4", "p_7"),
            ("p_4", "p_5", "p_8")
        ]
        
        for i, t in enumerate(triangles):
            tri_key = f"t_{i}"
            self.base_db.simplicial_complex[2][tri_key] = {"vertices": t}
            
    def test_integration_with_base_tcdb(self):
        """Test integration with base TopologicalCartesianDB"""
        # Arrange: Use mock TCDB with points added in setUp
        
        # Create a Hodge detector with the base_db
        detector = HodgeAnomalyTCDB()
        detector.base_db = self.base_db
        
        # Act: Detect anomalies
        results = detector.detect_spatial_anomalies(k_neighbors=3, max_dim=2)
        
        # Assert: Verify integration works
        self.assertIn('anomalous_points', results)
        self.assertIn('anomaly_indices', results)
        self.assertGreater(len(results['anomalous_points']), 0)

    def test_hodge_laplace_integrated_analysis(self):
        """Test integrated analysis using Hodge Laplacians."""
        # Arrange
        hodge_db = HodgeLaplaceTCDB(self.base_db)
        
        # Act
        hodge_db.build_laplacian_operators()
        betti_numbers = hodge_db.get_betti_numbers()
        
        # Assert
        # We should be able to compute meaningful Betti numbers from the base_db
        self.assertEqual(len(betti_numbers), hodge_db.max_dimension + 1)
        # Should have a single connected component
        self.assertGreaterEqual(betti_numbers[0], 1)


if __name__ == '__main__':
    unittest.main()