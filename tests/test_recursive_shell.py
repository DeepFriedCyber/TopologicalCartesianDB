"""
Test-driven development tests for the Recursive Shell extension to TCDB.
These tests define the expected behavior and interface for the extension.
"""

import unittest
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockTCDB:
    """Mock TCDB class for testing."""
    def __init__(self):
        self.points = {}
        self._storage = {}

class TestRecursiveShellTCDB(unittest.TestCase):
    """
    Tests for RecursiveShellTCDB which implements a recursive shell structure
    for spatial indexing and query optimization.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.base_db = MockTCDB()
        
        # Create a grid of points for testing
        for x in range(5):
            for y in range(5):
                point_id = f"p_{x}_{y}"
                self.base_db.points[point_id] = np.array([float(x), float(y)])

    def test_initialization(self):
        """Test initialization of RecursiveShellTCDB."""
        # Arrange & Act
        from extensions.recursive_shell import RecursiveShellTCDB
        shell_db = RecursiveShellTCDB(self.base_db, shell_constant=0.5)
        
        # Assert
        self.assertIs(shell_db.base_db, self.base_db)
        self.assertEqual(shell_db.shell_constant, 0.5)
        self.assertFalse(shell_db.is_built)
        self.assertEqual(len(shell_db.shells), 0)

    def test_build_recursive_shells(self):
        """Test building the recursive shell structure."""
        # Arrange
        from extensions.recursive_shell import RecursiveShellTCDB
        shell_db = RecursiveShellTCDB(self.base_db, shell_constant=0.5)
        
        # Act
        shell_db.build_recursive_shells()
        
        # Assert
        self.assertTrue(shell_db.is_built)
        self.assertGreater(len(shell_db.shells), 0)
        self.assertGreater(len(shell_db.radii), 0)
        
        # Center should be approximately at (2, 2) for our 5x5 grid
        self.assertIsNotNone(shell_db.center)
        np.testing.assert_allclose(shell_db.center, [2.0, 2.0], atol=0.1)
        
        # All points should be assigned to shells
        total_points = sum(len(points) for points in shell_db.shells.values())
        self.assertEqual(total_points, len(self.base_db.points))

    def test_find_shell_index(self):
        """Test finding shell index for a distance."""
        # Arrange
        from extensions.recursive_shell import RecursiveShellTCDB
        shell_db = RecursiveShellTCDB(self.base_db)
        
        # Define test radii
        radii = [1.0, 2.0, 4.0, 8.0]
        
        # Act & Assert
        # Points inside first shell
        self.assertEqual(shell_db._find_shell_index(0.5, radii), 0)
        self.assertEqual(shell_db._find_shell_index(0.99, radii), 0)
        
        # Points in other shells
        self.assertEqual(shell_db._find_shell_index(1.5, radii), 1)
        self.assertEqual(shell_db._find_shell_index(3.0, radii), 2)
        self.assertEqual(shell_db._find_shell_index(7.0, radii), 3)
        
        # Points outside all shells
        self.assertEqual(shell_db._find_shell_index(10.0, radii), 4)

    def test_determine_relevant_shells(self):
        """Test determining which shells are relevant for a query."""
        # Arrange
        from extensions.recursive_shell import RecursiveShellTCDB
        shell_db = RecursiveShellTCDB(self.base_db)
        shell_db.radii = [1.0, 2.0, 4.0, 8.0]
        
        # Act & Assert
        # Query at distance 3 with radius 0.5
        relevant_shells = shell_db._determine_relevant_shells(3.0, 0.5)
        self.assertIn(2, relevant_shells)  # Should include shell 2 (radius 4)
        
        # Query at distance 1.5 with radius 1.0
        relevant_shells = shell_db._determine_relevant_shells(1.5, 1.0)
        self.assertIn(0, relevant_shells)  # Should include shell 0 (radius 1)
        self.assertIn(1, relevant_shells)  # Should include shell 1 (radius 2)
        
        # Query at distance 0.5 with radius 0.2
        relevant_shells = shell_db._determine_relevant_shells(0.5, 0.2)
        self.assertIn(0, relevant_shells)  # Should only include shell 0
        self.assertNotIn(1, relevant_shells)  # Should not include other shells
        
        # Query at distance 10 with radius 3
        relevant_shells = shell_db._determine_relevant_shells(10.0, 3.0)
        self.assertIn(3, relevant_shells)  # Should include shell 3 (radius 8)
        self.assertIn(4, relevant_shells)  # Should include points outside all shells

    def test_query_optimized(self):
        """Test optimized spatial query."""
        # Arrange
        from extensions.recursive_shell import RecursiveShellTCDB
        shell_db = RecursiveShellTCDB(self.base_db)
        shell_db.build_recursive_shells()
        
        query_point = np.array([2.0, 2.0])
        radius = 1.0
        
        # Act
        results = shell_db.query_optimized(query_point, radius)
        
        # Assert
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # All results should be within radius of query point
        for point_id in results:
            point = self.base_db.points[point_id]
            distance = np.sqrt(np.sum((point - query_point) ** 2))
            self.assertLessEqual(distance, radius)
        
        # All points within radius should be in results
        expected_results = []
        for point_id, point in self.base_db.points.items():
            distance = np.sqrt(np.sum((point - query_point) ** 2))
            if distance <= radius:
                expected_results.append(point_id)
        
        self.assertEqual(set(results), set(expected_results))

    def test_query_on_empty_shells(self):
        """Test querying when shells haven't been built."""
        # Arrange
        from extensions.recursive_shell import RecursiveShellTCDB
        shell_db = RecursiveShellTCDB(self.base_db)
        
        # Act & Assert
        with self.assertRaises(ValueError):
            shell_db.query_optimized(np.array([2.0, 2.0]), 1.0)

    def test_get_shell_statistics(self):
        """Test getting statistics about the shell structure."""
        # Arrange
        from extensions.recursive_shell import RecursiveShellTCDB
        shell_db = RecursiveShellTCDB(self.base_db, shell_constant=0.7)
        shell_db.build_recursive_shells()
        
        # Act
        stats = shell_db.get_shell_statistics()
        
        # Assert
        self.assertIn('num_shells', stats)
        self.assertIn('shell_constant', stats)
        self.assertIn('total_points', stats)
        self.assertIn('points_per_shell', stats)
        
        self.assertEqual(stats['shell_constant'], 0.7)
        self.assertEqual(stats['total_points'], len(self.base_db.points))


class TestAdaptiveRecursiveShellTCDB(unittest.TestCase):
    """
    Tests for AdaptiveRecursiveShellTCDB which extends RecursiveShellTCDB
    with adaptive shell constant selection.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.base_db = MockTCDB()
        
        # Create clustered data for better testing of adaptive properties
        # Cluster 1: around (1,1)
        for i in range(10):
            x = 1.0 + 0.2 * np.random.randn()
            y = 1.0 + 0.2 * np.random.randn()
            self.base_db.points[f'c1_{i}'] = np.array([x, y])
            
        # Cluster 2: around (5,5)
        for i in range(10):
            x = 5.0 + 0.2 * np.random.randn()
            y = 5.0 + 0.2 * np.random.randn()
            self.base_db.points[f'c2_{i}'] = np.array([x, y])
            
        # Cluster 3: around (2,8)
        for i in range(10):
            x = 2.0 + 0.2 * np.random.randn()
            y = 8.0 + 0.2 * np.random.randn()
            self.base_db.points[f'c3_{i}'] = np.array([x, y])

    def test_initialization(self):
        """Test initialization of AdaptiveRecursiveShellTCDB."""
        # Arrange & Act
        from extensions.recursive_shell import AdaptiveRecursiveShellTCDB
        adaptive_db = AdaptiveRecursiveShellTCDB(
            self.base_db, min_shells=3, max_shells=10
        )
        
        # Assert
        self.assertIs(adaptive_db.base_db, self.base_db)
        self.assertEqual(adaptive_db.min_shells, 3)
        self.assertEqual(adaptive_db.max_shells, 10)

    def test_build_adaptive_shells(self):
        """Test building adaptive shell structure."""
        # Arrange
        from extensions.recursive_shell import AdaptiveRecursiveShellTCDB
        adaptive_db = AdaptiveRecursiveShellTCDB(
            self.base_db, min_shells=3, max_shells=8
        )
        
        # Act
        adaptive_db.build_recursive_shells()
        
        # Assert
        self.assertTrue(adaptive_db.is_built)
        self.assertGreater(len(adaptive_db.shells), 0)
        
        # Shell count should be within specified range
        self.assertGreaterEqual(len(adaptive_db.radii), adaptive_db.min_shells)
        self.assertLessEqual(len(adaptive_db.radii), adaptive_db.max_shells)
        
        # All points should be assigned to shells
        total_points = sum(len(points) for points in adaptive_db.shells.values())
        self.assertEqual(total_points, len(self.base_db.points))

    def test_find_optimal_shell_constant(self):
        """Test finding optimal shell constant."""
        # Arrange
        from extensions.recursive_shell import AdaptiveRecursiveShellTCDB
        adaptive_db = AdaptiveRecursiveShellTCDB(
            self.base_db, min_shells=3, max_shells=8
        )
        adaptive_db.center = np.array([3.0, 3.0])  # Set center manually
        
        points = np.array([self.base_db.points[p] for p in self.base_db.points])
        max_dist = 10.0  # Arbitrary large value
        
        # Act
        optimal_constant = adaptive_db._find_optimal_shell_constant(points, max_dist)
        
        # Assert
        self.assertGreater(optimal_constant, 0.0)
        self.assertLess(optimal_constant, 2.0)  # Reasonable range
        import unittest
        import numpy as np
        import os
        import sys
        from typing import List, Dict, Tuple, Any, Set
        from collections import defaultdict

        # Add parent directory to path for imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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
                from extensions.hodge_laplace import HodgeLaplaceTCDB
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


        if __name__ == '__main__':
            unittest.main()
        
        fixed_points_per_shell = np.array([
            count for shell_idx, count in fixed_stats['points_per_shell'].items()
            if count > 0
        ])
        
        # Calculate coefficient of variation (std/mean) - lower is more uniform
        adaptive_cv = np.std(adaptive_points_per_shell) / np.mean(adaptive_points_per_shell)
        fixed_cv = np.std(fixed_points_per_shell) / np.mean(fixed_points_per_shell)
        
        # Assert - adaptive should generally provide more uniform distribution
        # But since this involves randomness, allow some margin
        self.assertLessEqual(adaptive_cv, fixed_cv * 1.5)


if __name__ == '__main__':
    unittest.main()