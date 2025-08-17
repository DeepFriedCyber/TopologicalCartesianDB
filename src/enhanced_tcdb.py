from mpmath import mp
import numpy as np
import time
import random
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from collections import defaultdict
from .cube import Cube

class EnhancedTopologicalCartesianDB:
    """
    An enhanced database system that uses geometric cubes in n-dimensional space 
    with vector operations, Parseval's theorem verification, and query provenance.
    """
    # System constraints for security and resource management
    MAX_VECTORS = 1_000_000
    MAX_DIMENSIONS = 1000
    MAX_RADIUS = 1_000_000
    MAX_CUBES = 1_000_000
    
    def __init__(self, dimensions=2, cell_size=1.0, custom_limits=None):
        """
        Initialize an empty database with specified dimensions.
        
        Args:
            dimensions: Number of dimensions for vectors (default is 2)
            cell_size: Cell size for spatial indexing
            custom_limits: Optional dict with custom MAX_* values
        """
        # Apply custom limits if provided
        if custom_limits:
            for limit_name, limit_value in custom_limits.items():
                if hasattr(self, limit_name) and limit_name.startswith('MAX_'):
                    setattr(self, limit_name, limit_value)
        
        # Validate dimensions
        if dimensions > self.MAX_DIMENSIONS:
            raise ValueError(f"Dimensions cannot exceed {self.MAX_DIMENSIONS}")
        if dimensions < 1:
            raise ValueError("Dimensions must be at least 1")
            
        self.dimensions = dimensions
        self.cubes = []
        self.vectors = {}  # Store vectors by ID
        self.data = {}
        self.spatial_index = defaultdict(list)  # Grid-based spatial index
        self.cell_size = cell_size
        
        # Set high precision for exact arithmetic
        mp.dps = 50
    
    def add_cube(self, min_coords: List[float], max_coords: List[float], 
                data: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new cube to the database with optional associated data.
        
        Args:
            min_coords: List of minimum coordinates for each dimension
            max_coords: List of maximum coordinates for each dimension
            data: Optional metadata to associate with this cube
            
        Returns:
            The ID of the newly added cube
        """
        # Check dimensions
        if len(min_coords) != self.dimensions or len(max_coords) != self.dimensions:
            raise ValueError(f"Coordinates must have {self.dimensions} dimensions")
            
        # Create new cube
        cube = Cube(min_coords, max_coords, data)
        cube_id = len(self.cubes)
        self.cubes.append(cube)
        
        # Store associated data
        if data:
            self.data[cube_id] = data
            
        # Index the cube
        self._index_cube(cube_id, cube)
        
        return cube_id
    
    def _index_cube(self, cube_id: int, cube: Cube) -> None:
        """
        Add a cube to the spatial index for efficient querying.
        
        Args:
            cube_id: The ID of the cube
            cube: The Cube instance
        """
        dimensions = self.dimensions
        
        # Determine grid cells that this cube intersects with
        min_cells = [int(cube.min_coords[i] / self.cell_size) for i in range(dimensions)]
        max_cells = [int(cube.max_coords[i] / self.cell_size) for i in range(dimensions)]
        
        # Add cube to all intersecting grid cells
        if dimensions == 2:
            for i in range(min_cells[0], max_cells[0] + 1):
                for j in range(min_cells[1], max_cells[1] + 1):
                    self.spatial_index[(i, j)].append(cube_id)
        elif dimensions == 3:
            for i in range(min_cells[0], max_cells[0] + 1):
                for j in range(min_cells[1], max_cells[1] + 1):
                    for k in range(min_cells[2], max_cells[2] + 1):
                        self.spatial_index[(i, j, k)].append(cube_id)
        else:
            # For higher dimensions, use a simpler approach
            for i in range(min_cells[0], max_cells[0] + 1):
                for j in range(min_cells[1], max_cells[1] + 1):
                    self.spatial_index[(i, j, '*')].append(cube_id)
                        
    def insert_vector(self, vec_id: str, vector: List[float]) -> None:
        """
        Insert a vector into the database.
        
        Args:
            vec_id: Unique identifier for the vector
            vector: The vector to insert
            
        Raises:
            ValueError: If vector dimensions don't match or values are invalid
            RuntimeError: If database capacity is exceeded
        """
        # Validate vector count
        if len(self.vectors) >= self.MAX_VECTORS:
            raise RuntimeError(f"Database capacity exceeded (max {self.MAX_VECTORS} vectors)")
            
        # Validate dimensions
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions")
            
        # Validate vector values
        for val in vector:
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Vector contains invalid value: {val}")
        
        # Store the vector
        self.vectors[vec_id] = np.array(vector, dtype=float)
        
        # For 2D vectors, also index in spatial grid
        if self.dimensions == 2:
            x, y = vector
            cell_x = int(x / self.cell_size)
            cell_y = int(y / self.cell_size)
            self.spatial_index[(cell_x, cell_y)].append((vec_id, vector))
    
    def insert(self, x: float, y: float) -> None:
        """
        Insert a 2D point into the database (compatibility method).
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self.dimensions != 2:
            raise ValueError("insert(x, y) only supported in 2D mode")
        
        point_id = f"point_{len(self.vectors)}"
        self.insert_vector(point_id, [x, y])
    
    def query_vector(self, query_vector: List[float], radius: float) -> List[Tuple[str, List[float]]]:
        """
        Query vectors within a specified radius of the query vector.
        
        Args:
            query_vector: The center point for the query
            radius: The radius within which to find vectors
            
        Returns:
            List of (vector_id, vector) tuples within the radius
            
        Raises:
            ValueError: If radius or query vector is invalid
        """
        # Validate radius
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        if radius > self.MAX_RADIUS:
            raise ValueError(f"Radius exceeds maximum allowed value ({self.MAX_RADIUS})")
            
        # Validate query vector dimensions
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector must have {self.dimensions} dimensions")
            
        # Validate query vector values
        for val in query_vector:
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Query vector contains invalid value: {val}")
        
        results = []
        if self.dimensions == 2:
            # Use spatial grid for 2D
            x, y = query_vector
            min_x = int((x - radius) / self.cell_size)
            max_x = int((x + radius) / self.cell_size)
            min_y = int((y - radius) / self.cell_size)
            max_y = int((y + radius) / self.cell_size)
            
            for cell_x in range(min_x, max_x + 1):
                for cell_y in range(min_y, max_y + 1):
                    for item in self.spatial_index.get((cell_x, cell_y), []):
                        # Item could be a cube_id or (vec_id, vector) tuple
                        if isinstance(item, tuple):
                            vec_id, vector = item
                            # Calculate distance
                            dist_sq = sum((query_vector[i] - vector[i])**2 for i in range(self.dimensions))
                            if dist_sq <= radius**2:
                                results.append((vec_id, vector))
        else:
            # Linear scan for n-D (can be optimized later)
            query_vec = np.array(query_vector)
            for vec_id, vector in self.vectors.items():
                dist_sq = np.sum((query_vec - vector)**2)
                if dist_sq <= radius**2:
                    results.append((vec_id, vector.tolist()))
        
        return results
    
    def query(self, x: float, y: float, radius: float) -> List[Tuple[float, float]]:
        """
        Query points within a specified radius (compatibility method for 2D).
        
        Args:
            x: X coordinate of query center
            y: Y coordinate of query center
            radius: Radius to search within
            
        Returns:
            List of (x, y) points within the radius
        """
        if self.dimensions != 2:
            raise ValueError("query(x, y, radius) only supported in 2D mode")
            
        results = self.query_vector([x, y], radius)
        return [(float(vector[0]), float(vector[1])) for _, vector in results]
    
    def query_with_provenance(self, x: float, y: float, radius: float) -> Dict[str, Any]:
        """
        Query points with detailed provenance information showing the work.
        
        Args:
            x: X coordinate of query center
            y: Y coordinate of query center
            radius: Radius to search within
            
        Returns:
            Dict with results and provenance information
        """
        if self.dimensions != 2:
            raise ValueError("query_with_provenance currently only supported in 2D mode")
            
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        
        provenance = {
            'query_point': (x, y),
            'radius': radius,
            'energy_breakdown': [],
            'parseval_compliance': None,
            'points_examined': 0,
            'points_found': 0
        }
        
        results = []
        total_energy = 0.0
        
        # Perform query with detailed tracking
        for vec_id, vector in self.vectors.items():
            if len(vector) != 2:  # Skip non-2D vectors
                continue
                
            provenance['points_examined'] += 1
            
            # Calculate distance and energy contribution
            px, py = vector
            dx = px - x
            dy = py - y
            dist_sq = dx**2 + dy**2
            
            # Track energy breakdown
            energy_contrib = [dx**2, dy**2]
            provenance['energy_breakdown'].append({
                'point': (float(px), float(py)),
                'energy_contributions': [float(contrib) for contrib in energy_contrib],
                'total_energy': float(dist_sq)
            })
            
            if dist_sq <= radius**2:
                results.append((float(px), float(py)))
                provenance['points_found'] += 1
                total_energy += dist_sq
        
        # Verify Parseval compliance for results
        if results:
            # For 2D: ||x||² = Σ|cᵢ|² should hold for each point
            parseval_verified = all(
                abs(point[0]**2 + point[1]**2 - 
                    sum(prov['energy_contributions'][0] + prov['energy_contributions'][1] 
                        for prov in provenance['energy_breakdown'] 
                        if prov['point'] == point)) < 1e-10
                for point in results
            )
            provenance['parseval_compliance'] = {
                'verified': parseval_verified,
                'total_energy': float(total_energy),
                'tolerance': 1e-10
            }
        
        return {
            'results': results,
            'provenance': provenance
        }
    
    def verify_parseval_equality(self, vector: List[float], basis: Optional[List[List[float]]] = None) -> bool:
        """
        Verify if Parseval's equality holds for a given vector and basis.
        
        Args:
            vector: The vector to check
            basis: Optional orthonormal basis (if None, standard basis is used)
            
        Returns:
            True if Parseval's equality holds within numerical precision
        """
        dimensions = len(vector)
        
        if basis is None:
            basis = self._create_standard_basis(dimensions)
        
        # Calculate vector norm squared
        vector_norm_sq = sum(v**2 for v in vector)
        
        # Calculate sum of squared projection coefficients
        coefficients = self._project_vector(vector, basis)
        coeff_sum_sq = sum(c**2 for c in coefficients)
        
        # Check equality within numerical precision
        epsilon = 1e-10
        return abs(vector_norm_sq - coeff_sum_sq) < epsilon
    
    def _create_standard_basis(self, dimensions: int) -> List[List[float]]:
        """Create standard orthonormal basis for given dimensions."""
        basis = []
        for i in range(dimensions):
            vector = [0.0] * dimensions
            vector[i] = 1.0
            basis.append(vector)
        return basis
    
    def _project_vector(self, vector: List[float], basis: List[List[float]]) -> List[float]:
        """Project a vector onto an orthonormal basis."""
        coefficients = []
        for basis_vector in basis:
            # Calculate inner product
            inner_prod = sum(v * b for v, b in zip(vector, basis_vector))
            coefficients.append(inner_prod)
        return coefficients
        
    def optimize_spatial_index(self, new_cell_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize the spatial index by rebuilding it, optionally with a new cell size.
        
        Args:
            new_cell_size: Optional new cell size for the grid
            
        Returns:
            Dictionary with optimization metrics
        """
        old_cell_size = self.cell_size
        if new_cell_size:
            self.cell_size = new_cell_size
            
        start_time = time.time()
        
        # Save vectors
        vectors = self.vectors.copy()
        vector_count = len(vectors)
        
        # Clear and rebuild index
        old_index_size = len(self.spatial_index)
        self.spatial_index = defaultdict(list)
        self.vectors = {}
        
        # Reinsert all vectors
        for vec_id, vector in vectors.items():
            self.insert_vector(vec_id, vector.tolist())
        
        end_time = time.time()
        
        # Calculate metrics
        new_index_size = len(self.spatial_index)
        time_taken = end_time - start_time
        
        return {
            'vectors_reindexed': vector_count,
            'old_cell_size': old_cell_size,
            'new_cell_size': self.cell_size,
            'old_index_size': old_index_size,
            'new_index_size': new_index_size, 
            'time_taken': time_taken
        }
    
    def query_vector_optimized(self, query_vector: List[float], radius: float, 
                            use_early_termination: bool = True) -> Dict[str, Any]:
        """
        Optimized vector query with early termination and energy tracking.
        
        Args:
            query_vector: The center point for the query
            radius: The radius within which to find vectors
            use_early_termination: Whether to use early termination optimization
            
        Returns:
            Dict containing results and provenance information
        """
        # Validate radius
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        if radius > self.MAX_RADIUS:
            raise ValueError(f"Radius exceeds maximum allowed value ({self.MAX_RADIUS})")
            
        # Validate query vector dimensions
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector must have {self.dimensions} dimensions")
            
        # Validate query vector values
        for val in query_vector:
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Query vector contains invalid value: {val}")
        
        radius_sq = radius**2
        results = []
        provenance = {
            'early_terminations': 0,
            'dimensions_checked': [],
            'energy_tracking': []
        }
        
        for vec_id, vector in self.vectors.items():
            dist_sq = 0.0
            energy_log = []
            dims_checked = 0
            
            if use_early_termination:
                # Process dimensions separately for early termination
                for dim in range(self.dimensions):
                    dim_energy = (float(vector[dim]) - query_vector[dim])**2
                    energy_log.append(dim_energy)
                    dist_sq += dim_energy
                    dims_checked += 1
                    
                    if dist_sq > radius_sq:
                        provenance['early_terminations'] += 1
                        provenance['dimensions_checked'].append(dims_checked)
                        break
            else:
                # Process all dimensions
                for dim in range(self.dimensions):
                    dim_energy = (float(vector[dim]) - query_vector[dim])**2
                    energy_log.append(dim_energy)
                    dist_sq += dim_energy
                dims_checked = self.dimensions
            
            # Track energy regardless of whether point is within radius
            provenance['energy_tracking'].append({
                'vector_id': vec_id,
                'energy_by_dimension': energy_log[:],
                'dimensions_checked': dims_checked,
                'total_energy': dist_sq,
                'within_radius': dist_sq <= radius_sq
            })
                
            if dist_sq <= radius_sq:
                results.append((vec_id, vector.tolist()))
        
        return {
            'results': results,
            'provenance': provenance
        }
    
    def find_optimal_cell_size(self, sample_size: int = 100) -> float:
        """
        Find the optimal cell size for the current dataset by benchmarking.
        
        Args:
            sample_size: Number of sample queries to run for each cell size
            
        Returns:
            The optimal cell size value
        """
        if len(self.vectors) < 10:
            return self.cell_size  # Not enough data to optimize
            
        # Save current cell size
        original_cell_size = self.cell_size
        
        # Test different cell sizes
        cell_sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        query_times = {}
        
        # Generate random query points
        if self.dimensions == 2:
            # For 2D, use the actual data range for realistic queries
            x_values = [vector[0] for vector in self.vectors.values()]
            y_values = [vector[1] for vector in self.vectors.values()]
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)
            
            query_points = [
                [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
                for _ in range(sample_size)
            ]
        else:
            # For higher dimensions, use random points
            query_points = [
                [random.uniform(-100, 100) for _ in range(self.dimensions)]
                for _ in range(sample_size)
            ]
            
        # Standard radius for testing
        radius = 10.0
        
        for cell_size in cell_sizes:
            # Rebuild index with this cell size
            self.optimize_spatial_index(cell_size)
            
            # Run queries and measure time
            start_time = time.time()
            for query_point in query_points:
                self.query_vector(query_point, radius)
            total_time = time.time() - start_time
            query_times[cell_size] = total_time / sample_size
            
        # Find optimal cell size
        optimal_cell_size = min(query_times.items(), key=lambda x: x[1])[0]
        
        # Restore original cell size if needed
        if optimal_cell_size != original_cell_size:
            self.optimize_spatial_index(optimal_cell_size)
            
        return optimal_cell_size
        
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage by cleaning up empty grid cells and 
        consolidating the spatial index.
        
        Returns:
            Dictionary with optimization metrics
        """
        # Track metrics
        metrics = {
            'cells_before': len(self.spatial_index),
            'vectors': len(self.vectors),
            'cells_removed': 0,
            'memory_saved': 0
        }
        
        # Calculate bytes per cell (implementation-specific)
        bytes_per_cell = 48 + (self.dimensions * 8)  # Estimate based on key size + list overhead
        
        # Find empty cells
        empty_cells = []
        for cell, items in self.spatial_index.items():
            if not items:
                empty_cells.append(cell)
        
        # Delete empty cells
        for cell in empty_cells:
            del self.spatial_index[cell]
        
        # Update metrics
        metrics['cells_removed'] = len(empty_cells)
        metrics['memory_saved'] = len(empty_cells) * bytes_per_cell
        metrics['cells_after'] = len(self.spatial_index)
        
        return metrics
        
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the spatial index and memory usage.
        
        Returns:
            Dictionary with index statistics
        """
        # Count items in each cell
        cell_counts = {}
        for cell, items in self.spatial_index.items():
            cell_counts[cell] = len(items)
            
        # Calculate statistics
        cell_count = len(self.spatial_index)
        non_empty_cells = sum(1 for count in cell_counts.values() if count > 0)
        empty_cells = cell_count - non_empty_cells
        
        if non_empty_cells > 0:
            avg_per_cell = sum(cell_counts.values()) / non_empty_cells
        else:
            avg_per_cell = 0
            
        max_per_cell = max(cell_counts.values()) if cell_counts else 0
        
        # Estimate memory usage
        cell_key_size = 24  # Tuple overhead + 2 integers
        vector_size = 24 + (self.dimensions * 8)  # Numpy array overhead + float64 per dimension
        
        # Calculate total memory
        memory_bytes = (
            cell_count * cell_key_size +  # Cell keys
            len(self.vectors) * vector_size +  # Vector data
            sum(cell_counts.values()) * 16  # References in spatial index (8 bytes per reference)
        )
        
        return {
            'cell_count': cell_count,
            'total_vectors': len(self.vectors),
            'non_empty_cells': non_empty_cells,
            'empty_cells': empty_cells,
            'avg_vectors_per_cell': avg_per_cell,
            'max_vectors_per_cell': max_per_cell, 
            'estimated_memory_bytes': memory_bytes
        }
