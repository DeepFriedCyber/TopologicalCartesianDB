from mpmath import mp
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from collections import defaultdict
from .cube import Cube

class TopologicalCartesianDB:
    """
    A database system that uses geometric cubes in n-dimensional space for data storage and querying.
    Supports Hilbert space operations and verification of Parseval's theorem.
    """
    def __init__(self):
        """Initialize an empty database."""
        self.cubes = []
        self.data = {}
        self.spatial_index = defaultdict(list)  # Grid-based spatial index
        self.cell_size = 1.0  # Default cell size for spatial indexing
        
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
    
    def add_cube_from_center(self, center: List[float], sizes: Union[float, List[float]],
                          data: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new cube defined by its center and size(s).
        
        Args:
            center: Center coordinates of the cube
            sizes: Either a single size for all dimensions, or a list of sizes per dimension
            data: Optional metadata to associate with this cube
            
        Returns:
            The ID of the newly added cube
        """
        cube = Cube.from_center_and_size(center, sizes, data)
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
        dimensions = cube.dimensions
        
        # Determine grid cells that this cube intersects with
        min_cells = [int(cube.min_coords[i] / self.cell_size) for i in range(dimensions)]
        max_cells = [int(cube.max_coords[i] / self.cell_size) for i in range(dimensions)]
        
        # Add cube to all intersecting grid cells
        for i in range(min_cells[0], max_cells[0] + 1):
            for j in range(min_cells[1], max_cells[1] + 1):
                if dimensions > 2:
                    for k in range(min_cells[2], max_cells[2] + 1):
                        if dimensions > 3:
                            # For dimensions > 3, we simplify by just using the min and max cells
                            # This is less efficient but handles n-dimensional cases
                            self.spatial_index[(i, j, k, '*')].append(cube_id)
                        else:
                            self.spatial_index[(i, j, k)].append(cube_id)
                else:
                    self.spatial_index[(i, j)].append(cube_id)
    
    def query_point(self, point: List[float]) -> List[int]:
        """
        Find all cubes that contain a given point.
        
        Args:
            point: Coordinates of the point to query
            
        Returns:
            List of IDs of cubes containing the point
        """
        dimensions = len(point)
        
        # Find grid cell for the point
        cell_coords = [int(point[i] / self.cell_size) for i in range(dimensions)]
        
        # Get candidate cubes from spatial index
        if dimensions > 3:
            cell_key = tuple(cell_coords[:3]) + ('*',)
        else:
            cell_key = tuple(cell_coords[:dimensions])
            
        candidates = set(self.spatial_index[cell_key])
        
        # Filter candidates to those that actually contain the point
        result = []
        for cube_id in candidates:
            if self.cubes[cube_id].contains_point(point):
                result.append(cube_id)
                
        return result
    
    def query_region(self, min_coords: List[float], max_coords: List[float]) -> List[int]:
        """
        Find all cubes that intersect with a given region.
        
        Args:
            min_coords: Minimum coordinates of the query region
            max_coords: Maximum coordinates of the query region
            
        Returns:
            List of IDs of intersecting cubes
        """
        query_cube = Cube(min_coords, max_coords)
        dimensions = query_cube.dimensions
        
        # Find grid cells that the query region intersects with
        min_cells = [int(min_coords[i] / self.cell_size) for i in range(dimensions)]
        max_cells = [int(max_coords[i] / self.cell_size) for i in range(dimensions)]
        
        # Collect candidate cubes from all intersecting cells
        candidates = set()
        
        # Handle different dimensionality cases
        if dimensions > 3:
            # Simplified approach for higher dimensions
            for i in range(min_cells[0], max_cells[0] + 1):
                for j in range(min_cells[1], max_cells[1] + 1):
                    for k in range(min_cells[2], max_cells[2] + 1):
                        cell_key = (i, j, k, '*')
                        candidates.update(self.spatial_index[cell_key])
        elif dimensions == 3:
            for i in range(min_cells[0], max_cells[0] + 1):
                for j in range(min_cells[1], max_cells[1] + 1):
                    for k in range(min_cells[2], max_cells[2] + 1):
                        cell_key = (i, j, k)
                        candidates.update(self.spatial_index[cell_key])
        else:  # dimensions == 2
            for i in range(min_cells[0], max_cells[0] + 1):
                for j in range(min_cells[1], max_cells[1] + 1):
                    cell_key = (i, j)
                    candidates.update(self.spatial_index[cell_key])
        
        # Filter candidates to those that actually intersect with the query region
        result = []
        for cube_id in candidates:
            if self.cubes[cube_id].intersects(query_cube):
                result.append(cube_id)
                
        return result
    
    def get_total_volume(self) -> float:
        """
        Calculate total volume of all cubes.
        
        Returns:
            Total volume
        """
        return sum(cube.volume() for cube in self.cubes)
    
    def get_cube_data(self, cube_id: int) -> Dict[str, Any]:
        """
        Get data associated with a cube.
        
        Args:
            cube_id: The ID of the cube
            
        Returns:
            Data associated with the cube
        """
        return self.data.get(cube_id, {})
    
    # Hilbert space and Parseval's theorem methods
    
    def create_orthonormal_basis(self, dimensions: int) -> List[List[float]]:
        """
        Create an orthonormal basis for a given number of dimensions.
        
        Args:
            dimensions: Number of dimensions
            
        Returns:
            List of orthonormal basis vectors
        """
        basis = []
        for i in range(dimensions):
            vector = [0.0] * dimensions
            vector[i] = 1.0  # Standard basis vector
            basis.append(vector)
        return basis
    
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
            basis = self.create_orthonormal_basis(dimensions)
        
        # Create a temporary cube to use its methods
        # (Using unit cube from origin for simplicity)
        temp_cube = Cube([0] * dimensions, [1] * dimensions)
        
        return temp_cube.verify_parseval_equality(vector, basis)
    
    def project_vector(self, vector: List[float], basis: Optional[List[List[float]]] = None) -> List[float]:
        """
        Project a vector onto an orthonormal basis.
        
        Args:
            vector: The vector to project
            basis: Optional orthonormal basis (if None, standard basis is used)
            
        Returns:
            The projection coefficients
        """
        dimensions = len(vector)
        
        if basis is None:
            basis = self.create_orthonormal_basis(dimensions)
        
        # Create a temporary cube to use its methods
        temp_cube = Cube([0] * dimensions, [1] * dimensions)
        
        return temp_cube.project_vector_onto_basis(vector, basis)
