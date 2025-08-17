from mpmath import mp
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

class Cube:
    """
    A geometric cube in n-dimensional space defined by its minimum and maximum coordinates.
    Supports arbitrary dimensions for Hilbert space operations and Parseval theorem applications.
    """
    def __init__(self, min_coords: List[float], max_coords: List[float], data: Optional[Dict[str, Any]] = None):
        """
        Initialize a cube with min and max coordinates along each dimension.
        
        Args:
            min_coords: List of minimum coordinates for each dimension
            max_coords: List of maximum coordinates for each dimension
            data: Optional metadata associated with this cube
        """
        if len(min_coords) != len(max_coords):
            raise ValueError("Dimension mismatch: min_coords and max_coords must have same length")
            
        for i in range(len(min_coords)):
            if min_coords[i] > max_coords[i]:
                raise ValueError(f"Invalid coordinates: min_coords[{i}] > max_coords[{i}]")
        
        self.dimensions = len(min_coords)
        self.min_coords = mp.matrix(min_coords)
        self.max_coords = mp.matrix(max_coords)
        self.data = data or {}
        
        # Set high precision for exact arithmetic
        mp.dps = 50
    
    @classmethod
    def from_center_and_size(cls, center: List[float], sizes: Union[float, List[float]], 
                          data: Optional[Dict[str, Any]] = None) -> 'Cube':
        """
        Create a cube from center point and size(s).
        
        Args:
            center: Center coordinates of the cube
            sizes: Either a single size for all dimensions, or a list of sizes per dimension
            data: Optional metadata associated with this cube
            
        Returns:
            A new Cube instance
        """
        if isinstance(sizes, (int, float)):
            sizes = [sizes] * len(center)
        
        if len(center) != len(sizes):
            raise ValueError("Dimension mismatch: center and sizes must have same length")
            
        min_coords = [center[i] - sizes[i]/2 for i in range(len(center))]
        max_coords = [center[i] + sizes[i]/2 for i in range(len(center))]
        
        return cls(min_coords, max_coords, data)
    
    def volume(self) -> float:
        """
        Calculate the volume of the cube.
        
        Returns:
            The volume as a float
        """
        vol = mp.mpf(1)
        for i in range(self.dimensions):
            vol *= (self.max_coords[i] - self.min_coords[i])
        return float(vol)
    
    def contains_point(self, point: List[float]) -> bool:
        """
        Check if a point is contained within this cube.
        
        Args:
            point: The coordinates of the point to check
            
        Returns:
            True if the point is inside the cube, False otherwise
        """
        if len(point) != self.dimensions:
            raise ValueError(f"Dimension mismatch: point has {len(point)} dimensions, cube has {self.dimensions}")
            
        for i in range(self.dimensions):
            if point[i] < self.min_coords[i] or point[i] > self.max_coords[i]:
                return False
        return True
    
    def intersects(self, other: 'Cube') -> bool:
        """
        Check if this cube intersects with another cube.
        
        Args:
            other: Another Cube instance
            
        Returns:
            True if the cubes intersect, False otherwise
        """
        if self.dimensions != other.dimensions:
            raise ValueError("Dimension mismatch: cubes must have same dimensions")
            
        for i in range(self.dimensions):
            if (self.max_coords[i] < other.min_coords[i] or 
                self.min_coords[i] > other.max_coords[i]):
                return False
        return True
    
    def get_center(self) -> List[float]:
        """
        Get the center coordinates of the cube.
        
        Returns:
            List of center coordinates
        """
        return [(self.min_coords[i] + self.max_coords[i]) / 2 for i in range(self.dimensions)]
    
    def get_sizes(self) -> List[float]:
        """
        Get the size of the cube along each dimension.
        
        Returns:
            List of sizes
        """
        return [self.max_coords[i] - self.min_coords[i] for i in range(self.dimensions)]
    
    def subdivide(self) -> List['Cube']:
        """
        Subdivide this cube into 2^n smaller cubes (where n is the number of dimensions).
        
        Returns:
            List of new Cube instances
        """
        center = self.get_center()
        subcubes = []
        
        # Generate all combinations of min/max for subcubes
        # For n dimensions, we'll have 2^n subcubes
        for i in range(2**self.dimensions):
            binary = format(i, f'0{self.dimensions}b')  # Convert to binary representation
            
            subcube_min = []
            subcube_max = []
            
            for dim in range(self.dimensions):
                if binary[dim] == '0':
                    subcube_min.append(self.min_coords[dim])
                    subcube_max.append(center[dim])
                else:
                    subcube_min.append(center[dim])
                    subcube_max.append(self.max_coords[dim])
            
            subcubes.append(Cube(subcube_min, subcube_max))
            
        return subcubes
    
    def __str__(self) -> str:
        """String representation of the cube"""
        return f"Cube(dimensions={self.dimensions}, min={self.min_coords}, max={self.max_coords})"
    
    # Methods for Parseval's theorem and Hilbert spaces
    
    def project_vector_onto_basis(self, vector: List[float], basis_vectors: List[List[float]]) -> List[float]:
        """
        Project a vector onto an orthonormal basis within the cube's coordinate system.
        This is useful for implementing Parseval's theorem.
        
        Args:
            vector: The vector to project
            basis_vectors: A list of orthonormal basis vectors
            
        Returns:
            The projection coefficients
        """
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector dimension mismatch: got {len(vector)}, expected {self.dimensions}")
        
        for basis_vector in basis_vectors:
            if len(basis_vector) != self.dimensions:
                raise ValueError(f"Basis vector dimension mismatch: got {len(basis_vector)}, expected {self.dimensions}")
        
        # Calculate projection coefficients
        coefficients = []
        for basis_vector in basis_vectors:
            # Inner product
            inner_prod = sum(v * b for v, b in zip(vector, basis_vector))
            coefficients.append(inner_prod)
            
        return coefficients
    
    def verify_parseval_equality(self, vector: List[float], basis_vectors: List[List[float]]) -> bool:
        """
        Verify Parseval's equality for a vector with respect to an orthonormal basis.
        
        Args:
            vector: The vector to verify
            basis_vectors: A list of orthonormal basis vectors
            
        Returns:
            True if Parseval's equality holds within numerical precision
        """
        # Calculate vector norm squared
        vector_norm_sq = sum(v**2 for v in vector)
        
        # Calculate sum of squared projection coefficients
        coefficients = self.project_vector_onto_basis(vector, basis_vectors)
        coeff_sum_sq = sum(c**2 for c in coefficients)
        
        # Check equality within numerical precision
        epsilon = 1e-10
        return abs(vector_norm_sq - coeff_sum_sq) < epsilon
