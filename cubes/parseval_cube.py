"""
ParsevalCube module for adding vector operations with Parseval's theorem verification.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from .cube_adapter import CubeAdapter

class ParsevalCube(CubeAdapter):
    """
    Adds n-dimensional vector support with Parseval compliance verification.
    
    This cube extends the core database with support for:
    1. n-dimensional vectors
    2. Parseval's theorem verification
    3. Vector operations and projections
    """
    
    def __init__(self, source_db: Any, dimensions: int = 2):
        """
        Initialize the Parseval cube.
        
        Args:
            source_db: The source database or adapter to wrap
            dimensions: Number of dimensions for vectors
        
        Raises:
            ValueError: If dimensions is less than 1
        """
        super().__init__(source_db)
        if dimensions < 1:
            raise ValueError("Dimensions must be at least 1")
            
        self.dimensions = dimensions
        self.vectors: Dict[str, np.ndarray] = {}  # Store vectors by ID
    
    def insert_vector(self, vec_id: str, vector: List[float]) -> None:
        """
        Insert an n-dimensional vector with validation.
        
        Args:
            vec_id: Unique identifier for the vector
            vector: Vector coordinates
            
        Raises:
            ValueError: If vector dimension doesn't match or has invalid values
        """
        # Validate dimensions
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions")
        
        # Convert to numpy array
        vector_array = np.array(vector, dtype=float)
        
        # Validate values
        if not np.all(np.isfinite(vector_array)):
            raise ValueError("Vector contains infinite or NaN values")
        
        # Calculate energy
        energy = np.sum(vector_array**2)
        if not np.isfinite(energy):
            raise ValueError("Vector has infinite energy")
            
        # Store the vector
        self.vectors[vec_id] = vector_array
        
        # If 2D, also insert into the source database
        if self.dimensions == 2:
            x, y = vector
            self.source.insert(x, y, data={'vec_id': vec_id})
    
    def query_vector(self, query_vector: List[float], radius: float) -> List[Tuple[str, List[float]]]:
        """
        Query vectors within a specified radius of the query vector.
        
        Args:
            query_vector: The center point for the query
            radius: The radius within which to find vectors
            
        Returns:
            List of (vector_id, vector) tuples within the radius
            
        Raises:
            ValueError: If dimensions don't match or radius is invalid
        """
        # Validate dimensions
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector must have {self.dimensions} dimensions")
            
        # Validate radius
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        
        # Convert to numpy array
        query_array = np.array(query_vector, dtype=float)
        radius_sq = radius**2
        
        # Find all vectors within radius
        results = []
        for vec_id, vector in self.vectors.items():
            dist_sq = np.sum((query_array - vector)**2)
            if dist_sq <= radius_sq:
                results.append((vec_id, vector.tolist()))
        
        return results
    
    def verify_parseval_equality(self, vector: List[float], basis: Optional[List[List[float]]] = None) -> bool:
        """
        Verify if Parseval's equality holds for a given vector and basis.
        
        Args:
            vector: The vector to check
            basis: Optional orthonormal basis (if None, standard basis is used)
            
        Returns:
            True if Parseval's equality holds within numerical precision
        """
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions")
        
        # Create standard basis if not provided
        if basis is None:
            basis = self._create_standard_basis()
        elif len(basis) != self.dimensions:
            raise ValueError(f"Basis must have {self.dimensions} vectors")
            
        # Convert to numpy arrays
        vector_array = np.array(vector, dtype=float)
        basis_arrays = [np.array(b, dtype=float) for b in basis]
        
        # Calculate vector norm squared
        vector_norm_sq = np.sum(vector_array**2)
        
        # Calculate sum of squared projection coefficients
        coefficients = self._project_vector(vector_array, basis_arrays)
        coeff_sum_sq = np.sum(np.square(coefficients))
        
        # Check equality within numerical precision
        epsilon = 1e-10
        return abs(vector_norm_sq - coeff_sum_sq) < epsilon
    
    def _create_standard_basis(self) -> List[np.ndarray]:
        """Create standard orthonormal basis for current dimensions."""
        basis = []
        for i in range(self.dimensions):
            vector = np.zeros(self.dimensions)
            vector[i] = 1.0
            basis.append(vector)
        return basis
    
    def _project_vector(self, vector: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
        """Project a vector onto a basis."""
        coefficients = []
        for basis_vector in basis:
            # Calculate inner product
            inner_prod = np.dot(vector, basis_vector)
            coefficients.append(inner_prod)
        return np.array(coefficients)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ParsevalCube.
        
        Returns:
            Dictionary with statistics
        """
        # Get base stats
        if hasattr(self.source, 'get_stats'):
            stats = self.source.get_stats()
        else:
            stats = {}
        
        # Add Parseval cube stats
        stats.update({
            'adapter_type': self.__class__.__name__,
            'dimensions': self.dimensions,
            'vector_count': len(self.vectors)
        })
        
        return stats
