"""
ProvenanceCube module for adding "Show Your Work" functionality.
"""
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from .cube_adapter import CubeAdapter

class ProvenanceCube(CubeAdapter):
    """
    Adds "show your work" functionality to any source database.
    
    This cube adds detailed provenance tracking to query operations,
    showing how results were calculated and providing verification
    of mathematical properties.
    """
    
    def __init__(self, source_db: Any):
        """
        Initialize the Provenance cube.
        
        Args:
            source_db: The source database or adapter to wrap
        """
        super().__init__(source_db)
        self.query_log = []
    
    def query_with_provenance(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute query with full provenance tracking.
        
        This method adds detailed tracking information to any query,
        recording how the results were calculated and verifying
        mathematical properties.
        
        Args:
            *args: Arguments to pass to the underlying query method
            **kwargs: Keyword arguments to pass to the underlying query method
            
        Returns:
            Dictionary containing results and detailed provenance information
        """
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        # Determine which query method to call
        if hasattr(self.source, 'query_vector') and len(args) >= 2:
            # This looks like a vector query
            results = self.source.query_vector(*args, **kwargs)
            query_type = 'vector'
        else:
            # Default to standard query
            results = self.source.query(*args, **kwargs)
            query_type = 'point'
        
        end_time = time.time()
        
        # Build basic provenance record
        provenance = {
            'query_id': query_id,
            'query_type': query_type,
            'timestamp': start_time,
            'execution_time': end_time - start_time,
            'parameters': {'args': args, 'kwargs': kwargs},
            'result_count': len(results) if results else 0
        }
        
        # Add class hierarchy
        if hasattr(self, '__class_hierarchy__'):
            provenance['class_hierarchy'] = self.__class_hierarchy__
            
        # Add energy breakdown for vector queries
        if query_type == 'vector' and len(args) >= 2:
            provenance['energy_breakdown'] = self._analyze_energy(results, args[0])
            
            # Check Parseval compliance if source supports it
            provenance['parseval_compliance'] = self._verify_parseval(results)
        
        # Log this query
        self.query_log.append(provenance)
        
        return {
            'results': results,
            'provenance': provenance
        }
    
    def _analyze_energy(self, results: List[Tuple[str, List[float]]], query_vector: List[float]) -> List[Dict[str, Any]]:
        """
        Analyze energy components of vector distance calculations.
        
        Args:
            results: List of (vector_id, vector) tuples
            query_vector: The query vector
            
        Returns:
            List of energy breakdown dictionaries
        """
        if not results:
            return []
        
        energy_breakdown = []
        query_array = np.array(query_vector, dtype=float)
        
        for vec_id, vector in results:
            vector_array = np.array(vector, dtype=float)
            
            # Calculate dimension-by-dimension energy
            dim_energy = (vector_array - query_array)**2
            total_energy = np.sum(dim_energy)
            
            energy_breakdown.append({
                'vector_id': vec_id,
                'vector': vector,
                'energy_by_dimension': dim_energy.tolist(),
                'total_energy': float(total_energy)
            })
            
        return energy_breakdown
    
    def _verify_parseval(self, results: List[Tuple[str, List[float]]]) -> Optional[Dict[str, Any]]:
        """
        Verify Parseval's theorem for results if supported.
        
        Args:
            results: List of (vector_id, vector) tuples
            
        Returns:
            Dictionary with Parseval compliance information or None if not applicable
        """
        if not hasattr(self.source, 'verify_parseval_equality'):
            return None
            
        # Check each vector for Parseval compliance
        compliance = {'vectors_checked': 0, 'vectors_compliant': 0}
        
        for _, vector in results:
            compliance['vectors_checked'] += 1
            if self.source.verify_parseval_equality(vector):
                compliance['vectors_compliant'] += 1
                
        compliance['compliance_rate'] = (compliance['vectors_compliant'] / 
                                        max(1, compliance['vectors_checked']))
                
        return compliance
        
    def get_query_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all queries executed.
        
        Returns:
            List of query provenance records
        """
        return self.query_log
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ProvenanceCube.
        
        Returns:
            Dictionary with statistics
        """
        # Get base stats
        if hasattr(self.source, 'get_stats'):
            stats = self.source.get_stats()
        else:
            stats = {}
        
        # Add provenance stats
        stats.update({
            'adapter_type': self.__class__.__name__,
            'queries_tracked': len(self.query_log)
        })
        
        return stats
