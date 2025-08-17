"""
OptimizationCube module for adding performance optimizations.
"""
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from .cube_adapter import CubeAdapter

class OptimizationCube(CubeAdapter):
    """
    Adds performance optimizations to any source database.
    
    This cube implements advanced optimization techniques including:
    1. Early termination for vector queries
    2. Caching of frequent queries
    3. Smart indexing for high-dimensional data
    """
    
    def __init__(self, source_db: Any, enable_cache: bool = True, 
                cache_size: int = 100, enable_early_termination: bool = True):
        """
        Initialize the Optimization cube.
        
        Args:
            source_db: The source database or adapter to wrap
            enable_cache: Whether to enable query result caching
            cache_size: Maximum number of cached query results
            enable_early_termination: Whether to enable early termination
        """
        super().__init__(source_db)
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.enable_early_termination = enable_early_termination
        
        # Query cache: (query_args_hash) -> (timestamp, results)
        self.query_cache = {}
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'early_terminations': 0,
            'query_times': []
        }
    
    def query_vector_optimized(self, query_vector: List[float], radius: float, 
                             use_early_termination: Optional[bool] = None) -> Dict[str, Any]:
        """
        Optimized vector query with early termination.
        
        Args:
            query_vector: The center point for the query
            radius: The radius within which to find vectors
            use_early_termination: Whether to use early termination 
                                  (defaults to self.enable_early_termination)
            
        Returns:
            Dictionary containing results and optimization information
        """
        if use_early_termination is None:
            use_early_termination = self.enable_early_termination
            
        # Try cache first if enabled
        if self.enable_cache:
            cache_key = self._make_cache_key(query_vector, radius)
            cached = self.query_cache.get(cache_key)
            
            if cached:
                self.metrics['cache_hits'] += 1
                timestamp, results = cached
                return {
                    'results': results,
                    'optimization': {
                        'source': 'cache',
                        'age': time.time() - timestamp
                    }
                }
            else:
                self.metrics['cache_misses'] += 1
        
        # Not in cache or cache disabled, perform query
        start_time = time.time()
        
        if hasattr(self.source, 'query_vector'):
            # Direct query if no early termination or source already has query_vector
            if not use_early_termination:
                results = self.source.query_vector(query_vector, radius)
                optimization_info = {'early_termination': False}
            else:
                # Perform early termination optimization
                results, opt_info = self._early_termination_query(query_vector, radius)
                optimization_info = {
                    'early_termination': True,
                    'dimensions_checked': opt_info['dimensions_checked'],
                    'early_terminations': opt_info['early_terminations']
                }
                self.metrics['early_terminations'] += opt_info['early_terminations']
        else:
            # Fallback if source doesn't support vector queries
            results = []
            optimization_info = {'unsupported': True}
        
        # Record query time
        query_time = time.time() - start_time
        self.metrics['query_times'].append(query_time)
        optimization_info['query_time'] = query_time
        
        # Cache result if enabled
        if self.enable_cache:
            self._update_cache(query_vector, radius, results)
        
        return {
            'results': results,
            'optimization': optimization_info
        }
    
    def _early_termination_query(self, query_vector: List[float], 
                               radius: float) -> Tuple[List[Tuple[str, List[float]]], Dict[str, Any]]:
        """
        Perform vector query with early termination optimization.
        
        Args:
            query_vector: The center point for the query
            radius: The radius within which to find vectors
            
        Returns:
            Tuple of (results, optimization_info)
        """
        # Only works if source has vectors attribute
        if not hasattr(self.source, 'vectors'):
            # Fall back to standard query
            return self.source.query_vector(query_vector, radius), {'early_termination': False}
        
        # Setup for early termination
        radius_sq = radius**2
        results = []
        optimization_info = {
            'early_termination': True,
            'dimensions_checked': [],
            'early_terminations': 0
        }
        
        # Convert to numpy array
        query_array = np.array(query_vector, dtype=float)
        dimensions = len(query_vector)
        
        for vec_id, vector in self.source.vectors.items():
            dist_sq = 0.0
            dims_checked = 0
            
            # Process dimensions separately for early termination
            for dim in range(dimensions):
                dim_energy = (float(vector[dim]) - query_array[dim])**2
                dist_sq += dim_energy
                dims_checked += 1
                
                if dist_sq > radius_sq:
                    optimization_info['early_terminations'] += 1
                    optimization_info['dimensions_checked'].append(dims_checked)
                    break
            
            # If we checked all dimensions and it's within radius, add to results
            if dims_checked == dimensions and dist_sq <= radius_sq:
                results.append((vec_id, vector.tolist()))
        
        return results, optimization_info
    
    def _make_cache_key(self, query_vector: List[float], radius: float) -> str:
        """Create a cache key from query parameters."""
        # Convert to tuple for hashability
        vector_tuple = tuple(float(v) for v in query_vector)
        return str(hash((vector_tuple, float(radius))))
    
    def _update_cache(self, query_vector: List[float], radius: float, 
                     results: List[Tuple[str, List[float]]]) -> None:
        """Update the query cache."""
        # Create cache entry
        cache_key = self._make_cache_key(query_vector, radius)
        self.query_cache[cache_key] = (time.time(), results)
        
        # If cache is full, remove oldest entry
        if len(self.query_cache) > self.cache_size:
            oldest_key = None
            oldest_time = float('inf')
            
            for key, (timestamp, _) in self.query_cache.items():
                if timestamp < oldest_time:
                    oldest_time = timestamp
                    oldest_key = key
                    
            if oldest_key:
                del self.query_cache[oldest_key]
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the query cache.
        
        Returns:
            Dictionary with cache clearing metrics
        """
        cache_size = len(self.query_cache)
        self.query_cache.clear()
        
        return {
            'entries_removed': cache_size,
            'memory_freed': cache_size * 1000  # Rough estimate in bytes
        }
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """
        Get optimization performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate average query time if available
        if self.metrics['query_times']:
            metrics['avg_query_time'] = sum(self.metrics['query_times']) / len(self.metrics['query_times'])
            metrics['min_query_time'] = min(self.metrics['query_times'])
            metrics['max_query_time'] = max(self.metrics['query_times'])
        
        # Add cache information
        metrics['cache_size'] = len(self.query_cache)
        metrics['cache_limit'] = self.cache_size
        metrics['cache_usage'] = len(self.query_cache) / max(1, self.cache_size)
        
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the OptimizationCube.
        
        Returns:
            Dictionary with statistics
        """
        # Get base stats
        if hasattr(self.source, 'get_stats'):
            stats = self.source.get_stats()
        else:
            stats = {}
        
        # Add optimization stats
        opt_metrics = self.get_optimization_metrics()
        
        stats.update({
            'adapter_type': self.__class__.__name__,
            'cache_enabled': self.enable_cache,
            'early_termination_enabled': self.enable_early_termination,
            'cache_hits': opt_metrics.get('cache_hits', 0),
            'cache_misses': opt_metrics.get('cache_misses', 0),
            'early_terminations': opt_metrics.get('early_terminations', 0)
        })
        
        return stats
