#!/usr/bin/env python3
"""
Optimized Spatial Engine

Addresses performance optimization feedback:
1. Spatial indexing for cube regions (KD-Tree, R-Tree)
2. Optimized distance calculations
3. Cube load balancing
4. Efficient nearest neighbor search
5. Caching and memoization
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import pickle
import hashlib

logger = logging.getLogger(__name__)

# Import spatial indexing libraries with fallbacks
SPATIAL_LIBS_AVAILABLE = {}

try:
    from sklearn.neighbors import KDTree, BallTree, NearestNeighbors
    SPATIAL_LIBS_AVAILABLE['sklearn'] = True
    logger.info("✅ Scikit-learn spatial indexing available")
except ImportError:
    SPATIAL_LIBS_AVAILABLE['sklearn'] = False
    logger.warning("❌ Scikit-learn not available")

try:
    import rtree
    from rtree import index
    SPATIAL_LIBS_AVAILABLE['rtree'] = True
    logger.info("✅ R-tree spatial indexing available")
except ImportError:
    SPATIAL_LIBS_AVAILABLE['rtree'] = False
    logger.warning("❌ R-tree not available")

try:
    import faiss
    SPATIAL_LIBS_AVAILABLE['faiss'] = True
    logger.info("✅ FAISS approximate search available")
except ImportError:
    SPATIAL_LIBS_AVAILABLE['faiss'] = False
    logger.warning("❌ FAISS not available")

try:
    import annoy
    SPATIAL_LIBS_AVAILABLE['annoy'] = True
    logger.info("✅ Annoy approximate search available")
except ImportError:
    SPATIAL_LIBS_AVAILABLE['annoy'] = False
    logger.warning("❌ Annoy not available")


@dataclass
class SpatialIndexConfig:
    """Configuration for spatial indexing"""
    index_type: str = 'auto'  # 'kdtree', 'balltree', 'rtree', 'faiss', 'annoy', 'auto'
    leaf_size: int = 30
    metric: str = 'euclidean'
    n_trees: int = 10  # For Annoy
    rebuild_threshold: int = 1000  # Rebuild index after this many additions
    cache_size: int = 10000
    use_approximate: bool = True


@dataclass
class CubeLoadMetrics:
    """Metrics for cube load balancing"""
    cube_id: str
    current_load: int
    max_capacity: int
    processing_time_avg: float
    queue_length: int
    success_rate: float
    last_update: float


class OptimizedDistanceCalculator:
    """Optimized distance calculations with vectorization and caching"""
    
    def __init__(self, cache_size: int = 10000):
        self.distance_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_lock = threading.Lock()
    
    def calculate_distance_batch(self, points1: np.ndarray, points2: np.ndarray, 
                                metric: str = 'euclidean', weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized batch distance calculation"""
        
        if metric == 'euclidean':
            if weights is not None:
                # Weighted Euclidean distance
                diff = points1 - points2
                weighted_diff = diff * weights
                distances = np.sqrt(np.sum(weighted_diff ** 2, axis=1))
            else:
                # Standard Euclidean distance
                distances = np.linalg.norm(points1 - points2, axis=1)
        
        elif metric == 'manhattan':
            if weights is not None:
                diff = np.abs(points1 - points2) * weights
                distances = np.sum(diff, axis=1)
            else:
                distances = np.sum(np.abs(points1 - points2), axis=1)
        
        elif metric == 'cosine':
            # Cosine distance
            dot_products = np.sum(points1 * points2, axis=1)
            norms1 = np.linalg.norm(points1, axis=1)
            norms2 = np.linalg.norm(points2, axis=1)
            
            # Avoid division by zero
            valid_mask = (norms1 > 0) & (norms2 > 0)
            distances = np.ones(len(points1))
            distances[valid_mask] = 1 - (dot_products[valid_mask] / (norms1[valid_mask] * norms2[valid_mask]))
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return distances
    
    def calculate_distance_cached(self, point1: np.ndarray, point2: np.ndarray, 
                                 metric: str = 'euclidean', weights: Optional[np.ndarray] = None) -> float:
        """Calculate distance with caching"""
        
        # Create cache key
        key_data = (point1.tobytes(), point2.tobytes(), metric, 
                   weights.tobytes() if weights is not None else None)
        cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
        
        with self.cache_lock:
            if cache_key in self.distance_cache:
                self.cache_hits += 1
                return self.distance_cache[cache_key]
            
            self.cache_misses += 1
        
        # Calculate distance
        distance = self.calculate_distance_batch(
            point1.reshape(1, -1), point2.reshape(1, -1), metric, weights
        )[0]
        
        # Cache result
        with self.cache_lock:
            if len(self.distance_cache) >= self.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.distance_cache.keys())[:self.cache_size // 4]
                for old_key in oldest_keys:
                    del self.distance_cache[old_key]
            
            self.distance_cache[cache_key] = distance
        
        return distance
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.distance_cache)
        }


class SpatialIndex:
    """Unified spatial index interface supporting multiple backends"""
    
    def __init__(self, config: SpatialIndexConfig):
        self.config = config
        self.index = None
        self.points = None
        self.document_ids = []
        self.index_type = self._select_index_type()
        self.build_time = 0.0
        self.query_count = 0
        self.total_query_time = 0.0
        
        logger.info(f"Initialized spatial index with type: {self.index_type}")
    
    def _select_index_type(self) -> str:
        """Select the best available index type"""
        if self.config.index_type != 'auto':
            if self._is_index_available(self.config.index_type):
                return self.config.index_type
            else:
                logger.warning(f"Requested index type {self.config.index_type} not available")
        
        # Auto-selection priority
        if SPATIAL_LIBS_AVAILABLE['faiss']:
            return 'faiss'
        elif SPATIAL_LIBS_AVAILABLE['sklearn']:
            return 'kdtree'
        elif SPATIAL_LIBS_AVAILABLE['annoy']:
            return 'annoy'
        elif SPATIAL_LIBS_AVAILABLE['rtree']:
            return 'rtree'
        else:
            return 'linear'  # Fallback to linear search
    
    def _is_index_available(self, index_type: str) -> bool:
        """Check if an index type is available"""
        availability_map = {
            'kdtree': SPATIAL_LIBS_AVAILABLE['sklearn'],
            'balltree': SPATIAL_LIBS_AVAILABLE['sklearn'],
            'faiss': SPATIAL_LIBS_AVAILABLE['faiss'],
            'annoy': SPATIAL_LIBS_AVAILABLE['annoy'],
            'rtree': SPATIAL_LIBS_AVAILABLE['rtree']
        }
        return availability_map.get(index_type, False)
    
    def build_index(self, points: np.ndarray, document_ids: List[str]):
        """Build spatial index from points"""
        start_time = time.time()
        
        self.points = points.copy()
        self.document_ids = document_ids.copy()
        
        if self.index_type == 'kdtree':
            self._build_kdtree()
        elif self.index_type == 'balltree':
            self._build_balltree()
        elif self.index_type == 'faiss':
            self._build_faiss()
        elif self.index_type == 'annoy':
            self._build_annoy()
        elif self.index_type == 'rtree':
            self._build_rtree()
        else:
            # Linear search fallback
            self.index = None
        
        self.build_time = time.time() - start_time
        logger.info(f"Built {self.index_type} index in {self.build_time:.3f}s for {len(points)} points")
    
    def _build_kdtree(self):
        """Build KD-Tree index"""
        self.index = KDTree(self.points, leaf_size=self.config.leaf_size, metric=self.config.metric)
    
    def _build_balltree(self):
        """Build Ball Tree index"""
        self.index = BallTree(self.points, leaf_size=self.config.leaf_size, metric=self.config.metric)
    
    def _build_faiss(self):
        """Build FAISS index"""
        dimension = self.points.shape[1]
        
        if self.config.metric == 'euclidean':
            self.index = faiss.IndexFlatL2(dimension)
        elif self.config.metric == 'cosine':
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            # Normalize points for cosine similarity
            faiss.normalize_L2(self.points)
        else:
            # Default to L2
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(self.points.astype(np.float32))
    
    def _build_annoy(self):
        """Build Annoy index"""
        dimension = self.points.shape[1]
        
        if self.config.metric == 'euclidean':
            metric = 'euclidean'
        elif self.config.metric == 'cosine':
            metric = 'angular'
        else:
            metric = 'euclidean'
        
        self.index = annoy.AnnoyIndex(dimension, metric)
        
        for i, point in enumerate(self.points):
            self.index.add_item(i, point)
        
        self.index.build(self.config.n_trees)
    
    def _build_rtree(self):
        """Build R-Tree index"""
        self.index = index.Index()
        
        for i, point in enumerate(self.points):
            # R-tree expects (min_x, min_y, max_x, max_y, ...) format
            # For points, min and max are the same
            bounds = []
            for coord in point:
                bounds.extend([coord, coord])
            
            self.index.insert(i, bounds)
    
    def query_nearest(self, query_point: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Query k nearest neighbors"""
        start_time = time.time()
        
        if self.index is None:
            # Linear search fallback
            results = self._linear_search(query_point, k)
        elif self.index_type in ['kdtree', 'balltree']:
            results = self._query_sklearn(query_point, k)
        elif self.index_type == 'faiss':
            results = self._query_faiss(query_point, k)
        elif self.index_type == 'annoy':
            results = self._query_annoy(query_point, k)
        elif self.index_type == 'rtree':
            results = self._query_rtree(query_point, k)
        else:
            results = self._linear_search(query_point, k)
        
        query_time = time.time() - start_time
        self.query_count += 1
        self.total_query_time += query_time
        
        return results
    
    def _query_sklearn(self, query_point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Query sklearn-based index"""
        distances, indices = self.index.query([query_point], k=k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.document_ids):
                results.append((self.document_ids[idx], float(dist)))
        
        return results
    
    def _query_faiss(self, query_point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Query FAISS index"""
        query_vector = query_point.reshape(1, -1).astype(np.float32)
        
        if self.config.metric == 'cosine':
            faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.document_ids) and idx >= 0:
                results.append((self.document_ids[idx], float(dist)))
        
        return results
    
    def _query_annoy(self, query_point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Query Annoy index"""
        indices, distances = self.index.get_nns_by_vector(query_point, k, include_distances=True)
        
        results = []
        for idx, dist in zip(indices, distances):
            if idx < len(self.document_ids):
                results.append((self.document_ids[idx], float(dist)))
        
        return results
    
    def _query_rtree(self, query_point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Query R-Tree index"""
        # R-tree doesn't directly support k-NN, so we use a radius search
        # and then sort by distance
        
        # Create search bounds (point query)
        bounds = []
        for coord in query_point:
            bounds.extend([coord, coord])
        
        # Get all candidates (this is inefficient for R-tree)
        candidates = list(self.index.intersection(bounds))
        
        # Calculate distances and sort
        distances = []
        for idx in candidates:
            if idx < len(self.points):
                dist = np.linalg.norm(query_point - self.points[idx])
                distances.append((idx, dist))
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        
        results = []
        for idx, dist in distances[:k]:
            if idx < len(self.document_ids):
                results.append((self.document_ids[idx], dist))
        
        return results
    
    def _linear_search(self, query_point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Fallback linear search"""
        distances = []
        
        for i, point in enumerate(self.points):
            dist = np.linalg.norm(query_point - point)
            distances.append((i, dist))
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        
        results = []
        for idx, dist in distances[:k]:
            if idx < len(self.document_ids):
                results.append((self.document_ids[idx], dist))
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_query_time = self.total_query_time / max(1, self.query_count)
        
        return {
            'index_type': self.index_type,
            'build_time': self.build_time,
            'total_points': len(self.points) if self.points is not None else 0,
            'query_count': self.query_count,
            'avg_query_time': avg_query_time,
            'total_query_time': self.total_query_time
        }


class CubeLoadBalancer:
    """Load balancer for distributing work across multiple cubes"""
    
    def __init__(self):
        self.cube_metrics = {}
        self.load_history = defaultdict(deque)
        self.balancing_strategy = 'least_loaded'  # 'least_loaded', 'round_robin', 'performance_based'
        self.metrics_lock = threading.Lock()
    
    def register_cube(self, cube_id: str, max_capacity: int = 1000):
        """Register a cube for load balancing"""
        with self.metrics_lock:
            self.cube_metrics[cube_id] = CubeLoadMetrics(
                cube_id=cube_id,
                current_load=0,
                max_capacity=max_capacity,
                processing_time_avg=0.0,
                queue_length=0,
                success_rate=1.0,
                last_update=time.time()
            )
        
        logger.info(f"Registered cube {cube_id} with capacity {max_capacity}")
    
    def select_cube(self, task_requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select the best cube for a task"""
        with self.metrics_lock:
            if not self.cube_metrics:
                return None
            
            if self.balancing_strategy == 'least_loaded':
                return self._select_least_loaded()
            elif self.balancing_strategy == 'round_robin':
                return self._select_round_robin()
            elif self.balancing_strategy == 'performance_based':
                return self._select_performance_based()
            else:
                return self._select_least_loaded()
    
    def _select_least_loaded(self) -> str:
        """Select cube with lowest current load"""
        available_cubes = [
            (cube_id, metrics) for cube_id, metrics in self.cube_metrics.items()
            if metrics.current_load < metrics.max_capacity
        ]
        
        if not available_cubes:
            # All cubes at capacity, select least loaded
            cube_id = min(self.cube_metrics.keys(), 
                         key=lambda x: self.cube_metrics[x].current_load)
            return cube_id
        
        # Select cube with lowest load ratio
        cube_id = min(available_cubes, 
                     key=lambda x: x[1].current_load / x[1].max_capacity)[0]
        return cube_id
    
    def _select_round_robin(self) -> str:
        """Select cube using round-robin strategy"""
        cube_ids = list(self.cube_metrics.keys())
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected_cube = cube_ids[self._round_robin_index % len(cube_ids)]
        self._round_robin_index += 1
        
        return selected_cube
    
    def _select_performance_based(self) -> str:
        """Select cube based on performance metrics"""
        # Calculate performance score for each cube
        scores = {}
        
        for cube_id, metrics in self.cube_metrics.items():
            load_factor = 1.0 - (metrics.current_load / metrics.max_capacity)
            speed_factor = 1.0 / max(0.001, metrics.processing_time_avg)  # Faster is better
            success_factor = metrics.success_rate
            
            # Combined score
            score = load_factor * 0.4 + speed_factor * 0.3 + success_factor * 0.3
            scores[cube_id] = score
        
        # Select cube with highest score
        best_cube = max(scores.keys(), key=lambda x: scores[x])
        return best_cube
    
    def update_cube_metrics(self, cube_id: str, processing_time: float, 
                           success: bool, queue_length: int = 0):
        """Update metrics for a cube after task completion"""
        with self.metrics_lock:
            if cube_id not in self.cube_metrics:
                return
            
            metrics = self.cube_metrics[cube_id]
            
            # Update processing time (exponential moving average)
            alpha = 0.1
            if metrics.processing_time_avg == 0:
                metrics.processing_time_avg = processing_time
            else:
                metrics.processing_time_avg = (
                    alpha * processing_time + (1 - alpha) * metrics.processing_time_avg
                )
            
            # Update success rate
            if success:
                metrics.success_rate = min(1.0, metrics.success_rate + 0.01)
            else:
                metrics.success_rate = max(0.0, metrics.success_rate - 0.05)
            
            # Update other metrics
            metrics.queue_length = queue_length
            metrics.last_update = time.time()
            
            # Store in history
            self.load_history[cube_id].append({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'success': success,
                'load': metrics.current_load
            })
            
            # Keep only recent history
            if len(self.load_history[cube_id]) > 1000:
                self.load_history[cube_id].popleft()
    
    def increment_load(self, cube_id: str):
        """Increment load for a cube"""
        with self.metrics_lock:
            if cube_id in self.cube_metrics:
                self.cube_metrics[cube_id].current_load += 1
    
    def decrement_load(self, cube_id: str):
        """Decrement load for a cube"""
        with self.metrics_lock:
            if cube_id in self.cube_metrics:
                self.cube_metrics[cube_id].current_load = max(
                    0, self.cube_metrics[cube_id].current_load - 1
                )
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        with self.metrics_lock:
            stats = {
                'total_cubes': len(self.cube_metrics),
                'balancing_strategy': self.balancing_strategy,
                'cube_details': {}
            }
            
            total_capacity = 0
            total_load = 0
            
            for cube_id, metrics in self.cube_metrics.items():
                total_capacity += metrics.max_capacity
                total_load += metrics.current_load
                
                stats['cube_details'][cube_id] = {
                    'current_load': metrics.current_load,
                    'max_capacity': metrics.max_capacity,
                    'load_ratio': metrics.current_load / metrics.max_capacity,
                    'avg_processing_time': metrics.processing_time_avg,
                    'success_rate': metrics.success_rate,
                    'queue_length': metrics.queue_length
                }
            
            stats['overall_load_ratio'] = total_load / max(1, total_capacity)
            stats['total_capacity'] = total_capacity
            stats['total_load'] = total_load
            
            return stats


class OptimizedSpatialEngine:
    """
    Optimized spatial engine that integrates all performance improvements.
    """
    
    def __init__(self, config: Optional[SpatialIndexConfig] = None):
        self.config = config or SpatialIndexConfig()
        self.spatial_index = SpatialIndex(self.config)
        self.distance_calculator = OptimizedDistanceCalculator(self.config.cache_size)
        self.load_balancer = CubeLoadBalancer()
        
        self.documents = {}
        self.coordinate_points = []
        self.needs_rebuild = False
        self.last_rebuild = 0
        
        logger.info("OptimizedSpatialEngine initialized")
    
    def add_documents(self, documents: Dict[str, Dict[str, Any]]):
        """Add documents with automatic index management"""
        new_points = []
        new_doc_ids = []
        
        for doc_id, doc_data in documents.items():
            if doc_id not in self.documents:
                coords = doc_data.get('coordinates', {})
                point = [coords.get('domain', 0.5), coords.get('complexity', 0.5), 
                        coords.get('task_type', 0.5)]
                
                new_points.append(point)
                new_doc_ids.append(doc_id)
                self.documents[doc_id] = doc_data
        
        if new_points:
            self.coordinate_points.extend(new_points)
            
            # Check if rebuild is needed
            if (len(new_points) > self.config.rebuild_threshold or 
                time.time() - self.last_rebuild > 3600):  # Rebuild every hour
                self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild spatial index"""
        if not self.coordinate_points:
            return
        
        points_array = np.array(self.coordinate_points)
        doc_ids = list(self.documents.keys())
        
        self.spatial_index.build_index(points_array, doc_ids)
        self.last_rebuild = time.time()
        self.needs_rebuild = False
        
        logger.info(f"Rebuilt spatial index with {len(points_array)} points")
    
    def optimized_search(self, query_coords: Dict[str, float], 
                        k: int = 10, use_load_balancing: bool = True) -> List[Dict[str, Any]]:
        """Optimized search with spatial indexing and load balancing"""
        
        # Convert query to point
        query_point = np.array([
            query_coords.get('domain', 0.5),
            query_coords.get('complexity', 0.5),
            query_coords.get('task_type', 0.5)
        ])
        
        # Select processing cube if load balancing is enabled
        selected_cube = None
        if use_load_balancing:
            selected_cube = self.load_balancer.select_cube()
            if selected_cube:
                self.load_balancer.increment_load(selected_cube)
        
        try:
            start_time = time.time()
            
            # Use spatial index for nearest neighbor search
            if self.spatial_index.index is not None:
                nearest_results = self.spatial_index.query_nearest(query_point, k * 2)
            else:
                # Fallback to linear search
                nearest_results = self._linear_search_fallback(query_point, k * 2)
            
            # Convert to detailed results
            detailed_results = []
            for doc_id, distance in nearest_results[:k]:
                if doc_id in self.documents:
                    doc_data = self.documents[doc_id]
                    
                    # Calculate similarity from distance
                    similarity = max(0.0, 1.0 - distance / 2.0)  # Normalize distance
                    
                    result = {
                        'document_id': doc_id,
                        'content': doc_data.get('content', ''),
                        'coordinates': doc_data.get('coordinates', {}),
                        'distance': distance,
                        'similarity_score': similarity,
                        'processing_cube': selected_cube,
                        'search_method': 'spatial_index'
                    }
                    
                    detailed_results.append(result)
            
            processing_time = time.time() - start_time
            
            # Update load balancer metrics
            if selected_cube:
                self.load_balancer.update_cube_metrics(
                    selected_cube, processing_time, True
                )
                self.load_balancer.decrement_load(selected_cube)
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"Optimized search failed: {e}")
            
            # Update load balancer with failure
            if selected_cube:
                self.load_balancer.update_cube_metrics(
                    selected_cube, 0.0, False
                )
                self.load_balancer.decrement_load(selected_cube)
            
            return []
    
    def _linear_search_fallback(self, query_point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Fallback linear search when spatial index is not available"""
        distances = []
        
        for doc_id, doc_data in self.documents.items():
            coords = doc_data.get('coordinates', {})
            doc_point = np.array([
                coords.get('domain', 0.5),
                coords.get('complexity', 0.5),
                coords.get('task_type', 0.5)
            ])
            
            distance = np.linalg.norm(query_point - doc_point)
            distances.append((doc_id, distance))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'spatial_index': self.spatial_index.get_performance_stats(),
            'distance_calculator': self.distance_calculator.get_cache_stats(),
            'load_balancer': self.load_balancer.get_load_statistics(),
            'total_documents': len(self.documents),
            'index_needs_rebuild': self.needs_rebuild,
            'last_rebuild': self.last_rebuild
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the optimized spatial engine
    config = SpatialIndexConfig(
        index_type='auto',
        cache_size=1000,
        use_approximate=True
    )
    
    engine = OptimizedSpatialEngine(config)
    
    print("Optimized Spatial Engine Demo")
    print("=" * 40)
    
    # Add sample documents
    sample_docs = {}
    for i in range(100):
        doc_id = f"doc_{i}"
        sample_docs[doc_id] = {
            'content': f'Document {i} content',
            'coordinates': {
                'domain': np.random.random(),
                'complexity': np.random.random(),
                'task_type': np.random.random()
            }
        }
    
    # Register cubes for load balancing
    engine.load_balancer.register_cube('cube_1', 50)
    engine.load_balancer.register_cube('cube_2', 75)
    engine.load_balancer.register_cube('cube_3', 100)
    
    # Add documents
    start_time = time.time()
    engine.add_documents(sample_docs)
    add_time = time.time() - start_time
    
    print(f"Added {len(sample_docs)} documents in {add_time:.3f}s")
    
    # Test search
    query_coords = {'domain': 0.5, 'complexity': 0.7, 'task_type': 0.3}
    
    start_time = time.time()
    results = engine.optimized_search(query_coords, k=5)
    search_time = time.time() - start_time
    
    print(f"\nSearch completed in {search_time:.3f}s")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['document_id']}: similarity={result['similarity_score']:.3f}, "
              f"distance={result['distance']:.3f}, cube={result['processing_cube']}")
    
    # Performance report
    report = engine.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Index type: {report['spatial_index']['index_type']}")
    print(f"  Build time: {report['spatial_index']['build_time']:.3f}s")
    print(f"  Cache hit rate: {report['distance_calculator']['hit_rate']:.3f}")
    print(f"  Load balancer cubes: {report['load_balancer']['total_cubes']}")
    print(f"  Overall load ratio: {report['load_balancer']['overall_load_ratio']:.3f}")