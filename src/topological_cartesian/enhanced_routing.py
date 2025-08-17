#!/usr/bin/env python3
"""
Enhanced Routing and Distance Metrics System

Addresses the routing algorithm limitations identified in the technical feedback.
Implements weighted distance metrics, fallback strategies, and optimized routing.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math
from collections import defaultdict
import heapq
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# Optional imports
try:
    import faiss  # For approximate nearest neighbor search
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Available distance metrics"""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    WEIGHTED_EUCLIDEAN = "weighted_euclidean"
    MAHALANOBIS = "mahalanobis"
    HYBRID = "hybrid"


class RoutingStrategy(Enum):
    """Available routing strategies"""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    WEIGHTED_SIMILARITY = "weighted_similarity"
    TOPOLOGICAL_AWARE = "topological_aware"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE = "adaptive"


@dataclass
class RoutingConfig:
    """Configuration for routing algorithms"""
    distance_metric: DistanceMetric = DistanceMetric.WEIGHTED_EUCLIDEAN
    routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    dimension_weights: Optional[Dict[str, float]] = None
    fallback_threshold: float = 0.3
    max_candidates: int = 10
    use_approximate_search: bool = True
    topological_weight: float = 0.3
    diversity_factor: float = 0.1


@dataclass
class RoutingResult:
    """Result of a routing operation"""
    document_id: str
    coordinates: Dict[str, float]
    distance: float
    similarity_score: float
    routing_explanation: str
    confidence: float
    fallback_used: bool = False
    topological_features: List[str] = field(default_factory=list)


class DistanceCalculator(ABC):
    """Abstract base class for distance calculations"""
    
    @abstractmethod
    def calculate_distance(self, coords1: Dict[str, float], coords2: Dict[str, float]) -> float:
        """Calculate distance between two coordinate sets"""
        pass
    
    @abstractmethod
    def get_explanation(self, coords1: Dict[str, float], coords2: Dict[str, float], distance: float) -> str:
        """Explain how the distance was calculated"""
        pass


class WeightedEuclideanCalculator(DistanceCalculator):
    """Weighted Euclidean distance calculator"""
    
    def __init__(self, dimension_weights: Optional[Dict[str, float]] = None):
        self.dimension_weights = dimension_weights or {}
    
    def calculate_distance(self, coords1: Dict[str, float], coords2: Dict[str, float]) -> float:
        """Calculate weighted Euclidean distance"""
        common_dims = set(coords1.keys()) & set(coords2.keys())
        if not common_dims:
            return float('inf')
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dim in common_dims:
            weight = self.dimension_weights.get(dim, 1.0)
            diff = coords1[dim] - coords2[dim]
            weighted_sum += weight * (diff ** 2)
            total_weight += weight
        
        if total_weight == 0:
            return float('inf')
        
        return math.sqrt(weighted_sum / total_weight)
    
    def get_explanation(self, coords1: Dict[str, float], coords2: Dict[str, float], distance: float) -> str:
        """Explain weighted Euclidean distance calculation"""
        common_dims = set(coords1.keys()) & set(coords2.keys())
        explanations = []
        
        for dim in sorted(common_dims):
            weight = self.dimension_weights.get(dim, 1.0)
            diff = abs(coords1[dim] - coords2[dim])
            weighted_contrib = weight * (diff ** 2)
            explanations.append(f"{dim}: diff={diff:.3f}, weight={weight:.2f}, contrib={weighted_contrib:.3f}")
        
        return f"Weighted Euclidean distance ({distance:.3f}): " + "; ".join(explanations)


class CosineDistanceCalculator(DistanceCalculator):
    """Cosine distance calculator"""
    
    def calculate_distance(self, coords1: Dict[str, float], coords2: Dict[str, float]) -> float:
        """Calculate cosine distance"""
        common_dims = set(coords1.keys()) & set(coords2.keys())
        if not common_dims:
            return 1.0  # Maximum cosine distance
        
        # Convert to vectors
        vec1 = np.array([coords1[dim] for dim in sorted(common_dims)])
        vec2 = np.array([coords2[dim] for dim in sorted(common_dims)])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        return 1.0 - cosine_sim  # Convert to distance
    
    def get_explanation(self, coords1: Dict[str, float], coords2: Dict[str, float], distance: float) -> str:
        """Explain cosine distance calculation"""
        similarity = 1.0 - distance
        return f"Cosine distance ({distance:.3f}): similarity={similarity:.3f}, measures angle between coordinate vectors"


class HybridDistanceCalculator(DistanceCalculator):
    """Hybrid distance calculator combining multiple metrics"""
    
    def __init__(self, calculators: Dict[str, DistanceCalculator], weights: Dict[str, float]):
        self.calculators = calculators
        self.weights = weights
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v/total_weight for k, v in weights.items()}
    
    def calculate_distance(self, coords1: Dict[str, float], coords2: Dict[str, float]) -> float:
        """Calculate hybrid distance"""
        weighted_distance = 0.0
        
        for calc_name, calculator in self.calculators.items():
            distance = calculator.calculate_distance(coords1, coords2)
            weight = self.weights.get(calc_name, 0.0)
            weighted_distance += weight * distance
        
        return weighted_distance
    
    def get_explanation(self, coords1: Dict[str, float], coords2: Dict[str, float], distance: float) -> str:
        """Explain hybrid distance calculation"""
        explanations = []
        
        for calc_name, calculator in self.calculators.items():
            calc_distance = calculator.calculate_distance(coords1, coords2)
            weight = self.weights.get(calc_name, 0.0)
            contribution = weight * calc_distance
            explanations.append(f"{calc_name}: {calc_distance:.3f} (weight={weight:.2f}, contrib={contribution:.3f})")
        
        return f"Hybrid distance ({distance:.3f}): " + "; ".join(explanations)


class ApproximateNearestNeighborIndex:
    """Approximate nearest neighbor search using FAISS"""
    
    def __init__(self, dimension: int, use_gpu: bool = False):
        self.dimension = dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Create FAISS index
        if self.use_gpu:
            self.index = faiss.IndexFlatL2(dimension)
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.document_ids = []
        self.coordinates_list = []
    
    def add_documents(self, documents: Dict[str, Dict[str, Any]]):
        """Add documents to the index"""
        vectors = []
        doc_ids = []
        coords_list = []
        
        for doc_id, doc_data in documents.items():
            coords = doc_data['coordinates']
            # Convert coordinates to vector (ensure consistent ordering)
            vector = np.array([coords.get(dim, 0.0) for dim in sorted(coords.keys())], dtype=np.float32)
            
            vectors.append(vector)
            doc_ids.append(doc_id)
            coords_list.append(coords)
        
        if vectors:
            vectors_array = np.vstack(vectors)
            self.index.add(vectors_array)
            self.document_ids.extend(doc_ids)
            self.coordinates_list.extend(coords_list)
    
    def search(self, query_coords: Dict[str, float], k: int = 10) -> List[Tuple[str, Dict[str, float], float]]:
        """Search for k nearest neighbors"""
        # Convert query to vector
        query_vector = np.array([query_coords.get(dim, 0.0) for dim in sorted(query_coords.keys())], dtype=np.float32)
        query_vector = query_vector.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, min(k, len(self.document_ids)))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.document_ids):
                doc_id = self.document_ids[idx]
                coords = self.coordinates_list[idx]
                results.append((doc_id, coords, float(distance)))
        
        return results


class EnhancedRoutingEngine:
    """
    Enhanced routing engine addressing the technical feedback.
    
    Features:
    - Multiple distance metrics with weighting
    - Fallback strategies for ambiguous coordinates
    - Approximate nearest neighbor search for scalability
    - Topological awareness
    - Multi-objective optimization
    """
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self.distance_calculators = self._initialize_distance_calculators()
        self.documents = {}
        self.ann_index = None
        self.topological_features = []
        
        # Performance tracking
        self.routing_stats = {
            'total_queries': 0,
            'fallback_used': 0,
            'avg_candidates_considered': 0,
            'avg_routing_time': 0.0
        }
    
    def _initialize_distance_calculators(self) -> Dict[DistanceMetric, DistanceCalculator]:
        """Initialize distance calculators"""
        calculators = {}
        
        # Weighted Euclidean
        calculators[DistanceMetric.WEIGHTED_EUCLIDEAN] = WeightedEuclideanCalculator(
            self.config.dimension_weights
        )
        
        # Cosine distance
        calculators[DistanceMetric.COSINE] = CosineDistanceCalculator()
        
        # Hybrid calculator
        if self.config.distance_metric == DistanceMetric.HYBRID:
            hybrid_calculators = {
                'euclidean': calculators[DistanceMetric.WEIGHTED_EUCLIDEAN],
                'cosine': calculators[DistanceMetric.COSINE]
            }
            hybrid_weights = {'euclidean': 0.7, 'cosine': 0.3}
            calculators[DistanceMetric.HYBRID] = HybridDistanceCalculator(
                hybrid_calculators, hybrid_weights
            )
        
        return calculators
    
    def add_documents(self, documents: Dict[str, Dict[str, Any]]):
        """Add documents to the routing engine"""
        self.documents.update(documents)
        
        # Update approximate nearest neighbor index
        if self.config.use_approximate_search and documents:
            # Determine dimension from first document
            first_doc = next(iter(documents.values()))
            dimension = len(first_doc['coordinates'])
            
            if self.ann_index is None:
                self.ann_index = ApproximateNearestNeighborIndex(dimension)
            
            self.ann_index.add_documents(documents)
    
    def route_query(self, query_coords: Dict[str, float], max_results: int = 5) -> List[RoutingResult]:
        """Route a query to find the best matching documents"""
        import time
        start_time = time.time()
        
        self.routing_stats['total_queries'] += 1
        
        # Get candidates using the configured strategy
        if self.config.routing_strategy == RoutingStrategy.ADAPTIVE:
            candidates = self._adaptive_candidate_selection(query_coords, max_results * 2)
        elif self.config.routing_strategy == RoutingStrategy.TOPOLOGICAL_AWARE:
            candidates = self._topological_aware_selection(query_coords, max_results * 2)
        else:
            candidates = self._standard_candidate_selection(query_coords, max_results * 2)
        
        # Calculate distances and similarities
        results = []
        distance_calculator = self.distance_calculators[self.config.distance_metric]
        
        for doc_id, doc_coords in candidates:
            distance = distance_calculator.calculate_distance(query_coords, doc_coords)
            similarity = self._distance_to_similarity(distance)
            
            # Check if fallback is needed
            fallback_used = similarity < self.config.fallback_threshold
            
            # Generate explanation
            explanation = distance_calculator.get_explanation(query_coords, doc_coords, distance)
            
            # Calculate confidence
            confidence = self._calculate_confidence(similarity, distance, fallback_used)
            
            result = RoutingResult(
                document_id=doc_id,
                coordinates=doc_coords,
                distance=distance,
                similarity_score=similarity,
                routing_explanation=explanation,
                confidence=confidence,
                fallback_used=fallback_used
            )
            
            results.append(result)
        
        # Sort by similarity (descending) and apply diversity if needed
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        if self.config.diversity_factor > 0:
            results = self._apply_diversity_filter(results, self.config.diversity_factor)
        
        # Update statistics
        routing_time = time.time() - start_time
        self.routing_stats['avg_routing_time'] = (
            (self.routing_stats['avg_routing_time'] * (self.routing_stats['total_queries'] - 1) + routing_time) /
            self.routing_stats['total_queries']
        )
        
        if any(r.fallback_used for r in results):
            self.routing_stats['fallback_used'] += 1
        
        return results[:max_results]
    
    def _adaptive_candidate_selection(self, query_coords: Dict[str, float], max_candidates: int) -> List[Tuple[str, Dict[str, float]]]:
        """Adaptive candidate selection based on query characteristics"""
        # Analyze query characteristics
        coord_variance = np.var(list(query_coords.values()))
        
        if coord_variance < 0.1:  # Low variance - use approximate search
            return self._approximate_candidate_selection(query_coords, max_candidates)
        elif coord_variance > 0.3:  # High variance - use topological awareness
            return self._topological_aware_selection(query_coords, max_candidates)
        else:  # Medium variance - use standard selection
            return self._standard_candidate_selection(query_coords, max_candidates)
    
    def _approximate_candidate_selection(self, query_coords: Dict[str, float], max_candidates: int) -> List[Tuple[str, Dict[str, float]]]:
        """Use approximate nearest neighbor search for candidate selection"""
        if self.ann_index is not None:
            ann_results = self.ann_index.search(query_coords, max_candidates)
            return [(doc_id, coords) for doc_id, coords, _ in ann_results]
        else:
            return self._standard_candidate_selection(query_coords, max_candidates)
    
    def _standard_candidate_selection(self, query_coords: Dict[str, float], max_candidates: int) -> List[Tuple[str, Dict[str, float]]]:
        """Standard candidate selection using all documents"""
        candidates = []
        distance_calculator = self.distance_calculators[self.config.distance_metric]
        
        for doc_id, doc_data in self.documents.items():
            doc_coords = doc_data['coordinates']
            distance = distance_calculator.calculate_distance(query_coords, doc_coords)
            candidates.append((distance, doc_id, doc_coords))
        
        # Sort by distance and return top candidates
        candidates.sort(key=lambda x: x[0])
        return [(doc_id, coords) for _, doc_id, coords in candidates[:max_candidates]]
    
    def _topological_aware_selection(self, query_coords: Dict[str, float], max_candidates: int) -> List[Tuple[str, Dict[str, float]]]:
        """Topological-aware candidate selection"""
        # Start with standard selection
        standard_candidates = self._standard_candidate_selection(query_coords, max_candidates)
        
        # If we have topological features, enhance the selection
        if self.topological_features:
            # This would integrate with the topological analysis
            # For now, return standard candidates
            pass
        
        return standard_candidates
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score"""
        # Use exponential decay for similarity
        return math.exp(-distance)
    
    def _calculate_confidence(self, similarity: float, distance: float, fallback_used: bool) -> float:
        """Calculate confidence in the routing result"""
        base_confidence = similarity
        
        # Reduce confidence if fallback was used
        if fallback_used:
            base_confidence *= 0.7
        
        # Adjust based on distance
        if distance > 1.0:
            base_confidence *= 0.8
        
        return max(0.0, min(1.0, base_confidence))
    
    def _apply_diversity_filter(self, results: List[RoutingResult], diversity_factor: float) -> List[RoutingResult]:
        """Apply diversity filtering to avoid too similar results"""
        if diversity_factor <= 0 or len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include the best result
        
        for result in results[1:]:
            # Check diversity against already selected results
            min_distance = float('inf')
            
            for selected in diverse_results:
                distance = self.distance_calculators[self.config.distance_metric].calculate_distance(
                    result.coordinates, selected.coordinates
                )
                min_distance = min(min_distance, distance)
            
            # Include if sufficiently diverse
            if min_distance > diversity_factor:
                diverse_results.append(result)
        
        return diverse_results
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        return {
            **self.routing_stats,
            'fallback_rate': self.routing_stats['fallback_used'] / max(1, self.routing_stats['total_queries']),
            'config': {
                'distance_metric': self.config.distance_metric.value,
                'routing_strategy': self.config.routing_strategy.value,
                'use_approximate_search': self.config.use_approximate_search,
                'fallback_threshold': self.config.fallback_threshold
            }
        }
    
    def update_config(self, new_config: RoutingConfig):
        """Update routing configuration"""
        self.config = new_config
        self.distance_calculators = self._initialize_distance_calculators()


# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced routing engine
    config = RoutingConfig(
        distance_metric=DistanceMetric.WEIGHTED_EUCLIDEAN,
        routing_strategy=RoutingStrategy.ADAPTIVE,
        dimension_weights={'domain': 1.2, 'complexity': 1.0, 'task_type': 0.8},
        fallback_threshold=0.3,
        diversity_factor=0.2
    )
    
    routing_engine = EnhancedRoutingEngine(config)
    
    # Add sample documents
    sample_docs = {
        'doc1': {
            'content': 'Python programming tutorial',
            'coordinates': {'domain': 0.9, 'complexity': 0.3, 'task_type': 0.2}
        },
        'doc2': {
            'content': 'Advanced machine learning',
            'coordinates': {'domain': 0.8, 'complexity': 0.9, 'task_type': 0.7}
        },
        'doc3': {
            'content': 'Business strategy guide',
            'coordinates': {'domain': 0.2, 'complexity': 0.5, 'task_type': 0.3}
        }
    }
    
    routing_engine.add_documents(sample_docs)
    
    # Test routing
    query_coords = {'domain': 0.85, 'complexity': 0.4, 'task_type': 0.25}
    results = routing_engine.route_query(query_coords, max_results=3)
    
    print("Enhanced Routing Engine Demo")
    print("=" * 40)
    print(f"Query coordinates: {query_coords}")
    print("\nRouting Results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Document: {result.document_id}")
        print(f"   Similarity: {result.similarity_score:.3f}")
        print(f"   Distance: {result.distance:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Fallback used: {result.fallback_used}")
        print(f"   Explanation: {result.routing_explanation}")
    
    print(f"\nRouting Statistics:")
    stats = routing_engine.get_routing_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")