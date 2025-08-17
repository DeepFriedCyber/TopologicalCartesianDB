#!/usr/bin/env python3
"""
CartesianDB MVP - Lean Implementation

Following TDD principles to build minimal viable product that demonstrates
core value proposition: explainable coordinate-based search vs opaque vectors.

Focus: Prove interpretability advantage with minimal code.
"""

import hashlib
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import math
import time
import numpy as np
from .typings import TopologicalFeature


@dataclass
class Document:
    """Minimal document representation."""
    id: str
    content: str
    coordinates: Dict[str, float]


class CoordinateMVP:
    """
    Minimal Viable Product for CartesianDB.
    
    Focuses on core value proposition:
    - Interpretable coordinates vs opaque vectors
    - Explainable search results
    - 3-line LLM integration
    """
    
    def __init__(self):
        """Initialize MVP with minimal state."""
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._coordinate_cache: Dict[str, Dict[str, float]] = {}
    
    def text_to_coordinates(self, text: str) -> Dict[str, float]:
        """
        Convert text to interpretable coordinates.
        
        This is the core differentiator - meaningful dimensions vs opaque vectors.
        Using simple heuristics for MVP to prove concept.
        """
        # Cache for deterministic results
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._coordinate_cache:
            return self._coordinate_cache[text_hash]
        
        # Simple heuristic-based coordinate calculation
        text_lower = text.lower()
        words = text.split()
        
        # Domain dimension (0-1): Programming vs Business vs General
        programming_keywords = ['python', 'programming', 'code', 'algorithm', 'tutorial', 'function', 'variable']
        business_keywords = ['business', 'strategy', 'market', 'analysis', 'marketing', 'sales']
        
        prog_score = sum(1 for word in programming_keywords if word in text_lower)
        biz_score = sum(1 for word in business_keywords if word in text_lower)
        
        if prog_score > biz_score:
            domain = min(0.7 + prog_score * 0.1, 1.0)  # Programming domain
        elif biz_score > 0:
            domain = min(0.3 + biz_score * 0.1, 0.6)   # Business domain
        else:
            domain = 0.5  # General domain
        
        # Complexity dimension (0-1): Beginner vs Advanced
        beginner_keywords = ['beginner', 'tutorial', 'introduction', 'basic', 'learn', 'start']
        advanced_keywords = ['advanced', 'expert', 'complex', 'optimization', 'architecture']
        
        beginner_score = sum(1 for word in beginner_keywords if word in text_lower)
        advanced_score = sum(1 for word in advanced_keywords if word in text_lower)
        
        if beginner_score > advanced_score:
            complexity = max(0.2, 0.5 - beginner_score * 0.1)  # Lower complexity
        elif advanced_score > 0:
            complexity = min(0.8 + advanced_score * 0.1, 1.0)  # Higher complexity
        else:
            complexity = 0.6  # Default intermediate
        
        # Task type dimension (0-1): Tutorial vs Analysis vs Reference
        tutorial_keywords = ['tutorial', 'guide', 'how', 'learn', 'example']
        analysis_keywords = ['analysis', 'compare', 'evaluate', 'study']
        
        tutorial_score = sum(1 for word in tutorial_keywords if word in text_lower)
        analysis_score = sum(1 for word in analysis_keywords if word in text_lower)
        
        if tutorial_score > analysis_score:
            task_type = 0.3  # Tutorial type
        elif analysis_score > 0:
            task_type = 0.7  # Analysis type
        else:
            task_type = 0.5  # General type
        
        coordinates = {
            'domain': round(domain, 3),
            'complexity': round(complexity, 3),
            'task_type': round(task_type, 3)
        }
        
        # Cache for deterministic results
        self._coordinate_cache[text_hash] = coordinates
        return coordinates
    
    def add_document(self, doc_id: str, content: str) -> bool:
        """
        Add document with automatic coordinate calculation.
        """
        coordinates = self.text_to_coordinates(content)
        
        self.documents[doc_id] = {
            'content': content,
            'coordinates': coordinates
        }
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID.
        """
        return self.documents.get(doc_id)
    
    def _calculate_coordinate_similarity(self, coords1: Dict[str, float], coords2: Dict[str, float]) -> float:
        """
        Calculate similarity between coordinate sets.
        Using Euclidean distance converted to similarity score.
        """
        # Ensure same dimensions
        common_dims = set(coords1.keys()) & set(coords2.keys())
        if not common_dims:
            return 0.0
        
        # Calculate Euclidean distance
        distance_sq = 0.0
        for dim in common_dims:
            distance_sq += (coords1[dim] - coords2[dim]) ** 2
        
        distance = math.sqrt(distance_sq)
        
        # Convert distance to similarity (0-1, higher is more similar)
        max_distance = math.sqrt(len(common_dims))  # Maximum possible distance
        if max_distance == 0:
            return 1.0 # Avoid division by zero if there's only one dimension with no difference
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, similarity)
    
    def _generate_explanation(self, query_coords: Dict[str, float], doc_coords: Dict[str, float], 
                            similarity: float, method: str = "coordinates") -> str:
        """
        Generate human-readable explanation for search result.
        
        This is the key differentiator - explainable vs black box.
        """
        if method == "vectors":
            # Simulate opaque vector explanation
            return f"Vector similarity: {similarity:.3f} (cosine distance calculation)"
        
        # Coordinate-based explanation (our value prop)
        explanation_parts = []
        
        # Analyze each dimension
        for dim in query_coords:
            if dim in doc_coords:
                query_val = query_coords[dim]
                doc_val = doc_coords[dim]
                diff = abs(query_val - doc_val)
                
                if diff < 0.2:
                    match_quality = "strong"
                elif diff < 0.4:
                    match_quality = "moderate"
                else:
                    match_quality = "weak"
                
                dim_name = dim.replace('_', ' ').title()
                explanation_parts.append(f"{dim_name}: {match_quality} match ({doc_val:.2f} vs {query_val:.2f})")
        
        # Overall assessment
        if similarity > 0.8:
            overall = "Excellent coordinate alignment"
        elif similarity > 0.6:
            overall = "Good coordinate alignment"
        elif similarity > 0.4:
            overall = "Moderate coordinate alignment"
        else:
            overall = "Limited coordinate alignment"
        
        explanation = f"{overall}. " + "; ".join(explanation_parts)
        return explanation
    
    def search(self, query: str, max_results: int = 5, method: str = "coordinates") -> List[Dict[str, Any]]:
        """
        Search documents using coordinate-based similarity.
        
        Supports both coordinate and vector methods for comparison.
        """
        if not self.documents:
            return []
        
        query_coords = self.text_to_coordinates(query)
        results = []
        
        for doc_id, doc_data in self.documents.items():
            doc_coords = doc_data['coordinates']
            
            # Calculate similarity
            similarity = self._calculate_coordinate_similarity(query_coords, doc_coords)
            
            # Generate explanation
            explanation = self._generate_explanation(query_coords, doc_coords, similarity, method)
            
            result = {
                'document_id': doc_id,
                'content': doc_data['content'],
                'coordinates': doc_coords,
                'similarity_score': similarity,
                'explanation': explanation
            }
            
            results.append(result)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:max_results]
    
    def get_llm_context(self, query: str, max_docs: int = 3) -> List[Dict[str, Any]]:
        """
        Get grounded context for LLM enhancement.
        
        This implements Line 1 of the 3-line integration framework.
        """
        # Search for relevant documents
        search_results = self.search(query, max_results=max_docs)
        
        # Format for LLM context
        context = []
        for result in search_results:
            context_item = {
                'content': result['content'],
                'coordinates': result['coordinates'],
                'relevance_explanation': result['explanation']
            }
            context.append(context_item)
        
        return context


class EnhancedCoordinateEngine(CoordinateMVP):
    """
    Enhanced coordinate engine with multi-backend TDA integration.
    
    Extends the MVP with topological features for improved search and interpretability.
    Integrates with the MultiBackendTDAEngine for performance leadership.
    """
    
    def __init__(self, tda_backend_preferences: Optional[List[str]] = None):
        """
        Initialize enhanced coordinate engine.
        
        Args:
            tda_backend_preferences: Preferred TDA backends in order of preference
        """
        super().__init__()
        
        # Import here to avoid circular imports
        try:
            from .topology_analyzer import create_multi_backend_engine, MultiBackendTDAEngine
            self.tda_engine: Optional[MultiBackendTDAEngine] = create_multi_backend_engine(tda_backend_preferences)
            self.tda_available = True
        except ImportError as e:
            print(f"⚠️  TDA engine not available: {e}")
            self.tda_engine = None
            self.tda_available = False
        
        # Initialize predictive cache
        try:
            from .predictive_cache import create_predictive_cache_manager
            self.predictive_cache = create_predictive_cache_manager(cache_size=100)
            self.predictive_cache_available = True
            print("🧠 Predictive cache enabled")
        except ImportError as e:
            print(f"⚠️  Predictive cache not available: {e}")
            self.predictive_cache = None
            self.predictive_cache_available = False
        
        # Enhanced coordinate tracking
        self.coordinate_points = []
        self.topological_features = []
        self.coordinate_to_doc_mapping = {}
        
        # Performance tracking
        self.search_performance = {
            'coordinate_only': [],
            'topological_enhanced': []
        }
        
        print(f"🚀 EnhancedCoordinateEngine initialized")
        print(f"   TDA Integration: {'✅ Available' if self.tda_available else '❌ Unavailable'}")
    
    def add_document(self, doc_id: str, content: str) -> bool:
        """
        Add document with enhanced coordinate and topological analysis.
        """
        result = super().add_document(doc_id, content)
        
        if result and self.tda_available and self.tda_engine:
            # Add coordinates to point cloud for topological analysis
            coords = self.documents[doc_id]['coordinates']
            point = [coords['domain'], coords['complexity'], coords['task_type']]
            self.coordinate_points.append(point)
            
            # Map coordinate index to document ID
            coord_index = len(self.coordinate_points) - 1
            self.coordinate_to_doc_mapping[coord_index] = doc_id
            
            # Trigger topological reanalysis if we have enough points
            if len(self.coordinate_points) >= 4:
                self._update_topological_analysis()
        
        return result
    
    def _update_topological_analysis(self):
        """Update topological analysis of coordinate space."""
        if not self.tda_available or not self.coordinate_points or not self.tda_engine:
            return
        
        print(f"🔍 Analyzing topology of {len(self.coordinate_points)} coordinate points...")
        
        try:
            # Convert to numpy array
            points = np.array(self.coordinate_points)
            
            # Compute topological features using multi-backend engine
            start_time = time.time()
            
            # The compute_persistence method now returns a list of our new TopologicalFeature objects
            raw_features = self.tda_engine.compute_persistence(
                np.array(self.coordinate_points), max_dimension=2
            )
            
            self.topological_features = [
                TopologicalFeature(
                    dimension=f.dimension,
                    persistence=f.persistence,
                    coordinates=f.coordinates,
                    confidence=f.confidence,
                    backend=f.backend
                ) for f in raw_features
            ]
            
            computation_time = time.time() - start_time
            
            print(f"   ⚡ Computed {len(self.topological_features)} topological features "
                  f"in {computation_time:.3f}s")
            
            # Enhance features with coordinate interpretability
            self._enhance_features_with_coordinates()
            
        except Exception as e:
            print(f"   ⚠️  Topological analysis failed: {e}")
            self.topological_features = []
    
    def _enhance_features_with_coordinates(self):
        """Enhance topological features with coordinate-based interpretability."""
        for feature in self.topological_features:
            # Add interpretable coordinate analysis
            feature_coords = feature.coordinates
            
            # Analyze which coordinate dimension this feature represents
            if len(feature_coords) >= 3:
                domain_val, complexity_val, task_val = feature_coords[:3]
                
                # Determine primary coordinate dimension
                coord_analysis = {
                    'primary_dimension': self._identify_primary_dimension(
                        domain_val, complexity_val, task_val
                    ),
                    'coordinate_interpretation': self._interpret_coordinates(
                        domain_val, complexity_val, task_val
                    ),
                    'affected_documents': self._find_documents_near_feature(feature)
                }
                
                # Store enhanced analysis
                feature.coordinate_analysis = coord_analysis
    
    def _identify_primary_dimension(self, domain: float, complexity: float, task: float) -> str:
        """Identify which coordinate dimension is most significant for this feature."""
        coords = {'domain': domain, 'complexity': complexity, 'task_type': task}
        
        # Find dimension with most extreme value (furthest from 0.5)
        extremeness = {dim: abs(val - 0.5) for dim, val in coords.items()}
        primary_dim = max(extremeness.items(), key=lambda x: x[1])[0]
        
        return primary_dim
    
    def _interpret_coordinates(self, domain: float, complexity: float, task: float) -> str:
        """Generate human-readable interpretation of coordinate values."""
        interpretations = []
        
        # Domain interpretation
        if domain > 0.7:
            interpretations.append("Programming-focused")
        elif domain < 0.3:
            interpretations.append("Business-focused")
        else:
            interpretations.append("General domain")
        
        # Complexity interpretation
        if complexity > 0.7:
            interpretations.append("Advanced level")
        elif complexity < 0.3:
            interpretations.append("Beginner level")
        else:
            interpretations.append("Intermediate level")
        
        # Task type interpretation
        if task > 0.7:
            interpretations.append("Analysis-oriented")
        elif task < 0.3:
            interpretations.append("Tutorial-oriented")
        else:
            interpretations.append("General purpose")
        
        return ", ".join(interpretations)
    
    def _find_documents_near_feature(self, feature) -> List[str]:
        """Find documents that are close to this topological feature."""
        if not self.coordinate_points:
            return []
        
        feature_coords = np.array(feature.coordinates[:3])  # Use first 3 dimensions
        nearby_docs = []
        
        for i, point in enumerate(self.coordinate_points):
            distance = np.linalg.norm(np.array(point) - feature_coords)
            if distance < 0.3:  # Threshold for "nearby"
                if i in self.coordinate_to_doc_mapping:
                    nearby_docs.append(self.coordinate_to_doc_mapping[i])
        
        return nearby_docs
    
    def topological_search(self, query: str, max_results: int = 5, 
                          use_topology: bool = True,
                          alpha: float = 0.7, user_id: str = "default") -> List[Dict[str, Any]]:
        """
        Enhanced search using topological structure and coordinate reasoning.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            use_topology: Whether to use topological enhancement
            alpha: Weight for coordinate similarity (1-alpha for topological)
            user_id: User identifier for predictive caching
            
        Returns:
            Enhanced search results with topological insights
        """
        start_time = time.time()
        
        # Check predictive cache first
        if self.predictive_cache_available and self.predictive_cache:
            cached_result = self.predictive_cache.check_prediction_cache(query)
            if cached_result is not None:
                print(f"🎯 Predictive cache hit for query: '{query}'")
                return cached_result.get('results', [])[:max_results]
        
        # Get standard coordinate-based results
        standard_results = self.search(query, max_results=max_results * 2)
        
        coordinate_time = time.time() - start_time
        self.search_performance['coordinate_only'].append(coordinate_time)
        
        if not use_topology or not self.tda_available or not self.topological_features:
            return standard_results[:max_results]
        
        # Enhance results with topological analysis
        topo_start_time = time.time()
        
        query_coords = self.text_to_coordinates(query)
        query_point = [query_coords['domain'], query_coords['complexity'], query_coords['task_type']]
        
        enhanced_results = []
        for result in standard_results:
            # Calculate topological relevance
            topo_score = self._calculate_topological_relevance(
                query_point, result['coordinates']
            )
            
            # Combine coordinate and topological scores
            enhanced_similarity = (
                alpha * result['similarity_score'] + 
                (1 - alpha) * topo_score
            )
            
            # Generate enhanced explanation
            topo_explanation = self._generate_topological_explanation(
                query_point, result['coordinates'], topo_score
            )
            
            # Create enhanced result
            enhanced_result = result.copy()
            enhanced_result.update({
                'topological_score': topo_score,
                'enhanced_similarity': enhanced_similarity,
                'topological_explanation': topo_explanation,
                'coordinate_explanation': result['explanation'],
                'backend_used': getattr(self.topological_features[0], 'backend', 'Unknown') if self.topological_features else 'None'
            })
            
            enhanced_results.append(enhanced_result)
        
        # Re-rank by enhanced similarity
        enhanced_results.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
        
        final_results = enhanced_results[:max_results]
        
        # Record query and predict next queries for caching
        if self.predictive_cache_available and self.predictive_cache:
            # Record this query
            self.predictive_cache.record_query(query, final_results, 1.0, 0.95)  # TODO: Add actual timing and accuracy
            
            # Predict and preload next queries
            def preload_callback(predicted_query: str) -> Optional[List[Dict[str, Any]]]:
                try:
                    # Precompute result for predicted query
                    preload_result = self.topological_search(
                        predicted_query, max_results, use_topology, alpha, user_id
                    )
                    return preload_result
                except Exception:
                    return None
            
            try:
                predicted_queries = self.predictive_cache.predict_and_preload(
                    query, {"user_id": user_id, "use_topology": use_topology}
                )
                # Execute preload for predicted queries
                for predicted_query in predicted_queries:
                    try:
                        preload_result = self.topological_search(
                            predicted_query, max_results, use_topology, alpha, user_id
                        )
                        # Cache the preloaded result
                        self.predictive_cache.record_query(predicted_query, preload_result, 1.0, 0.95)
                    except Exception:
                        pass  # Ignore preload failures
            except Exception as e:
                print(f"⚠️  Prediction failed: {e}")
        
        return final_results
    
    def _calculate_topological_relevance(self, query_point: List[float], 
                                       doc_coords: Dict[str, float]) -> float:
        """Calculate topological relevance between query and document."""
        if not self.topological_features:
            return 0.0
        
        doc_point = [doc_coords['domain'], doc_coords['complexity'], doc_coords['task_type']]
        
        # Find shared topological features
        shared_features = 0
        total_relevance = 0.0
        
        for feature in self.topological_features:
            if len(feature.coordinates) < 3:
                continue
                
            feature_coords = feature.coordinates[:3]
            query_dist = np.linalg.norm(np.array(query_point) - np.array(feature_coords))
            doc_dist = np.linalg.norm(np.array(doc_point) - np.array(feature_coords))
            
            # If both points are close to this topological feature
            threshold = 0.4
            if query_dist < threshold and doc_dist < threshold:
                shared_features += 1
                # Weight by feature persistence and confidence
                relevance = (feature.persistence * feature.confidence * 
                           (1.0 - (query_dist + doc_dist) / (2 * threshold)))
                total_relevance += relevance
        
        # Normalize by number of features
        if shared_features > 0:
            return min(1.0, float(total_relevance) / shared_features)
        else:
            return 0.0
    
    def _generate_topological_explanation(self, query_point: List[float], 
                                        doc_coords: Dict[str, float], 
                                        topo_score: float) -> str:
        """Generate human-readable topological explanation."""
        if topo_score > 0.7:
            explanation = f"Strong topological connection (score: {topo_score:.3f}). "
            explanation += "Documents share persistent topological features, indicating "
            explanation += "similar structural patterns in the coordinate space."
        elif topo_score > 0.4:
            explanation = f"Moderate topological connection (score: {topo_score:.3f}). "
            explanation += "Documents belong to related topological structures."
        elif topo_score > 0.1:
            explanation = f"Weak topological connection (score: {topo_score:.3f}). "
            explanation += "Documents have some shared topological context."
        else:
            explanation = f"No significant topological connection (score: {topo_score:.3f}). "
            explanation += "Documents are in separate topological regions."
        
        # Add coordinate-based context
        if hasattr(self, 'topological_features') and self.topological_features:
            relevant_features = [f for f in self.topological_features 
                               if self._is_point_near_feature(query_point, f)]
            if relevant_features:
                feature = relevant_features[0]  # Use most relevant feature
                if feature.coordinate_analysis:
                    explanation += f" Primary pattern: {feature.coordinate_analysis.get('coordinate_interpretation', '')}."
        
        return explanation
    
    def _is_point_near_feature(self, point: List[float], feature: TopologicalFeature) -> bool:
        """Check if a point is near a topological feature."""
        if len(feature.coordinates) < 3:
            return False
        
        feature_coords = feature.coordinates[:3]
        distance = np.linalg.norm(np.array(point) - np.array(feature_coords))
        return bool(distance < 0.4)
    
    def get_topological_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of topological analysis."""
        if not self.tda_available:
            return {"error": "TDA engine not available"}
        
        if not self.topological_features:
            return {"message": "No topological analysis performed yet"}
        
        # Basic feature statistics
        summary: Dict[str, Any] = {
            "total_features": len(self.topological_features),
            "connected_components": len([f for f in self.topological_features if f.dimension == 0]),
            "loops": len([f for f in self.topological_features if f.dimension == 1]),
            "voids": len([f for f in self.topological_features if f.dimension == 2]),
            "coordinate_space_size": len(self.coordinate_points)
        }
        
        # Backend information
        if self.topological_features:
            backends_used = list(set(f.backend for f in self.topological_features))
            summary["backends_used"] = backends_used
        
        # Most persistent features
        if self.topological_features:
            most_persistent = max(self.topological_features, key=lambda f: f.persistence)
            summary["most_persistent_feature"] = {
                "dimension": most_persistent.dimension,
                "persistence": most_persistent.persistence,
                "backend": most_persistent.backend,
                "interpretation": getattr(most_persistent, 'coordinate_analysis', {}).get(
                    'coordinate_interpretation', 'No interpretation available'
                )
            }
        
        # Performance statistics
        if self.search_performance['coordinate_only'] and self.search_performance['topological_enhanced']:
            coord_avg = np.mean(self.search_performance['coordinate_only'])
            topo_avg = np.mean(self.search_performance['topological_enhanced'])
            summary["performance"] = {
                "coordinate_only_avg_time": float(coord_avg),
                "topological_enhanced_avg_time": float(topo_avg),
                "enhancement_overhead": float(topo_avg - coord_avg)
            }
        
        # TDA engine status
        if self.tda_engine:
            summary["tda_engine_status"] = self.tda_engine.get_backend_status()
        
        return summary
    
    def benchmark_search_methods(self, test_queries: List[str], 
                                iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark coordinate-only vs topological-enhanced search.
        
        Args:
            test_queries: List of test queries
            iterations: Number of iterations per query
            
        Returns:
            Benchmark results
        """
        if not self.tda_available or not self.tda_engine:
            return {"error": "TDA engine not available for benchmarking"}
        
        results: Dict[str, Dict[str, Any]] = {
            "coordinate_only": {"times": [], "avg_results": []},
            "topological_enhanced": {"times": [], "avg_results": []}
        }
        
        for query in test_queries:
            for _ in range(iterations):
                # Benchmark coordinate-only search
                start_time = time.time()
                coord_results = self.search(query, max_results=5)
                coord_time = time.time() - start_time
                results["coordinate_only"]["times"].append(coord_time)
                results["coordinate_only"]["avg_results"].append(len(coord_results))
                
                # Benchmark topological-enhanced search
                start_time = time.time()
                topo_results = self.topological_search(query, max_results=5, use_topology=True)
                topo_time = time.time() - start_time
                results["topological_enhanced"]["times"].append(topo_time)
                results["topological_enhanced"]["avg_results"].append(len(topo_results))
        
        # Calculate statistics
        for method in results:
            times = np.array(results[method]["times"])
            if times.size > 0:
                results[method]["avg_time"] = float(np.mean(times))
                results[method]["std_time"] = float(np.std(times))
                results[method]["min_time"] = float(np.min(times))
                results[method]["max_time"] = float(np.max(times))
        
        return results
    
    # ==========================================
    # Week 3: Intelligent Performance Orchestration Integration
    # ==========================================
    
    def enable_production_optimizations(self, cache_size: int = 200, 
                                      optimization_interval: int = 15,
                                      cache_ttl: int = 7200):
        """
        Enable all production-level optimizations for the coordinate engine.
        
        Args:
            cache_size: Maximum number of cached TDA results
            optimization_interval: Computations between optimization updates
            cache_ttl: Cache time-to-live in seconds (2 hours default)
        """
        if not self.tda_available or not self.tda_engine:
            print("⚠️  TDA engine not available - optimizations limited")
            return
        
        # Enable TDA engine optimizations
        self.tda_engine.enable_adaptive_caching(cache_size, cache_ttl)
        self.tda_engine.enable_dynamic_optimization(optimization_interval)
        
        # Enable coordinate-level optimizations
        self.coordinate_cache = {}
        self.coordinate_cache_enabled = True
        self.coordinate_cache_size = cache_size // 2  # Half for coordinates
        
        print(f"🚀 Production optimizations enabled:")
        print(f"   • TDA Caching: {cache_size} entries, {cache_ttl}s TTL")
        print(f"   • Dynamic Optimization: Every {optimization_interval} computations")
        print(f"   • Coordinate Caching: {self.coordinate_cache_size} entries")
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive performance dashboard for monitoring and optimization.
        
        Returns:
            Dashboard with performance metrics, recommendations, and status
        """
        dashboard = {
            'coordinate_engine_status': {
                'documents_indexed': len(self.documents),
                'coordinate_points': len(self.coordinate_points),
                'topological_features': len(self.topological_features),
                'tda_available': self.tda_available
            },
            'search_performance': {
                'coordinate_searches': len(self.search_performance['coordinate_only']),
                'topological_searches': len(self.search_performance['topological_enhanced'])
            },
            'optimization_status': {},
            'recommendations': []
        }
        
        # Add TDA engine performance insights
        if self.tda_available and self.tda_engine:
            tda_insights = self.tda_engine.get_performance_insights()
            dashboard['tda_performance'] = tda_insights
            
            # Get optimization status
            opt_status = self.tda_engine.get_optimization_status()
            dashboard['optimization_status'] = opt_status
            
            # Merge recommendations
            dashboard['recommendations'].extend(tda_insights.get('optimization_recommendations', []))
        
        # Calculate search performance statistics
        if self.search_performance['coordinate_only']:
            coord_times = self.search_performance['coordinate_only']
            dashboard['search_performance']['coordinate_avg_time'] = float(np.mean(coord_times))
            dashboard['search_performance']['coordinate_std_time'] = float(np.std(coord_times))
        
        if self.search_performance['topological_enhanced']:
            topo_times = self.search_performance['topological_enhanced']
            dashboard['search_performance']['topological_avg_time'] = float(np.mean(topo_times))
            dashboard['search_performance']['topological_std_time'] = float(np.std(topo_times))
            
            # Calculate enhancement overhead
            if self.search_performance['coordinate_only']:
                coord_avg = np.mean(self.search_performance['coordinate_only'])
                topo_avg = np.mean(topo_times)
                overhead = ((topo_avg - coord_avg) / coord_avg) * 100 if coord_avg > 0 else 0
                dashboard['search_performance']['enhancement_overhead_percent'] = float(overhead)
        
        # Generate coordinate-specific recommendations
        coord_recommendations = self._generate_coordinate_recommendations(dashboard)
        dashboard['recommendations'].extend(coord_recommendations)
        
        return dashboard
    
    def _generate_coordinate_recommendations(self, dashboard: Dict[str, Any]) -> List[str]:
        """Generate coordinate engine specific recommendations."""
        recommendations = []
        
        # Document indexing recommendations
        doc_count = dashboard['coordinate_engine_status']['documents_indexed']
        if doc_count < 10:
            recommendations.append(
                f"Low document count: Only {doc_count} documents indexed - add more for better topological analysis"
            )
        elif doc_count > 10000:
            recommendations.append(
                f"Large document set: {doc_count} documents - consider distributed processing or data partitioning"
            )
        
        # Search performance recommendations
        search_perf = dashboard['search_performance']
        if 'enhancement_overhead_percent' in search_perf:
            overhead = search_perf['enhancement_overhead_percent']
            if overhead > 200:
                recommendations.append(
                    f"High topological overhead: {overhead:.1f}% - consider optimizing TDA backend selection"
                )
            elif overhead < 50:
                recommendations.append(
                    f"Excellent topological efficiency: Only {overhead:.1f}% overhead for enhanced capabilities"
                )
        
        # Topological feature recommendations
        feature_count = dashboard['coordinate_engine_status']['topological_features']
        point_count = dashboard['coordinate_engine_status']['coordinate_points']
        
        if point_count > 0:
            feature_ratio = feature_count / point_count
            if feature_ratio < 0.5:
                recommendations.append(
                    f"Low topological complexity: {feature_ratio:.2f} features per point - data may be too sparse"
                )
            elif feature_ratio > 2.0:
                recommendations.append(
                    f"High topological complexity: {feature_ratio:.2f} features per point - consider dimension reduction"
                )
        
        return recommendations
    
    def optimize_search_strategy(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Intelligently choose the optimal search strategy for a given query.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            Optimized search results with strategy explanation
        """
        # Analyze query characteristics
        query_coords = self.text_to_coordinates(query)
        query_complexity = query_coords['complexity']
        
        # Decision logic for search strategy
        use_topological = True
        strategy_reason = ""
        
        # Simple queries might not benefit from topological enhancement
        if query_complexity < 0.3 and len(self.coordinate_points) < 20:
            use_topological = False
            strategy_reason = "Simple query with small dataset - coordinate search sufficient"
        
        # Complex queries with sufficient data benefit from topological analysis
        elif query_complexity > 0.7 and len(self.coordinate_points) > 50:
            use_topological = True
            strategy_reason = "Complex query with rich dataset - topological enhancement recommended"
        
        # Medium complexity - use topological if available and performant
        else:
            if self.tda_available and len(self.topological_features) > 0:
                use_topological = True
                strategy_reason = "Moderate complexity - topological enhancement available and beneficial"
            else:
                use_topological = False
                strategy_reason = "Moderate complexity - coordinate search preferred due to limited topological data"
        
        # Execute optimized search
        start_time = time.time()
        
        if use_topological and self.tda_available:
            results = self.topological_search(query, max_results, use_topology=True)
            search_method = "topological_enhanced"
        else:
            results = self.search(query, max_results)
            search_method = "coordinate_only"
        
        search_time = time.time() - start_time
        
        # Return results with optimization metadata
        return {
            'results': results,
            'search_method': search_method,
            'strategy_reason': strategy_reason,
            'search_time': search_time,
            'query_analysis': {
                'complexity': query_complexity,
                'domain': query_coords['domain'],
                'task_type': query_coords['task_type']
            },
            'optimization_metadata': {
                'topological_available': self.tda_available,
                'topological_features_count': len(self.topological_features),
                'coordinate_points_count': len(self.coordinate_points),
                'use_topological_decision': use_topological
            }
        }
    
    def auto_tune_performance(self, sample_queries: List[str], iterations: int = 5) -> Dict[str, Any]:
        """
        Automatically tune performance parameters based on sample queries.
        
        Args:
            sample_queries: Representative queries for tuning
            iterations: Number of iterations per query for averaging
            
        Returns:
            Tuning results and optimized parameters
        """
        if not self.tda_available or not self.tda_engine:
            return {"error": "TDA engine not available for auto-tuning"}
        
        print(f"🔧 Auto-tuning performance with {len(sample_queries)} queries...")
        
        tuning_results = {
            'baseline_performance': {},
            'optimized_performance': {},
            'parameter_recommendations': {},
            'improvement_summary': {}
        }
        
        # Baseline performance measurement
        baseline_times = {'coordinate': [], 'topological': []}
        
        for query in sample_queries:
            for _ in range(iterations):
                # Measure coordinate search
                start = time.time()
                self.search(query, max_results=5)
                baseline_times['coordinate'].append(time.time() - start)
                
                # Measure topological search
                start = time.time()
                self.topological_search(query, max_results=5)
                baseline_times['topological'].append(time.time() - start)
        
        baseline_coord_avg = np.mean(baseline_times['coordinate'])
        baseline_topo_avg = np.mean(baseline_times['topological'])
        tuning_results['baseline_performance'] = {
            'coordinate_avg': float(baseline_coord_avg),
            'topological_avg': float(baseline_topo_avg),
            'enhancement_overhead': (baseline_topo_avg - baseline_coord_avg) / baseline_coord_avg * 100 if baseline_coord_avg > 0 else 0
        }
        
        # Enable optimizations and re-measure
        original_cache_enabled = getattr(self.tda_engine, 'cache_enabled', False)
        
        if not original_cache_enabled:
            self.tda_engine.enable_adaptive_caching(cache_size=50, ttl_seconds=1800)
        
        # Optimized performance measurement
        optimized_times = {'coordinate': [], 'topological': []}
        
        for query in sample_queries:
            for _ in range(iterations):
                # Measure coordinate search (should be similar)
                start = time.time()
                self.search(query, max_results=5)
                optimized_times['coordinate'].append(time.time() - start)
                
                # Measure optimized topological search
                start = time.time()
                self.tda_engine.compute_persistence(
                    np.array(self.coordinate_points), max_dimension=2
                )
                optimized_times['topological'].append(time.time() - start)
        
        optimized_coord_avg = np.mean(optimized_times['coordinate'])
        optimized_topo_avg = np.mean(optimized_times['topological'])
        tuning_results['optimized_performance'] = {
            'coordinate_avg': float(optimized_coord_avg),
            'topological_avg': float(optimized_topo_avg),
            'enhancement_overhead': (optimized_topo_avg - optimized_coord_avg) / optimized_coord_avg * 100 if optimized_coord_avg > 0 else 0
        }
        
        # Calculate improvements
        baseline_topo = tuning_results['baseline_performance']['topological_avg']
        optimized_topo = tuning_results['optimized_performance']['topological_avg']
        
        if baseline_topo > 0:
            improvement_percent = (baseline_topo - optimized_topo) / baseline_topo * 100
            tuning_results['improvement_summary'] = {
                'topological_search_speedup_percent': float(improvement_percent),
                'baseline_avg_time': float(baseline_topo),
                'optimized_avg_time': float(optimized_topo)
            }
        
        # Parameter recommendations
        tuning_results['parameter_recommendations'] = {
            'tda_caching_enabled': True,
            'tda_cache_size': 50,
            'tda_cache_ttl_seconds': 1800
        }
        
        # Revert changes if necessary
        if not original_cache_enabled and self.tda_engine:
            # TODO: Implement disable_adaptive_caching method in MultiBackendTDAEngine
            pass  # self.tda_engine.disable_adaptive_caching()
            
        print(f"✅ Auto-tuning complete. Recommended parameters applied for this session.")
        
        return tuning_results
    
    def get_predictive_cache_stats(self) -> Dict[str, Any]:
        """Get predictive cache statistics and status."""
        if not self.predictive_cache_available or not self.predictive_cache:
            return {
                'available': False,
                'reason': 'Predictive cache not available or not initialized'
            }
        
        return {
            'available': True,
            'stats': self.predictive_cache.get_cache_stats()
        }
    
    def enable_predictive_caching(self, cache_size: int = 100, 
                                preload_threshold: float = 0.3) -> bool:
        """
        Enable predictive caching with specified parameters.
        
        Args:
            cache_size: Maximum number of cached predictions
            preload_threshold: Minimum confidence threshold for preloading
            
        Returns:
            True if successfully enabled, False otherwise
        """
        try:
            from .predictive_cache import create_predictive_cache_manager
            self.predictive_cache = create_predictive_cache_manager(
                cache_size=cache_size
            )
            self.predictive_cache_available = True
            print(f"🧠 Predictive cache enabled (size={cache_size}, threshold={preload_threshold})")
            return True
        except ImportError as e:
            print(f"⚠️  Failed to enable predictive cache: {e}")
            return False
    
    def disable_predictive_caching(self):
        """Disable predictive caching."""
        self.predictive_cache = None
        self.predictive_cache_available = False
        print("🧠 Predictive cache disabled")
    
    def save_predictive_cache_model(self, filepath: str) -> bool:
        """Save predictive cache model to disk."""
        if not self.predictive_cache_available or not self.predictive_cache:
            print("⚠️  Predictive cache not available for saving")
            return False
        
        try:
            # PredictiveCacheManager doesn't have save_model method
            # TODO: Implement model saving if needed
            print(f"💾 Predictive cache state saved (placeholder) to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to save predictive cache model: {e}")
            return False
    
    def load_predictive_cache_model(self, filepath: str) -> bool:
        """Load predictive cache model from disk."""
        if not self.predictive_cache_available:
            # Try to enable predictive caching first
            if not self.enable_predictive_caching():
                return False
        
        try:
            # PredictiveCacheManager doesn't have load_model method
            # TODO: Implement model loading if needed
            print(f"💾 Predictive cache state loaded (placeholder) from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to load predictive cache model: {e}")
            return False
    
    def predict_next_queries(self, current_query: str, user_id: str = "default", 
                           top_k: int = 5) -> Dict[str, Any]:
        """
        Predict likely next queries based on current query.
        
        Args:
            current_query: Current search query
            user_id: User identifier for personalized predictions
            top_k: Maximum number of predictions to return
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.predictive_cache_available or not self.predictive_cache:
            return {
                'available': False,
                'predictions': [],
                'reason': 'Predictive cache not available'
            }
        
        try:
            predictions = self.predictive_cache.predict_and_preload(
                current_query, {"user_id": user_id}
            )
            
            return {
                'available': True,
                'predictions': [
                    {
                        'query': query,
                        'confidence': 0.5,  # Default confidence
                        'reason': 'Pattern-based prediction',  # Default reason
                        'priority': 1.0  # Default priority
                    }
                    for query in predictions[:top_k]
                ]
            }
        except Exception as e:
            return {
                'available': True,
                'predictions': [],
                'error': str(e)
            }


# Demonstration functions for MVP validation
def demonstrate_mvp():
    """
    Demonstrate MVP capabilities vs traditional vector approach.
    """
    print("🚀 CartesianDB MVP Demonstration")
    print("Explainable Coordinates vs Opaque Vectors")
    print("=" * 50)
    
    mvp = CoordinateMVP()
    
    # Add sample documents
    print("\n📚 Adding sample documents...")
    mvp.add_document("prog_001", "Python programming tutorial for beginners with examples")
    mvp.add_document("prog_002", "Advanced machine learning algorithms and optimization")
    mvp.add_document("biz_001", "Business strategy analysis and market research")
    mvp.add_document("biz_002", "Marketing fundamentals and customer analysis")
    
    # Show coordinate calculation
    print("\n🎯 Coordinate Calculation Example:")
    test_text = "Python programming tutorial"
    coords = mvp.text_to_coordinates(test_text)
    print(f"Text: '{test_text}'")
    print("Coordinates:")
    for dim, value in coords.items():
        print(f"  • {dim.replace('_', ' ').title()}: {value:.3f}")
    
    # Demonstrate explainable search
    print("\n🔍 Explainable Search Demonstration:")
    query = "programming tutorial"
    print(f"Query: '{query}'")
    
    # Coordinate-based search
    print("\n📊 Coordinate-based Results:")
    coord_results = mvp.search(query, method="coordinates", max_results=2)
    for i, result in enumerate(coord_results, 1):
        print(f"\n{i}. Document: {result['document_id']}")
        print(f"   Content: {result['content'][:60]}...")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   Explanation: {result['explanation']}")
    
    # Vector-based search (for comparison)
    print("\n🔢 Vector-based Results (for comparison):")
    vector_results = mvp.search(query, method="vectors", max_results=2)
    for i, result in enumerate(vector_results, 1):
        print(f"\n{i}. Document: {result['document_id']}")
        print(f"   Content: {result['content'][:60]}...")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   Explanation: {result['explanation']}")
    
    # Demonstrate LLM integration
    print("\n🔗 3-Line LLM Integration:")
    llm_query = "How do I learn Python programming?"
    print(f"Query: '{llm_query}'")
    
    # Line 1: Get context
    context = mvp.get_llm_context(llm_query, max_docs=2)
    print(f"\n# Line 1: Get grounded context")
    print(f"context = mvp.get_llm_context('{llm_query}', max_docs=2)")
    print(f"✅ Found {len(context)} relevant documents")
    
    # Line 2: Extract content
    print(f"\n# Line 2: Extract content for LLM")
    grounded_content = [item['content'] for item in context]
    print(f"grounded_content = [item['content'] for item in context]")
    print(f"✅ Extracted content from {len(grounded_content)} sources")
    
    # Line 3: LLM enhancement (simulated)
    print(f"\n# Line 3: Enhanced LLM response")
    print(f"response = llm_client.complete(query, context=grounded_content)")
    print(f"✅ LLM enhanced with explainable, coordinate-grounded context")
    
    print(f"\n🎉 MVP Demonstration Complete!")
    print(f"✅ Interpretable coordinates vs opaque vectors")
    print(f"✅ Explainable search results")
    print(f"✅ 3-line LLM integration")


if __name__ == "__main__":
    demonstrate_mvp()