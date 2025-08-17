#!/usr/bin/env python3
"""
Optimized Search Engine

Addresses search speed and precision issues by:
1. Approximate nearest neighbor search (FAISS, Annoy)
2. Hybrid search combining semantic and coordinate-based approaches
3. Query expansion and relevance feedback
4. Adaptive indexing and caching strategies
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import heapq
from collections import defaultdict, deque
import threading
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
SEARCH_LIBS_AVAILABLE = {}

try:
    import faiss
    SEARCH_LIBS_AVAILABLE['faiss'] = True
except ImportError:
    SEARCH_LIBS_AVAILABLE['faiss'] = False

try:
    from annoy import AnnoyIndex
    SEARCH_LIBS_AVAILABLE['annoy'] = True
except ImportError:
    SEARCH_LIBS_AVAILABLE['annoy'] = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    SEARCH_LIBS_AVAILABLE['sklearn'] = True
except ImportError:
    SEARCH_LIBS_AVAILABLE['sklearn'] = False


@dataclass
class SearchResult:
    """Enhanced search result with detailed scoring"""
    document_id: str
    similarity_score: float
    coordinate_distance: float
    semantic_similarity: float
    hybrid_score: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'similarity_score': self.similarity_score,
            'coordinate_distance': self.coordinate_distance,
            'semantic_similarity': self.semantic_similarity,
            'hybrid_score': self.hybrid_score,
            'explanation': self.explanation,
            'metadata': self.metadata
        }


@dataclass
class SearchConfig:
    """Search engine configuration"""
    index_type: str = 'auto'  # 'faiss', 'annoy', 'sklearn', 'auto'
    similarity_metric: str = 'cosine'  # 'cosine', 'euclidean', 'hybrid'
    coordinate_weight: float = 0.3  # Weight for coordinate-based similarity
    semantic_weight: float = 0.7  # Weight for semantic similarity
    use_query_expansion: bool = True
    use_relevance_feedback: bool = True
    cache_size: int = 10000
    index_rebuild_threshold: int = 1000
    approximate_search: bool = True
    search_ef: int = 50  # FAISS search parameter
    n_trees: int = 10  # Annoy parameter


class SearchIndex(ABC):
    """Abstract base class for search indices"""
    
    @abstractmethod
    def build_index(self, embeddings: np.ndarray, document_ids: List[str]):
        """Build the search index"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def add_documents(self, embeddings: np.ndarray, document_ids: List[str]):
        """Add new documents to the index"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        pass


class FAISSIndex(SearchIndex):
    """FAISS-based approximate nearest neighbor index"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.index = None
        self.document_ids = []
        self.dimension = None
        self.is_built = False
    
    def build_index(self, embeddings: np.ndarray, document_ids: List[str]):
        """Build FAISS index"""
        if not SEARCH_LIBS_AVAILABLE['faiss']:
            raise ImportError("FAISS not available")
        
        self.dimension = embeddings.shape[1]
        self.document_ids = document_ids.copy()
        
        # Choose index type based on size and requirements
        if len(embeddings) < 1000:
            # Use exact search for small datasets
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        else:
            # Use approximate search for larger datasets
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.dimension), self.dimension, nlist)
            
            # Train the index
            self.index.train(embeddings.astype(np.float32))
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype(np.float32))
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(10, nlist)  # Number of clusters to search
        
        self.is_built = True
        logger.info(f"Built FAISS index with {len(embeddings)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search using FAISS"""
        if not self.is_built:
            return []
        
        # Normalize query
        query_norm = query_embedding.copy().astype(np.float32)
        faiss.normalize_L2(query_norm.reshape(1, -1))
        
        # Search
        similarities, indices = self.index.search(query_norm.reshape(1, -1), k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx < len(self.document_ids):
                results.append((idx, float(sim)))
        
        return results
    
    def add_documents(self, embeddings: np.ndarray, document_ids: List[str]):
        """Add documents to FAISS index"""
        if not self.is_built:
            self.build_index(embeddings, document_ids)
            return
        
        # Normalize new embeddings
        faiss.normalize_L2(embeddings.astype(np.float32))
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        self.document_ids.extend(document_ids)
        
        logger.info(f"Added {len(document_ids)} documents to FAISS index")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'index_type': 'FAISS',
            'total_documents': len(self.document_ids),
            'dimension': self.dimension,
            'is_built': self.is_built,
            'is_trained': hasattr(self.index, 'is_trained') and self.index.is_trained
        }


class AnnoyIndex(SearchIndex):
    """Annoy-based approximate nearest neighbor index"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.index = None
        self.document_ids = []
        self.dimension = None
        self.is_built = False
    
    def build_index(self, embeddings: np.ndarray, document_ids: List[str]):
        """Build Annoy index"""
        if not SEARCH_LIBS_AVAILABLE['annoy']:
            raise ImportError("Annoy not available")
        
        self.dimension = embeddings.shape[1]
        self.document_ids = document_ids.copy()
        
        # Create Annoy index
        metric = 'angular' if self.config.similarity_metric == 'cosine' else 'euclidean'
        self.index = AnnoyIndex(self.dimension, metric)
        
        # Add embeddings
        for i, embedding in enumerate(embeddings):
            self.index.add_item(i, embedding)
        
        # Build index
        self.index.build(self.config.n_trees)
        self.is_built = True
        
        logger.info(f"Built Annoy index with {len(embeddings)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search using Annoy"""
        if not self.is_built:
            return []
        
        # Search
        indices, distances = self.index.get_nns_by_vector(
            query_embedding.tolist(), k, include_distances=True
        )
        
        # Convert distances to similarities
        results = []
        for idx, dist in zip(indices, distances):
            if idx < len(self.document_ids):
                # Convert distance to similarity (higher is better)
                similarity = 1.0 / (1.0 + dist) if dist > 0 else 1.0
                results.append((idx, similarity))
        
        return results
    
    def add_documents(self, embeddings: np.ndarray, document_ids: List[str]):
        """Add documents to Annoy index (requires rebuild)"""
        # Annoy requires rebuilding the entire index
        all_embeddings = np.vstack([self._get_all_embeddings(), embeddings])
        all_document_ids = self.document_ids + document_ids
        
        self.build_index(all_embeddings, all_document_ids)
    
    def _get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings from current index (placeholder)"""
        # This is a limitation of Annoy - we'd need to store embeddings separately
        # For now, return empty array
        return np.array([]).reshape(0, self.dimension) if self.dimension else np.array([])
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'index_type': 'Annoy',
            'total_documents': len(self.document_ids),
            'dimension': self.dimension,
            'is_built': self.is_built,
            'n_trees': self.config.n_trees
        }


class SklearnIndex(SearchIndex):
    """Scikit-learn based exact nearest neighbor index"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.index = None
        self.embeddings = None
        self.document_ids = []
        self.is_built = False
    
    def build_index(self, embeddings: np.ndarray, document_ids: List[str]):
        """Build sklearn index"""
        if not SEARCH_LIBS_AVAILABLE['sklearn']:
            raise ImportError("Scikit-learn not available")
        
        self.embeddings = embeddings.copy()
        self.document_ids = document_ids.copy()
        
        # Choose metric
        metric = 'cosine' if self.config.similarity_metric == 'cosine' else 'euclidean'
        
        # Create NearestNeighbors index
        self.index = NearestNeighbors(
            n_neighbors=min(50, len(embeddings)),
            metric=metric,
            algorithm='auto',
            n_jobs=-1
        )
        
        self.index.fit(embeddings)
        self.is_built = True
        
        logger.info(f"Built sklearn index with {len(embeddings)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search using sklearn"""
        if not self.is_built:
            return []
        
        # Find neighbors
        distances, indices = self.index.kneighbors(
            query_embedding.reshape(1, -1), 
            n_neighbors=min(k, len(self.document_ids))
        )
        
        # Convert distances to similarities
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.document_ids):
                # Convert distance to similarity
                if self.config.similarity_metric == 'cosine':
                    similarity = 1.0 - dist  # Cosine distance to similarity
                else:
                    similarity = 1.0 / (1.0 + dist)  # Euclidean distance to similarity
                
                results.append((idx, float(similarity)))
        
        return results
    
    def add_documents(self, embeddings: np.ndarray, document_ids: List[str]):
        """Add documents to sklearn index (requires rebuild)"""
        if not self.is_built:
            self.build_index(embeddings, document_ids)
            return
        
        # Combine with existing embeddings
        all_embeddings = np.vstack([self.embeddings, embeddings])
        all_document_ids = self.document_ids + document_ids
        
        self.build_index(all_embeddings, all_document_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'index_type': 'Sklearn',
            'total_documents': len(self.document_ids),
            'dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'is_built': self.is_built,
            'metric': self.config.similarity_metric
        }


class HybridSearchEngine:
    """Hybrid search engine combining multiple approaches"""
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.semantic_index = None
        self.coordinate_index = None
        self.documents = {}
        self.embeddings = {}
        self.coordinates = {}
        self.query_cache = {}
        self.relevance_feedback = defaultdict(list)
        self.search_stats = defaultdict(int)
        self.lock = threading.RLock()
        
        # Initialize search index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the appropriate search index"""
        
        if self.config.index_type == 'auto':
            # Auto-select best available index
            if SEARCH_LIBS_AVAILABLE['faiss']:
                index_type = 'faiss'
            elif SEARCH_LIBS_AVAILABLE['annoy']:
                index_type = 'annoy'
            elif SEARCH_LIBS_AVAILABLE['sklearn']:
                index_type = 'sklearn'
            else:
                raise ImportError("No search libraries available")
        else:
            index_type = self.config.index_type
        
        # Create index
        if index_type == 'faiss':
            self.semantic_index = FAISSIndex(self.config)
        elif index_type == 'annoy':
            self.semantic_index = AnnoyIndex(self.config)
        elif index_type == 'sklearn':
            self.semantic_index = SklearnIndex(self.config)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        logger.info(f"Initialized {index_type} search index")
    
    def add_documents(self, documents: Dict[str, Dict[str, Any]]):
        """Add documents with embeddings and coordinates"""
        
        with self.lock:
            new_doc_ids = []
            new_embeddings = []
            new_coordinates = []
            
            for doc_id, doc_data in documents.items():
                if doc_id not in self.documents:
                    self.documents[doc_id] = doc_data
                    
                    # Extract or generate embedding
                    if 'embedding' in doc_data:
                        embedding = doc_data['embedding']
                    else:
                        # Generate embedding from content (placeholder)
                        content = doc_data.get('content', '')
                        embedding = self._generate_embedding(content)
                    
                    # Extract coordinates
                    coordinates = doc_data.get('coordinates', {})
                    
                    self.embeddings[doc_id] = embedding
                    self.coordinates[doc_id] = coordinates
                    
                    new_doc_ids.append(doc_id)
                    new_embeddings.append(embedding)
                    new_coordinates.append(coordinates)
            
            if new_embeddings:
                # Add to semantic index
                embeddings_array = np.array(new_embeddings)
                self.semantic_index.add_documents(embeddings_array, new_doc_ids)
                
                logger.info(f"Added {len(new_doc_ids)} documents to search engine")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (placeholder)"""
        # This would use the actual embedding model
        # For now, return random embedding
        return np.random.random(384)
    
    def search(self, query: str, query_embedding: np.ndarray = None, 
              query_coordinates: Dict[str, float] = None, k: int = 10) -> List[SearchResult]:
        """Hybrid search combining semantic and coordinate-based approaches"""
        
        with self.lock:
            # Check cache
            cache_key = f"{query}_{k}_{self.config.coordinate_weight}_{self.config.semantic_weight}"
            if cache_key in self.query_cache:
                self.search_stats['cache_hits'] += 1
                return self.query_cache[cache_key]
            
            self.search_stats['cache_misses'] += 1
            
            # Generate query embedding if not provided
            if query_embedding is None:
                query_embedding = self._generate_embedding(query)
            
            # Semantic search
            semantic_results = self._semantic_search(query_embedding, k * 2)  # Get more candidates
            
            # Coordinate search (if coordinates provided)
            coordinate_results = []
            if query_coordinates:
                coordinate_results = self._coordinate_search(query_coordinates, k * 2)
            
            # Combine and rank results
            final_results = self._combine_results(
                semantic_results, coordinate_results, query_embedding, query_coordinates, k
            )
            
            # Apply query expansion if enabled
            if self.config.use_query_expansion and len(final_results) < k:
                expanded_results = self._expand_query_search(query, query_embedding, k)
                final_results = self._merge_results(final_results, expanded_results, k)
            
            # Apply relevance feedback if available
            if self.config.use_relevance_feedback and query in self.relevance_feedback:
                final_results = self._apply_relevance_feedback(final_results, query)
            
            # Cache results
            if len(self.query_cache) < self.config.cache_size:
                self.query_cache[cache_key] = final_results
            
            return final_results
    
    def _semantic_search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Perform semantic search"""
        
        if not self.semantic_index.is_built:
            return []
        
        # Search semantic index
        index_results = self.semantic_index.search(query_embedding, k)
        
        # Convert index results to document IDs
        results = []
        for idx, similarity in index_results:
            if idx < len(self.semantic_index.document_ids):
                doc_id = self.semantic_index.document_ids[idx]
                results.append((doc_id, similarity))
        
        return results
    
    def _coordinate_search(self, query_coordinates: Dict[str, float], k: int) -> List[Tuple[str, float]]:
        """Perform coordinate-based search"""
        
        results = []
        query_coord_array = np.array([query_coordinates.get(dim, 0.5) for dim in ['domain', 'complexity', 'task_type']])
        
        for doc_id, doc_coordinates in self.coordinates.items():
            if doc_coordinates:
                doc_coord_array = np.array([doc_coordinates.get(dim, 0.5) for dim in ['domain', 'complexity', 'task_type']])
                
                # Calculate coordinate distance
                distance = np.linalg.norm(query_coord_array - doc_coord_array)
                similarity = 1.0 / (1.0 + distance)  # Convert to similarity
                
                results.append((doc_id, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _combine_results(self, semantic_results: List[Tuple[str, float]], 
                        coordinate_results: List[Tuple[str, float]],
                        query_embedding: np.ndarray, query_coordinates: Dict[str, float],
                        k: int) -> List[SearchResult]:
        """Combine semantic and coordinate search results"""
        
        # Create score dictionaries
        semantic_scores = {doc_id: score for doc_id, score in semantic_results}
        coordinate_scores = {doc_id: score for doc_id, score in coordinate_results}
        
        # Get all unique document IDs
        all_doc_ids = set(semantic_scores.keys()) | set(coordinate_scores.keys())
        
        combined_results = []
        
        for doc_id in all_doc_ids:
            semantic_sim = semantic_scores.get(doc_id, 0.0)
            coordinate_sim = coordinate_scores.get(doc_id, 0.0)
            
            # Calculate hybrid score
            hybrid_score = (self.config.semantic_weight * semantic_sim + 
                           self.config.coordinate_weight * coordinate_sim)
            
            # Calculate coordinate distance for explanation
            coord_distance = 0.0
            if query_coordinates and doc_id in self.coordinates:
                doc_coords = self.coordinates[doc_id]
                if doc_coords:
                    query_array = np.array([query_coordinates.get(dim, 0.5) for dim in ['domain', 'complexity', 'task_type']])
                    doc_array = np.array([doc_coords.get(dim, 0.5) for dim in ['domain', 'complexity', 'task_type']])
                    coord_distance = float(np.linalg.norm(query_array - doc_array))
            
            # Create explanation
            explanation = f"Semantic: {semantic_sim:.3f}, Coordinate: {coordinate_sim:.3f}, Hybrid: {hybrid_score:.3f}"
            
            result = SearchResult(
                document_id=doc_id,
                similarity_score=hybrid_score,
                coordinate_distance=coord_distance,
                semantic_similarity=semantic_sim,
                hybrid_score=hybrid_score,
                explanation=explanation,
                metadata={
                    'semantic_weight': self.config.semantic_weight,
                    'coordinate_weight': self.config.coordinate_weight
                }
            )
            
            combined_results.append(result)
        
        # Sort by hybrid score and return top k
        combined_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return combined_results[:k]
    
    def _expand_query_search(self, query: str, query_embedding: np.ndarray, k: int) -> List[SearchResult]:
        """Expand query and search for additional results"""
        
        # Simple query expansion: add related terms (placeholder)
        expanded_queries = [
            query + " tutorial",
            query + " guide",
            query + " introduction",
            query + " advanced"
        ]
        
        expanded_results = []
        
        for expanded_query in expanded_queries:
            expanded_embedding = self._generate_embedding(expanded_query)
            results = self._semantic_search(expanded_embedding, k // 2)
            
            for doc_id, similarity in results:
                result = SearchResult(
                    document_id=doc_id,
                    similarity_score=similarity * 0.8,  # Reduce score for expanded queries
                    coordinate_distance=0.0,
                    semantic_similarity=similarity,
                    hybrid_score=similarity * 0.8,
                    explanation=f"Query expansion: {expanded_query}",
                    metadata={'expanded_query': expanded_query}
                )
                expanded_results.append(result)
        
        return expanded_results
    
    def _merge_results(self, results1: List[SearchResult], results2: List[SearchResult], k: int) -> List[SearchResult]:
        """Merge two result lists"""
        
        # Combine results, avoiding duplicates
        seen_docs = set()
        merged_results = []
        
        for result in results1 + results2:
            if result.document_id not in seen_docs:
                seen_docs.add(result.document_id)
                merged_results.append(result)
        
        # Sort by hybrid score and return top k
        merged_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return merged_results[:k]
    
    def _apply_relevance_feedback(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Apply relevance feedback to boost relevant documents"""
        
        feedback = self.relevance_feedback[query]
        relevant_docs = {doc_id for doc_id, is_relevant in feedback if is_relevant}
        
        # Boost scores for relevant documents
        for result in results:
            if result.document_id in relevant_docs:
                result.hybrid_score *= 1.2  # 20% boost
                result.explanation += " (relevance boosted)"
        
        # Re-sort results
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results
    
    def add_relevance_feedback(self, query: str, doc_id: str, is_relevant: bool):
        """Add relevance feedback for a query-document pair"""
        
        with self.lock:
            self.relevance_feedback[query].append((doc_id, is_relevant))
            
            # Clear cache for this query
            cache_keys_to_remove = [key for key in self.query_cache.keys() if key.startswith(query)]
            for key in cache_keys_to_remove:
                del self.query_cache[key]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        
        with self.lock:
            stats = {
                'total_documents': len(self.documents),
                'cache_size': len(self.query_cache),
                'cache_hits': self.search_stats['cache_hits'],
                'cache_misses': self.search_stats['cache_misses'],
                'relevance_feedback_queries': len(self.relevance_feedback),
                'index_stats': self.semantic_index.get_stats() if self.semantic_index else {}
            }
            
            if stats['cache_hits'] + stats['cache_misses'] > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            else:
                stats['cache_hit_rate'] = 0.0
            
            return stats
    
    def clear_cache(self):
        """Clear search cache"""
        with self.lock:
            self.query_cache.clear()
            logger.info("Search cache cleared")
    
    def save_index(self, filepath: str):
        """Save search index to file"""
        
        index_data = {
            'config': self.config,
            'documents': self.documents,
            'embeddings': self.embeddings,
            'coordinates': self.coordinates,
            'relevance_feedback': dict(self.relevance_feedback),
            'search_stats': dict(self.search_stats)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Search index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load search index from file"""
        
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.config = index_data['config']
        self.documents = index_data['documents']
        self.embeddings = index_data['embeddings']
        self.coordinates = index_data['coordinates']
        self.relevance_feedback = defaultdict(list, index_data['relevance_feedback'])
        self.search_stats = defaultdict(int, index_data['search_stats'])
        
        # Rebuild semantic index
        if self.embeddings:
            doc_ids = list(self.embeddings.keys())
            embeddings_array = np.array([self.embeddings[doc_id] for doc_id in doc_ids])
            self.semantic_index.build_index(embeddings_array, doc_ids)
        
        logger.info(f"Search index loaded from {filepath}")


if __name__ == "__main__":
    # Demonstration of optimized search engine
    print("Optimized Search Engine Demo")
    print("=" * 35)
    
    # Create search engine
    config = SearchConfig(
        index_type='auto',
        similarity_metric='cosine',
        coordinate_weight=0.3,
        semantic_weight=0.7,
        use_query_expansion=True,
        use_relevance_feedback=True
    )
    
    search_engine = HybridSearchEngine(config)
    
    # Add sample documents
    sample_documents = {
        'doc1': {
            'content': 'Python machine learning tutorial with scikit-learn',
            'coordinates': {'domain': 0.9, 'complexity': 0.6, 'task_type': 0.3},
            'embedding': np.random.random(384)
        },
        'doc2': {
            'content': 'Basic HTML CSS web development guide',
            'coordinates': {'domain': 0.8, 'complexity': 0.2, 'task_type': 0.1},
            'embedding': np.random.random(384)
        },
        'doc3': {
            'content': 'Advanced neural network architecture design',
            'coordinates': {'domain': 0.95, 'complexity': 0.9, 'task_type': 0.8},
            'embedding': np.random.random(384)
        },
        'doc4': {
            'content': 'Business strategy and market analysis',
            'coordinates': {'domain': 0.4, 'complexity': 0.7, 'task_type': 0.8},
            'embedding': np.random.random(384)
        },
        'doc5': {
            'content': 'Creative writing and storytelling techniques',
            'coordinates': {'domain': 0.2, 'complexity': 0.5, 'task_type': 0.3},
            'embedding': np.random.random(384)
        }
    }
    
    search_engine.add_documents(sample_documents)
    
    print(f"Added {len(sample_documents)} documents to search engine")
    
    # Test searches
    test_queries = [
        ("machine learning python", {'domain': 0.9, 'complexity': 0.6, 'task_type': 0.4}),
        ("web development tutorial", {'domain': 0.8, 'complexity': 0.3, 'task_type': 0.2}),
        ("business analysis", {'domain': 0.4, 'complexity': 0.6, 'task_type': 0.7}),
    ]
    
    print("\nSearch Results:")
    print("-" * 20)
    
    for query, query_coords in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Coordinates: {query_coords}")
        
        start_time = time.time()
        results = search_engine.search(query, query_coordinates=query_coords, k=3)
        search_time = time.time() - start_time
        
        print(f"Search time: {search_time:.4f}s")
        print("Results:")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.document_id}")
            print(f"     Score: {result.hybrid_score:.3f}")
            print(f"     Explanation: {result.explanation}")
    
    # Add relevance feedback
    search_engine.add_relevance_feedback("machine learning python", "doc1", True)
    search_engine.add_relevance_feedback("machine learning python", "doc2", False)
    
    print(f"\nAdded relevance feedback")
    
    # Test search with feedback
    results = search_engine.search("machine learning python", 
                                  query_coordinates={'domain': 0.9, 'complexity': 0.6, 'task_type': 0.4}, 
                                  k=3)
    
    print(f"\nSearch with relevance feedback:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.document_id}: {result.hybrid_score:.3f}")
        print(f"     {result.explanation}")
    
    # Show statistics
    stats = search_engine.get_search_stats()
    print(f"\nSearch Engine Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.3f}")
    print(f"  Index type: {stats['index_stats'].get('index_type', 'Unknown')}")
    
    print(f"\n✅ Optimized search engine demo completed!")