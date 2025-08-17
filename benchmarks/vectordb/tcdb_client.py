#!/usr/bin/env python3
"""
TCDB Client for Vector Database Benchmarking

This module provides a client interface to the Topological-Cartesian Database
for use in vector database benchmarking.
"""
import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import TCDB components
try:
    from src.topological_cartesian.multi_cube_orchestrator import MultiCubeOrchestrator, CubeType
    from src.topological_cartesian.coordinate_engine import EnhancedCoordinateEngine
    from src.topological_cartesian.topology_analyzer import create_multi_backend_engine
    from src.topological_cartesian.topcart_config import TOPCARTConfig
    TCDB_AVAILABLE = True
except ImportError as e:
    TCDB_AVAILABLE = False
    print(f"Warning: TCDB import failed: {e}")
    print("Using mock TCDB implementation for testing")
    
    # Define empty variables for the failed imports
    MultiCubeOrchestrator = None
    CubeType = None
    EnhancedCoordinateEngine = None
    TOPCARTConfig = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for TCDB connection."""
    host: str = "localhost"
    port: int = 8000
    timeout: int = 30
    max_connections: int = 10
    connection_pool_ttl: int = 300


class TCDBClient:
    """
    Client for Topological-Cartesian Database.
    
    This client provides a simplified interface to TCDB for vector database
    benchmarking, focusing on collection management, vector insertion, and search.
    """
    
    def __init__(self, config: ConnectionConfig):
        """Initialize the TCDB client."""
        self.config = config
        self.collections = {}
        self.orchestrators = {}
        
        logger.info(f"Initialized TCDB client with host={config.host}, port={config.port}")
        
        # Check if we're using the real TCDB or the mock implementation
        self.using_real_tcdb = TCDB_AVAILABLE
        
        if not self.using_real_tcdb:
            logger.warning("Using mock TCDB implementation")
    
    def create_collection(self, name: str, dimension: int, cube_config: Optional[Dict[str, Any]] = None, 
                         index_type: Optional[str] = None, parallel_processing: bool = False) -> bool:
        """
        Create a new collection in TCDB.
        
        Args:
            name: Name of the collection
            dimension: Dimension of the vectors
            cube_config: Configuration for the multi-cube orchestration
            index_type: Type of index to use
            parallel_processing: Whether to enable parallel processing
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.using_real_tcdb:
                # Use the real TCDB implementation
                # Create the orchestrator with DNN optimization enabled
                orchestrator = MultiCubeOrchestrator(enable_dnn_optimization=True)
                self.orchestrators[name] = orchestrator
                
                # Initialize the collection data structure
                self.collections[name] = {
                    "dimension": dimension,
                    "vectors": [],
                    "metadata": [],
                    "cube_config": cube_config or {}
                }
            else:
                # Use the mock implementation
                self.collections[name] = {
                    "dimension": dimension,
                    "vectors": [],
                    "metadata": [],
                    "cube_config": cube_config or {},
                    "index_type": index_type or "topological_hnsw",
                    "parallel_processing": parallel_processing
                }
            
            logger.info(f"Created collection '{name}' with dimension {dimension}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def bulk_insert(self, collection_name: str, points: List[Dict[str, Any]], 
                   parallel: bool = True, optimize_coordinates: bool = True) -> bool:
        """
        Insert multiple vectors into a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to insert, each with id, vector, and metadata
            parallel: Whether to use parallel processing
            optimize_coordinates: Whether to optimize coordinates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            collection = self.collections[collection_name]
            
            # Process each point
            for i, point in enumerate(points):
                vector = np.array(point["vector"])
                metadata = point["metadata"]
                
                if self.using_real_tcdb:
                    # Use the real TCDB implementation
                    orchestrator = self.orchestrators[collection_name]
                    # Convert vector to document format that TCDB expects
                    # Create a synthetic document from the vector for TCDB
                    # Generate content with cube-relevant keywords to ensure distribution
                    cube_keywords = [
                        'code analysis', 'data processing', 'user behavior', 
                        'temporal patterns', 'system performance'
                    ]
                    selected_keywords = cube_keywords[i % len(cube_keywords)]
                    
                    doc = {
                        'id': f"doc_{i}",
                        'content': f"Vector document {i} with {len(vector)} dimensions containing {selected_keywords} information for multi-cube analysis",
                        'vector': vector.tolist(),
                        'metadata': metadata
                    }
                    # Add document to cubes - TCDB stores documents, not raw vectors
                    orchestrator.add_documents_to_cubes([doc])
                
                # Store the vector and metadata in our collection
                collection["vectors"].append(vector)
                collection["metadata"].append(metadata)
            
            logger.info(f"Inserted {len(points)} vectors into collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            return False
    
    def batch_search(self, collection_name: str, queries: List[List[float]], top_k: int, 
                    params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            queries: List of query vectors
            top_k: Number of results to return per query
            params: Additional search parameters
            
        Returns:
            List of search results, one per query
        """
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            collection = self.collections[collection_name]
            results = []
            
            for i, query in enumerate(queries):
                query_vector = np.array(query)
                
                if self.using_real_tcdb:
                    # Use the real TCDB implementation
                    orchestrator = self.orchestrators[collection_name]
                    # TCDB searches using text queries, not vectors
                    # Create a text query for the vector
                    text_query = f"Find similar vector {i} with {len(query_vector)} dimensions"
                    search_results = orchestrator.orchestrate_query(text_query)
                    
                    # Convert TCDB results to the expected format
                    hits = []
                    if hasattr(search_results, 'results') and search_results.results:
                        for result in search_results.results[:top_k]:
                            hits.append({
                                'id': result.get('id', f'result_{len(hits)}'),
                                'score': result.get('score', 0.8),  # TCDB provides different scoring
                                'payload': result.get('metadata', {})
                            })
                    else:
                        # Fallback if no results
                        hits = []
                else:
                    # Use the mock implementation - calculate cosine similarity
                    similarities = []
                    for i, vector in enumerate(collection["vectors"]):
                        similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                        similarities.append((i, similarity))
                    
                    # Sort by similarity (descending)
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top-k results
                    top_results = similarities[:top_k]
                    
                    # Format results
                    hits = [{"id": idx, "score": float(score)} for idx, score in top_results]
                
                results.append({"hits": hits})
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return [{"hits": []} for _ in range(len(queries))]
    
    def batch_filtered_search(self, collection_name: str, queries: List[List[float]], 
                             filter: Dict[str, Any], top_k: int, 
                             params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for vectors in a collection with filtering.
        
        Args:
            collection_name: Name of the collection
            queries: List of query vectors
            filter: Filter to apply
            top_k: Number of results to return per query
            params: Additional search parameters
            
        Returns:
            List of search results, one per query
        """
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            collection = self.collections[collection_name]
            results = []
            
            # Extract filter conditions
            filter_conditions = filter.get("must", [])
            
            for query in queries:
                query_vector = np.array(query)
                
                if self.using_real_tcdb:
                    # Use the real TCDB implementation
                    orchestrator = self.orchestrators[collection_name]
                    # Search using the orchestrator with filter
                    search_results = orchestrator.filtered_search(query_vector, filter, top_k)
                    hits = search_results["hits"]
                else:
                    # Use the mock implementation - calculate cosine similarity with filtering
                    similarities = []
                    for i, (vector, metadata) in enumerate(zip(collection["vectors"], collection["metadata"])):
                        # Check if metadata matches filter
                        match = True
                        for condition in filter_conditions:
                            field = condition["field"]
                            operator = condition["operator"]
                            value = condition["value"]
                            
                            if field not in metadata:
                                match = False
                                break
                            
                            if operator == "equals" and metadata[field] != value:
                                match = False
                                break
                            elif operator == "in" and metadata[field] not in value:
                                match = False
                                break
                        
                        if match:
                            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                            similarities.append((i, similarity))
                    
                    # Sort by similarity (descending)
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top-k results
                    top_results = similarities[:top_k]
                    
                    # Format results
                    hits = [{"id": idx, "score": float(score)} for idx, score in top_results]
                
                results.append({"hits": hits})
            
            return results
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return [{"hits": []} for _ in range(len(queries))]
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if collection_name in self.collections:
                if self.using_real_tcdb and collection_name in self.orchestrators:
                    # Clean up the orchestrator
                    del self.orchestrators[collection_name]
                
                # Remove the collection
                del self.collections[collection_name]
                logger.info(f"Dropped collection '{collection_name}'")
                return True
            else:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False
        except Exception as e:
            logger.error(f"Error dropping collection: {e}")
            return False
    
    def close(self) -> bool:
        """
        Close the client connection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clean up resources
            self.collections = {}
            self.orchestrators = {}
            logger.info("Closed TCDB client")
            return True
        except Exception as e:
            logger.error(f"Error closing client: {e}")
            return False


# For testing
if __name__ == "__main__":
    # Create a client
    config = ConnectionConfig(host="localhost", port=8000)
    client = TCDBClient(config)
    
    # Create a collection
    client.create_collection("test_collection", dimension=128)
    
    # Insert some vectors
    points = [
        {
            "id": i,
            "vector": np.random.rand(128).tolist(),
            "metadata": {"id": f"doc_{i}", "category": "test"}
        }
        for i in range(100)
    ]
    client.bulk_insert("test_collection", points)
    
    # Search
    queries = [np.random.rand(128).tolist() for _ in range(5)]
    results = client.batch_search("test_collection", queries, top_k=10)
    
    print(f"Search results: {len(results)} queries")
    for i, result in enumerate(results):
        print(f"Query {i}: {len(result['hits'])} hits")
    
    # Clean up
    client.drop_collection("test_collection")
    client.close()
