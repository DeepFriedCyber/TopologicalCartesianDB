#!/usr/bin/env python3
"""
Vector Database Benchmarking Framework

This module provides a comprehensive benchmarking framework for comparing
the Topological-Cartesian Database (TCDB) against other vector databases
using public datasets.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import requests
import zipfile
import tarfile
import argparse
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import TCDB client
from benchmarks.vectordb.tcdb_client import TCDBClient, ConnectionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VectorDBBench")


@dataclass
class ModelConfig:
    """Configuration for the embedding model."""
    name: str
    model_path: str
    dimension: int
    normalize: bool = True
    batch_size: int = 32
    device: str = "cpu"  # "cpu" or "cuda"


@dataclass
class DatasetConfig:
    """Configuration for the dataset."""
    name: str  # Name of the public dataset
    subset: Optional[str] = None  # Subset of the dataset if applicable
    max_samples: Optional[int] = None  # Limit number of samples
    query_count: int = 100  # Number of queries to use


@dataclass
class DatabaseConfig:
    """Configuration for a vector database."""
    name: str
    connection_params: Dict[str, Any] = field(default_factory=dict)
    index_params: Dict[str, Any] = field(default_factory=dict)
    search_params: Dict[str, Any] = field(default_factory=dict)
    collection_name: str = "benchmark_collection"


@dataclass
class BenchmarkConfig:
    """Main configuration for the benchmark."""
    model_config: ModelConfig
    dataset_config: DatasetConfig
    database_configs: List[DatabaseConfig]
    metrics: List[str] = field(default_factory=lambda: ["latency", "throughput", "recall@10"])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 100, 1000])
    top_k_values: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    output_dir: str = "./benchmark_results"

class PublicDatasetLoader:
    """Loader for public vector search datasets."""
    
    DATASETS = {
        "msmarco": {
            "description": "MS MARCO Passage Ranking dataset",
            "url": "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
            "queries_url": "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",
            "dimension": 768,  # When using BERT-based embeddings
            "format": "tsv"
        },
        "sift1m": {
            "description": "SIFT1M dataset with 1M 128-dimensional SIFT features",
            "url": "http://corpus-texmex.irisa.fr/ftp/sift.tar.gz",
            "dimension": 128,
            "format": "binary"
        },
        "glove": {
            "description": "GloVe word embeddings",
            "url": "https://nlp.stanford.edu/data/glove.6B.zip",
            "dimension": 300,  # Using 300d embeddings
            "format": "txt"
        },
        "dbpedia": {
            "description": "DBpedia entity embeddings",
            "url": "https://zenodo.org/record/5794544/files/dbpedia.zip",
            "dimension": 768,
            "format": "npy"
        },
        "beir-scifact": {
            "description": "BEIR SciFact dataset",
            "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
            "dimension": 768,
            "format": "jsonl"
        }
    }
    
    def __init__(self, config: DatasetConfig):
        """Initialize the dataset loader."""
        self.config = config
        self.dataset_info = self.DATASETS.get(config.name.lower())
        
        if not self.dataset_info:
            raise ValueError(f"Unknown dataset: {config.name}. Available datasets: {list(self.DATASETS.keys())}")
        
        self.data_dir = os.path.join("datasets", config.name)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_dataset(self):
        """Download the dataset if not already available."""
        dataset_file = os.path.join(self.data_dir, "dataset.zip")
        
        if not os.path.exists(dataset_file):
            logger.info(f"Downloading dataset {self.config.name} from {self.dataset_info['url']}")
            
            response = requests.get(self.dataset_info['url'], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dataset_file, 'wb') as f, tqdm(
                desc=f"Downloading {self.config.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            # Extract the dataset
            logger.info(f"Extracting dataset {self.config.name}")
            
            if dataset_file.endswith('.zip'):
                with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
            elif dataset_file.endswith('.tar.gz') or dataset_file.endswith('.tgz'):
                with tarfile.open(dataset_file, 'r:gz') as tar_ref:
                    tar_ref.extractall(self.data_dir)
            
            logger.info(f"Dataset {self.config.name} downloaded and extracted")
    
    def load_dataset(self):
        """Load the dataset and return texts, embeddings, and metadata."""
        # Download the dataset if needed
        self.download_dataset()
        
        # Load the dataset based on its format
        if self.dataset_info['format'] == 'tsv':
            return self._load_tsv_dataset()
        elif self.dataset_info['format'] == 'binary':
            return self._load_binary_dataset()
        elif self.dataset_info['format'] == 'txt':
            return self._load_txt_dataset()
        elif self.dataset_info['format'] == 'npy':
            return self._load_npy_dataset()
        elif self.dataset_info['format'] == 'jsonl':
            return self._load_jsonl_dataset()
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_info['format']}")
    
    def _load_tsv_dataset(self):
        """Load a TSV dataset (like MS MARCO)."""
        # Implementation for loading TSV datasets
        # This is a simplified example - actual implementation would depend on the specific dataset structure
        
        collection_file = os.path.join(self.data_dir, "collection.tsv")
        
        if not os.path.exists(collection_file):
            collection_file = self._find_file("collection.tsv")
        
        if not collection_file:
            raise FileNotFoundError(f"Could not find collection.tsv in {self.data_dir}")
        
        # Load the collection
        texts = []
        metadata = []
        
        with open(collection_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading dataset")):
                if self.config.max_samples and i >= self.config.max_samples:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_id = parts[0]
                    text = parts[1]
                    
                    texts.append(text)
                    metadata.append({"id": doc_id})
        
        logger.info(f"Loaded {len(texts)} documents from {collection_file}")
        
        # For MS MARCO, we need to load queries separately
        queries = []
        query_file = os.path.join(self.data_dir, "queries.tsv")
        
        if not os.path.exists(query_file):
            query_file = self._find_file("queries.tsv")
        
        if query_file:
            with open(query_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc="Loading queries")):
                    if i >= self.config.query_count:
                        break
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        query_text = parts[1]
                        queries.append(query_text)
            
            logger.info(f"Loaded {len(queries)} queries from {query_file}")
        
        # Return texts, embeddings (None, will be generated later), metadata, and queries
        return texts, None, metadata, queries
    
    def _load_binary_dataset(self):
        """Load a binary dataset (like SIFT1M)."""
        # Implementation for loading binary datasets
        # This would require specific code for each binary format
        
        # For SIFT1M, we would load the binary files directly
        # This is a placeholder - actual implementation would be more complex
        
        # Return placeholder data
        logger.warning("Binary dataset loading not fully implemented")
        
        # Generate some random vectors for demonstration
        vectors = np.random.randn(10000, self.dataset_info['dimension'])
        metadata = [{"id": i} for i in range(10000)]
        queries = np.random.randn(self.config.query_count, self.dataset_info['dimension'])
        
        return None, vectors, metadata, queries
    
    def _load_txt_dataset(self):
        """Load a text dataset (like GloVe)."""
        # Implementation for loading text datasets
        
        # For GloVe, find the appropriate dimension file
        glove_file = self._find_file(f"glove.6B.{self.dataset_info['dimension']}d.txt")
        
        if not glove_file:
            raise FileNotFoundError(f"Could not find GloVe embeddings file in {self.data_dir}")
        
        # Load the embeddings
        texts = []
        vectors = []
        metadata = []
        
        with open(glove_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading GloVe embeddings")):
                if self.config.max_samples and i >= self.config.max_samples:
                    break
                
                parts = line.strip().split(' ')
                word = parts[0]
                embedding = np.array([float(val) for val in parts[1:]])
                
                texts.append(word)
                vectors.append(embedding)
                metadata.append({"id": i, "word": word})
        
        logger.info(f"Loaded {len(texts)} word embeddings from {glove_file}")
        
        # Convert to numpy array
        vectors = np.array(vectors)
        
        # Generate queries (random subset of vectors)
        query_indices = np.random.choice(len(vectors), min(self.config.query_count, len(vectors)), replace=False)
        queries = vectors[query_indices]
        
        return texts, vectors, metadata, queries
    
    def _load_npy_dataset(self):
        """Load a NumPy dataset."""
        # Implementation for loading NumPy datasets
        
        # Find the embeddings file
        embeddings_file = self._find_file(".npy")
        
        if not embeddings_file:
            raise FileNotFoundError(f"Could not find .npy file in {self.data_dir}")
        
        # Load the embeddings
        vectors = np.load(embeddings_file)
        
        # Limit the number of samples if needed
        if self.config.max_samples and self.config.max_samples < len(vectors):
            vectors = vectors[:self.config.max_samples]
        
        logger.info(f"Loaded {len(vectors)} embeddings from {embeddings_file}")
        
        # Generate metadata
        metadata = [{"id": i} for i in range(len(vectors))]
        
        # Generate queries (random subset of vectors)
        query_indices = np.random.choice(len(vectors), min(self.config.query_count, len(vectors)), replace=False)
        queries = vectors[query_indices]
        
        return None, vectors, metadata, queries
    
    def _load_jsonl_dataset(self):
        """Load a JSONL dataset (like BEIR)."""
        # Implementation for loading JSONL datasets
        
        # Find the corpus file
        corpus_file = self._find_file("corpus.jsonl")
        
        if not corpus_file:
            raise FileNotFoundError(f"Could not find corpus.jsonl in {self.data_dir}")
        
        # Load the corpus
        texts = []
        metadata = []
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading corpus")):
                if self.config.max_samples and i >= self.config.max_samples:
                    break
                
                data = json.loads(line)
                doc_id = data.get('_id')
                title = data.get('title', '')
                text = data.get('text', '')
                
                full_text = title + " " + text if title else text
                texts.append(full_text)
                metadata.append({"id": doc_id, "title": title})
        
        logger.info(f"Loaded {len(texts)} documents from {corpus_file}")
        
        # Load queries
        queries = []
        queries_file = self._find_file("queries.jsonl")
        
        if queries_file:
            with open(queries_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc="Loading queries")):
                    if i >= self.config.query_count:
                        break
                    
                    data = json.loads(line)
                    query_text = data.get('text', '')
                    queries.append(query_text)
            
            logger.info(f"Loaded {len(queries)} queries from {queries_file}")
        
        # Return texts, embeddings (None, will be generated later), metadata, and queries
        return texts, None, metadata, queries
    
    def _find_file(self, pattern):
        """Find a file in the dataset directory that matches the pattern."""
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if pattern in file:
                    return os.path.join(root, file)
        return None


class EmbeddingModel:
    """Wrapper for embedding models to generate vector embeddings."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the embedding model."""
        self.config = config
        self.model = self._load_model()
        logger.info(f"Loaded embedding model: {config.name}")
    
    def _load_model(self):
        """Load the embedding model based on configuration."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.config.model_path, device=self.config.device)
            return model
        except ImportError:
            logger.error("sentence-transformers package not found. Please install it.")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        batch_size = self.config.batch_size
        
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            
            # Normalize if required
            if self.config.normalize:
                batch_embeddings = self._normalize(batch_embeddings)
                
            embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(embeddings)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.config.dimension

class TCDBConnector:
    """Connector for Topological-Cartesian Database."""
    
    def __init__(self, config: DatabaseConfig, dimension: int):
        """Initialize the TCDB connector."""
        self.config = config
        self.dimension = dimension
        self.collection_name = config.collection_name
        self.tcdb_client = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to TCDB."""
        try:
            # Create connection configuration
            conn_config = ConnectionConfig(
                host=self.config.connection_params.get("host", "localhost"),
                port=self.config.connection_params.get("port", 8000),
                timeout=self.config.connection_params.get("timeout", 30),
                max_connections=self.config.connection_params.get("max_connections", 10),
                connection_pool_ttl=self.config.connection_params.get("connection_pool_ttl", 300)
            )
            
            # Initialize client
            self.tcdb_client = TCDBClient(conn_config)
            self.is_connected = True
            logger.info(f"Connected to TCDB at {conn_config.host}:{conn_config.port}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to TCDB: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create a collection in TCDB."""
        try:
            # Configure the multi-cube orchestration system
            cube_config = {
                "dimension": self.dimension,
                "distance_metric": self.config.index_params.get("distance_metric", "cosine"),
                "cube_count": self.config.connection_params.get("multi_cube_config", {}).get("cube_count", 5),
                "coordinate_system": self.config.connection_params.get("multi_cube_config", {}).get("coordinate_system", "topological-cartesian"),
                "optimization_level": self.config.connection_params.get("multi_cube_config", {}).get("optimization_level", "maximum")
            }
            
            # Create collection
            self.tcdb_client.create_collection(
                name=self.collection_name,
                dimension=self.dimension,
                cube_config=cube_config,
                index_type=self.config.index_params.get("index_type", "topological_hnsw"),
                parallel_processing=self.dimension > 128
            )
            
            logger.info(f"Created TCDB collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating TCDB collection: {e}")
            return False
    
    def insert_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        """Insert vectors into TCDB."""
        try:
            start_time = time.time()
            total_vectors = len(vectors)
            
            # Calculate optimal batch size
            optimal_batch_size = min(
                batch_size,
                max(100, int(10000000 / (self.dimension * 4)))  # 4 bytes per float
            )
            
            # Process in batches
            for i in tqdm(range(0, total_vectors, optimal_batch_size), desc="Inserting into TCDB"):
                end_idx = min(i + optimal_batch_size, total_vectors)
                batch_vectors = vectors[i:end_idx]
                batch_metadata = metadata[i:end_idx]
                
                # Convert to TCDB's format
                tcdb_points = [
                    {
                        "id": i + j,
                        "vector": vector.tolist(),
                        "metadata": meta
                    }
                    for j, (vector, meta) in enumerate(zip(batch_vectors, batch_metadata))
                ]
                
                # Insert batch
                self.tcdb_client.bulk_insert(
                    collection_name=self.collection_name,
                    points=tcdb_points,
                    parallel=True,
                    optimize_coordinates=True
                )
            
            elapsed = time.time() - start_time
            
            result = {
                "total_vectors": total_vectors,
                "batch_size": optimal_batch_size,
                "time": elapsed,
                "throughput": total_vectors / elapsed
            }
            
            logger.info(f"Inserted {total_vectors} vectors in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            return {
                "error": str(e),
                "total_vectors": len(vectors),
                "batch_size": batch_size,
                "time": 0,
                "throughput": 0
            }
    
    def search_vectors(self, queries: np.ndarray, top_k: int) -> Dict[str, Any]:
        """Search for vectors in TCDB."""
        try:
            start_time = time.time()
            
            # Configure search parameters
            search_params = {
                "ef_search": min(top_k * 4, 128),
                "optimize_coordinates": True,
                "use_multi_cube": True,
                "parallel": len(queries) > 10
            }
            
            # Batch queries for optimal throughput
            batch_size = min(10, len(queries))
            results = []
            
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size].tolist()
                
                # Search
                batch_results = self.tcdb_client.batch_search(
                    collection_name=self.collection_name,
                    queries=batch_queries,
                    top_k=top_k,
                    params=search_params
                )
                
                # Process results
                for query_result in batch_results:
                    results.append([hit["id"] for hit in query_result["hits"]])
            
            elapsed = time.time() - start_time
            
            result = {
                "num_queries": len(queries),
                "top_k": top_k,
                "time": elapsed,
                "latency": elapsed / len(queries),
                "qps": len(queries) / elapsed,
                "results": results
            }
            
            logger.info(f"Searched {len(queries)} queries in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {
                "error": str(e),
                "num_queries": len(queries),
                "top_k": top_k,
                "time": 0,
                "latency": 0,
                "qps": 0,
                "results": []
            }
    
    def filtered_search(self, queries: np.ndarray, filter_dict: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Search with filters in TCDB."""
        try:
            start_time = time.time()
            
            # Convert filter_dict to TCDB's filter format
            tcdb_filter = self._build_filter(filter_dict)
            
            # Configure search parameters
            search_params = {
                "ef_search": min(top_k * 8, 256),
                "optimize_coordinates": True,
                "use_multi_cube": True,
                "parallel": len(queries) > 5,
                "filter_optimization": "precompute"
            }
            
            # Batch queries for optimal throughput
            batch_size = min(5, len(queries))
            results = []
            
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size].tolist()
                
                # Search with filter
                batch_results = self.tcdb_client.batch_filtered_search(
                    collection_name=self.collection_name,
                    queries=batch_queries,
                    filter=tcdb_filter,
                    top_k=top_k,
                    params=search_params
                )
                
                # Process results
                for query_result in batch_results:
                    results.append([hit["id"] for hit in query_result["hits"]])
            
            elapsed = time.time() - start_time
            
            result = {
                "num_queries": len(queries),
                "top_k": top_k,
                "filter": filter_dict,
                "time": elapsed,
                "latency": elapsed / len(queries),
                "qps": len(queries) / elapsed,
                "results": results
            }
            
            logger.info(f"Filtered search for {len(queries)} queries in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return {
                "error": str(e),
                "num_queries": len(queries),
                "top_k": top_k,
                "filter": filter_dict,
                "time": 0,
                "latency": 0,
                "qps": 0,
                "results": []
            }
    
    def _build_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build a filter structure for TCDB."""
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, str):
                conditions.append({
                    "field": key,
                    "operator": "equals",
                    "value": value,
                    "boost": 1.0
                })
            elif isinstance(value, (int, float)):
                conditions.append({
                    "field": key,
                    "operator": "equals",
                    "value": value,
                    "boost": 1.0
                })
            elif isinstance(value, list):
                conditions.append({
                    "field": key,
                    "operator": "in",
                    "value": value,
                    "boost": 1.0
                })
        
        return {
            "must": conditions,
            "execution_strategy": "optimized"
        }
    
    def drop_collection(self) -> bool:
        """Drop the TCDB collection."""
        try:
            self.tcdb_client.drop_collection(self.collection_name)
            logger.info(f"Dropped TCDB collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error dropping collection: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from TCDB."""
        try:
            if self.tcdb_client:
                self.tcdb_client.close()
            self.is_connected = False
            logger.info("Disconnected from TCDB")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            return False


class FAISSConnector:
    """Connector for FAISS vector database."""
    
    def __init__(self, config: DatabaseConfig, dimension: int):
        """Initialize the FAISS connector."""
        self.config = config
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to FAISS (initialize)."""
        try:
            import faiss
            self.is_connected = True
            logger.info("Connected to FAISS")
            return True
        except ImportError:
            logger.error("FAISS package not found. Please install it.")
            return False
        except Exception as e:
            logger.error(f"Error connecting to FAISS: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create a FAISS index."""
        try:
            import faiss
            
            # Create index based on configuration
            index_type = self.config.index_params.get("index_type", "Flat")
            metric_type = self.config.index_params.get("metric_type", "L2")
            
            if metric_type.lower() == "cosine":
                metric = faiss.METRIC_INNER_PRODUCT
            else:
                metric = faiss.METRIC_L2
            
            if index_type == "Flat":
                self.index = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
            elif index_type == "IVF":
                nlist = self.config.index_params.get("nlist", 100)
                quantizer = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric)
                self.index.train_mode = True  # Need to train this index
            elif index_type == "HNSW":
                m = self.config.index_params.get("M", 16)
                self.index = faiss.IndexHNSWFlat(self.dimension, m, metric)
            else:
                logger.warning(f"Unsupported index type: {index_type}, using Flat")
                self.index = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"Created FAISS index: {index_type} with metric {metric_type}")
            return True
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False
    
    def insert_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        """Insert vectors into FAISS."""
        try:
            start_time = time.time()
            
            # Store metadata
            self.metadata = metadata
            
            # Normalize vectors if using cosine similarity
            metric_type = self.config.index_params.get("metric_type", "L2")
            if metric_type.lower() == "cosine":
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms
            
            # Train the index if needed
            index_type = self.config.index_params.get("index_type", "Flat")
            if index_type == "IVF" and hasattr(self.index, 'train_mode') and self.index.train_mode:
                self.index.train(vectors)
                self.index.train_mode = False
            
            # Add vectors to the index
            self.index.add(vectors.astype(np.float32))
            
            elapsed = time.time() - start_time
            
            result = {
                "total_vectors": len(vectors),
                "batch_size": batch_size,
                "time": elapsed,
                "throughput": len(vectors) / elapsed
            }
            
            logger.info(f"Inserted {len(vectors)} vectors in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            return {
                "error": str(e),
                "total_vectors": len(vectors),
                "batch_size": batch_size,
                "time": 0,
                "throughput": 0
            }
    
    def search_vectors(self, queries: np.ndarray, top_k: int) -> Dict[str, Any]:
        """Search for vectors in FAISS."""
        try:
            start_time = time.time()
            
            # Normalize queries if using cosine similarity
            metric_type = self.config.index_params.get("metric_type", "L2")
            if metric_type.lower() == "cosine":
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                queries = queries / norms
            
            # Search
            distances, indices = self.index.search(queries.astype(np.float32), top_k)
            
            # Format results
            results = []
            for i in range(len(queries)):
                query_results = []
                for j in range(top_k):
                    if j < len(indices[i]) and indices[i][j] >= 0 and indices[i][j] < len(self.metadata):
                        query_results.append(int(indices[i][j]))
                results.append(query_results)
            
            elapsed = time.time() - start_time
            
            result = {
                "num_queries": len(queries),
                "top_k": top_k,
                "time": elapsed,
                "latency": elapsed / len(queries),
                "qps": len(queries) / elapsed,
                "results": results
            }
            
            logger.info(f"Searched {len(queries)} queries in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return {
                "error": str(e),
                "num_queries": len(queries),
                "top_k": top_k,
                "time": 0,
                "latency": 0,
                "qps": 0,
                "results": []
            }
    
    def filtered_search(self, queries: np.ndarray, filter_dict: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Search with filters in FAISS (post-filtering)."""
        try:
            start_time = time.time()
            
            # FAISS doesn't support filtering directly, so we need to do post-filtering
            # First, get more results than needed
            expanded_top_k = min(top_k * 10, self.index.ntotal)
            distances, indices = self.index.search(queries.astype(np.float32), expanded_top_k)
            
            # Format results with post-filtering
            results = []
            for i in range(len(queries)):
                query_results = []
                for j in range(expanded_top_k):
                    if j < len(indices[i]) and indices[i][j] >= 0 and indices[i][j] < len(self.metadata):
                        idx = int(indices[i][j])
                        meta = self.metadata[idx]
                        
                        # Check if metadata matches filter
                        match = True
                        for key, value in filter_dict.items():
                            if key not in meta or meta[key] != value:
                                match = False
                                break
                        
                        if match:
                            query_results.append(idx)
                            if len(query_results) >= top_k:
                                break
                
                results.append(query_results)
            
            elapsed = time.time() - start_time
            
            result = {
                "num_queries": len(queries),
                "top_k": top_k,
                "filter": filter_dict,
                "time": elapsed,
                "latency": elapsed / len(queries),
                "qps": len(queries) / elapsed,
                "results": results
            }
            
            logger.info(f"Filtered search for {len(queries)} queries in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return {
                "error": str(e),
                "num_queries": len(queries),
                "top_k": top_k,
                "filter": filter_dict,
                "time": 0,
                "latency": 0,
                "qps": 0,
                "results": []
            }
    
    def drop_collection(self) -> bool:
        """Drop the FAISS index."""
        try:
            self.index = None
            self.metadata = []
            logger.info("Dropped FAISS index")
            return True
        except Exception as e:
            logger.error(f"Error dropping index: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from FAISS."""
        try:
            self.index = None
            self.is_connected = False
            logger.info("Disconnected from FAISS")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            return False

class VectorDBBench:
    """Main benchmarking class for vector databases."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark."""
        self.config = config
        self.results = {}
        self.model = None
        self.connectors = {}
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up file logging."""
        log_file = os.path.join(self.config.output_dir, "benchmark.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        self.model = EmbeddingModel(self.config.model_config)
    
    def _initialize_connectors(self):
        """Initialize database connectors."""
        for db_config in self.config.database_configs:
            if db_config.name.lower() == "tcdb":
                connector = TCDBConnector(db_config, self.config.model_config.dimension)
            elif db_config.name.lower() == "faiss":
                connector = FAISSConnector(db_config, self.config.model_config.dimension)
            else:
                logger.warning(f"Unsupported database: {db_config.name}")
                continue
            
            self.connectors[db_config.name] = connector
    
    def _load_dataset(self):
        """Load the dataset."""
        loader = PublicDatasetLoader(self.config.dataset_config)
        texts, embeddings, metadata, queries = loader.load_dataset()
        
        # Generate embeddings if not provided
        if embeddings is None:
            self._initialize_model()
            embeddings = self.model.generate_embeddings(texts)
        
        # Generate query embeddings if needed
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            query_embeddings = self.model.generate_embeddings(queries)
        else:
            query_embeddings = queries
        
        return texts, embeddings, metadata, query_embeddings
    
    def run(self):
        """Run the benchmark."""
        logger.info("Starting VectorDBBench")
        
        # Load dataset
        logger.info("Loading dataset")
        texts, embeddings, metadata, queries = self._load_dataset()
        logger.info(f"Dataset loaded: {len(embeddings)} vectors, {len(queries)} queries")
        
        # Initialize database connectors
        logger.info("Initializing database connectors")
        self._initialize_connectors()
        
        # Run benchmarks for each database
        for db_name, connector in self.connectors.items():
            logger.info(f"Benchmarking {db_name}")
            
            # Connect to database
            if not connector.connect():
                logger.error(f"Failed to connect to {db_name}")
                continue
            
            # Create collection
            if not connector.create_collection():
                logger.error(f"Failed to create collection in {db_name}")
                connector.disconnect()
                continue
            
            # Benchmark insertion
            logger.info(f"Benchmarking insertion for {db_name}")
            insertion_results = {}
            
            for batch_size in self.config.batch_sizes:
                logger.info(f"Insertion batch size: {batch_size}")
                result = connector.insert_vectors(embeddings, metadata, batch_size)
                insertion_results[batch_size] = result
            
            self.results[f"{db_name}_insertion"] = insertion_results
            
            # Benchmark search
            logger.info(f"Benchmarking search for {db_name}")
            search_results = {}
            
            for top_k in self.config.top_k_values:
                logger.info(f"Search top_k: {top_k}")
                result = connector.search_vectors(queries, top_k)
                search_results[top_k] = result
            
            self.results[f"{db_name}_search"] = search_results
            
            # Benchmark filtered search
            logger.info(f"Benchmarking filtered search for {db_name}")
            
            # Create a simple filter
            filter_dict = {"id": metadata[0]["id"]}
            
            filtered_result = connector.filtered_search(queries, filter_dict, self.config.top_k_values[0])
            self.results[f"{db_name}_filtered_search"] = filtered_result
            
            # Drop collection
            connector.drop_collection()
            
            # Disconnect
            connector.disconnect()
        
        # Save results
        self._save_results()
        
        # Visualize results
        self._visualize_results()
        
        logger.info("Benchmark completed")
        return self.results
    
    def _save_results(self):
        """Save benchmark results to file."""
        result_file = os.path.join(self.config.output_dir, "benchmark_results.json")
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
    
    def _visualize_results(self):
        """Generate visualizations of benchmark results."""
        # Create directory for visualizations
        viz_dir = os.path.join(self.config.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Insertion throughput comparison
        self._plot_insertion_throughput(viz_dir)
        
        # Search QPS comparison
        self._plot_search_qps(viz_dir)
        
        # Search latency comparison
        self._plot_search_latency(viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _plot_insertion_throughput(self, viz_dir):
        """Plot insertion throughput comparison."""
        plt.figure(figsize=(10, 6))
        
        for db_name in self.connectors.keys():
            insertion_results = self.results.get(f"{db_name}_insertion", {})
            batch_sizes = []
            throughputs = []
            
            for batch_size, result in insertion_results.items():
                if "throughput" in result:
                    batch_sizes.append(int(batch_size))
                    throughputs.append(result["throughput"])
            
            if batch_sizes:
                plt.plot(batch_sizes, throughputs, marker='o', label=db_name)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (vectors/s)')
        plt.title('Insertion Throughput Comparison')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(viz_dir, "insertion_throughput.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_search_qps(self, viz_dir):
        """Plot search QPS comparison."""
        plt.figure(figsize=(10, 6))
        
        for db_name in self.connectors.keys():
            search_results = self.results.get(f"{db_name}_search", {})
            top_ks = []
            qps_values = []
            
            for top_k, result in search_results.items():
                if "qps" in result:
                    top_ks.append(int(top_k))
                    qps_values.append(result["qps"])
            
            if top_ks:
                plt.plot(top_ks, qps_values, marker='o', label=db_name)
        
        plt.xlabel('Top-K')
        plt.ylabel('QPS (queries/s)')
        plt.title('Search QPS Comparison')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(viz_dir, "search_qps.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_search_latency(self, viz_dir):
        """Plot search latency comparison."""
        plt.figure(figsize=(10, 6))
        
        for db_name in self.connectors.keys():
            search_results = self.results.get(f"{db_name}_search", {})
            top_ks = []
            latencies = []
            
            for top_k, result in search_results.items():
                if "latency" in result:
                    top_ks.append(int(top_k))
                    latencies.append(result["latency"] * 1000)  # Convert to ms
            
            if top_ks:
                plt.plot(top_ks, latencies, marker='o', label=db_name)
        
        plt.xlabel('Top-K')
        plt.ylabel('Latency (ms)')
        plt.title('Search Latency Comparison')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(viz_dir, "search_latency.png"), dpi=300, bbox_inches='tight')
        plt.close()


def create_benchmark_config(dataset_name="glove", max_samples=10000):
    """Create a benchmark configuration."""
    # Create model config
    model_config = ModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        normalize=True
    )
    
    # Create dataset config
    dataset_config = DatasetConfig(
        name=dataset_name,
        max_samples=max_samples,
        query_count=100
    )
    
    # Create database configs
    tcdb_config = DatabaseConfig(
        name="TCDB",
        connection_params={
            "host": "localhost",
            "port": 8000,
            "multi_cube_config": {
                "cube_count": 5,
                "coordinate_system": "topological-cartesian",
                "optimization_level": "maximum"
            }
        },
        index_params={
            "distance_metric": "cosine",
            "index_type": "topological_hnsw"
        }
    )
    
    faiss_config = DatabaseConfig(
        name="FAISS",
        index_params={"index_type": "HNSW", "metric_type": "cosine"}
    )
    
    # Create benchmark config
    benchmark_config = BenchmarkConfig(
        model_config=model_config,
        dataset_config=dataset_config,
        database_configs=[tcdb_config, faiss_config],
        output_dir=f"./benchmark_results_{dataset_name}"
    )
    
    return benchmark_config


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="VectorDBBench with public datasets")
    
    parser.add_argument("--dataset", type=str, default="glove", 
                        choices=["msmarco", "sift1m", "glove", "dbpedia", "beir-scifact"],
                        help="Public dataset to use")
    parser.add_argument("--max_samples", type=int, default=10000, 
                        help="Maximum number of samples to use from the dataset")
    parser.add_argument("--query_count", type=int, default=100, 
                        help="Number of queries to use")
    
    args = parser.parse_args()
    
    # Create benchmark config
    config = create_benchmark_config(args.dataset, args.max_samples)
    config.dataset_config.query_count = args.query_count
    
    # Run benchmark
    benchmark = VectorDBBench(config)
    results = benchmark.run()
    
    return results


if __name__ == "__main__":
    main()
