"""
Simple Vector Database Benchmark

A lightweight benchmarking tool for vector databases with FAISS implementation.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorDBBench")

class EmbeddingModel:
    """Wrapper for embedding models to generate vector embeddings."""
    
    def __init__(self, model_name: str, dimension: int = 384):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.dimension = dimension
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
            return model
        except ImportError:
            logger.error("sentence-transformers package not found. Please install it.")
            raise
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True)
            embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
        
        return all_embeddings

def generate_synthetic_dataset(size: int, dimension: int, query_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset for benchmarking."""
    logger.info(f"Generating synthetic dataset with {size} vectors of dimension {dimension}")
    
    # Generate random vectors
    vectors = np.random.randn(size, dimension).astype(np.float32)
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # Generate query vectors
    query_indices = np.random.choice(size, query_size, replace=False)
    queries = vectors[query_indices].copy()
    
    # Add some noise to queries
    noise = np.random.randn(*queries.shape) * 0.1
    queries += noise
    
    # Normalize queries
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / norms
    
    logger.info(f"Generated {size} vectors and {query_size} queries")
    return vectors, queries

class FAISSBenchmark:
    """Benchmark for FAISS vector database."""
    
    def __init__(self, dimension: int, output_dir: str = "./benchmark_results"):
        """Initialize the benchmark."""
        self.dimension = dimension
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if FAISS is available
        try:
            import faiss
            self.faiss_available = True
            logger.info("FAISS is available")
        except ImportError:
            logger.error("faiss-cpu package not found. Please install it.")
            self.faiss_available = False
    
    def benchmark_index_types(self, vectors: np.ndarray, queries: np.ndarray, 
                             index_types: List[str], top_k_values: List[int]) -> Dict[str, Any]:
        """Benchmark different FAISS index types."""
        if not self.faiss_available:
            logger.error("FAISS is not available. Skipping benchmark.")
            return {}
        
        import faiss
        
        results = {}
        
        for index_type in index_types:
            logger.info(f"Benchmarking FAISS index type: {index_type}")
            
            # Create index
            if index_type == "Flat":
                index = faiss.IndexFlatIP(self.dimension)
            elif index_type == "IVF":
                nlist = 100
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                # Train the index
                logger.info("Training IVF index...")
                index.train(vectors)
            elif index_type == "HNSW":
                M = 16  # Number of connections per layer
                index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
            else:
                logger.warning(f"Unsupported index type: {index_type}, using Flat")
                index = faiss.IndexFlatIP(self.dimension)
            
            # Add vectors to index
            start_time = time.time()
            index.add(vectors)
            index_time = time.time() - start_time
            
            logger.info(f"Added {len(vectors)} vectors to index in {index_time:.2f} seconds")
            
            # Benchmark search for different top_k values
            search_results = {}
            
            for k in top_k_values:
                logger.info(f"Benchmarking search with top_k={k}")
                
                # Set search parameters if applicable
                if hasattr(index, 'nprobe'):
                    index.nprobe = 10
                
                # Perform search
                start_time = time.time()
                distances, indices = index.search(queries, k)
                search_time = time.time() - start_time
                
                qps = len(queries) / search_time
                latency = search_time / len(queries)
                
                search_results[k] = {
                    "time": search_time,
                    "qps": qps,
                    "latency": latency
                }
                
                logger.info(f"Search with top_k={k}: QPS={qps:.2f}, Latency={latency*1000:.2f}ms")
            
            results[index_type] = {
                "index_time": index_time,
                "index_throughput": len(vectors) / index_time,
                "search": search_results
            }
        
        self.results = results
        return results
    
    def save_results(self):
        """Save benchmark results to file."""
        result_file = os.path.join(self.output_dir, "faiss_benchmark_results.json")
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
    
    def visualize_results(self):
        """Generate visualizations of benchmark results."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Create directory for visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Extract data for plotting
        index_types = list(self.results.keys())
        
        # Plot indexing throughput
        index_throughputs = [self.results[idx_type]["index_throughput"] for idx_type in index_types]
        
        plt.figure(figsize=(10, 6))
        plt.bar(index_types, index_throughputs)
        plt.title("FAISS Indexing Throughput by Index Type")
        plt.xlabel("Index Type")
        plt.ylabel("Vectors per Second")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, "faiss_index_throughput.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot search QPS for different top_k values
        top_k_values = list(next(iter(self.results.values()))["search"].keys())
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(index_types))
        width = 0.8 / len(top_k_values)
        
        for i, k in enumerate(top_k_values):
            qps_values = [self.results[idx_type]["search"][k]["qps"] for idx_type in index_types]
            plt.bar(x + i * width - 0.4 + width/2, qps_values, width, label=f'top-{k}')
        
        plt.title("FAISS Search Performance (QPS) by Index Type and top-k")
        plt.xlabel("Index Type")
        plt.ylabel("Queries per Second")
        plt.xticks(x, index_types)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, "faiss_search_qps.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot search latency for different top_k values
        plt.figure(figsize=(12, 6))
        
        for i, k in enumerate(top_k_values):
            latency_values = [self.results[idx_type]["search"][k]["latency"] * 1000 for idx_type in index_types]
            plt.bar(x + i * width - 0.4 + width/2, latency_values, width, label=f'top-{k}')
        
        plt.title("FAISS Search Performance (Latency) by Index Type and top-k")
        plt.xlabel("Index Type")
        plt.ylabel("Latency (ms)")
        plt.xticks(x, index_types)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, "faiss_search_latency.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")

def main():
    """Main function to run the benchmark from command line."""
    parser = argparse.ArgumentParser(description="Simple Vector Database Benchmark")
    
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model to use")
    parser.add_argument("--dimension", type=int, default=384,
                        help="Embedding dimension")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset file (text file with one document per line)")
    parser.add_argument("--synthetic_size", type=int, default=10000,
                        help="Size of synthetic dataset")
    parser.add_argument("--query_size", type=int, default=100,
                        help="Number of queries")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Generate or load dataset
    if args.dataset:
        # Load real dataset
        logger.info(f"Loading dataset from {args.dataset}")
        with open(args.dataset, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        
        # Limit dataset size if needed
        if len(texts) > args.synthetic_size:
            texts = texts[:args.synthetic_size]
        
        # Generate embeddings
        model = EmbeddingModel(args.model, args.dimension)
        vectors = model.generate_embeddings(texts)
        
        # Generate queries (use a subset of the dataset)
        query_indices = np.random.choice(len(vectors), args.query_size, replace=False)
        queries = vectors[query_indices].copy()
        
        # Add some noise to queries
        noise = np.random.randn(*queries.shape) * 0.1
        queries += noise
        
        # Normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / norms
    else:
        # Generate synthetic dataset
        vectors, queries = generate_synthetic_dataset(args.synthetic_size, args.dimension, args.query_size)
    
    # Run FAISS benchmark
    benchmark = FAISSBenchmark(args.dimension, args.output_dir)
    benchmark.benchmark_index_types(
        vectors, 
        queries, 
        index_types=["Flat", "IVF", "HNSW"], 
        top_k_values=[1, 10, 100]
    )
    
    # Save and visualize results
    benchmark.save_results()
    benchmark.visualize_results()

if __name__ == "__main__":
    main()
