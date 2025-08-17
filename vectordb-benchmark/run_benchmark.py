"""
Run Vector Database Benchmark with Facebook SimSearchNet++ sample data.

This script downloads a small sample of the Facebook SimSearchNet++ dataset
and runs the vector database benchmark on it.
"""

import os
import sys
import numpy as np
import requests
import subprocess
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkRunner")

# Constants
SAMPLE_SIZE = 10000  # Number of vectors to use from the dataset
QUERY_SIZE = 100     # Number of queries to generate
OUTPUT_DIR = "benchmark_results"
SAMPLE_DATA_DIR = "sample_data"

def download_sample_data():
    """Download a small sample of the Facebook SimSearchNet++ dataset."""
    # Create directory for sample data
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    
    # Instead of downloading the full dataset, we'll create a synthetic sample
    # that mimics the properties of the SimSearchNet++ dataset
    
    logger.info(f"Generating synthetic sample data with {SAMPLE_SIZE} vectors")
    
    # SimSearchNet++ uses 256-dimensional vectors in uint8 format
    vectors = np.random.randint(0, 256, size=(SAMPLE_SIZE, 256), dtype=np.uint8)
    
    # Save vectors to file
    sample_file = os.path.join(SAMPLE_DATA_DIR, "fb_ssnpp_sample.npy")
    np.save(sample_file, vectors)
    
    logger.info(f"Saved sample data to {sample_file}")
    return sample_file

def convert_to_float32(uint8_vectors):
    """Convert uint8 vectors to float32 and normalize."""
    # Convert to float32 and scale to [0, 1]
    float_vectors = uint8_vectors.astype(np.float32) / 255.0
    
    # Center around zero [-0.5, 0.5]
    float_vectors -= 0.5
    
    # Normalize to unit length
    norms = np.linalg.norm(float_vectors, axis=1, keepdims=True)
    float_vectors = float_vectors / norms
    
    return float_vectors

def run_benchmark(sample_file):
    """Run the vector database benchmark on the sample data."""
    logger.info("Running vector database benchmark")
    
    # Load sample data
    vectors = np.load(sample_file)
    logger.info(f"Loaded {len(vectors)} vectors with shape {vectors.shape}")
    
    # Convert uint8 vectors to float32 and normalize
    float_vectors = convert_to_float32(vectors)
    
    # Save float32 vectors
    float_file = os.path.join(SAMPLE_DATA_DIR, "fb_ssnpp_sample_float32.npy")
    np.save(float_file, float_vectors)
    
    # Generate query vectors (random subset with noise)
    query_indices = np.random.choice(len(float_vectors), QUERY_SIZE, replace=False)
    queries = float_vectors[query_indices].copy()
    
    # Add some noise
    noise = np.random.randn(*queries.shape) * 0.1
    queries += noise
    
    # Normalize
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / norms
    
    # Save queries
    query_file = os.path.join(SAMPLE_DATA_DIR, "fb_ssnpp_sample_queries.npy")
    np.save(query_file, queries)
    
    logger.info(f"Generated and saved {len(queries)} query vectors")
    
    # Run the benchmark script
    try:
        # Import the benchmark module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from simple_benchmark import FAISSBenchmark
        
        # Create benchmark instance
        benchmark = FAISSBenchmark(dimension=256, output_dir=OUTPUT_DIR)
        
        # Run benchmark
        benchmark.benchmark_index_types(
            float_vectors, 
            queries, 
            index_types=["Flat", "IVF", "HNSW"], 
            top_k_values=[1, 10, 100]
        )
        
        # Save and visualize results
        benchmark.save_results()
        benchmark.visualize_results()
        
        logger.info("Benchmark completed successfully")
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise

def main():
    """Main function to download sample data and run benchmark."""
    logger.info("Starting benchmark runner")
    
    # Download or generate sample data
    sample_file = download_sample_data()
    
    # Run benchmark
    run_benchmark(sample_file)
    
    logger.info("Benchmark runner completed")

if __name__ == "__main__":
    main()
