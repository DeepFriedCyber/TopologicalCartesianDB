
"""
Compare different vector databases using the same dataset.

This script benchmarks multiple vector databases (FAISS, Qdrant, etc.)
using the same dataset and compares their performance.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
import requests

# --- Public dataset loader for .fvecs files ---
def read_fvecs(filename):
    """Read .fvecs file format used by ANN benchmarks (float32 vectors)."""
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(-1, dim + 1)
        return data[:, 1:].astype(np.float32)

def download_file(url, dest_path):
    """Download a file from a URL to a local path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path
    """Read .fvecs file format used by ANN benchmarks (float32 vectors)."""
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(-1, dim + 1)
        return data[:, 1:].astype(np.float32)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorDBComparison")

# Constants
OUTPUT_DIR = "comparison_results"
SAMPLE_DATA_DIR = "sample_data"

def load_or_generate_data(dimension=256, num_vectors=10000, num_queries=100):
    """Load existing sample data or generate new data."""
    # Check if sample data exists
    vector_file = os.path.join(SAMPLE_DATA_DIR, f"sample_vectors_{dimension}d.npy")
    query_file = os.path.join(SAMPLE_DATA_DIR, f"sample_queries_{dimension}d.npy")
    
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    
    if os.path.exists(vector_file) and os.path.exists(query_file):
        logger.info(f"Loading existing sample data from {vector_file}")
        vectors = np.load(vector_file)
        queries = np.load(query_file)
        
        # Verify dimensions
        if vectors.shape[1] != dimension or len(vectors) < num_vectors or len(queries) < num_queries:
            logger.info("Existing data doesn't match requested dimensions, generating new data")
            vectors, queries = generate_data(dimension, num_vectors, num_queries)
            np.save(vector_file, vectors)
            np.save(query_file, queries)
    else:
        logger.info("Generating new sample data")
        vectors, queries = generate_data(dimension, num_vectors, num_queries)
        np.save(vector_file, vectors)
        np.save(query_file, queries)
    
    return vectors[:num_vectors], queries[:num_queries]

def generate_data(dimension, num_vectors, num_queries):
    """Generate synthetic vector data."""
    # Generate random vectors
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # Generate query vectors
    query_indices = np.random.choice(num_vectors, num_queries, replace=False)
    queries = vectors[query_indices].copy()
    
    # Add some noise to queries
    noise = np.random.randn(*queries.shape) * 0.1
    queries += noise
    
    # Normalize queries
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / norms
    
    logger.info(f"Generated {num_vectors} vectors and {num_queries} queries with dimension {dimension}")
    return vectors, queries

def benchmark_faiss(vectors, queries, top_k_values):
    """Benchmark FAISS with different index types."""
    try:
        import faiss
        logger.info("Benchmarking FAISS")
        
        results = {}
        dimension = vectors.shape[1]
        
        # Test different index types
        index_types = {
            "Flat": faiss.IndexFlatIP(dimension),
            "IVF": faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, 100, faiss.METRIC_INNER_PRODUCT),
            "HNSW": faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
        }
        
        for name, index in index_types.items():
            logger.info(f"Testing FAISS {name} index")
            
            # Train if needed
            if hasattr(index, 'train'):
                logger.info(f"Training {name} index")
                index.train(vectors)
            
            # Add vectors
            start_time = time.time()
            index.add(vectors)
            index_time = time.time() - start_time
            
            # Test search with different k values
            search_results = {}
            for k in top_k_values:
                logger.info(f"Testing search with top_k={k}")
                
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
            
            results[f"FAISS_{name}"] = {
                "index_time": index_time,
                "index_throughput": len(vectors) / index_time,
                "search": search_results
            }
        
        return results
    except ImportError:
        logger.error("FAISS not available. Install with: pip install faiss-cpu")
        return {}

def benchmark_qdrant(vectors, queries, top_k_values):
    """Benchmark Qdrant vector database."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
    except ImportError:
        logger.error("Qdrant client not available. Install with: pip install qdrant-client")
        return {}
    logger.info("Benchmarking Qdrant")
    # Connect to Qdrant Docker service (default: localhost:6333)
    client = QdrantClient(host="localhost", port=6333)
    # Create collection
    dimension = vectors.shape[1]
    collection_name = "benchmark_collection"
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE
        )
    )
    # Insert vectors
    start_time = time.time()
    batch_size = 1000
    for i in tqdm(range(0, len(vectors), batch_size), desc="Inserting into Qdrant"):
        end_idx = min(i + batch_size, len(vectors))
        batch_vectors = vectors[i:end_idx]
        points = [
            PointStruct(
                id=j + i,
                vector=vector.tolist(),
                payload={"id": j + i}
            )
            for j, vector in enumerate(batch_vectors)
        ]
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    index_time = time.time() - start_time
    # Test search with different k values
    search_results = {}
    for k in top_k_values:
        logger.info(f"Testing search with top_k={k}")
        start_time = time.time()
        for query in tqdm(queries, desc=f"Searching with top_k={k}"):
            _ = client.search(
                collection_name=collection_name,
                query_vector=query.tolist(),
                limit=k
            )
        search_time = time.time() - start_time
        qps = len(queries) / search_time
        latency = search_time / len(queries)
        search_results[k] = {
            "time": search_time,
            "qps": qps,
            "latency": latency
        }
        logger.info(f"Search with top_k={k}: QPS={qps:.2f}, Latency={latency*1000:.2f}ms")
    results = {
        "Qdrant": {
            "index_time": index_time,
            "index_throughput": len(vectors) / index_time,
            "search": search_results
        }
    }
    return results

def benchmark_topological_cartesian_db(vectors, queries, top_k_values):
    """Benchmark Topological-Cartesian-DB."""
    # Import TCDB client
    import time
    try:
        from benchmarks.vectordb.tcdb_client import TCDBClient, ConnectionConfig
    except ImportError:
        logger.error("TCDB client not found. Skipping TCDB benchmark.")
        return {}

    # Setup connection config (adjust host/port as needed)
    config = ConnectionConfig(host="localhost", port=8000)
    client = TCDBClient(config)
    collection_name = "benchmark_collection"
    dimension = len(vectors[0]) if len(vectors) > 0 else 0

    # Create collection
    client.create_collection(collection_name, dimension=dimension)

    # Insert vectors
    points = [
        {"id": i, "vector": v.tolist() if hasattr(v, 'tolist') else list(v), "metadata": {}} for i, v in enumerate(vectors)
    ]
    start_index = time.time()
    client.bulk_insert(collection_name, points)
    index_time = time.time() - start_index

    search_results = {}
    for k in top_k_values:
        start = time.time()
        queries_list = [q.tolist() if hasattr(q, 'tolist') else list(q) for q in queries]
        client.batch_search(collection_name, queries_list, top_k=k)
        search_time = time.time() - start
        qps = len(queries) / search_time if search_time > 0 else 0
        latency = search_time / len(queries) if len(queries) > 0 else 0
        search_results[k] = {"time": search_time, "qps": qps, "latency": latency}

    # Optionally drop collection to clean up
    try:
        client.drop_collection(collection_name)
    except Exception:
        pass

    return {
        "TopologicalCartesianDB": {
            "index_time": index_time,
            "index_throughput": len(vectors) / index_time if index_time > 0 else 0,
            "search": search_results
        }
    }

def benchmark_weaviate(vectors, queries, top_k_values):
    """Benchmark Weaviate vector database (Docker)."""
    import requests
    import uuid
    logger.info("Benchmarking Weaviate (REST API)")
    candidate_ports = globals().get('WEAVIATE_PORTS', [8081, 8082, 8080])
    base_url = None
    for port in candidate_ports:
        try:
            url = f"http://localhost:{port}/v1/.well-known/ready"
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                base_url = f"http://localhost:{port}"
                logger.info(f"Using Weaviate at {base_url}")
                break
        except Exception:
            continue
    if base_url is None:
        logger.error(f"Could not connect to Weaviate on any of ports {candidate_ports}. Skipping Weaviate benchmark.")
        return {}
    class_name = "BenchmarkObject"
    # Remove class if exists
    schema_url = f"{base_url}/v1/schema/{class_name}"
    requests.delete(schema_url)
    # Create class
    dim = vectors.shape[1]
    class_obj = {
        "class": class_name,
        "vectorIndexType": "hnsw",
        "vectorizer": "none",
        "properties": []
    }
    requests.post(f"{base_url}/v1/schema", json=class_obj)
    # Insert vectors
    start_time = time.time()
    for v in vectors:
        obj = {
            "class": class_name,
            "vector": v.tolist(),
            "id": str(uuid.uuid4()),
            "properties": {}
        }
        requests.post(f"{base_url}/v1/objects", json=obj)
    index_time = time.time() - start_time
    # Search
    search_results = {}
    for k in top_k_values:
        logger.info(f"Testing search with top_k={k}")
        start_time = time.time()
        for q in queries:
            search_body = {
                "nearVector": {"vector": q.tolist()},
                "limit": k
            }
            requests.post(f"{base_url}/v1/objects/{class_name}/search", json=search_body)
        search_time = time.time() - start_time
        qps = len(queries) / search_time
        latency = search_time / len(queries)
        search_results[k] = {"time": search_time, "qps": qps, "latency": latency}
        logger.info(f"Search with top_k={k}: QPS={qps:.2f}, Latency={latency*1000:.2f}ms")
    results = {
        "Weaviate": {
            "index_time": index_time,
            "index_throughput": len(vectors) / index_time,
            "search": search_results
        }
    }
    return results

def save_and_visualize_results(results, output_dir=OUTPUT_DIR):
    """Save results to file and generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON
    result_file = os.path.join(output_dir, "comparison_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {result_file}")

    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    if not results:
        logger.warning("No benchmark results to visualize.")
        return

    db_names = list(results.keys())
    if not db_names:
        logger.warning("No database results to visualize.")
        return

    # Plot indexing throughput
    index_throughputs = [results[db].get("index_throughput", 0) for db in db_names]

    plt.figure(figsize=(10, 6))
    plt.bar(db_names, index_throughputs)
    plt.title("Vector Database Indexing Throughput Comparison")
    plt.xlabel("Database")
    plt.ylabel("Vectors per Second")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig(os.path.join(viz_dir, "indexing_throughput.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot search QPS for different top_k values
    # Get all top_k values from the first database with a 'search' key
    first_db = next((v for v in results.values() if "search" in v and v["search"]), None)
    if not first_db:
        logger.warning("No search results to visualize.")
        return
    top_k_values = list(first_db["search"].keys())
    if not top_k_values:
        logger.warning("No top_k values found in search results.")
        return

    plt.figure(figsize=(12, 6))

    x = np.arange(len(db_names))
    width = 0.8 / len(top_k_values)

    for i, k in enumerate(top_k_values):
        qps_values = [results[db].get("search", {}).get(k, {}).get("qps", 0) for db in db_names]
        plt.bar(x + i * width - 0.4 + width/2, qps_values, width, label=f'top-{k}')

    plt.title("Vector Database Search Performance (QPS) Comparison")
    plt.xlabel("Database")
    plt.ylabel("Queries per Second")
    plt.xticks(x, db_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig(os.path.join(viz_dir, "search_qps.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot search latency for different top_k values
    plt.figure(figsize=(12, 6))

    for i, k in enumerate(top_k_values):
        latency_values = [results[db].get("search", {}).get(k, {}).get("latency", 0) * 1000 for db in db_names]  # Convert to ms
        plt.bar(x + i * width - 0.4 + width/2, latency_values, width, label=f'top-{k}')

    plt.title("Vector Database Search Performance (Latency) Comparison")
    plt.xlabel("Database")
    plt.ylabel("Latency (ms)")
    plt.xticks(x, db_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig(os.path.join(viz_dir, "search_latency.png"), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualizations saved to {viz_dir}")

def main():
    """Main function to run the comparison benchmark."""

    parser = argparse.ArgumentParser(description="Vector Database Comparison Benchmark")
    parser.add_argument("--dimension", type=int, default=256,
                        help="Vector dimension (default: 256)")
    parser.add_argument("--num_vectors", type=int, default=10000,
                        help="Number of vectors (default: 10000)")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Number of queries (default: 100)")
    parser.add_argument("--top_k", type=str, default="1,10,100",
                        help="Comma-separated list of top-k values (default: 1,10,100)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory (default: comparison_results)")
    parser.add_argument("--public_data_path", type=str, default=None,
                        help="Path to public dataset directory (expects sift_base.fvecs and sift_query.fvecs)")

    args = parser.parse_args()

    # Parse top_k values
    top_k_values = [int(k) for k in args.top_k.split(",")]

    # Load public data if specified, else synthetic
    if args.public_data_path:
        base_path = os.path.join(args.public_data_path, "sift_base.fvecs")
        query_path = os.path.join(args.public_data_path, "sift_query.fvecs")
        base_url = "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/"
        # Download if missing
        if not os.path.exists(base_path):
            logger.info(f"Downloading {base_path} from HuggingFace...")
            download_file(base_url + "sift_base.fvecs", base_path)
        if not os.path.exists(query_path):
            logger.info(f"Downloading {query_path} from HuggingFace...")
            download_file(base_url + "sift_query.fvecs", query_path)
        vectors = read_fvecs(base_path)
        queries = read_fvecs(query_path)
        if vectors is None or queries is None:
            logger.error("Failed to load vectors or queries from .fvecs files.")
            return
        vectors = vectors[:args.num_vectors]
        queries = queries[:args.num_queries]
        logger.info(f"Loaded public dataset: {base_path} ({vectors.shape}), {query_path} ({queries.shape})")
    else:
        vectors, queries = load_or_generate_data(args.dimension, args.num_vectors, args.num_queries)

    # Run benchmarks
    results = {}

    # Benchmark FAISS
    faiss_results = benchmark_faiss(vectors, queries, top_k_values)
    results.update(faiss_results)

    # Benchmark Qdrant
    qdrant_results = benchmark_qdrant(vectors, queries, top_k_values)
    results.update(qdrant_results)

    # Benchmark Weaviate (try ports 8083, 8081, 8082, 8080)
    global WEAVIATE_PORTS
    WEAVIATE_PORTS = [8083, 8081, 8082, 8080]
    weaviate_results = benchmark_weaviate(vectors, queries, top_k_values)
    results.update(weaviate_results)

    # Benchmark Topological-Cartesian-DB
    tcdb_results = benchmark_topological_cartesian_db(vectors, queries, top_k_values)
    results.update(tcdb_results)

    # Save and visualize results
    save_and_visualize_results(results, args.output_dir)

    logger.info("Comparison benchmark completed")

if __name__ == "__main__":
    main()
