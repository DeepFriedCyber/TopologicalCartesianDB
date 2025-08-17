"""
Run a comprehensive benchmark of vector databases with multiple configurations.

This script benchmarks vector databases with different:
- Vector dimensions
- Dataset sizes
- Query counts
- Index types

The results are saved and visualized for easy comparison.
"""

import os
import subprocess
import logging
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComprehensiveBenchmark")

# Benchmark configurations
DIMENSIONS = [128, 256, 512]
DATASET_SIZES = [1000, 10000, 100000]
QUERY_COUNTS = [10, 100]
TOP_K_VALUES = [1, 10, 100]

# Output directory
BASE_OUTPUT_DIR = "comprehensive_benchmark_results"

def run_benchmark(dimension, num_vectors, num_queries, top_k_values, output_dir):
    """Run a single benchmark with the given configuration."""
    logger.info(f"Running benchmark: dim={dimension}, vectors={num_vectors}, queries={num_queries}")
    
    # Convert top_k_values to comma-separated string
    top_k_str = ",".join(map(str, top_k_values))
    
    # Build command
    cmd = [
        "python", "vectordb-benchmark/compare_vector_dbs.py",
        "--dimension", str(dimension),
        "--num_vectors", str(num_vectors),
        "--num_queries", str(num_queries),
        "--top_k", top_k_str,
        "--output_dir", output_dir
    ]
    
    # Run benchmark
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Benchmark completed successfully: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed: {e}")
        return False

def collect_results(base_output_dir):
    """Collect and aggregate results from all benchmarks."""
    logger.info("Collecting benchmark results")
    
    aggregated_results = {
        "dimensions": {},
        "dataset_sizes": {},
        "query_counts": {}
    }
    
    # Collect results by dimension
    for dimension in DIMENSIONS:
        dimension_results = {"index_throughput": {}, "search_qps": {}, "search_latency": {}}
        
        for num_vectors in DATASET_SIZES:
            for num_queries in QUERY_COUNTS:
                output_dir = os.path.join(base_output_dir, f"dim_{dimension}_vec_{num_vectors}_q_{num_queries}")
                result_file = os.path.join(output_dir, "comparison_results.json")
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    # Extract metrics
                    for db_name, db_results in results.items():
                        # Index throughput
                        if db_name not in dimension_results["index_throughput"]:
                            dimension_results["index_throughput"][db_name] = []
                        dimension_results["index_throughput"][db_name].append(db_results["index_throughput"])
                        
                        # Search QPS (use top_k=10)
                        if "10" in db_results["search"]:
                            if db_name not in dimension_results["search_qps"]:
                                dimension_results["search_qps"][db_name] = []
                            dimension_results["search_qps"][db_name].append(db_results["search"]["10"]["qps"])
                        
                        # Search latency (use top_k=10)
                        if "10" in db_results["search"]:
                            if db_name not in dimension_results["search_latency"]:
                                dimension_results["search_latency"][db_name] = []
                            dimension_results["search_latency"][db_name].append(db_results["search"]["10"]["latency"] * 1000)  # Convert to ms
        
        # Calculate averages
        for metric, db_values in dimension_results.items():
            for db_name, values in db_values.items():
                if values:
                    db_values[db_name] = sum(values) / len(values)
        
        aggregated_results["dimensions"][dimension] = dimension_results
    
    # Collect results by dataset size
    for num_vectors in DATASET_SIZES:
        size_results = {"index_throughput": {}, "search_qps": {}, "search_latency": {}}
        
        for dimension in DIMENSIONS:
            for num_queries in QUERY_COUNTS:
                output_dir = os.path.join(base_output_dir, f"dim_{dimension}_vec_{num_vectors}_q_{num_queries}")
                result_file = os.path.join(output_dir, "comparison_results.json")
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    # Extract metrics
                    for db_name, db_results in results.items():
                        # Index throughput
                        if db_name not in size_results["index_throughput"]:
                            size_results["index_throughput"][db_name] = []
                        size_results["index_throughput"][db_name].append(db_results["index_throughput"])
                        
                        # Search QPS (use top_k=10)
                        if "10" in db_results["search"]:
                            if db_name not in size_results["search_qps"]:
                                size_results["search_qps"][db_name] = []
                            size_results["search_qps"][db_name].append(db_results["search"]["10"]["qps"])
                        
                        # Search latency (use top_k=10)
                        if "10" in db_results["search"]:
                            if db_name not in size_results["search_latency"]:
                                size_results["search_latency"][db_name] = []
                            size_results["search_latency"][db_name].append(db_results["search"]["10"]["latency"] * 1000)  # Convert to ms
        
        # Calculate averages
        for metric, db_values in size_results.items():
            for db_name, values in db_values.items():
                if values:
                    db_values[db_name] = sum(values) / len(values)
        
        aggregated_results["dataset_sizes"][num_vectors] = size_results
    
    # Collect results by query count
    for num_queries in QUERY_COUNTS:
        query_results = {"search_qps": {}, "search_latency": {}}
        
        for dimension in DIMENSIONS:
            for num_vectors in DATASET_SIZES:
                output_dir = os.path.join(base_output_dir, f"dim_{dimension}_vec_{num_vectors}_q_{num_queries}")
                result_file = os.path.join(output_dir, "comparison_results.json")
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    # Extract metrics
                    for db_name, db_results in results.items():
                        # Search QPS (use top_k=10)
                        if "10" in db_results["search"]:
                            if db_name not in query_results["search_qps"]:
                                query_results["search_qps"][db_name] = []
                            query_results["search_qps"][db_name].append(db_results["search"]["10"]["qps"])
                        
                        # Search latency (use top_k=10)
                        if "10" in db_results["search"]:
                            if db_name not in query_results["search_latency"]:
                                query_results["search_latency"][db_name] = []
                            query_results["search_latency"][db_name].append(db_results["search"]["10"]["latency"] * 1000)  # Convert to ms
        
        # Calculate averages
        for metric, db_values in query_results.items():
            for db_name, values in db_values.items():
                if values:
                    db_values[db_name] = sum(values) / len(values)
        
        aggregated_results["query_counts"][num_queries] = query_results
    
    return aggregated_results

def visualize_aggregated_results(aggregated_results, base_output_dir):
    """Generate visualizations for aggregated results."""
    logger.info("Generating visualizations for aggregated results")
    
    # Create directory for visualizations
    viz_dir = os.path.join(base_output_dir, "aggregated_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize results by dimension
    visualize_by_parameter(
        aggregated_results["dimensions"],
        DIMENSIONS,
        "Dimension",
        os.path.join(viz_dir, "by_dimension")
    )
    
    # Visualize results by dataset size
    visualize_by_parameter(
        aggregated_results["dataset_sizes"],
        DATASET_SIZES,
        "Dataset Size",
        os.path.join(viz_dir, "by_dataset_size")
    )
    
    # Visualize results by query count
    visualize_by_parameter(
        aggregated_results["query_counts"],
        QUERY_COUNTS,
        "Query Count",
        os.path.join(viz_dir, "by_query_count"),
        metrics=["search_qps", "search_latency"]  # Only these metrics are relevant for query count
    )
    
    logger.info(f"Visualizations saved to {viz_dir}")

def visualize_by_parameter(results, parameter_values, parameter_name, output_dir, 
                          metrics=["index_throughput", "search_qps", "search_latency"]):
    """Generate visualizations for results by a specific parameter."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all database names
    all_db_names = set()
    for param_value in parameter_values:
        if param_value in results:
            for metric in metrics:
                if metric in results[param_value]:
                    all_db_names.update(results[param_value][metric].keys())
    
    db_names = sorted(list(all_db_names))
    
    # Generate plots for each metric
    for metric in metrics:
        # Set plotting variables only for known metrics
        if metric == "index_throughput":
            title = f"Indexing Throughput by {parameter_name}"
            ylabel = "Vectors per Second"
            filename = f"indexing_throughput_by_{parameter_name.lower().replace(' ', '_')}.png"
            log_scale = True
        elif metric == "search_qps":
            title = f"Search Performance (QPS) by {parameter_name}"
            ylabel = "Queries per Second"
            filename = f"search_qps_by_{parameter_name.lower().replace(' ', '_')}.png"
            log_scale = True
        elif metric == "search_latency":
            title = f"Search Performance (Latency) by {parameter_name}"
            ylabel = "Latency (ms)"
            filename = f"search_latency_by_{parameter_name.lower().replace(' ', '_')}.png"
            log_scale = True
        else:
            # Skip unknown metrics
            continue

        plt.figure(figsize=(12, 6))

        # Prepare data for plotting
        x = np.arange(len(parameter_values))
        width = 0.8 / len(db_names) if db_names else 0.8

        # Collect all values for all dbs for this metric
        all_values = []
        for i, db_name in enumerate(db_names):
            values = []
            for param_value in parameter_values:
                if param_value in results and metric in results[param_value] and db_name in results[param_value][metric]:
                    values.append(results[param_value][metric][db_name])
                else:
                    values.append(0)  # Use 0 for missing values
            all_values.extend(values)
            plt.bar(x + i * width - 0.4 + width/2, values, width, label=db_name)

        plt.title(title)
        plt.xlabel(parameter_name)
        plt.ylabel(ylabel)
        plt.xticks(x, parameter_values)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Only set log scale if all values are positive and not empty
        if log_scale and all(v > 0 for v in all_values):
            plt.yscale('log')

        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        if not db_names:
            logger.warning(f"No results found for {parameter_name}. Skipping plot.")
            continue

def main():
    """Main function to run comprehensive benchmarks."""
    logger.info("Starting comprehensive benchmark")
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{BASE_OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run benchmarks for all configurations
    for dimension, num_vectors, num_queries in itertools.product(DIMENSIONS, DATASET_SIZES, QUERY_COUNTS):
        benchmark_output_dir = os.path.join(output_dir, f"dim_{dimension}_vec_{num_vectors}_q_{num_queries}")
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        run_benchmark(dimension, num_vectors, num_queries, TOP_K_VALUES, benchmark_output_dir)
    
    # Collect and visualize aggregated results
    aggregated_results = collect_results(output_dir)
    
    # Save aggregated results
    aggregated_file = os.path.join(output_dir, "aggregated_results.json")
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Visualize aggregated results
    visualize_aggregated_results(aggregated_results, output_dir)
    
    logger.info(f"Comprehensive benchmark completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
