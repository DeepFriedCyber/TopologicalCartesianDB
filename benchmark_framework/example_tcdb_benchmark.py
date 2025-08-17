import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark_framework.tcdb_client import TCDBBenchmarkClient
from benchmark_framework.runner import BenchmarkRunner
from benchmark_utils.dataset import DatasetLoader

# Example config for TCDB
TCDB_CONFIG = {
    "host": "localhost",
    "port": 8080,
    # Add other required TCDB connection parameters here
}

def main():
    dataset_name = "sift-small"
    dataset_path = os.path.join("benchmark_datasets", "sift-128-euclidean.hdf5")
    sample_size = 1000
    output_dir = "tcdb_benchmark_results"
    db_client = TCDBBenchmarkClient(TCDB_CONFIG)
    runner = BenchmarkRunner(db_client, DatasetLoader())
    runner.run(dataset_name, dataset_path, sample_size)
    runner.save(output_dir, prefix=dataset_name)
    print(f"Benchmark for {dataset_name} complete. Results saved to {output_dir}.")

if __name__ == "__main__":
    main()
