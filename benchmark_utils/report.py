import json
import os
from typing import List
from benchmark_utils.result import BenchmarkResult

def save_results(results: List[BenchmarkResult], output_dir: str, prefix: str = "benchmark"):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    with open(json_path, 'w') as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    return json_path

def generate_summary_md(results: List[BenchmarkResult], output_dir: str, prefix: str = "benchmark"):
    md_path = os.path.join(output_dir, f"{prefix}_summary.md")
    with open(md_path, 'w') as f:
        f.write(f"# Benchmark Summary\n\n")
        datasets = set(r.dataset_name for r in results)
        for dataset in datasets:
            f.write(f"## Dataset: {dataset}\n\n")
            insert_results = [r for r in results if r.dataset_name == dataset and r.operation == "insert"]
            f.write("### Insertion Performance\n\n")
            f.write("| Database | Vectors/sec | Time (s) |\n")
            f.write("|----------|-------------|----------|\n")
            for result in insert_results:
                insert_time = result.vectors_count / result.throughput if result.throughput > 0 else 0
                f.write(f"| {result.database_name} | {result.throughput:.1f} | {insert_time:.3f} |\n")
            f.write("\n### Search Performance\n\n")
            search_results = [r for r in results if r.dataset_name == dataset and r.operation == "search"]
            for top_k in [1, 5, 10]:
                f.write(f"#### Top-{top_k} Search\n\n")
                f.write(f"| Database | QPS | Recall@{top_k} |\n")
                f.write("|----------|-----|--------|\n")
                for result in search_results:
                    if result.queries_per_second and result.recall is not None:
                        f.write(f"| {result.database_name} | {result.queries_per_second:.1f} | {result.recall:.1%} |\n")
            f.write("\n")
    return md_path
