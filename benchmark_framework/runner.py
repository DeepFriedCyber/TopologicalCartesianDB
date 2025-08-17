from typing import Optional
import time
from benchmark_utils.result import BenchmarkResult
from benchmark_utils.dataset import DatasetLoader
from benchmark_utils.report import save_results, generate_summary_md

class BenchmarkRunner:
    def __init__(self, db_client, dataset_loader=None):
        self.db_client = db_client
        self.dataset_loader = dataset_loader or DatasetLoader()
        self.results = []

    def run(self, dataset_name: str, dataset_path: str, sample_size: Optional[int] = None, top_k_list=[1, 5, 10]):
        # Load dataset
        train_vectors, test_queries, ground_truth = self.dataset_loader.load_hdf5(dataset_path, sample_size)
        dimension = train_vectors.shape[1]
        collection_name = f"{dataset_name}_benchmark_{int(time.time())}"
        self.db_client.create_collection(collection_name, dimension)
        # Insert vectors
        insert_start = time.time()
        insert_success, insert_throughput = self.db_client.bulk_insert(train_vectors.tolist())
        insert_time = time.time() - insert_start
        self.results.append(BenchmarkResult(
            database_name=type(self.db_client).__name__,
            dataset_name=dataset_name,
            operation="insert",
            vectors_count=len(train_vectors),
            dimension=dimension,
            elapsed_time=insert_time,
            throughput=insert_throughput,
            optimization_notes="Standard insert"
        ))
        # Search
        for top_k in top_k_list:
            search_start = time.time()
            search_results, qps = self.db_client.batch_search(test_queries.tolist(), top_k)
            search_time = time.time() - search_start
            recall = None
            if ground_truth is not None:
                recall = self._calculate_recall(search_results, ground_truth, top_k)
            self.results.append(BenchmarkResult(
                database_name=type(self.db_client).__name__,
                dataset_name=dataset_name,
                operation="search",
                vectors_count=len(train_vectors),
                dimension=dimension,
                elapsed_time=search_time,
                throughput=qps,
                queries_per_second=qps,
                recall=recall,
                optimization_notes=f"Top-{top_k} search"
            ))
        self.db_client.cleanup()

    def _calculate_recall(self, search_results, ground_truth, top_k):
        # Simple recall@k calculation
        correct = 0
        total = len(search_results)
        for i, result in enumerate(search_results):
            hits = result.get('hits', [])
            gt = set(ground_truth[i][:top_k])
            retrieved = set(h['id'] for h in hits[:top_k])
            correct += len(gt & retrieved) > 0
        return correct / total if total > 0 else None

    def save(self, output_dir: str, prefix: str = "benchmark"):
        save_results(self.results, output_dir, prefix)
        generate_summary_md(self.results, output_dir, prefix)
