import numpy as np
from benchmark_framework.db_client_base import DBClientBase
from benchmarks.vectordb.tcdb_client import TCDBClient, ConnectionConfig

class TCDBBenchmarkClient(DBClientBase):
    def __init__(self, config: dict):
        self.config = config
        self.client = TCDBClient(ConnectionConfig(**config))
        self.collection_name = None
    def create_collection(self, name: str, dimension: int) -> bool:
        self.collection_name = name
        return self.client.create_collection(name, dimension)
    def bulk_insert(self, vectors: np.ndarray) -> tuple:
        # TCDB expects a list of dicts: {"id": i, "vector": vector.tolist()}
        points = [
            {"id": i, "vector": vector.tolist() if hasattr(vector, 'tolist') else vector} for i, vector in enumerate(vectors)
        ]
        if not isinstance(self.collection_name, str):
            raise ValueError("collection_name must be a string before bulk_insert")
        result = self.client.bulk_insert(self.collection_name, points)
        # If result is bool, return (result, 0.0)
        if isinstance(result, tuple):
            return result
        elif isinstance(result, bool):
            return result, 0.0
        else:
            return False, 0.0
    def batch_search(self, queries: np.ndarray, top_k: int) -> tuple:
        # TCDB expects queries as list of lists
        if not isinstance(self.collection_name, str):
            raise ValueError("collection_name must be a string before batch_search")
        if isinstance(queries, np.ndarray):
            queries_list = queries.tolist()
        elif isinstance(queries, list):
            queries_list = queries
        else:
            queries_list = list(queries)
        result = self.client.batch_search(self.collection_name, queries_list, top_k)
        if isinstance(result, tuple):
            return result
        elif isinstance(result, list):
            # Assume only results, no qps
            return result, 0.0
        else:
            return [], 0.0
    def cleanup(self):
        if self.collection_name:
            self.client.drop_collection(self.collection_name)
