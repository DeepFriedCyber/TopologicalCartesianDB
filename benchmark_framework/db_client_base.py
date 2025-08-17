from typing import Any, Tuple
import numpy as np

class DBClientBase:
    """Base interface for database clients."""
    def create_collection(self, name: str, dimension: int) -> bool:
        raise NotImplementedError
    def bulk_insert(self, vectors: np.ndarray) -> Tuple[bool, float]:
        raise NotImplementedError
    def batch_search(self, queries: np.ndarray, top_k: int) -> Tuple[list, float]:
        raise NotImplementedError
    def cleanup(self):
        """Optional cleanup after test."""
        pass
