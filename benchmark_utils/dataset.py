import numpy as np
import h5py
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Loads synthetic and public datasets for benchmarking."""
    def __init__(self, cache_dir: str = "./benchmark_datasets"):
        self.cache_dir = cache_dir
    
    def load_hdf5(self, filepath: str, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load vectors, queries, and ground truth from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            train_vectors = np.array(f['train'])
            if sample_size and train_vectors.shape[0] > sample_size:
                indices = np.random.choice(train_vectors.shape[0], sample_size, replace=False)
                train_vectors = train_vectors[indices]
            test_queries = np.array(f['test'])
            if sample_size and test_queries.shape[0] > min(100, sample_size // 10):
                query_size = min(100, sample_size // 10)
                test_queries = test_queries[:query_size]
            ground_truth = None
            if 'neighbors' in f:
                neighbors_obj = f['neighbors']
                if isinstance(neighbors_obj, h5py.Dataset):
                    try:
                        ground_truth = np.array(neighbors_obj[:test_queries.shape[0]])
                    except Exception as e:
                        logger.warning(f"Could not load ground_truth from neighbors: {e}")
                        ground_truth = None
            return train_vectors, test_queries, ground_truth

    def generate_synthetic(self, size: int, dimension: int, query_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        vectors = np.random.randn(size, dimension).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        query_indices = np.random.choice(size, query_size, replace=False)
        queries = vectors[query_indices].copy()
        noise = np.random.randn(*queries.shape) * 0.1
        queries += noise
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / norms
        return vectors, queries
