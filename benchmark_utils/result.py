from dataclasses import dataclass
from typing import Optional

@dataclass
class BenchmarkResult:
    database_name: str
    dataset_name: str
    operation: str
    vectors_count: int
    dimension: int
    elapsed_time: float
    throughput: float
    queries_per_second: Optional[float] = None
    recall: Optional[float] = None
    optimization_notes: Optional[str] = None
    error_message: Optional[str] = None
