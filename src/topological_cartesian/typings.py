from typing import Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class TopologicalFeature:
    """Represents a topological feature with enhanced analysis."""
    dimension: int
    persistence: float
    coordinates: List[float]
    confidence: float = 1.0
    backend: str = "Unknown"
    coordinate_analysis: Dict[str, Any] = field(default_factory=dict)
