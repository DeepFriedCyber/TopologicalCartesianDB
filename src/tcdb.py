from typing import List, Tuple

class TopologicalCartesianDB:
    """Minimal spatial database with TDD, validation, resource limits, and docstrings."""
    __slots__ = ['points']
    MAX_POINTS = 1000000
    MAX_RADIUS = 1e6
    DISTANCE_TOLERANCE = 1e-10

    def __init__(self):
        self.points = set()

    def insert(self, x: float, y: float):
        """Insert a unique point (x, y). Raises TypeError or RuntimeError on error."""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Coordinates must be numeric")
        if len(self.points) >= self.MAX_POINTS:
            raise RuntimeError("Database full")
        self.points.add((x, y))

    def query(self, x: float, y: float, radius: float) -> List[Tuple[float, float]]:
        """Find points within `radius` of (x, y). Raises ValueError for invalid radius."""
        if not isinstance(radius, (int, float)):
            raise TypeError("Radius must be numeric")
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        if radius > self.MAX_RADIUS:
            raise ValueError("Radius too large")
        result = []
        radius_sq = radius ** 2
        for px, py in self.points:
            dist_sq = (px - x) ** 2 + (py - y) ** 2
            if (dist_sq - radius_sq) <= self.DISTANCE_TOLERANCE:
                if radius == 0 and (px, py) != (x, y):
                    continue
                result.append((px, py))
        return result
