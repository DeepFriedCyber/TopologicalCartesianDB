"""
Core module for TopologicalCartesianDB with minimal functionality.
This module provides the foundational spatial database capabilities.
"""
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional, Union

class TopologicalCartesianDB:
    """
    Core 2D spatial database with minimal dependencies.
    Provides efficient spatial indexing and querying capabilities.
    """
    
    def __init__(self, cell_size: float = 1.0):
        """
        Initialize an empty database with spatial indexing.
        
        Args:
            cell_size: Size of cells for spatial grid indexing
        """
        self.points: Dict[Tuple[float, float], Any] = {}  # Points with associated data
        self.grid = defaultdict(list)  # Grid-based spatial index
        self.cell_size = cell_size
    
    def insert(self, x: float, y: float, data: Any = None) -> None:
        """
        Insert a 2D point with optional associated data.
        
        Args:
            x: X coordinate
            y: Y coordinate
            data: Optional data to associate with this point
        
        Raises:
            TypeError: If coordinates are not numeric
        """
        # Input validation
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise TypeError("Coordinates must be numeric")
        
        # Skip if point already exists
        if (x, y) in self.points:
            return
        
        # Store point and data
        self.points[(x, y)] = data
        
        # Add to spatial index
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        self.grid[(cell_x, cell_y)].append((x, y))
    
    def query(self, x: float, y: float, radius: float) -> List[Tuple[float, float]]:
        """
        Query points within a specified radius of the given coordinates.
        
        Args:
            x: X coordinate of query center
            y: Y coordinate of query center
            radius: Radius to search within
            
        Returns:
            List of points within the radius
            
        Raises:
            ValueError: If radius is negative
            TypeError: If coordinates are not numeric
        """
        # Input validation
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise TypeError("Coordinates must be numeric")
        if not isinstance(radius, (int, float)):
            raise TypeError("Radius must be numeric")
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        
        # Determine grid cells to search
        min_x = int((x - radius) / self.cell_size)
        max_x = int((x + radius) / self.cell_size)
        min_y = int((y - radius) / self.cell_size)
        max_y = int((y + radius) / self.cell_size)
        
        # Search cells
        results = []
        for cell_x in range(min_x, max_x + 1):
            for cell_y in range(min_y, max_y + 1):
                for point in self.grid.get((cell_x, cell_y), []):
                    px, py = point
                    if (px - x)**2 + (py - y)**2 <= radius**2:
                        results.append(point)
        
        return results
    
    def query_with_data(self, x: float, y: float, radius: float) -> List[Tuple[Tuple[float, float], Any]]:
        """
        Query points and their associated data within a specified radius.
        
        Args:
            x: X coordinate of query center
            y: Y coordinate of query center
            radius: Radius to search within
            
        Returns:
            List of (point, data) tuples within the radius
        """
        points = self.query(x, y, radius)
        return [(point, self.points.get(point)) for point in points]
    
    def get_all_points(self) -> List[Tuple[float, float]]:
        """
        Get all points in the database.
        
        Returns:
            List of all stored points
        """
        return list(self.points.keys())
    
    def clear(self) -> None:
        """Clear all points from the database."""
        self.points.clear()
        self.grid.clear()
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimize database internals for better performance.
        
        Returns:
            Dictionary with optimization metrics
        """
        # Find and remove empty cells
        empty_cells = [cell for cell, points in self.grid.items() if not points]
        for cell in empty_cells:
            del self.grid[cell]
        
        return {
            'empty_cells_removed': len(empty_cells),
            'total_cells': len(self.grid),
            'total_points': len(self.points)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        cells = list(self.grid.keys())
        if cells:
            min_x = min(cell[0] for cell in cells)
            max_x = max(cell[0] for cell in cells)
            min_y = min(cell[1] for cell in cells)
            max_y = max(cell[1] for cell in cells)
            grid_size = (max_x - min_x + 1, max_y - min_y + 1)
        else:
            grid_size = (0, 0)
        
        return {
            'total_points': len(self.points),
            'total_cells': len(self.grid),
            'grid_size': grid_size,
            'points_per_cell_avg': len(self.points) / max(1, len(self.grid)),
            'cell_size': self.cell_size
        }
