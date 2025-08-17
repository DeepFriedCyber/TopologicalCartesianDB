"""
Test module for memory management in EnhancedTopologicalCartesianDB.
"""
import unittest
import gc
import random
from src.enhanced_tcdb import EnhancedTopologicalCartesianDB

class TestMemoryManagement(unittest.TestCase):
    """Tests for memory optimization and management features."""
    
    def test_optimize_memory(self):
        """Test that optimize_memory removes empty grid cells."""
        db = EnhancedTopologicalCartesianDB(dimensions=2, cell_size=1.0)
        
        # Insert points that will create cells in specific areas
        for i in range(100):
            # Points in quadrant 1 (positive x, positive y)
            db.insert_vector(f"q1_{i}", [random.uniform(1, 10), random.uniform(1, 10)])
            
            # Points in quadrant 3 (negative x, negative y)
            db.insert_vector(f"q3_{i}", [random.uniform(-10, -1), random.uniform(-10, -1)])
        
        # Check initial cell distribution
        cells_before = len(db.spatial_index)
        self.assertGreater(cells_before, 0)
        
        print(f"Cells before deletion: {cells_before}")
        
        # Delete all points in quadrant 1
        to_delete = [key for key in db.vectors.keys() if key.startswith("q1_")]
        for key in to_delete:
            del db.vectors[key]
        
        # This should leave empty cells in the spatial index
        # Run memory optimization
        result = db.optimize_memory()
        
        # Verify cells were removed
        cells_after = len(db.spatial_index)
        print(f"Cells after optimization: {cells_after}")
        print(f"Cells removed: {result['cells_removed']}")
        print(f"Memory saved (estimated): {result['memory_saved']} bytes")
        
        self.assertLess(cells_after, cells_before)
        self.assertGreater(result['cells_removed'], 0)
        self.assertGreater(result['memory_saved'], 0)
        
        # Force garbage collection
        gc.collect()
    
    def test_index_efficiency(self):
        """Test the efficiency of the spatial index."""
        db = EnhancedTopologicalCartesianDB(dimensions=2, cell_size=5.0)
        
        # Insert 10,000 points in a normal distribution
        print("Inserting 10,000 points...")
        for i in range(10000):
            x = random.gauss(0, 20)  # Normal distribution around 0
            y = random.gauss(0, 20)
            db.insert_vector(f"vec_{i}", [x, y])
        
        # Get index statistics
        stats = db.get_index_statistics()
        
        print(f"Total points: {stats['total_vectors']}")
        print(f"Total cells: {stats['cell_count']}")
        print(f"Points per cell (avg): {stats['avg_vectors_per_cell']:.2f}")
        print(f"Empty cells: {stats['empty_cells']}")
        print(f"Memory usage (estimated): {stats['estimated_memory_bytes'] / (1024*1024):.2f} MB")
        
        # Run optimization
        print("\nOptimizing memory...")
        result = db.optimize_memory()
        print(f"Cells removed: {result['cells_removed']}")
        print(f"Memory saved: {result['memory_saved'] / 1024:.2f} KB")
        
        # Get updated statistics
        stats_after = db.get_index_statistics()
        print(f"\nAfter optimization:")
        print(f"Total cells: {stats_after['cell_count']}")
        print(f"Empty cells: {stats_after['empty_cells']}")
        print(f"Memory usage (estimated): {stats_after['estimated_memory_bytes'] / (1024*1024):.2f} MB")
        
        # Verify empty cells were removed
        self.assertEqual(stats_after['empty_cells'], 0, 
                        "After optimization, there should be no empty cells")

if __name__ == '__main__':
    unittest.main()
