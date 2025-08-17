import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from tcdb import TopologicalCartesianDB

# Constants for resource limits
def test_insert_duplicate_points():
    db = TopologicalCartesianDB()
    db.insert(1, 1)
    db.insert(1, 1)
    assert len(db.points) == 1  # Should not allow duplicates

def test_query_negative_radius():
    db = TopologicalCartesianDB()
    with pytest.raises(ValueError):
        db.query(0, 0, -1)

def test_insert_non_numeric():
    db = TopologicalCartesianDB()
    with pytest.raises(TypeError):
        db.insert('a', 1)
    with pytest.raises(TypeError):
        db.insert(1, 'b')

def test_query_zero_radius():
    db = TopologicalCartesianDB()
    db.insert(2, 2)
    result = db.query(2, 2, 0)
    assert result == [(2, 2)]

def test_max_points_limit():
    db = TopologicalCartesianDB()
    for i in range(db.MAX_POINTS):
        db.insert(i, i)
    with pytest.raises(RuntimeError):
        db.insert(db.MAX_POINTS, db.MAX_POINTS)

def test_max_radius_limit():
    db = TopologicalCartesianDB()
    db.insert(0, 0)
    with pytest.raises(ValueError):
        db.query(0, 0, db.MAX_RADIUS + 1)

def test_floating_point_tolerance():
    db = TopologicalCartesianDB()
    db.insert(0, 1)
    # Point at radius boundary
    result = db.query(0, 0, 1.0)
    assert (0, 1) in result

def test_docstrings_and_type_hints():
    assert TopologicalCartesianDB.__doc__ is not None
    assert hasattr(TopologicalCartesianDB, 'insert')
    assert hasattr(TopologicalCartesianDB, 'query')

# Add more tests for batch operations, spatial indexing, and security as features are implemented
