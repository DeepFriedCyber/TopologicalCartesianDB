import pytest
from tcdb import Simplex

def test_simplex_volume_precision():
    """Test simplex volume calculation with exact arithmetic"""
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]  # Unit tetrahedron
    ]
    simplex = Simplex(vertices)
    # Volume should be exactly 1/6 (no floating-point errors)
    assert simplex.volume() == pytest.approx(1/6, abs=1e-15)
