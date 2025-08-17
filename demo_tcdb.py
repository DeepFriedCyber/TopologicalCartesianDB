from tcdb import Simplex, TopologicalCartesianDB
import numpy as np
from mpmath import mp

def main():
    print("Topological Cartesian Database Demo")
    print("==================================")
    
    # Create a database
    db = TopologicalCartesianDB()
    
    # Create some tetrahedrons
    print("\nAdding tetrahedrons to the database...")
    
    # A unit tetrahedron
    unit_tetra = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    id1 = db.add_simplex(unit_tetra, {"name": "unit", "color": "red"})
    
    # A translated tetrahedron
    translated_tetra = [
        [1, 1, 1],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ]
    id2 = db.add_simplex(translated_tetra, {"name": "translated", "color": "blue"})
    
    # A scaled tetrahedron
    scaled_tetra = [
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ]
    id3 = db.add_simplex(scaled_tetra, {"name": "scaled", "color": "green"})
    
    # Calculate and display volumes
    print("\nCalculated volumes:")
    for i, simplex in enumerate(db.simplices):
        volume = simplex.volume()
        print(f"Tetrahedron {i} volume: {float(volume)}")
    
    total_volume = db.get_total_volume()
    print(f"\nTotal volume of all tetrahedrons: {float(total_volume)}")
    
    # Query points
    print("\nQuerying points:")
    query_points = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 0],
        [3, 3, 3]
    ]
    
    for point in query_points:
        simplex_ids = db.query_point(point)
        if simplex_ids:
            print(f"Point {point} belongs to simplex(es): {simplex_ids}")
            # Print data associated with these simplices
            for id in simplex_ids:
                if id in db.data:
                    print(f"  Simplex {id} data: {db.data[id]}")
        else:
            print(f"Point {point} not found in any simplex")
    
if __name__ == "__main__":
    main()
