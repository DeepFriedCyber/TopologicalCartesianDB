from src.topological_cartesian_db import TopologicalCartesianDB
import numpy as np
from mpmath import mp

def main():
    print("Topological Cartesian Database Demo - Cube Implementation")
    print("======================================================")
    
    # Create a database
    db = TopologicalCartesianDB()
    
    # Create some cubes
    print("\nAdding cubes to the database...")
    
    # A unit cube at origin
    unit_cube_id = db.add_cube(
        [0, 0, 0],
        [1, 1, 1],
        {"name": "unit_cube", "color": "red"}
    )
    
    # A translated cube
    translated_cube_id = db.add_cube(
        [2, 2, 2],
        [3, 3, 3],
        {"name": "translated_cube", "color": "blue"}
    )
    
    # A scaled cube
    scaled_cube_id = db.add_cube(
        [0, 0, 0],
        [2, 2, 2],
        {"name": "scaled_cube", "color": "green"}
    )
    
    # Calculate and display volumes
    print("\nCalculated volumes:")
    for i, cube in enumerate(db.cubes):
        volume = cube.volume()
        print(f"Cube {i} volume: {float(volume)}")
    
    total_volume = db.get_total_volume()
    print(f"\nTotal volume of all cubes: {float(total_volume)}")
    
    # Query points
    print("\nQuerying points:")
    query_points = [
        [0.5, 0.5, 0.5],  # Inside unit and scaled cubes
        [2.5, 2.5, 2.5],  # Inside translated cube
        [1.5, 1.5, 1.5],  # Inside scaled cube only
        [4, 4, 4]         # Outside all cubes
    ]
    
    for point in query_points:
        cube_ids = db.query_point(point)
        if cube_ids:
            print(f"Point {point} belongs to cube(s): {cube_ids}")
            # Print data associated with these cubes
            for id in cube_ids:
                if id in db.data:
                    print(f"  Cube {id} data: {db.data[id]}")
        else:
            print(f"Point {point} not found in any cube")
    
    # Demonstrate Parseval's theorem
    print("\nDemonstrating Parseval's theorem:")
    
    # Create a vector (3-4-5 triangle)
    vector = [3.0, 4.0, 0.0]
    print(f"Vector: {vector}")
    print(f"Vector norm squared: {sum(v**2 for v in vector)}")
    
    # Standard basis
    basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    
    # Project vector onto basis
    coefficients = db.project_vector(vector, basis)
    print(f"Projection coefficients: {coefficients}")
    print(f"Sum of squared coefficients: {sum(c**2 for c in coefficients)}")
    
    # Verify Parseval's theorem
    is_valid = db.verify_parseval_equality(vector, basis)
    print(f"Parseval's theorem verification: {is_valid}")
    
    # Region query example
    print("\nQuerying region:")
    region_min = [0.5, 0.5, 0.5]
    region_max = [2.5, 2.5, 2.5]
    print(f"Query region: {region_min} to {region_max}")
    
    cube_ids = db.query_region(region_min, region_max)
    print(f"Cubes in region: {cube_ids}")
    
if __name__ == "__main__":
    main()
