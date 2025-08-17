from src.enhanced_tcdb import EnhancedTopologicalCartesianDB
import numpy as np
from mpmath import mp
import json

def pretty_print_json(data):
    """Print JSON data with nice formatting."""
    print(json.dumps(data, indent=2))

def main():
    print("Enhanced Topological Cartesian Database Demo")
    print("===========================================")
    print("\nThis demo shows vector operations and Parseval's theorem integration")
    
    # Create a 2D database
    print("\n1. Creating 2D database and inserting vectors...")
    db_2d = EnhancedTopologicalCartesianDB(dimensions=2)
    
    # Insert some vectors
    db_2d.insert_vector("triangle", [3.0, 4.0])  # 3-4-5 triangle
    db_2d.insert_vector("unit_x", [1.0, 0.0])    # Unit vector along x-axis
    db_2d.insert_vector("unit_y", [0.0, 1.0])    # Unit vector along y-axis
    db_2d.insert_vector("origin", [0.0, 0.0])    # Origin
    
    print("Inserted 4 vectors in 2D space")
    
    # Query vectors
    print("\n2. Querying vectors within radius 5 of origin...")
    results = db_2d.query_vector([0.0, 0.0], 5.0)
    print(f"Found {len(results)} vectors:")
    for vec_id, vector in results:
        norm = np.sqrt(sum(v**2 for v in vector))
        print(f"  - {vec_id}: {vector} (norm: {norm})")
    
    # Demonstrate Parseval's theorem
    print("\n3. Verifying Parseval's theorem for the triangle vector...")
    vector = [3.0, 4.0]
    is_valid = db_2d.verify_parseval_equality(vector)
    print(f"Vector: {vector}")
    print(f"Vector norm squared: {sum(v**2 for v in vector)}")
    
    # Standard basis
    basis = db_2d._create_standard_basis(2)
    print(f"Standard basis: {basis}")
    
    # Project vector onto basis
    coefficients = db_2d._project_vector(vector, basis)
    print(f"Projection coefficients: {coefficients}")
    print(f"Sum of squared coefficients: {sum(c**2 for c in coefficients)}")
    
    # Verify Parseval's theorem
    print(f"Parseval's theorem verification: {is_valid}")
    
    # Show query with provenance
    print("\n4. Query with 'Show Your Work' provenance...")
    result = db_2d.query_with_provenance(0.0, 0.0, 5.0)
    
    print("\nResults:")
    for point in result['results']:
        print(f"  Point: {point}")
    
    print("\nProvenance Information:")
    print(f"  Query point: {result['provenance']['query_point']}")
    print(f"  Radius: {result['provenance']['radius']}")
    print(f"  Points examined: {result['provenance']['points_examined']}")
    print(f"  Points found: {result['provenance']['points_found']}")
    
    print("\nEnergy Breakdown (first 2 entries):")
    for i, entry in enumerate(result['provenance']['energy_breakdown'][:2]):
        print(f"  Entry {i+1}:")
        print(f"    Point: {entry['point']}")
        print(f"    Energy contributions: {entry['energy_contributions']}")
        print(f"    Total energy: {entry['total_energy']}")
    
    print("\nParseval Compliance:")
    parseval = result['provenance']['parseval_compliance']
    if parseval:
        print(f"  Verified: {parseval['verified']}")
        print(f"  Total energy: {parseval['total_energy']}")
        print(f"  Tolerance: {parseval['tolerance']}")
    
    # Higher dimensional example
    print("\n5. Higher dimensional example (4D)...")
    db_4d = EnhancedTopologicalCartesianDB(dimensions=4)
    
    # Insert 4D vectors
    db_4d.insert_vector("vec1", [1.0, 2.0, 3.0, 4.0])
    db_4d.insert_vector("vec2", [4.0, 3.0, 2.0, 1.0])
    
    # Query 4D vectors
    results = db_4d.query_vector([0.0, 0.0, 0.0, 0.0], 10.0)
    print(f"Found {len(results)} vectors in 4D space:")
    for vec_id, vector in results:
        norm = np.sqrt(sum(v**2 for v in vector))
        print(f"  - {vec_id}: {vector} (norm: {norm})")
    
    # Verify Parseval's theorem in 4D
    vector_4d = [1.0, 2.0, 3.0, 4.0]
    is_valid = db_4d.verify_parseval_equality(vector_4d)
    print(f"\nParseval's theorem verification for 4D vector {vector_4d}: {is_valid}")
    print(f"Vector norm squared: {sum(v**2 for v in vector_4d)}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
