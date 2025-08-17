#!/usr/bin/env python3
"""
Debug script to understand why cube searches return no results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

def test_coordinate_engine_search():
    """Test coordinate engine search functionality"""
    
    print("🔍 Testing Coordinate Engine Search...")
    
    # Create coordinate engine
    engine = EnhancedCoordinateEngine()
    
    # Add some test documents
    test_docs = [
        ("doc1", "Vector document 1 with 128 dimensions containing data processing information for multi-cube analysis"),
        ("doc2", "Vector document 2 with 128 dimensions containing temporal patterns information for multi-cube analysis"),
        ("doc3", "Vector document 3 with 128 dimensions containing system performance information for multi-cube analysis")
    ]
    
    print(f"📄 Adding {len(test_docs)} test documents...")
    for doc_id, content in test_docs:
        engine.add_document(doc_id, content)
        print(f"   ✅ Added: {doc_id}")
    
    print(f"📊 Total documents in engine: {len(engine.documents)}")
    
    # Test different queries
    test_queries = [
        "Find similar vector 0 with 128 dimensions",
        "data processing analysis",
        "temporal patterns",
        "system performance"
    ]
    
    print(f"\n🔍 Testing {len(test_queries)} search queries:")
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        # Get query coordinates
        query_coords = engine.text_to_coordinates(query)
        print(f"   🎯 Query coordinates: {query_coords}")
        
        # Perform search
        results = engine.search(query, max_results=3)
        print(f"   📊 Results: {len(results)} found")
        
        if results:
            for i, result in enumerate(results):
                print(f"      {i+1}. {result['document_id']}: similarity={result['similarity_score']:.3f}")
                print(f"         Content: {result['content'][:60]}...")
                print(f"         Doc coords: {result['coordinates']}")
        else:
            print("      ❌ No results found")
            
            # Debug: check coordinates of stored documents
            print("      🔍 Debugging - Document coordinates:")
            for doc_id, doc_data in engine.documents.items():
                similarity = engine._calculate_coordinate_similarity(query_coords, doc_data['coordinates'])
                print(f"         {doc_id}: {doc_data['coordinates']} (similarity: {similarity:.3f})")

if __name__ == "__main__":
    test_coordinate_engine_search()
