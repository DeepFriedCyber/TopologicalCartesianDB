#!/usr/bin/env python3
"""
Populate Coordinate System
=========================

Add relevant knowledge to the coordinate system so it can actually
be used for knowledge retrieval in our tests.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

def populate_coordinate_system():
    """Populate coordinate system with relevant knowledge"""
    
    print("📥 Populating Coordinate System with Knowledge")
    print("=" * 60)
    print("🎯 Adding domain knowledge for coordinate retrieval testing")
    
    # Initialize coordinate engine
    coordinate_engine = EnhancedCoordinateEngine()
    
    # Define knowledge documents relevant to our test questions
    knowledge_docs = [
        {
            "id": "topology_ml_001",
            "content": "Topology in machine learning refers to the study of spatial properties preserved under continuous deformations. Topological data analysis (TDA) uses concepts from algebraic topology to analyze the shape of data. It helps identify persistent features in high-dimensional datasets, revealing hidden patterns and structures that traditional methods might miss."
        },
        {
            "id": "topology_ml_002", 
            "content": "Topological data analysis provides tools for understanding the global structure of data through persistent homology. This technique tracks topological features across multiple scales, identifying holes, connected components, and voids in data. Applications include analyzing neural networks, protein folding, and complex systems."
        },
        {
            "id": "coordinate_systems_001",
            "content": "Coordinate systems in document categorization provide a mathematical framework for representing documents in multi-dimensional space. Each dimension corresponds to specific features or topics, allowing for geometric interpretation of document relationships. This enables clustering, similarity search, and classification based on spatial proximity."
        },
        {
            "id": "coordinate_systems_002",
            "content": "Coordinate-based approaches for information retrieval use vector spaces where documents and queries are represented as points. The coordinate system allows for efficient similarity computation using distance metrics like cosine similarity or Euclidean distance. This geometric approach enables interpretable search results."
        },
        {
            "id": "tda_mathematics_001",
            "content": "The mathematical principles of topological data analysis include persistent homology, simplicial complexes, and filtrations. Persistent homology tracks the birth and death of topological features across parameter scales. Simplicial complexes provide discrete representations of continuous spaces for computational analysis."
        },
        {
            "id": "tda_applications_001",
            "content": "Topological data analysis applications span neuroscience, biology, materials science, and social networks. In neuroscience, TDA analyzes brain connectivity patterns. In biology, it studies protein structures and evolutionary relationships. In materials science, it characterizes porous materials and crystal structures."
        },
        {
            "id": "clustering_topology_001",
            "content": "Topological analysis improves data clustering by identifying the intrinsic shape and connectivity of data manifolds. Unlike traditional clustering methods that assume specific cluster shapes, topological approaches can detect arbitrary cluster topologies including rings, branches, and complex interconnected structures."
        },
        {
            "id": "pattern_recognition_001",
            "content": "Topological pattern recognition uses persistent features to identify robust patterns in noisy data. The method is particularly effective for recognizing patterns that persist across multiple scales and are invariant to continuous deformations. This makes it valuable for image analysis, signal processing, and time series analysis."
        },
        {
            "id": "machine_learning_topology_001",
            "content": "Machine learning benefits from topological analysis through improved feature extraction, dimensionality reduction, and model interpretability. Topological features provide stable representations that are robust to noise and outliers. This leads to more reliable classification and regression models."
        },
        {
            "id": "data_analysis_coordinates_001",
            "content": "Coordinate-based data analysis represents datasets in structured mathematical spaces where each dimension has semantic meaning. This approach enables interpretable analysis where the contribution of each feature to the final result can be understood and explained, unlike black-box vector embeddings."
        },
        {
            "id": "information_retrieval_001",
            "content": "Information retrieval using coordinate systems provides transparent and explainable search results. Each coordinate dimension corresponds to specific semantic concepts, allowing users to understand why documents are retrieved and how similarity is computed. This interpretability is crucial for applications requiring explainable AI."
        },
        {
            "id": "categorization_clustering_001",
            "content": "Document categorization and clustering using coordinate systems leverage geometric properties of the coordinate space. Documents with similar coordinates are grouped together, and the coordinate values explain the basis for categorization. This approach enables both automatic classification and human-interpretable category definitions."
        }
    ]
    
    print(f"📚 Adding {len(knowledge_docs)} knowledge documents...")
    
    # Add documents to coordinate system
    added_count = 0
    for doc in knowledge_docs:
        try:
            success = coordinate_engine.add_document(doc["id"], doc["content"])
            if success:
                added_count += 1
                print(f"   ✅ Added: {doc['id']}")
            else:
                print(f"   ❌ Failed: {doc['id']}")
        except Exception as e:
            print(f"   ❌ Error adding {doc['id']}: {e}")
    
    print(f"\n📊 Population Results:")
    print(f"   Documents attempted: {len(knowledge_docs)}")
    print(f"   Documents added: {added_count}")
    print(f"   Success rate: {added_count/len(knowledge_docs)*100:.1f}%")
    
    # Verify population
    print(f"\n🔍 Verifying coordinate system population...")
    
    try:
        # Check document count
        doc_count = len(coordinate_engine.documents)
        print(f"   Total documents in system: {doc_count}")
        
        # Test retrieval
        if hasattr(coordinate_engine, 'search_documents'):
            test_results = coordinate_engine.search_documents("topology", top_k=3)
            print(f"   Test search results: {len(test_results) if test_results else 0}")
        elif hasattr(coordinate_engine, 'get_document'):
            # Try to get a specific document
            test_doc = coordinate_engine.get_document("topology_ml_001")
            print(f"   Test document retrieval: {'Success' if test_doc else 'Failed'}")
        
        # Show sample documents
        if doc_count > 0:
            print(f"\n📋 Sample documents in system:")
            doc_ids = list(coordinate_engine.documents.keys())[:5]
            for doc_id in doc_ids:
                print(f"      - {doc_id}")
    
    except Exception as e:
        print(f"   ❌ Verification error: {e}")
    
    return coordinate_engine, added_count

def test_populated_system():
    """Test the populated coordinate system"""
    
    print(f"\n🧪 Testing Populated Coordinate System")
    print("=" * 50)
    
    # Populate system
    coordinate_engine, added_count = populate_coordinate_system()
    
    if added_count == 0:
        print("❌ No documents added - cannot test")
        return False
    
    # Test with a simple query
    print(f"\n🔍 Testing coordinate retrieval...")
    
    test_queries = [
        "topology and machine learning",
        "coordinate systems for categorization", 
        "topological data analysis mathematics"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        try:
            # Try different retrieval methods
            if hasattr(coordinate_engine, 'search_documents'):
                results = coordinate_engine.search_documents(query, top_k=3)
                print(f"      Results: {len(results) if results else 0}")
                
                if results:
                    for i, result in enumerate(results[:2]):
                        doc_id = result.get('id', 'Unknown')
                        content = result.get('content', '')[:100]
                        print(f"         {i+1}. {doc_id}: {content}...")
            
            elif hasattr(coordinate_engine, 'find_similar'):
                results = coordinate_engine.find_similar(query, top_k=3)
                print(f"      Results: {len(results) if results else 0}")
            
            else:
                print(f"      No search method available")
                
        except Exception as e:
            print(f"      Error: {e}")
    
    print(f"\n✅ Coordinate system populated and ready for testing!")
    return True

def main():
    """Populate coordinate system with knowledge"""
    
    print("📥 Coordinate System Population")
    print("=" * 60)
    print("🎯 Adding knowledge so coordinates can actually be used")
    
    success = test_populated_system()
    
    if success:
        print(f"\n🎉 Coordinate system successfully populated!")
        print(f"🚀 Ready to test coordinate usage in knowledge retrieval")
        print(f"\n📋 Next steps:")
        print(f"   1. Run coordinate test again: python crag_coordinate_test.py")
        print(f"   2. Check if coordinates are now being used")
        print(f"   3. Compare baseline vs coordinate-enhanced performance")
    else:
        print(f"\n❌ Population failed - check error messages above")

if __name__ == "__main__":
    main()