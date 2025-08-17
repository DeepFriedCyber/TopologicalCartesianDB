#!/usr/bin/env python3
"""
Test Coordinates Properly
=========================

Test coordinate system with persistent populated data to see if it actually works.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator, HybridCoordinateLLM
from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

def create_populated_coordinate_engine():
    """Create and populate coordinate engine with knowledge"""
    
    print("📥 Creating populated coordinate engine...")
    
    # Initialize coordinate engine
    coordinate_engine = EnhancedCoordinateEngine()
    
    # Add knowledge documents
    knowledge_docs = [
        {
            "id": "topology_ml_001",
            "content": "Topology in machine learning refers to the study of spatial properties preserved under continuous deformations. Topological data analysis (TDA) uses concepts from algebraic topology to analyze the shape of data. It helps identify persistent features in high-dimensional datasets, revealing hidden patterns and structures that traditional methods might miss."
        },
        {
            "id": "coordinate_systems_001",
            "content": "Coordinate systems in document categorization provide a mathematical framework for representing documents in multi-dimensional space. Each dimension corresponds to specific features or topics, allowing for geometric interpretation of document relationships. This enables clustering, similarity search, and classification based on spatial proximity."
        },
        {
            "id": "tda_mathematics_001",
            "content": "The mathematical principles of topological data analysis include persistent homology, simplicial complexes, and filtrations. Persistent homology tracks the birth and death of topological features across parameter scales. Simplicial complexes provide discrete representations of continuous spaces for computational analysis."
        }
    ]
    
    # Add documents
    for doc in knowledge_docs:
        coordinate_engine.add_document(doc["id"], doc["content"])
    
    print(f"✅ Added {len(knowledge_docs)} documents to coordinate engine")
    print(f"📊 Total documents: {len(coordinate_engine.documents)}")
    
    return coordinate_engine

def test_coordinate_search_directly():
    """Test coordinate search directly"""
    
    print("\n🔍 Testing Coordinate Search Directly")
    print("=" * 50)
    
    # Create populated engine
    coordinate_engine = create_populated_coordinate_engine()
    
    # Test direct search
    test_queries = [
        "topology machine learning",
        "coordinate systems categorization", 
        "mathematical principles topological"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        try:
            # Test direct search method
            results = coordinate_engine.search(query, max_results=3)
            print(f"      Direct search results: {len(results)}")
            
            if results:
                for i, result in enumerate(results):
                    doc_id = result.get('id', 'Unknown')
                    content = result.get('content', '')[:100]
                    similarity = result.get('similarity_score', 0)
                    print(f"         {i+1}. {doc_id} (sim: {similarity:.3f}): {content}...")
            else:
                print(f"         No results found")
                
        except Exception as e:
            print(f"      Error: {e}")
    
    return coordinate_engine

def test_llm_context_method():
    """Test get_llm_context method"""
    
    print("\n🧠 Testing LLM Context Method")
    print("=" * 50)
    
    # Create populated engine
    coordinate_engine = create_populated_coordinate_engine()
    
    test_queries = [
        "topology machine learning",
        "coordinate systems categorization"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        try:
            # Test LLM context method (this is what hybrid system uses)
            context = coordinate_engine.get_llm_context(query, max_docs=3)
            print(f"      LLM context results: {len(context)}")
            
            if context:
                for i, ctx in enumerate(context):
                    content = ctx.get('content', '')[:100]
                    coords = ctx.get('coordinates', {})
                    explanation = ctx.get('relevance_explanation', '')[:50]
                    print(f"         {i+1}. Content: {content}...")
                    print(f"             Coords: {list(coords.keys())[:3]}...")
                    print(f"             Explanation: {explanation}...")
            else:
                print(f"         No context found")
                
        except Exception as e:
            print(f"      Error: {e}")
    
    return coordinate_engine

def test_hybrid_system_with_populated_engine():
    """Test hybrid system with pre-populated coordinate engine"""
    
    print("\n🚀 Testing Hybrid System with Populated Engine")
    print("=" * 50)
    
    # Create populated engine
    coordinate_engine = create_populated_coordinate_engine()
    
    # Initialize hybrid system with populated engine
    try:
        ollama = OllamaLLMIntegrator(default_model="llama3.2:3b", timeout=60)
        hybrid = HybridCoordinateLLM(coordinate_engine, ollama)
        
        print("✅ Hybrid system initialized with populated coordinate engine")
        
        # Test with knowledge retrieval questions
        test_questions = [
            "What is the relationship between topology and machine learning?",
            "How do coordinate systems help with document categorization?"
        ]
        
        for question in test_questions:
            print(f"\n📝 Question: {question}")
            
            start_time = time.time()
            
            try:
                result = hybrid.process_query(
                    query=question,
                    model="llama3.2:3b",
                    temperature=0.1,
                    max_context_docs=3
                )
                
                processing_time = time.time() - start_time
                
                # Check coordinate usage
                coordinate_context = result.get("coordinate_context", [])
                coordinate_count = len(coordinate_context)
                
                print(f"   ⏱️ Time: {processing_time:.1f}s")
                print(f"   📊 Coordinate docs used: {coordinate_count}")
                
                if coordinate_count > 0:
                    print(f"   ✅ SUCCESS! Coordinates were used!")
                    for i, doc in enumerate(coordinate_context):
                        content = doc.get("content", "")[:100]
                        print(f"      {i+1}. {content}...")
                else:
                    print(f"   ❌ No coordinates used")
                
                # Show response preview
                response = result.get("llm_response", "")
                print(f"   📄 Response: {response[:150]}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Hybrid system error: {e}")

def main():
    """Test coordinate system properly"""
    
    print("🎯 Testing Coordinate System Properly")
    print("=" * 60)
    print("🔍 Using persistent populated coordinate engine")
    
    # Test 1: Direct coordinate search
    coordinate_engine = test_coordinate_search_directly()
    
    # Test 2: LLM context method
    test_llm_context_method()
    
    # Test 3: Full hybrid system
    test_hybrid_system_with_populated_engine()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   This test shows if coordinate system actually works")
    print(f"   If coordinates are used, we have a working system")
    print(f"   If not, we need to debug the search/retrieval mechanism")

if __name__ == "__main__":
    main()