#!/usr/bin/env python3
"""
Debug Coordinate System
======================

Investigate why the coordinate system isn't being used.
Check what's actually in the coordinate knowledge base.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

def debug_coordinate_system():
    """Debug coordinate system to see what's inside"""
    
    print("🔍 Debugging Coordinate System")
    print("=" * 60)
    print("🎯 Investigating why coordinates aren't being used")
    
    try:
        # Initialize coordinate engine
        print("\n🚀 Initializing coordinate engine...")
        coordinate_engine = EnhancedCoordinateEngine()
        
        # Check if it has any content
        print("\n📊 Checking coordinate system content...")
        
        # Try to access internal state
        if hasattr(coordinate_engine, 'knowledge_base'):
            kb = coordinate_engine.knowledge_base
            print(f"   Knowledge base type: {type(kb)}")
            
            if hasattr(kb, '__len__'):
                print(f"   Knowledge base size: {len(kb)}")
            
            if hasattr(kb, 'keys'):
                keys = list(kb.keys())
                print(f"   Knowledge base keys: {keys[:10]}...")  # First 10 keys
        
        # Check for vector store
        if hasattr(coordinate_engine, 'vector_store'):
            vs = coordinate_engine.vector_store
            print(f"   Vector store type: {type(vs)}")
            
            if hasattr(vs, '__len__'):
                print(f"   Vector store size: {len(vs)}")
        
        # Check for documents
        if hasattr(coordinate_engine, 'documents'):
            docs = coordinate_engine.documents
            print(f"   Documents type: {type(docs)}")
            
            if hasattr(docs, '__len__'):
                print(f"   Documents count: {len(docs)}")
                
                if len(docs) > 0:
                    print(f"   Sample document: {docs[0] if docs else 'None'}")
        
        # Try a simple search
        print("\n🔍 Testing coordinate search...")
        
        test_queries = [
            "topology",
            "machine learning", 
            "data analysis",
            "coordinate system",
            "mathematics"
        ]
        
        for query in test_queries:
            print(f"\n   Testing query: '{query}'")
            
            try:
                # Try to search for relevant documents
                if hasattr(coordinate_engine, 'search_similar'):
                    results = coordinate_engine.search_similar(query, top_k=3)
                    print(f"      Results: {len(results) if results else 0}")
                    
                    if results:
                        for i, result in enumerate(results[:2]):
                            preview = str(result)[:100]
                            print(f"         {i+1}. {preview}...")
                    else:
                        print(f"         No results found")
                        
                elif hasattr(coordinate_engine, 'retrieve_context'):
                    results = coordinate_engine.retrieve_context(query, max_docs=3)
                    print(f"      Results: {len(results) if results else 0}")
                    
                    if results:
                        for i, result in enumerate(results[:2]):
                            preview = str(result)[:100]
                            print(f"         {i+1}. {preview}...")
                    else:
                        print(f"         No results found")
                        
                else:
                    print(f"      No search method found")
                    
            except Exception as e:
                print(f"      Search error: {e}")
        
        # Check coordinate engine methods
        print(f"\n🔧 Coordinate engine methods:")
        methods = [method for method in dir(coordinate_engine) if not method.startswith('_')]
        for method in methods[:10]:  # First 10 methods
            print(f"   - {method}")
        
        if len(methods) > 10:
            print(f"   ... and {len(methods) - 10} more methods")
        
        # Check if coordinate engine is properly initialized
        print(f"\n✅ Coordinate engine status:")
        print(f"   Type: {type(coordinate_engine)}")
        print(f"   Initialized: {coordinate_engine is not None}")
        
        # Try to understand the coordinate system structure
        print(f"\n🏗️ Coordinate system structure:")
        for attr in ['knowledge_base', 'vector_store', 'documents', 'index', 'embeddings']:
            if hasattr(coordinate_engine, attr):
                value = getattr(coordinate_engine, attr)
                print(f"   {attr}: {type(value)} - {value is not None}")
            else:
                print(f"   {attr}: Not found")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run coordinate system debug"""
    
    print("🔍 Coordinate System Debug")
    print("=" * 60)
    print("🎯 Finding out why coordinates aren't being used")
    
    debug_coordinate_system()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. 📊 Check if coordinate system has any content")
    print(f"   2. 🔍 Verify knowledge base initialization")
    print(f"   3. 🎯 Add content to coordinate system if empty")
    print(f"   4. 🔧 Fix retrieval mechanism if broken")

if __name__ == "__main__":
    main()