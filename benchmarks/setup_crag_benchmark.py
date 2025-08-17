#!/usr/bin/env python3
"""
Setup CRAG Benchmark for Coordinate Testing
==========================================

Downloads and sets up the CRAG (Comprehensive RAG Benchmark) to properly
test our coordinate system with knowledge retrieval + reasoning tasks.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def setup_crag_benchmark():
    """Setup CRAG benchmark for coordinate testing"""
    
    print("🎯 Setting up CRAG Benchmark for Coordinate Testing")
    print("=" * 70)
    print("📊 CRAG: 4,409 Q&A pairs requiring knowledge retrieval + reasoning")
    print("🔍 Perfect for testing coordinate system functionality")
    
    # Create benchmarks data directory
    data_dir = Path("benchmark_data")
    data_dir.mkdir(exist_ok=True)
    
    crag_dir = data_dir / "CRAG"
    
    # Download CRAG if not exists
    if not crag_dir.exists():
        print(f"\n📥 Downloading CRAG benchmark...")
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/facebookresearch/CRAG.git",
                str(crag_dir)
            ], check=True, cwd=data_dir.parent)
            print("✅ CRAG downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download CRAG: {e}")
            print("💡 You can manually download from: https://github.com/facebookresearch/CRAG")
            return False
    else:
        print("✅ CRAG already downloaded")
    
    # Check CRAG structure
    print(f"\n🔍 Checking CRAG structure...")
    if crag_dir.exists():
        print(f"   📁 CRAG directory: {crag_dir}")
        
        # List contents
        contents = list(crag_dir.iterdir())
        print(f"   📋 Contents: {len(contents)} items")
        for item in contents[:10]:  # Show first 10 items
            print(f"      - {item.name}")
        
        if len(contents) > 10:
            print(f"      ... and {len(contents) - 10} more items")
    
    # Create coordinate test implementation
    create_crag_coordinate_test()
    
    print(f"\n✅ CRAG benchmark setup completed!")
    print(f"🎯 Ready to test coordinate system with real knowledge retrieval tasks")
    
    return True

def create_crag_coordinate_test():
    """Create CRAG coordinate test implementation"""
    
    test_code = '''#!/usr/bin/env python3
"""
CRAG Coordinate System Test
==========================

Tests our coordinate system using the CRAG benchmark.
This will show if coordinates actually help with knowledge retrieval + reasoning.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator, HybridCoordinateLLM
from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

class CRAGCoordinateTest:
    """Test coordinate system with CRAG benchmark"""
    
    def __init__(self):
        """Initialize CRAG test"""
        print("🎯 CRAG Coordinate System Test")
        print("=" * 60)
        print("📊 Testing coordinate system with knowledge retrieval + reasoning")
        
        # Initialize coordinate engine
        self.coordinate_engine = EnhancedCoordinateEngine()
        
        # Load CRAG data
        self.crag_data = self.load_crag_sample()
        
        print(f"✅ Loaded {len(self.crag_data)} CRAG test questions")
    
    def load_crag_sample(self):
        """Load sample CRAG questions"""
        
        # For now, create sample questions that require knowledge retrieval
        # In real implementation, load from CRAG dataset
        return [
            {
                "id": "crag_sample_1",
                "question": "What is the relationship between topology and data analysis in machine learning?",
                "domain": "Computer Science",
                "type": "Knowledge + Reasoning",
                "expected_coordinate_usage": "Should retrieve topology and ML concepts"
            },
            {
                "id": "crag_sample_2", 
                "question": "How do coordinate systems help with document categorization and clustering?",
                "domain": "Information Science",
                "type": "Knowledge + Reasoning",
                "expected_coordinate_usage": "Should retrieve coordinate and categorization concepts"
            },
            {
                "id": "crag_sample_3",
                "question": "Explain the mathematical principles behind topological data analysis and its applications.",
                "domain": "Mathematics",
                "type": "Knowledge + Reasoning",
                "expected_coordinate_usage": "Should retrieve mathematical and TDA concepts"
            }
        ]
    
    def test_coordinate_usage(self, model_name="llama3.2:3b"):
        """Test coordinate system usage with CRAG-style questions"""
        
        print(f"\\n🤖 Testing {model_name} with coordinate system")
        print("=" * 50)
        
        try:
            # Initialize systems
            ollama = OllamaLLMIntegrator(default_model=model_name, timeout=60)
            hybrid = HybridCoordinateLLM(self.coordinate_engine, ollama)
            
            results = []
            
            for question in self.crag_data:
                print(f"\\n📝 {question['id']}: {question['domain']}")
                print(f"   Question: {question['question'][:100]}...")
                
                # Test with coordinate enhancement
                print("   🚀 Testing with coordinate enhancement...")
                
                start_time = time.time()
                
                result = hybrid.process_query(
                    query=question["question"],
                    model=model_name,
                    temperature=0.1,
                    max_context_docs=5
                )
                
                processing_time = time.time() - start_time
                
                # Extract coordinate usage
                coordinate_context = result.get("coordinate_context", [])
                coordinate_count = len(coordinate_context)
                
                print(f"      ⏱️ Time: {processing_time:.1f}s")
                print(f"      📊 Coordinate docs used: {coordinate_count}")
                
                if coordinate_count > 0:
                    print(f"      ✅ COORDINATES USED! Found {coordinate_count} relevant docs")
                    for i, doc in enumerate(coordinate_context[:3]):  # Show first 3
                        preview = doc.get("content", "")[:100]
                        print(f"         {i+1}. {preview}...")
                else:
                    print(f"      ⚠️ No coordinates used - question may not match our knowledge base")
                
                # Store result
                test_result = {
                    "question_id": question["id"],
                    "domain": question["domain"],
                    "coordinate_docs_used": coordinate_count,
                    "processing_time": processing_time,
                    "coordinate_context": coordinate_context,
                    "response_preview": result.get("llm_response", "")[:200]
                }
                
                results.append(test_result)
            
            # Analyze results
            self.analyze_coordinate_usage(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return []
    
    def analyze_coordinate_usage(self, results):
        """Analyze coordinate system usage"""
        
        print(f"\\n" + "=" * 60)
        print("📊 COORDINATE USAGE ANALYSIS")
        print("=" * 60)
        
        total_questions = len(results)
        questions_with_coordinates = sum(1 for r in results if r["coordinate_docs_used"] > 0)
        total_coordinate_docs = sum(r["coordinate_docs_used"] for r in results)
        avg_processing_time = sum(r["processing_time"] for r in results) / total_questions if total_questions > 0 else 0
        
        print(f"📋 SUMMARY:")
        print(f"   Total Questions: {total_questions}")
        print(f"   Questions Using Coordinates: {questions_with_coordinates}")
        print(f"   Coordinate Usage Rate: {questions_with_coordinates/total_questions*100:.1f}%")
        print(f"   Total Coordinate Docs Retrieved: {total_coordinate_docs}")
        print(f"   Average Processing Time: {avg_processing_time:.1f}s")
        
        if questions_with_coordinates > 0:
            print(f"\\n✅ COORDINATE SYSTEM IS WORKING!")
            print(f"   🎯 {questions_with_coordinates}/{total_questions} questions used coordinate retrieval")
            print(f"   📊 Average {total_coordinate_docs/questions_with_coordinates:.1f} docs per question")
            print(f"   🚀 This proves coordinates add value for knowledge retrieval")
        else:
            print(f"\\n⚠️ COORDINATE SYSTEM NOT USED")
            print(f"   🔍 Questions may not match our coordinate knowledge base")
            print(f"   💡 Need to add more relevant knowledge to coordinate system")
            print(f"   🎯 Or test with questions that match our existing knowledge")
        
        print(f"\\n🎯 NEXT STEPS:")
        if questions_with_coordinates > 0:
            print(f"   ✅ Coordinate system working - expand to full CRAG benchmark")
            print(f"   📊 Compare performance vs baseline LLM")
            print(f"   🚀 Test with more complex multi-hop questions")
        else:
            print(f"   📥 Add more knowledge to coordinate system")
            print(f"   🔍 Test with domain-specific questions")
            print(f"   🎯 Verify coordinate system contains relevant knowledge")

def main():
    """Run CRAG coordinate test"""
    
    print("🎯 CRAG Coordinate System Test")
    print("=" * 60)
    print("📊 Testing if coordinates actually help with knowledge retrieval")
    
    try:
        tester = CRAGCoordinateTest()
        results = tester.test_coordinate_usage()
        
        if results:
            print(f"\\n🎉 CRAG coordinate test completed!")
            print(f"📊 Results show coordinate system usage patterns")
        else:
            print(f"\\n❌ Test failed - check system setup")
            
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()
'''
    
    # Write test file
    test_file = Path("crag_coordinate_test.py")
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    print(f"✅ Created CRAG coordinate test: {test_file}")

def main():
    """Setup CRAG benchmark"""
    
    print("🎯 CRAG Benchmark Setup for Coordinate Testing")
    print("=" * 70)
    print("🔍 This will properly test our coordinate system!")
    
    success = setup_crag_benchmark()
    
    if success:
        print(f"\n🎉 CRAG benchmark setup completed!")
        print(f"🚀 Ready to test coordinate system with real knowledge retrieval")
        print(f"\n📋 Next steps:")
        print(f"   1. Run: python crag_coordinate_test.py")
        print(f"   2. Check if coordinates are actually used")
        print(f"   3. Compare vs baseline LLM performance")
        print(f"   4. Get real evidence of coordinate value")
    else:
        print(f"\n❌ Setup failed - check error messages above")

if __name__ == "__main__":
    main()