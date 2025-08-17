#!/usr/bin/env python3
"""
CRAG Coordinate System Test
==========================

Tests our coordinate system using knowledge retrieval questions.
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
    """Test coordinate system with knowledge retrieval questions"""
    
    def __init__(self):
        """Initialize CRAG test"""
        print("🎯 CRAG-Style Coordinate System Test")
        print("=" * 60)
        print("📊 Testing coordinate system with knowledge retrieval + reasoning")
        
        # Initialize coordinate engine
        self.coordinate_engine = EnhancedCoordinateEngine()
        
        # Load test questions that should use coordinates
        self.test_questions = self.load_coordinate_test_questions()
        
        print(f"✅ Loaded {len(self.test_questions)} coordinate test questions")
    
    def load_coordinate_test_questions(self):
        """Load questions that should trigger coordinate usage"""
        
        return [
            {
                "id": "coord_test_1",
                "question": "What is the relationship between topology and data analysis in machine learning?",
                "domain": "Computer Science",
                "type": "Knowledge + Reasoning",
                "expected_coordinate_usage": "Should retrieve topology and ML concepts",
                "keywords": ["topology", "data analysis", "machine learning"]
            },
            {
                "id": "coord_test_2", 
                "question": "How do coordinate systems help with document categorization and clustering?",
                "domain": "Information Science",
                "type": "Knowledge + Reasoning",
                "expected_coordinate_usage": "Should retrieve coordinate and categorization concepts",
                "keywords": ["coordinate systems", "categorization", "clustering"]
            },
            {
                "id": "coord_test_3",
                "question": "Explain the mathematical principles behind topological data analysis and its applications.",
                "domain": "Mathematics",
                "type": "Knowledge + Reasoning",
                "expected_coordinate_usage": "Should retrieve mathematical and TDA concepts",
                "keywords": ["topological data analysis", "mathematics", "applications"]
            },
            {
                "id": "coord_test_4",
                "question": "What are the advantages of using coordinate-based approaches for information retrieval?",
                "domain": "Information Retrieval",
                "type": "Knowledge + Reasoning", 
                "expected_coordinate_usage": "Should retrieve IR and coordinate concepts",
                "keywords": ["coordinate-based", "information retrieval", "advantages"]
            },
            {
                "id": "coord_test_5",
                "question": "How does topological analysis improve data clustering and pattern recognition?",
                "domain": "Data Science",
                "type": "Knowledge + Reasoning",
                "expected_coordinate_usage": "Should retrieve topology and clustering concepts",
                "keywords": ["topological analysis", "clustering", "pattern recognition"]
            }
        ]
    
    def test_coordinate_usage(self, model_name="llama3.2:3b"):
        """Test coordinate system usage with knowledge retrieval questions"""
        
        print(f"\n🤖 Testing {model_name} with coordinate system")
        print("=" * 50)
        
        try:
            # Initialize systems
            ollama = OllamaLLMIntegrator(default_model=model_name, timeout=60)
            hybrid = HybridCoordinateLLM(self.coordinate_engine, ollama)
            
            results = []
            
            for question in self.test_questions:
                print(f"\n📝 {question['id']}: {question['domain']}")
                print(f"   Question: {question['question']}")
                print(f"   Keywords: {', '.join(question['keywords'])}")
                
                # Test with coordinate enhancement
                print("   🚀 Testing with coordinate enhancement...")
                
                start_time = time.time()
                
                try:
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
                            content = doc.get("content", "")
                            preview = content[:100] if content else "No content"
                            print(f"         {i+1}. {preview}...")
                    else:
                        print(f"      ⚠️ No coordinates used")
                        print(f"         💡 Question may not match coordinate knowledge base")
                    
                    # Get response preview
                    response = result.get("llm_response", "")
                    response_preview = response[:200] if response else "No response"
                    print(f"      📄 Response preview: {response_preview}...")
                    
                    # Store result
                    test_result = {
                        "question_id": question["id"],
                        "domain": question["domain"],
                        "keywords": question["keywords"],
                        "coordinate_docs_used": coordinate_count,
                        "processing_time": processing_time,
                        "coordinate_context": coordinate_context,
                        "response_preview": response_preview,
                        "full_response": response
                    }
                    
                    results.append(test_result)
                    
                except Exception as e:
                    print(f"      ❌ Error processing question: {e}")
                    test_result = {
                        "question_id": question["id"],
                        "domain": question["domain"],
                        "keywords": question["keywords"],
                        "coordinate_docs_used": 0,
                        "processing_time": time.time() - start_time,
                        "error": str(e)
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
        
        print(f"\n" + "=" * 60)
        print("📊 COORDINATE USAGE ANALYSIS")
        print("=" * 60)
        
        total_questions = len(results)
        questions_with_coordinates = sum(1 for r in results if r.get("coordinate_docs_used", 0) > 0)
        total_coordinate_docs = sum(r.get("coordinate_docs_used", 0) for r in results)
        avg_processing_time = sum(r.get("processing_time", 0) for r in results) / total_questions if total_questions > 0 else 0
        
        print(f"📋 SUMMARY:")
        print(f"   Total Questions: {total_questions}")
        print(f"   Questions Using Coordinates: {questions_with_coordinates}")
        print(f"   Coordinate Usage Rate: {questions_with_coordinates/total_questions*100:.1f}%")
        print(f"   Total Coordinate Docs Retrieved: {total_coordinate_docs}")
        print(f"   Average Processing Time: {avg_processing_time:.1f}s")
        
        if questions_with_coordinates > 0:
            avg_docs_per_question = total_coordinate_docs / questions_with_coordinates
            print(f"   Average Docs per Question (when used): {avg_docs_per_question:.1f}")
        
        print(f"\n📊 DETAILED BREAKDOWN:")
        for result in results:
            question_id = result.get("question_id", "Unknown")
            coord_count = result.get("coordinate_docs_used", 0)
            domain = result.get("domain", "Unknown")
            
            status = "✅ USED" if coord_count > 0 else "❌ NOT USED"
            print(f"   {question_id} ({domain}): {status} - {coord_count} docs")
        
        print(f"\n🎯 ANALYSIS:")
        
        if questions_with_coordinates > 0:
            print(f"   ✅ COORDINATE SYSTEM IS WORKING!")
            print(f"      🎯 {questions_with_coordinates}/{total_questions} questions used coordinate retrieval")
            print(f"      📊 Retrieved {total_coordinate_docs} total documents")
            print(f"      🚀 This proves coordinates add value for knowledge retrieval")
            
            if questions_with_coordinates == total_questions:
                print(f"      🏆 PERFECT: All questions used coordinate system!")
            else:
                unused = total_questions - questions_with_coordinates
                print(f"      ⚠️ {unused} questions didn't use coordinates")
                print(f"         💡 May need more relevant knowledge in coordinate system")
        else:
            print(f"   ❌ COORDINATE SYSTEM NOT USED")
            print(f"      🔍 Questions don't match coordinate knowledge base")
            print(f"      💡 Need to add more relevant knowledge to coordinate system")
            print(f"      🎯 Or verify coordinate system contains expected knowledge")
        
        print(f"\n🚀 NEXT STEPS:")
        if questions_with_coordinates > 0:
            print(f"   ✅ Coordinate system working - test with baseline comparison")
            print(f"   📊 Compare coordinate-enhanced vs baseline LLM performance")
            print(f"   🎯 Expand to larger benchmark datasets")
            print(f"   📈 Measure accuracy improvements")
        else:
            print(f"   📥 Add domain-specific knowledge to coordinate system")
            print(f"   🔍 Verify coordinate system initialization")
            print(f"   🎯 Test with simpler knowledge retrieval questions")
            print(f"   📊 Check coordinate system content and indexing")
        
        print(f"\n🎯 COORDINATE SYSTEM VALUE:")
        if questions_with_coordinates > 0:
            usage_rate = questions_with_coordinates/total_questions*100
            print(f"   📊 {usage_rate:.1f}% of knowledge questions used coordinates")
            print(f"   ✅ Proves coordinate system adds value for knowledge retrieval")
            print(f"   🚀 Ready for baseline vs enhanced performance comparison")
        else:
            print(f"   ⚠️ Coordinate system not utilized in current test")
            print(f"   🔧 Need to improve knowledge base or question matching")

def main():
    """Run CRAG-style coordinate test"""
    
    print("🎯 CRAG-Style Coordinate System Test")
    print("=" * 60)
    print("📊 Testing if coordinates actually help with knowledge retrieval")
    print("🔍 This will show if our coordinate system is working properly")
    
    try:
        tester = CRAGCoordinateTest()
        results = tester.test_coordinate_usage()
        
        if results:
            print(f"\n🎉 Coordinate test completed!")
            print(f"📊 Results show coordinate system usage patterns")
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"coordinate_usage_test_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved: {filename}")
        else:
            print(f"\n❌ Test failed - check system setup")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()