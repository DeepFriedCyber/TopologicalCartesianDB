#!/usr/bin/env python3
"""
Quick Model Comparison - Test 2 models on 1 problem
==================================================

Quick demonstration of multi-model testing for immediate verification.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator, HybridCoordinateLLM
from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

def quick_comparison():
    """Quick comparison of 2 models on 1 problem"""
    
    print("🚀 Quick Model Comparison Test")
    print("=" * 50)
    
    # Test problem
    test_problem = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    correct_answer = 18.0
    
    print(f"📝 Test Problem: {test_problem}")
    print(f"✅ Correct Answer: {correct_answer}")
    
    # Initialize coordinate engine
    coordinate_engine = EnhancedCoordinateEngine()
    
    # Models to test
    models = ["llama3.2:3b", "llama3.2:1b"]
    
    results = {}
    
    for model in models:
        print(f"\n🤖 Testing {model}...")
        
        try:
            # Initialize model
            ollama = OllamaLLMIntegrator(default_model=model)
            hybrid = HybridCoordinateLLM(coordinate_engine, ollama)
            
            # Create prompt
            prompt = f"""
MATH PROBLEM: {test_problem}

Solve step by step and end with:
FINAL ANSWER: [number]
"""
            
            start_time = time.time()
            
            # Process
            result = hybrid.process_query(
                query=prompt,
                model=model,
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            
            # Extract answer
            response = result.get('llm_response', '')
            
            # Simple answer extraction
            import re
            answer_match = re.search(r'FINAL ANSWER:\s*([0-9]+(?:\.[0-9]+)?)', response, re.IGNORECASE)
            if answer_match:
                predicted = float(answer_match.group(1))
            else:
                # Look for $18 or 18 in response
                money_match = re.search(r'\$?([0-9]+(?:\.[0-9]+)?)', response)
                predicted = float(money_match.group(1)) if money_match else 0.0
            
            is_correct = abs(predicted - correct_answer) < 0.01
            
            results[model] = {
                'predicted': predicted,
                'correct': is_correct,
                'time': processing_time,
                'response_preview': response[:200] + "..." if len(response) > 200 else response
            }
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"   {status}: {predicted} (expected: {correct_answer})")
            print(f"   ⏱️  Time: {processing_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[model] = {
                'predicted': 0.0,
                'correct': False,
                'time': 0.0,
                'error': str(e)
            }
    
    # Summary
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    
    for model, result in results.items():
        if 'error' not in result:
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {model}: {result['predicted']} ({result['time']:.1f}s)")
        else:
            print(f"❌ {model}: Error - {result['error']}")
    
    # Verification strength
    correct_models = sum(1 for r in results.values() if r.get('correct', False))
    total_models = len([r for r in results.values() if 'error' not in r])
    
    if total_models > 0:
        consensus_rate = correct_models / total_models
        print(f"\n🎯 Consensus Rate: {consensus_rate:.1%} ({correct_models}/{total_models} models correct)")
        
        if consensus_rate >= 0.8:
            print("✅ Strong consensus - high confidence in result")
        elif consensus_rate >= 0.5:
            print("⚠️ Moderate consensus - some uncertainty")
        else:
            print("❌ Low consensus - significant disagreement")
    
    print(f"\n🔬 VERIFICATION VALUE:")
    print(f"   ✅ Same problem tested across multiple models")
    print(f"   ✅ Independent model reasoning")
    print(f"   ✅ Consensus validation")
    print(f"   ✅ Reproducible results")

if __name__ == "__main__":
    quick_comparison()