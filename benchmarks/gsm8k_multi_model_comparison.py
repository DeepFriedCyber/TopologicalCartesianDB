#!/usr/bin/env python3
"""
GSM8K Multi-Model Comparison Benchmark
=====================================

Tests multiple Ollama models on the same GSM8K problems for comparative analysis.
This provides much stronger verifiable results by showing consistency across models.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator, HybridCoordinateLLM
from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

class MultiModelGSM8KBenchmark:
    """
    Comparative GSM8K benchmark across multiple Ollama models.
    
    This provides stronger evidence by testing the same problems
    with different models to show consistency and reliability.
    """
    
    def __init__(self, 
                 dataset_file: str = "gsm8k_test.jsonl",
                 test_problems: int = 5):
        """
        Initialize multi-model benchmark.
        
        Args:
            dataset_file: Path to GSM8K dataset
            test_problems: Number of problems to test (same for all models)
        """
        self.dataset_file = dataset_file
        self.test_problems = test_problems
        
        # Initialize coordinate engine (shared across all models)
        print("🚀 Initializing coordinate engine...")
        self.coordinate_engine = EnhancedCoordinateEngine()
        
        # Load test problems
        self.problems = self._load_problems()
        
        # Define models to test
        self.test_models = [
            {
                "name": "llama3.2:3b",
                "description": "Llama 3.2 3B - Balanced performance",
                "size": "2.0 GB",
                "expected_strength": "General reasoning"
            },
            {
                "name": "llama3.2:1b", 
                "description": "Llama 3.2 1B - Ultra-fast",
                "size": "1.3 GB",
                "expected_strength": "Speed"
            },
            {
                "name": "codellama:7b-code",
                "description": "CodeLlama 7B - Code specialized",
                "size": "3.8 GB", 
                "expected_strength": "Mathematical reasoning"
            },
            {
                "name": "mistral:latest",
                "description": "Mistral 7B - General purpose",
                "size": "4.1 GB",
                "expected_strength": "Balanced performance"
            },
            {
                "name": "gemma:2b",
                "description": "Gemma 2B - Google model",
                "size": "1.7 GB",
                "expected_strength": "Efficiency"
            }
        ]
        
        print(f"✅ Loaded {len(self.problems)} problems")
        print(f"🤖 Will test {len(self.test_models)} models")
    
    def _load_problems(self) -> List[Dict[str, Any]]:
        """Load GSM8K problems from dataset file"""
        problems = []
        
        if not os.path.exists(self.dataset_file):
            print(f"❌ Dataset file not found: {self.dataset_file}")
            return []
        
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > self.test_problems:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract numerical answer
                        answer_text = data['answer']
                        numerical_answer = self._extract_numerical_answer(answer_text)
                        
                        problem = {
                            'id': line_num,
                            'question': data['question'],
                            'answer_text': answer_text,
                            'numerical_answer': numerical_answer
                        }
                        
                        problems.append(problem)
                        
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON at line {line_num}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error loading problems: {e}")
            return []
        
        return problems
    
    def _extract_numerical_answer(self, answer_text: str) -> float:
        """Extract numerical answer from GSM8K answer text"""
        import re
        
        # GSM8K answers end with #### followed by the numerical answer
        match = re.search(r'####\s*([0-9,]+(?:\.[0-9]+)?)', answer_text)
        if match:
            # Remove commas and convert to float
            number_str = match.group(1).replace(',', '')
            try:
                return float(number_str)
            except ValueError:
                pass
        
        # Fallback: look for any number at the end
        numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', answer_text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                pass
        
        return 0.0
    
    def _test_model(self, model_info: Dict[str, str]) -> Dict[str, Any]:
        """Test a specific model on all problems"""
        model_name = model_info["name"]
        
        print(f"\n🤖 Testing model: {model_name}")
        print(f"   Description: {model_info['description']}")
        print(f"   Size: {model_info['size']}")
        print(f"   Expected strength: {model_info['expected_strength']}")
        
        # Initialize Ollama for this model
        try:
            ollama = OllamaLLMIntegrator(default_model=model_name)
            hybrid_system = HybridCoordinateLLM(self.coordinate_engine, ollama)
        except Exception as e:
            print(f"❌ Failed to initialize {model_name}: {e}")
            return {
                "model_name": model_name,
                "model_info": model_info,
                "error": str(e),
                "results": [],
                "summary": {
                    "total_problems": 0,
                    "correct_answers": 0,
                    "accuracy": 0.0,
                    "average_time": 0.0,
                    "total_time": 0.0
                }
            }
        
        results = []
        correct_count = 0
        total_time = 0.0
        
        for i, problem in enumerate(self.problems, 1):
            print(f"   📝 Problem {i}/{len(self.problems)}: {problem['question'][:50]}...")
            
            # Create math prompt
            math_prompt = f"""
MATH WORD PROBLEM - GSM8K BENCHMARK

Problem: {problem['question']}

INSTRUCTIONS:
1. Read the problem carefully and identify what needs to be calculated
2. Break down the problem into clear, logical steps
3. Perform the mathematical calculations step by step
4. Show your work clearly
5. Provide the final numerical answer

CRITICAL: Your response must end with the final numerical answer in this format:
FINAL ANSWER: [number]

Solve this step by step:
"""
            
            start_time = time.time()
            
            try:
                # Process with hybrid system
                result = hybrid_system.process_query(
                    query=math_prompt,
                    max_context_docs=2,
                    model=model_name,
                    temperature=0.1  # Low temperature for consistent math
                )
                
                processing_time = time.time() - start_time
                
                # Extract numerical answer
                llm_response = result.get('llm_response', '')
                predicted_answer = self._extract_llm_answer(llm_response)
                
                # Check correctness
                is_correct = abs(predicted_answer - problem['numerical_answer']) < 0.01
                if is_correct:
                    correct_count += 1
                
                total_time += processing_time
                
                problem_result = {
                    'problem_id': problem['id'],
                    'question': problem['question'],
                    'predicted_answer': predicted_answer,
                    'correct_answer': problem['numerical_answer'],
                    'correct': is_correct,
                    'processing_time': processing_time,
                    'llm_response': llm_response[:500] + "..." if len(llm_response) > 500 else llm_response,
                    'success': result.get('success', False)
                }
                
                results.append(problem_result)
                
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                print(f"      {status}: {predicted_answer} (expected: {problem['numerical_answer']}) - {processing_time:.1f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                total_time += processing_time
                
                problem_result = {
                    'problem_id': problem['id'],
                    'question': problem['question'],
                    'predicted_answer': 0.0,
                    'correct_answer': problem['numerical_answer'],
                    'correct': False,
                    'processing_time': processing_time,
                    'llm_response': '',
                    'success': False,
                    'error': str(e)
                }
                
                results.append(problem_result)
                print(f"      ❌ ERROR: {e} - {processing_time:.1f}s")
        
        # Calculate summary statistics
        accuracy = correct_count / len(self.problems) if self.problems else 0.0
        avg_time = total_time / len(self.problems) if self.problems else 0.0
        
        model_result = {
            "model_name": model_name,
            "model_info": model_info,
            "results": results,
            "summary": {
                "total_problems": len(self.problems),
                "correct_answers": correct_count,
                "accuracy": accuracy,
                "average_time": avg_time,
                "total_time": total_time
            }
        }
        
        print(f"   📊 {model_name} Results: {accuracy:.1%} accuracy ({correct_count}/{len(self.problems)}) - {avg_time:.1f}s avg")
        
        return model_result
    
    def _extract_llm_answer(self, llm_response: str) -> float:
        """Extract numerical answer from LLM response"""
        import re
        
        # Look for "FINAL ANSWER: number" pattern
        final_answer_match = re.search(r'FINAL ANSWER:\s*([0-9,]+(?:\.[0-9]+)?)', llm_response, re.IGNORECASE)
        if final_answer_match:
            try:
                return float(final_answer_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        # Look for numbers at the end of the response
        numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', llm_response)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                pass
        
        return 0.0
    
    def run_comparative_benchmark(self) -> Dict[str, Any]:
        """Run comparative benchmark across all models"""
        print("\n" + "=" * 80)
        print("🧮 GSM8K MULTI-MODEL COMPARATIVE BENCHMARK")
        print("📊 Testing multiple Ollama models on the same problems")
        print("🎯 This provides stronger verifiable evidence!")
        print("=" * 80)
        
        print(f"📊 Testing {len(self.problems)} problems across {len(self.test_models)} models")
        print(f"🎯 Same problems, different models = comparative validation")
        
        all_results = []
        
        for model_info in self.test_models:
            try:
                model_result = self._test_model(model_info)
                all_results.append(model_result)
            except Exception as e:
                print(f"❌ Failed to test {model_info['name']}: {e}")
                continue
        
        # Generate comparative analysis
        comparative_report = self._generate_comparative_report(all_results)
        
        # Save detailed report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"gsm8k_multi_model_comparison_{timestamp}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(comparative_report, f, indent=2)
        
        # Display results
        self._display_comparative_results(comparative_report)
        
        print(f"\n📄 Detailed report saved: {report_filename}")
        
        return comparative_report
    
    def _generate_comparative_report(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive comparative analysis"""
        
        # Calculate cross-model statistics
        successful_models = [r for r in all_results if r.get('summary', {}).get('total_problems', 0) > 0]
        
        if not successful_models:
            return {
                "error": "No models completed testing successfully",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "all_results": all_results
            }
        
        # Best performing model
        best_model = max(successful_models, key=lambda x: x['summary']['accuracy'])
        
        # Fastest model
        fastest_model = min(successful_models, key=lambda x: x['summary']['average_time'])
        
        # Problem-by-problem analysis
        problem_analysis = []
        for i in range(len(self.problems)):
            problem = self.problems[i]
            problem_results = []
            
            for model_result in successful_models:
                if i < len(model_result['results']):
                    result = model_result['results'][i]
                    problem_results.append({
                        'model': model_result['model_name'],
                        'correct': result['correct'],
                        'predicted_answer': result['predicted_answer'],
                        'processing_time': result['processing_time']
                    })
            
            # Calculate consensus
            correct_count = sum(1 for r in problem_results if r['correct'])
            consensus_rate = correct_count / len(problem_results) if problem_results else 0.0
            
            problem_analysis.append({
                'problem_id': problem['id'],
                'question': problem['question'],
                'correct_answer': problem['numerical_answer'],
                'model_results': problem_results,
                'consensus_rate': consensus_rate,
                'difficulty_assessment': 'Easy' if consensus_rate >= 0.8 else 'Medium' if consensus_rate >= 0.5 else 'Hard'
            })
        
        return {
            "benchmark_name": "GSM8K Multi-Model Comparative Analysis",
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_problems": len(self.problems),
            "models_tested": len(successful_models),
            "models_failed": len(all_results) - len(successful_models),
            
            "summary_statistics": {
                "best_accuracy": best_model['summary']['accuracy'],
                "best_model": best_model['model_name'],
                "fastest_model": fastest_model['model_name'],
                "fastest_time": fastest_model['summary']['average_time'],
                "accuracy_range": {
                    "min": min(r['summary']['accuracy'] for r in successful_models),
                    "max": max(r['summary']['accuracy'] for r in successful_models),
                    "average": sum(r['summary']['accuracy'] for r in successful_models) / len(successful_models)
                },
                "time_range": {
                    "min": min(r['summary']['average_time'] for r in successful_models),
                    "max": max(r['summary']['average_time'] for r in successful_models),
                    "average": sum(r['summary']['average_time'] for r in successful_models) / len(successful_models)
                }
            },
            
            "model_rankings": {
                "by_accuracy": sorted(successful_models, key=lambda x: x['summary']['accuracy'], reverse=True),
                "by_speed": sorted(successful_models, key=lambda x: x['summary']['average_time'])
            },
            
            "problem_analysis": problem_analysis,
            "detailed_results": all_results,
            
            "verification_info": {
                "dataset_source": "Official GSM8K from OpenAI",
                "same_problems_all_models": True,
                "reproducible": True,
                "open_source": True,
                "local_processing": True
            }
        }
    
    def _display_comparative_results(self, report: Dict[str, Any]):
        """Display comparative results in a clear format"""
        print("\n" + "=" * 80)
        print("🏆 GSM8K MULTI-MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            return
        
        summary = report['summary_statistics']
        
        print(f"📊 Models Tested: {report['models_tested']}")
        print(f"📝 Problems: {report['test_problems']}")
        print(f"🏆 Best Accuracy: {summary['best_accuracy']:.1%} ({summary['best_model']})")
        print(f"⚡ Fastest Model: {summary['fastest_model']} ({summary['fastest_time']:.1f}s)")
        
        print(f"\n📈 Accuracy Range: {summary['accuracy_range']['min']:.1%} - {summary['accuracy_range']['max']:.1%}")
        print(f"⏱️  Time Range: {summary['time_range']['min']:.1f}s - {summary['time_range']['max']:.1f}s")
        
        print(f"\n🏆 MODEL RANKINGS BY ACCURACY:")
        for i, model in enumerate(report['model_rankings']['by_accuracy'], 1):
            accuracy = model['summary']['accuracy']
            time_avg = model['summary']['average_time']
            print(f"   {i}. {model['model_name']}: {accuracy:.1%} ({model['summary']['correct_answers']}/{model['summary']['total_problems']}) - {time_avg:.1f}s avg")
        
        print(f"\n⚡ MODEL RANKINGS BY SPEED:")
        for i, model in enumerate(report['model_rankings']['by_speed'], 1):
            accuracy = model['summary']['accuracy']
            time_avg = model['summary']['average_time']
            print(f"   {i}. {model['model_name']}: {time_avg:.1f}s avg - {accuracy:.1%} accuracy")
        
        print(f"\n🎯 PROBLEM DIFFICULTY ANALYSIS:")
        problem_analysis = report['problem_analysis']
        easy_problems = sum(1 for p in problem_analysis if p['difficulty_assessment'] == 'Easy')
        medium_problems = sum(1 for p in problem_analysis if p['difficulty_assessment'] == 'Medium')
        hard_problems = sum(1 for p in problem_analysis if p['difficulty_assessment'] == 'Hard')
        
        print(f"   Easy (≥80% models correct): {easy_problems}")
        print(f"   Medium (50-79% models correct): {medium_problems}")
        print(f"   Hard (<50% models correct): {hard_problems}")
        
        print(f"\n✅ VERIFICATION STRENGTHS:")
        print(f"   ✅ Same problems tested across all models")
        print(f"   ✅ Official GSM8K dataset from OpenAI")
        print(f"   ✅ Complete reproducibility (open source)")
        print(f"   ✅ Local processing (privacy preserved)")
        print(f"   ✅ Multiple model validation")

def main():
    """Run multi-model comparative benchmark"""
    print("🚀 GSM8K Multi-Model Comparative Benchmark")
    print("=" * 60)
    print("Testing multiple Ollama models on the same problems")
    print("This provides much stronger verifiable evidence!")
    
    # Check if dataset exists
    dataset_file = "gsm8k_test.jsonl"
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        print("💡 Please ensure GSM8K dataset is available")
        return
    
    try:
        # Initialize and run comparative benchmark
        benchmark = MultiModelGSM8KBenchmark(
            dataset_file=dataset_file,
            test_problems=5  # Same 5 problems for all models
        )
        
        results = benchmark.run_comparative_benchmark()
        
        if 'error' not in results:
            print(f"\n🎉 Multi-model comparison completed successfully!")
            print(f"📊 This provides much stronger verifiable evidence")
            print(f"🔬 Multiple models tested on identical problems")
            print(f"✅ Results can be independently verified")
        else:
            print(f"\n❌ Comparison failed: {results['error']}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print(f"💡 Make sure Ollama is running: ollama serve")

if __name__ == "__main__":
    main()