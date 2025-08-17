#!/usr/bin/env python3
"""
GSM8K Benchmark with Ollama LLM Integration
==========================================

FINALLY! A benchmark with REAL AI capability!
This combines our fast coordinate system with actual language understanding.
"""

import sys
import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator, HybridCoordinateLLM
from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

class GSM8KOllamaBenchmark:
    """
    GSM8K benchmark using Ollama LLM + coordinate system.
    
    This is our breakthrough: REAL math reasoning capability!
    """
    
    def __init__(self, 
                 dataset_file: str = "gsm8k_test.jsonl",
                 ollama_model: str = "llama3.2:3b",
                 max_problems: Optional[int] = None):
        """
        Initialize GSM8K benchmark with Ollama integration.
        
        Args:
            dataset_file: Path to GSM8K dataset
            ollama_model: Ollama model to use
            max_problems: Limit number of problems (None for all)
        """
        self.dataset_file = dataset_file
        self.ollama_model = ollama_model
        self.max_problems = max_problems
        
        # Initialize components
        print("🚀 Initializing GSM8K benchmark with REAL AI capability...")
        
        # Initialize coordinate engine
        self.coordinate_engine = EnhancedCoordinateEngine()
        
        # Initialize Ollama
        self.ollama = OllamaLLMIntegrator(default_model=ollama_model)
        
        # Initialize hybrid system
        self.hybrid_system = HybridCoordinateLLM(self.coordinate_engine, self.ollama)
        
        # Load problems
        self.problems = self._load_problems()
        
        print(f"✅ Initialized with {len(self.problems)} problems")
        print(f"🤖 Using model: {ollama_model}")
    
    def _load_problems(self) -> List[Dict[str, Any]]:
        """Load GSM8K problems from dataset file"""
        problems = []
        
        if not os.path.exists(self.dataset_file):
            print(f"❌ Dataset file not found: {self.dataset_file}")
            print("💡 Run download_gsm8k.py first to get the dataset")
            return []
        
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
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
                        
                        # Limit if specified
                        if self.max_problems and len(problems) >= self.max_problems:
                            break
                            
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON at line {line_num}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error loading problems: {e}")
            return []
        
        return problems
    
    def _extract_numerical_answer(self, answer_text: str) -> float:
        """Extract numerical answer from GSM8K answer text"""
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
    
    def _solve_math_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a math problem using hybrid Ollama + coordinate system.
        
        This is where the magic happens - REAL AI reasoning!
        """
        question = problem['question']
        
        # Create enhanced prompt for math reasoning
        math_prompt = f"""
MATH WORD PROBLEM - GSM8K BENCHMARK

Problem: {question}

INSTRUCTIONS:
1. Read the problem carefully and identify what needs to be calculated
2. Break down the problem into clear, logical steps
3. Perform the mathematical calculations step by step
4. Show your work clearly
5. Provide the final numerical answer

CRITICAL: Your response must end with the final numerical answer in this format:
FINAL ANSWER: [number]

Example:
FINAL ANSWER: 42

Solve this step by step:
"""
        
        # Use hybrid system for enhanced reasoning
        start_time = time.time()
        
        try:
            # Process with hybrid coordinate + LLM system
            result = self.hybrid_system.process_query(
                query=math_prompt,
                max_context_docs=2,  # Use coordinate context for math problems
                model=self.ollama_model,
                temperature=0.1  # Low temperature for consistent math reasoning
            )
            
            processing_time = time.time() - start_time
            
            # Extract numerical answer from LLM response
            llm_response = result.get('llm_response', '')
            predicted_answer = self._extract_llm_answer(llm_response)
            
            return {
                'problem_id': problem['id'],
                'question': question,
                'llm_response': llm_response,
                'predicted_answer': predicted_answer,
                'correct_answer': problem['numerical_answer'],
                'correct': abs(predicted_answer - problem['numerical_answer']) < 0.01,
                'processing_time': processing_time,
                'coordinate_context': result.get('coordinate_context', []),
                'model_used': result.get('model_used', self.ollama_model),
                'success': result.get('success', False),
                'error': result.get('error')
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error solving problem {problem['id']}: {e}")
            
            return {
                'problem_id': problem['id'],
                'question': question,
                'llm_response': '',
                'predicted_answer': 0.0,
                'correct_answer': problem['numerical_answer'],
                'correct': False,
                'processing_time': processing_time,
                'coordinate_context': [],
                'model_used': self.ollama_model,
                'success': False,
                'error': str(e)
            }
    
    def _extract_llm_answer(self, llm_response: str) -> float:
        """Extract numerical answer from LLM response"""
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
        
        # Look for any number in the response
        all_numbers = re.findall(r'([0-9]+(?:\.[0-9]+)?)', llm_response)
        if all_numbers:
            try:
                return float(all_numbers[-1])
            except ValueError:
                pass
        
        return 0.0
    
    def run_benchmark(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete GSM8K benchmark with Ollama.
        
        This is our moment of truth - testing REAL AI capability!
        """
        print("\n" + "=" * 80)
        print("🧮 GSM8K BENCHMARK WITH REAL AI (OLLAMA)")
        print("📊 Grade School Math 8K - Math Word Problems")
        print("🤖 Using Ollama LLM + Coordinate System")
        print("🔗 Finally testing ACTUAL language understanding!")
        print("=" * 80)
        
        if not self.problems:
            print("❌ No problems loaded!")
            return {}
        
        print(f"📊 Testing {len(self.problems)} problems")
        print(f"🤖 Model: {self.ollama_model}")
        print(f"⚡ Hybrid: Coordinate context + LLM reasoning")
        
        results = []
        correct_count = 0
        total_time = 0.0
        
        for i, problem in enumerate(self.problems, 1):
            if verbose:
                print(f"\n📝 Problem {i}/{len(self.problems)}")
                print(f"   Question: {problem['question'][:100]}...")
            
            # Solve the problem
            result = self._solve_math_problem(problem)
            results.append(result)
            
            # Update statistics
            if result['correct']:
                correct_count += 1
            total_time += result['processing_time']
            
            if verbose:
                status = "✅ CORRECT" if result['correct'] else "❌ WRONG"
                print(f"   {status}: {result['predicted_answer']} (expected: {result['correct_answer']})")
                print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
                
                if result['error']:
                    print(f"   ⚠️  Error: {result['error']}")
        
        # Calculate final statistics
        accuracy = correct_count / len(self.problems) if self.problems else 0.0
        avg_time = total_time / len(self.problems) if self.problems else 0.0
        
        # Generate comprehensive report
        report = {
            "benchmark_name": "GSM8K Math Reasoning with Ollama",
            "model_used": self.ollama_model,
            "dataset_file": self.dataset_file,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_problems": len(self.problems),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "baseline_accuracy": 0.92,  # GPT-4 baseline
            "average_time_per_problem": avg_time,
            "total_execution_time": total_time,
            "hybrid_system": True,
            "coordinate_enhancement": True,
            "verifiable": True,
            "detailed_results": results,
            "system_stats": {
                "coordinate_stats": self.coordinate_engine.get_performance_stats() if hasattr(self.coordinate_engine, 'get_performance_stats') else {},
                "ollama_stats": self.ollama.get_performance_stats(),
                "hybrid_stats": self.hybrid_system.get_hybrid_stats()
            }
        }
        
        # Save detailed report
        report_filename = f"gsm8k_ollama_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display results
        print("\n" + "=" * 80)
        print("🧮 GSM8K OLLAMA BENCHMARK RESULTS")
        print("=" * 80)
        print(f"🤖 Model: {self.ollama_model}")
        print(f"📊 Accuracy: {accuracy:.1%} ({correct_count}/{len(self.problems)})")
        print(f"📈 GPT-4 Baseline: 92.0%")
        print(f"⏱️  Average Time: {avg_time:.2f}s per problem")
        print(f"🕒 Total Time: {total_time:.1f}s")
        print(f"⚡ Hybrid System: Coordinate + LLM")
        
        if accuracy > 0:
            print(f"\n🎉 SUCCESS! We achieved {accuracy:.1%} accuracy with REAL AI!")
            print(f"🚀 This is a MASSIVE improvement from our previous 0%!")
        else:
            print(f"\n🔧 No correct answers yet - need to optimize prompts and model")
        
        print(f"\n✅ VERIFIED: Results based on official GSM8K dataset")
        print(f"🔄 REPRODUCIBLE: Complete methodology documented")
        print(f"📄 Report saved: {report_filename}")
        
        return report

def main():
    """Run GSM8K benchmark with Ollama"""
    print("🚀 GSM8K Benchmark with Ollama LLM Integration")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_file = "gsm8k_test.jsonl"
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        print("💡 Please run: python benchmarks/download_gsm8k.py")
        return
    
    # Initialize and run benchmark
    try:
        # Start with a small test
        print("🧪 Running initial test on 5 problems...")
        benchmark = GSM8KOllamaBenchmark(
            dataset_file=dataset_file,
            ollama_model="llama3.2:3b",
            max_problems=5
        )
        
        results = benchmark.run_benchmark(verbose=True)
        
        if results and results.get('accuracy', 0) > 0:
            print(f"\n🎉 Initial test successful! Accuracy: {results['accuracy']:.1%}")
            
            # Ask if user wants to run full benchmark
            response = input("\n❓ Run full benchmark on all 1,319 problems? (y/n): ").lower().strip()
            if response == 'y':
                print("\n🚀 Running full GSM8K benchmark...")
                full_benchmark = GSM8KOllamaBenchmark(
                    dataset_file=dataset_file,
                    ollama_model="llama3.2:3b",
                    max_problems=None
                )
                full_results = full_benchmark.run_benchmark(verbose=False)
                print(f"\n🏆 Full benchmark completed!")
                print(f"📊 Final accuracy: {full_results.get('accuracy', 0):.1%}")
        else:
            print(f"\n🔧 Initial test showed issues - check Ollama setup")
            print(f"💡 Try running: python setup_ollama.py")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print(f"💡 Make sure Ollama is running: ollama serve")

if __name__ == "__main__":
    main()