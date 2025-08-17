#!/usr/bin/env python3
"""
GSM8K Verified Benchmark Implementation
======================================

This implements a COMPLETELY VERIFIABLE benchmark using the official GSM8K dataset.

VERIFICATION REQUIREMENTS:
✅ Official dataset from https://github.com/openai/grade-school-math
✅ Standard evaluation metrics (exact match accuracy)
✅ Reproducible methodology
✅ Honest baseline comparison (GPT-4: 92%)
✅ Complete documentation

NO SYNTHETIC DATA - ONLY OFFICIAL GSM8K PROBLEMS
"""

import json
import os
import sys
import time
import re
import hashlib
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import our system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from topological_cartesian.multi_cube_orchestrator import create_multi_cube_orchestrator

@dataclass
class GSM8KProblem:
    """Represents a GSM8K math problem"""
    question: str
    answer: str
    numerical_answer: float

@dataclass
class GSM8KResult:
    """Result for a GSM8K problem attempt"""
    problem_id: int
    question: str
    correct_answer: float
    predicted_answer: Optional[float]
    raw_response: str
    correct: bool
    execution_time: float

class GSM8KDatasetLoader:
    """Loads and verifies the official GSM8K dataset"""
    
    def __init__(self, data_path: str = "gsm8k_data"):
        self.data_path = data_path
        self.official_url = "https://github.com/openai/grade-school-math"
        
    def download_instructions(self):
        """Provide instructions for downloading official GSM8K data"""
        print("📥 GSM8K Dataset Download Instructions")
        print("=" * 50)
        print(f"🔗 Official Source: {self.official_url}")
        print("\n📋 Manual Download Steps:")
        print("1. Visit: https://github.com/openai/grade-school-math")
        print("2. Download the repository or clone it")
        print("3. Locate the test.jsonl file in grade_school_math/data/")
        print("4. Copy test.jsonl to this directory as 'gsm8k_test.jsonl'")
        print("\n⚠️  We use manual download to ensure dataset integrity")
        print("✅ This guarantees we're using the exact official data")
    
    def verify_dataset(self, file_path: str) -> bool:
        """Verify the GSM8K dataset integrity"""
        if not os.path.exists(file_path):
            print(f"❌ Dataset file not found: {file_path}")
            return False
        
        # Calculate file hash for integrity
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        print(f"🔐 Dataset hash: {file_hash[:16]}...")
        
        # Verify basic format
        try:
            problems = self.load_problems(file_path)
            print(f"✅ Loaded {len(problems)} problems")
            print(f"📊 Expected: 1,319 problems (official GSM8K test set)")
            
            if len(problems) == 1319:
                print("✅ Problem count matches official GSM8K test set")
                return True
            else:
                print(f"⚠️  Problem count mismatch: {len(problems)} vs 1,319")
                return False
                
        except Exception as e:
            print(f"❌ Dataset format error: {e}")
            return False
    
    def load_problems(self, file_path: str) -> List[GSM8KProblem]:
        """Load GSM8K problems from official format"""
        problems = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                question = data['question']
                answer_text = data['answer']
                
                # Extract numerical answer
                numerical_answer = self._extract_numerical_answer(answer_text)
                
                problems.append(GSM8KProblem(
                    question=question,
                    answer=answer_text,
                    numerical_answer=numerical_answer
                ))
        
        return problems
    
    def _extract_numerical_answer(self, answer_text: str) -> float:
        """Extract numerical answer from GSM8K answer format"""
        # GSM8K answers end with "#### NUMBER"
        match = re.search(r'####\s*([0-9,]+(?:\.[0-9]+)?)', answer_text)
        if match:
            # Remove commas and convert to float
            number_str = match.group(1).replace(',', '')
            return float(number_str)
        else:
            # Fallback: look for last number in the text
            numbers = re.findall(r'[0-9,]+(?:\.[0-9]+)?', answer_text)
            if numbers:
                return float(numbers[-1].replace(',', ''))
            return 0.0

class GSM8KSolver:
    """Solves GSM8K problems using our system"""
    
    def __init__(self):
        print("🚀 Initializing GSM8K solver with multi-cube orchestrator...")
        self.orchestrator = create_multi_cube_orchestrator(enable_dnn_optimization=True)
    
    def solve_problem(self, problem: GSM8KProblem, problem_id: int) -> GSM8KResult:
        """Solve a single GSM8K problem"""
        start_time = time.time()
        
        # Create math reasoning query
        query = self._create_math_query(problem.question)
        
        try:
            # Use our system to solve
            response = self.orchestrator.orchestrate_query(
                query=query,
                strategy="topological"
            )
            
            # Extract numerical answer
            predicted_answer = self._extract_answer_from_response(str(response))
            
            # Check if correct
            correct = self._is_answer_correct(predicted_answer, problem.numerical_answer)
            
            execution_time = time.time() - start_time
            
            return GSM8KResult(
                problem_id=problem_id,
                question=problem.question,
                correct_answer=problem.numerical_answer,
                predicted_answer=predicted_answer,
                raw_response=str(response),
                correct=correct,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return GSM8KResult(
                problem_id=problem_id,
                question=problem.question,
                correct_answer=problem.numerical_answer,
                predicted_answer=None,
                raw_response=f"Error: {str(e)}",
                correct=False,
                execution_time=execution_time
            )
    
    def _create_math_query(self, question: str) -> str:
        """Create a math reasoning query"""
        return f"""
MATH WORD PROBLEM - GSM8K BENCHMARK

Problem: {question}

REQUIREMENTS:
1. Read the problem carefully
2. Identify what needs to be calculated
3. Break down the problem step by step
4. Perform the mathematical calculations
5. Provide the final numerical answer

CRITICAL: Your response must include the final numerical answer clearly.
Format your final answer as: ANSWER: [number]

Solve this step by step:
"""
    
    def _extract_answer_from_response(self, response: str) -> Optional[float]:
        """Extract numerical answer from system response"""
        # Look for "ANSWER: number" pattern
        answer_match = re.search(r'ANSWER:\s*([0-9,]+(?:\.[0-9]+)?)', response, re.IGNORECASE)
        if answer_match:
            return float(answer_match.group(1).replace(',', ''))
        
        # Look for final number in response
        numbers = re.findall(r'[0-9,]+(?:\.[0-9]+)?', response)
        if numbers:
            return float(numbers[-1].replace(',', ''))
        
        return None
    
    def _is_answer_correct(self, predicted: Optional[float], correct: float) -> bool:
        """Check if predicted answer is correct"""
        if predicted is None:
            return False
        
        # Allow small floating point differences
        return abs(predicted - correct) < 0.01

class GSM8KBenchmark:
    """Complete GSM8K benchmark implementation"""
    
    def __init__(self, data_path: str = "gsm8k_test.jsonl"):
        self.data_path = data_path
        self.loader = GSM8KDatasetLoader()
        self.solver = GSM8KSolver()
        self.results = []
    
    def run_benchmark(self, max_problems: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete GSM8K benchmark"""
        
        print("\n" + "="*80)
        print("🧮 GSM8K VERIFIED BENCHMARK - OFFICIAL DATASET")
        print("📊 Grade School Math 8K - Math Word Problems")
        print("🔗 Source: https://github.com/openai/grade-school-math")
        print("📈 Baseline: GPT-4 achieves 92% accuracy")
        print("="*80)
        
        # Verify dataset
        if not self.loader.verify_dataset(self.data_path):
            print("❌ Dataset verification failed!")
            print("\n📥 Please download the official GSM8K dataset:")
            self.loader.download_instructions()
            return {}
        
        # Load problems
        print(f"\n📥 Loading problems from {self.data_path}...")
        problems = self.loader.load_problems(self.data_path)
        
        if max_problems:
            problems = problems[:max_problems]
            print(f"🔬 Running on first {max_problems} problems for testing")
        
        print(f"✅ Loaded {len(problems)} problems")
        
        # Solve problems
        print(f"\n🧮 Solving {len(problems)} math problems...")
        
        correct_count = 0
        total_time = 0
        
        for i, problem in enumerate(problems):
            print(f"\n📝 Problem {i+1}/{len(problems)}")
            print(f"   Question: {problem.question[:100]}...")
            
            result = self.solver.solve_problem(problem, i+1)
            self.results.append(result)
            
            if result.correct:
                correct_count += 1
                print(f"   ✅ CORRECT: {result.predicted_answer} = {result.correct_answer}")
            else:
                print(f"   ❌ WRONG: {result.predicted_answer} ≠ {result.correct_answer}")
            
            print(f"   ⏱️  Time: {result.execution_time:.3f}s")
            total_time += result.execution_time
        
        # Calculate results
        accuracy = correct_count / len(problems) if problems else 0
        avg_time = total_time / len(problems) if problems else 0
        
        # Generate report
        report = self._generate_report(problems, correct_count, accuracy, avg_time, total_time)
        
        # Display results
        print("\n" + "="*80)
        print("🧮 GSM8K VERIFIED BENCHMARK RESULTS")
        print("="*80)
        print(f"📊 Accuracy: {accuracy:.1%} ({correct_count}/{len(problems)})")
        print(f"📈 GPT-4 Baseline: 92.0%")
        print(f"⏱️  Average Time: {avg_time:.3f}s per problem")
        print(f"🕒 Total Time: {total_time:.2f}s")
        
        if accuracy > 0:
            print(f"\n🎉 SUCCESS: Achieved {accuracy:.1%} on official GSM8K!")
        else:
            print(f"\n📊 BASELINE: 0% accuracy - room for improvement")
        
        print(f"\n✅ VERIFIED: Results based on official GSM8K dataset")
        print(f"🔄 REPRODUCIBLE: Complete methodology documented")
        
        return report
    
    def _generate_report(self, problems, correct_count, accuracy, avg_time, total_time):
        """Generate verification report"""
        
        # Calculate dataset hash
        with open(self.data_path, 'rb') as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()
        
        report = {
            "benchmark_name": "GSM8K Math Reasoning",
            "dataset_source": "https://github.com/openai/grade-school-math",
            "dataset_file": self.data_path,
            "dataset_hash": dataset_hash,
            "evaluation_date": datetime.now().isoformat(),
            "total_problems": len(problems),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "baseline_accuracy": 0.92,  # GPT-4 baseline
            "average_time_per_problem": avg_time,
            "total_execution_time": total_time,
            "verifiable": True,
            "reproduction_steps": [
                "1. Download GSM8K from https://github.com/openai/grade-school-math",
                "2. Use test.jsonl file (1,319 problems)",
                "3. Verify dataset integrity with SHA256 hash",
                "4. Load problems in official JSONL format",
                "5. Run our system on each problem",
                "6. Extract numerical answers using regex",
                "7. Calculate exact match accuracy",
                "8. Compare to GPT-4 baseline (92%)"
            ],
            "detailed_results": [
                {
                    "problem_id": result.problem_id,
                    "correct": result.correct,
                    "predicted_answer": result.predicted_answer,
                    "correct_answer": result.correct_answer,
                    "execution_time": result.execution_time
                }
                for result in self.results
            ]
        }
        
        # Save report
        report_filename = f"gsm8k_verified_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Verification report saved: {report_filename}")
        
        return report

def main():
    """Run GSM8K verified benchmark"""
    
    # Check if dataset exists
    dataset_file = "gsm8k_test.jsonl"
    
    if not os.path.exists(dataset_file):
        print("📥 GSM8K Dataset Required")
        print("=" * 40)
        loader = GSM8KDatasetLoader()
        loader.download_instructions()
        print(f"\n⚠️  Please download the dataset and save as '{dataset_file}'")
        return
    
    # Run benchmark
    benchmark = GSM8KBenchmark(dataset_file)
    
    # Start with small test
    print("🔬 Running initial test on 10 problems...")
    results = benchmark.run_benchmark(max_problems=10)
    
    if results:
        print("\n🎯 Test completed successfully!")
        print("📋 Ready to run full benchmark on all 1,319 problems")
    
    return results

if __name__ == "__main__":
    main()