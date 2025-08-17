#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite
=============================

This module runs a complete benchmarking suite for the Topological Cartesian Cube system,
including:

1. VERSES AI Comparison (existing benchmarks)
2. Kaggle LLM Performance Benchmarking (new)
3. Custom performance tests
4. Comparative analysis and reporting

This provides a comprehensive evaluation against multiple baselines and datasets.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our benchmark modules
from verses_comparison_suite import VERSESBenchmarkSuite
from kaggle_llm_benchmark import KaggleLLMBenchmark
from setup_kaggle_data import KaggleDatasetSetup

class ComprehensiveBenchmarkSuite:
    """
    Master benchmark suite that orchestrates all testing approaches
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the comprehensive benchmark suite
        
        Args:
            config: Configuration dictionary for benchmark parameters
        """
        self.config = config or self.get_default_config()
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Initialize benchmark components
        self.verses_suite = None
        self.kaggle_benchmark = None
        
        print("🚀 Initializing Comprehensive Benchmark Suite")
        print("=" * 60)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default benchmark configuration"""
        return {
            "run_verses_benchmark": True,
            "run_kaggle_benchmark": True,
            "run_custom_tests": True,
            "verses_config": {
                "mastermind_trials": 10,
                "reasoning_trials": 5
            },
            "kaggle_config": {
                "use_synthetic_data": True,
                "test_scenarios": [
                    {"name": "lightweight", "context_length": 500, "complexity": 1, "queries": 100},
                    {"name": "medium", "context_length": 2000, "complexity": 3, "queries": 50},
                    {"name": "heavy", "context_length": 8000, "complexity": 5, "queries": 25}
                ]
            },
            "custom_config": {
                "stress_test": True,
                "scalability_test": True,
                "memory_efficiency_test": True
            },
            "output": {
                "save_detailed_results": True,
                "generate_visualizations": True,
                "create_summary_report": True
            }
        }
    
    def setup_benchmarks(self):
        """Initialize all benchmark components"""
        print("🔧 Setting up benchmark components...")
        
        # Setup VERSES benchmark
        if self.config["run_verses_benchmark"]:
            try:
                self.verses_suite = VERSESBenchmarkSuite()
                print("✅ VERSES benchmark suite initialized")
            except Exception as e:
                print(f"⚠️  VERSES benchmark setup failed: {e}")
                self.config["run_verses_benchmark"] = False
        
        # Setup Kaggle benchmark
        if self.config["run_kaggle_benchmark"]:
            try:
                # First ensure dataset is available
                dataset_setup = KaggleDatasetSetup()
                dataset_path = None
                
                if os.path.exists(dataset_setup.dataset_file):
                    dataset_path = dataset_setup.dataset_file
                elif not self.config["kaggle_config"]["use_synthetic_data"]:
                    print("📥 Setting up Kaggle dataset...")
                    if dataset_setup.setup_dataset():
                        dataset_path = dataset_setup.dataset_file
                
                self.kaggle_benchmark = KaggleLLMBenchmark(dataset_path)
                print("✅ Kaggle benchmark suite initialized")
                
            except Exception as e:
                print(f"⚠️  Kaggle benchmark setup failed: {e}")
                self.config["run_kaggle_benchmark"] = False
        
        print("🎯 Benchmark setup complete")
    
    def run_verses_benchmark(self) -> Dict[str, Any]:
        """Run the VERSES AI comparison benchmark"""
        print("\n" + "🔥" * 20 + " VERSES AI BENCHMARK " + "🔥" * 20)
        
        if not self.verses_suite:
            return {"error": "VERSES benchmark not available"}
        
        try:
            results = self.verses_suite.run_full_benchmark_suite()
            
            # Extract key metrics for summary
            summary = {
                "benchmark_type": "VERSES_AI_Comparison",
                "status": "completed",
                "mastermind_results": len(results.get("mastermind_benchmark", {}).get("results", [])),
                "reasoning_results": len(results.get("reasoning_benchmark", {}).get("results", [])),
                "vs_openai_o1": results.get("mastermind_benchmark", {}).get("comparison_vs_openai_o1", {}),
                "vs_deepseek_r1": results.get("reasoning_benchmark", {}).get("comparison_vs_deepseek_r1", {}),
                "execution_time": time.time() - self.start_time if self.start_time else 0
            }
            
            return {
                "summary": summary,
                "detailed_results": results
            }
            
        except Exception as e:
            print(f"❌ VERSES benchmark failed: {e}")
            return {"error": str(e), "benchmark_type": "VERSES_AI_Comparison"}
    
    def run_kaggle_benchmark(self) -> Dict[str, Any]:
        """Run the Kaggle LLM performance benchmark"""
        print("\n" + "📊" * 20 + " KAGGLE LLM BENCHMARK " + "📊" * 20)
        
        if not self.kaggle_benchmark:
            return {"error": "Kaggle benchmark not available"}
        
        try:
            results = self.kaggle_benchmark.run_full_kaggle_benchmark()
            
            # Extract key metrics for summary
            our_perf = results.get("performance_comparison", {}).get("our_performance", {})
            summary = {
                "benchmark_type": "Kaggle_LLM_Performance",
                "status": "completed",
                "test_scenarios": len(results.get("test_scenarios", [])),
                "our_avg_latency_ms": our_perf.get("avg_latency_ms", 0),
                "our_avg_throughput": our_perf.get("avg_throughput_qps", 0),
                "our_avg_accuracy": our_perf.get("avg_accuracy", 0),
                "models_compared": len(results.get("performance_comparison", {}).get("comparisons", [])),
                "visualization_file": results.get("visualization_file", ""),
                "execution_time": time.time() - self.start_time if self.start_time else 0
            }
            
            return {
                "summary": summary,
                "detailed_results": results
            }
            
        except Exception as e:
            print(f"❌ Kaggle benchmark failed: {e}")
            return {"error": str(e), "benchmark_type": "Kaggle_LLM_Performance"}
    
    def run_custom_tests(self) -> Dict[str, Any]:
        """Run custom performance tests"""
        print("\n" + "⚡" * 20 + " CUSTOM PERFORMANCE TESTS " + "⚡" * 20)
        
        custom_results = {
            "benchmark_type": "Custom_Performance_Tests",
            "tests": {}
        }
        
        config = self.config["custom_config"]
        
        # Stress test
        if config.get("stress_test", False):
            print("🔥 Running stress test...")
            stress_result = self.run_stress_test()
            custom_results["tests"]["stress_test"] = stress_result
        
        # Scalability test
        if config.get("scalability_test", False):
            print("📈 Running scalability test...")
            scalability_result = self.run_scalability_test()
            custom_results["tests"]["scalability_test"] = scalability_result
        
        # Memory efficiency test
        if config.get("memory_efficiency_test", False):
            print("💾 Running memory efficiency test...")
            memory_result = self.run_memory_efficiency_test()
            custom_results["tests"]["memory_efficiency_test"] = memory_result
        
        return custom_results
    
    def run_stress_test(self) -> Dict[str, Any]:
        """Run stress test with high load"""
        try:
            from topological_cartesian.multi_cube_orchestrator import create_multi_cube_orchestrator
            
            orchestrator = create_multi_cube_orchestrator(enable_dnn_optimization=True)
            
            # Generate high-load scenario
            num_concurrent_queries = 100
            query_complexity = 5
            
            start_time = time.time()
            successful_queries = 0
            failed_queries = 0
            
            for i in range(num_concurrent_queries):
                try:
                    query = f"Complex stress test query {i} with high computational demands"
                    result = orchestrator.orchestrate_query(query, strategy="adaptive")
                    successful_queries += 1
                except Exception:
                    failed_queries += 1
            
            execution_time = time.time() - start_time
            
            return {
                "status": "completed",
                "total_queries": num_concurrent_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": successful_queries / num_concurrent_queries,
                "execution_time": execution_time,
                "queries_per_second": num_concurrent_queries / execution_time,
                "avg_query_time": execution_time / num_concurrent_queries
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def run_scalability_test(self) -> Dict[str, Any]:
        """Test system scalability with increasing load"""
        try:
            from topological_cartesian.multi_cube_orchestrator import create_multi_cube_orchestrator
            
            orchestrator = create_multi_cube_orchestrator(enable_dnn_optimization=True)
            
            # Test with increasing query counts
            test_loads = [10, 25, 50, 100, 200]
            results = []
            
            for load in test_loads:
                start_time = time.time()
                successful = 0
                
                for i in range(load):
                    try:
                        query = f"Scalability test query {i} for load {load}"
                        result = orchestrator.orchestrate_query(query, strategy="adaptive")
                        successful += 1
                    except Exception:
                        pass
                
                execution_time = time.time() - start_time
                throughput = successful / execution_time if execution_time > 0 else 0
                
                results.append({
                    "load": load,
                    "successful_queries": successful,
                    "execution_time": execution_time,
                    "throughput": throughput,
                    "success_rate": successful / load
                })
            
            # Calculate scalability metrics
            throughputs = [r["throughput"] for r in results]
            scalability_factor = throughputs[-1] / throughputs[0] if throughputs[0] > 0 else 0
            
            return {
                "status": "completed",
                "test_loads": test_loads,
                "results": results,
                "scalability_factor": scalability_factor,
                "linear_scalability": scalability_factor >= 0.8  # 80% of linear scaling
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def run_memory_efficiency_test(self) -> Dict[str, Any]:
        """Test memory efficiency under various loads"""
        try:
            import psutil
            import gc
            
            from topological_cartesian.multi_cube_orchestrator import create_multi_cube_orchestrator
            
            # Get baseline memory
            gc.collect()
            baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            orchestrator = create_multi_cube_orchestrator(enable_dnn_optimization=True)
            
            # Test memory usage with different context sizes
            context_sizes = [1000, 5000, 10000, 20000]
            memory_results = []
            
            for context_size in context_sizes:
                gc.collect()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Generate large context query
                large_context = " ".join([f"context_item_{i}" for i in range(context_size)])
                query = f"Process large context with {context_size} items: {large_context[:100]}..."
                
                try:
                    result = orchestrator.orchestrate_query(query, strategy="adaptive")
                    success = True
                except Exception:
                    success = False
                
                gc.collect()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = end_memory - start_memory
                
                memory_results.append({
                    "context_size": context_size,
                    "memory_used_mb": memory_used,
                    "memory_per_item": memory_used / context_size if context_size > 0 else 0,
                    "success": success
                })
            
            # Calculate memory efficiency
            memory_per_item = [r["memory_per_item"] for r in memory_results if r["success"]]
            avg_memory_per_item = sum(memory_per_item) / len(memory_per_item) if memory_per_item else 0
            
            return {
                "status": "completed",
                "baseline_memory_mb": baseline_memory,
                "context_tests": memory_results,
                "avg_memory_per_item": avg_memory_per_item,
                "memory_efficient": avg_memory_per_item < 0.01  # Less than 0.01 MB per item
            }
            
        except ImportError:
            return {"status": "skipped", "reason": "psutil not available"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report"""
        print("\n" + "📋" * 20 + " GENERATING COMPREHENSIVE REPORT " + "📋" * 20)
        
        report = {
            "benchmark_suite": "Comprehensive Topological Cartesian Cube Evaluation",
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": self.end_time - self.start_time if self.start_time and self.end_time else 0,
            "configuration": self.config,
            "results": self.results,
            "summary": self.generate_executive_summary(),
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of all benchmark results"""
        summary = {
            "overall_status": "completed",
            "benchmarks_run": [],
            "key_findings": [],
            "performance_highlights": []
        }
        
        # Analyze VERSES results
        if "verses_benchmark" in self.results and "error" not in self.results["verses_benchmark"]:
            summary["benchmarks_run"].append("VERSES AI Comparison")
            
            verses_summary = self.results["verses_benchmark"]["summary"]
            vs_openai = verses_summary.get("vs_openai_o1", {})
            vs_deepseek = verses_summary.get("vs_deepseek_r1", {})
            
            if vs_openai:
                summary["key_findings"].append(f"Mastermind vs OpenAI o1: {vs_openai.get('speed_improvement', 'N/A')}")
            if vs_deepseek:
                summary["key_findings"].append(f"Reasoning vs DeepSeek R1: {vs_deepseek.get('speed_improvement', 'N/A')}")
        
        # Analyze Kaggle results
        if "kaggle_benchmark" in self.results and "error" not in self.results["kaggle_benchmark"]:
            summary["benchmarks_run"].append("Kaggle LLM Performance")
            
            kaggle_summary = self.results["kaggle_benchmark"]["summary"]
            summary["performance_highlights"].append(
                f"Average latency: {kaggle_summary.get('our_avg_latency_ms', 0):.1f}ms"
            )
            summary["performance_highlights"].append(
                f"Average accuracy: {kaggle_summary.get('our_avg_accuracy', 0):.1%}"
            )
            summary["performance_highlights"].append(
                f"Compared against {kaggle_summary.get('models_compared', 0)} LLM models"
            )
        
        # Analyze custom test results
        if "custom_tests" in self.results:
            summary["benchmarks_run"].append("Custom Performance Tests")
            
            custom_tests = self.results["custom_tests"].get("tests", {})
            
            if "stress_test" in custom_tests:
                stress = custom_tests["stress_test"]
                if stress.get("status") == "completed":
                    summary["performance_highlights"].append(
                        f"Stress test: {stress.get('success_rate', 0):.1%} success rate"
                    )
            
            if "scalability_test" in custom_tests:
                scalability = custom_tests["scalability_test"]
                if scalability.get("status") == "completed":
                    summary["performance_highlights"].append(
                        f"Scalability factor: {scalability.get('scalability_factor', 0):.2f}"
                    )
        
        return summary
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        # Analyze results and provide recommendations
        if "kaggle_benchmark" in self.results:
            kaggle_summary = self.results["kaggle_benchmark"].get("summary", {})
            avg_latency = kaggle_summary.get("our_avg_latency_ms", 0)
            
            if avg_latency > 100:
                recommendations.append("Consider optimizing query processing to reduce latency below 100ms")
            
            avg_accuracy = kaggle_summary.get("our_avg_accuracy", 0)
            if avg_accuracy < 0.8:
                recommendations.append("Focus on improving accuracy through better topological feature extraction")
        
        if "custom_tests" in self.results:
            custom_tests = self.results["custom_tests"].get("tests", {})
            
            if "scalability_test" in custom_tests:
                scalability = custom_tests["scalability_test"]
                if scalability.get("scalability_factor", 0) < 0.5:
                    recommendations.append("Investigate scalability bottlenecks for better performance under load")
            
            if "memory_efficiency_test" in custom_tests:
                memory = custom_tests["memory_efficiency_test"]
                if not memory.get("memory_efficient", True):
                    recommendations.append("Optimize memory usage for large context processing")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges across all benchmarks")
        
        return recommendations
    
    def save_results(self, report: Dict[str, Any]):
        """Save benchmark results to files"""
        if not self.config["output"]["save_detailed_results"]:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save comprehensive report
        report_file = f"comprehensive_benchmark_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Comprehensive report saved to: {report_file}")
        
        # Save summary report
        if self.config["output"]["create_summary_report"]:
            summary_file = f"benchmark_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                self.write_summary_report(f, report)
            
            print(f"📋 Summary report saved to: {summary_file}")
    
    def write_summary_report(self, file, report: Dict[str, Any]):
        """Write a human-readable summary report"""
        file.write("=" * 80 + "\n")
        file.write("TOPOLOGICAL CARTESIAN CUBE - COMPREHENSIVE BENCHMARK REPORT\n")
        file.write("=" * 80 + "\n\n")
        
        file.write(f"Benchmark Date: {report['timestamp']}\n")
        file.write(f"Total Execution Time: {report['total_execution_time']:.2f} seconds\n\n")
        
        # Executive Summary
        summary = report["summary"]
        file.write("EXECUTIVE SUMMARY\n")
        file.write("-" * 40 + "\n")
        file.write(f"Status: {summary['overall_status']}\n")
        file.write(f"Benchmarks Run: {', '.join(summary['benchmarks_run'])}\n\n")
        
        if summary["key_findings"]:
            file.write("Key Findings:\n")
            for finding in summary["key_findings"]:
                file.write(f"  • {finding}\n")
            file.write("\n")
        
        if summary["performance_highlights"]:
            file.write("Performance Highlights:\n")
            for highlight in summary["performance_highlights"]:
                file.write(f"  • {highlight}\n")
            file.write("\n")
        
        # Recommendations
        file.write("RECOMMENDATIONS\n")
        file.write("-" * 40 + "\n")
        for i, rec in enumerate(report["recommendations"], 1):
            file.write(f"{i}. {rec}\n")
        file.write("\n")
        
        # Detailed Results Summary
        file.write("DETAILED RESULTS SUMMARY\n")
        file.write("-" * 40 + "\n")
        
        for benchmark_name, benchmark_result in report["results"].items():
            file.write(f"\n{benchmark_name.upper().replace('_', ' ')}:\n")
            
            if "error" in benchmark_result:
                file.write(f"  Status: FAILED - {benchmark_result['error']}\n")
            else:
                file.write(f"  Status: COMPLETED\n")
                
                if "summary" in benchmark_result:
                    summary_data = benchmark_result["summary"]
                    for key, value in summary_data.items():
                        if key not in ["benchmark_type", "status"]:
                            file.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        
        file.write("\n" + "=" * 80 + "\n")
        file.write("End of Report\n")
        file.write("=" * 80 + "\n")
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final benchmark summary to console"""
        print("\n" + "🏆" * 30 + " FINAL RESULTS " + "🏆" * 30)
        print(f"Benchmark Suite: {report['benchmark_suite']}")
        print(f"Completion Time: {report['timestamp']}")
        print(f"Total Execution: {report['total_execution_time']:.2f} seconds")
        print("=" * 90)
        
        summary = report["summary"]
        print(f"\n📊 BENCHMARKS COMPLETED: {', '.join(summary['benchmarks_run'])}")
        
        if summary["key_findings"]:
            print(f"\n🔍 KEY FINDINGS:")
            for finding in summary["key_findings"]:
                print(f"  • {finding}")
        
        if summary["performance_highlights"]:
            print(f"\n⚡ PERFORMANCE HIGHLIGHTS:")
            for highlight in summary["performance_highlights"]:
                print(f"  • {highlight}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 90)
        print("🎉 Comprehensive benchmark suite completed successfully!")
        print("=" * 90)
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        print("🚀 Starting Comprehensive Benchmark Suite")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Setup all benchmarks
        self.setup_benchmarks()
        
        # Run VERSES benchmark
        if self.config["run_verses_benchmark"]:
            self.results["verses_benchmark"] = self.run_verses_benchmark()
        
        # Run Kaggle benchmark
        if self.config["run_kaggle_benchmark"]:
            self.results["kaggle_benchmark"] = self.run_kaggle_benchmark()
        
        # Run custom tests
        if self.config["run_custom_tests"]:
            self.results["custom_tests"] = self.run_custom_tests()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save results
        self.save_results(report)
        
        # Print final summary
        self.print_final_summary(report)
        
        return report

def create_config_from_args(args) -> Dict[str, Any]:
    """Create configuration from command line arguments"""
    config = {
        "run_verses_benchmark": not args.skip_verses,
        "run_kaggle_benchmark": not args.skip_kaggle,
        "run_custom_tests": not args.skip_custom,
        "verses_config": {
            "mastermind_trials": args.mastermind_trials,
            "reasoning_trials": args.reasoning_trials
        },
        "kaggle_config": {
            "use_synthetic_data": args.synthetic_data,
            "test_scenarios": [
                {"name": "lightweight", "context_length": 500, "complexity": 1, "queries": args.light_queries},
                {"name": "medium", "context_length": 2000, "complexity": 3, "queries": args.medium_queries},
                {"name": "heavy", "context_length": 8000, "complexity": 5, "queries": args.heavy_queries}
            ]
        },
        "custom_config": {
            "stress_test": not args.skip_stress,
            "scalability_test": not args.skip_scalability,
            "memory_efficiency_test": not args.skip_memory
        },
        "output": {
            "save_detailed_results": not args.no_save,
            "generate_visualizations": not args.no_viz,
            "create_summary_report": not args.no_summary
        }
    }
    
    return config

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Benchmark Suite for Topological Cartesian Cube System"
    )
    
    # Benchmark selection
    parser.add_argument("--skip-verses", action="store_true", 
                       help="Skip VERSES AI comparison benchmark")
    parser.add_argument("--skip-kaggle", action="store_true",
                       help="Skip Kaggle LLM performance benchmark")
    parser.add_argument("--skip-custom", action="store_true",
                       help="Skip custom performance tests")
    
    # VERSES configuration
    parser.add_argument("--mastermind-trials", type=int, default=10,
                       help="Number of Mastermind trials for VERSES benchmark")
    parser.add_argument("--reasoning-trials", type=int, default=5,
                       help="Number of reasoning trials for VERSES benchmark")
    
    # Kaggle configuration
    parser.add_argument("--synthetic-data", action="store_true",
                       help="Use synthetic data for Kaggle benchmark")
    parser.add_argument("--light-queries", type=int, default=100,
                       help="Number of lightweight queries")
    parser.add_argument("--medium-queries", type=int, default=50,
                       help="Number of medium complexity queries")
    parser.add_argument("--heavy-queries", type=int, default=25,
                       help="Number of heavy complexity queries")
    
    # Custom test configuration
    parser.add_argument("--skip-stress", action="store_true",
                       help="Skip stress test")
    parser.add_argument("--skip-scalability", action="store_true",
                       help="Skip scalability test")
    parser.add_argument("--skip-memory", action="store_true",
                       help="Skip memory efficiency test")
    
    # Output configuration
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save detailed results")
    parser.add_argument("--no-viz", action="store_true",
                       help="Don't generate visualizations")
    parser.add_argument("--no-summary", action="store_true",
                       help="Don't create summary report")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run benchmark suite
    suite = ComprehensiveBenchmarkSuite(config)
    results = suite.run_full_benchmark_suite()
    
    return results

if __name__ == "__main__":
    main()