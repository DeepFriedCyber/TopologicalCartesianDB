#!/usr/bin/env python3
"""
Validation Helper Functions
==========================

Utility functions to assist third-party validators in verifying
our performance claims and system architecture.
"""

import time
import json
import random
import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

@dataclass
class ValidationResult:
    """Result of a validation check"""
    test_name: str
    passed: bool
    details: str
    measured_value: Any = None
    expected_value: Any = None
    variance: float = 0.0

class SystemValidator:
    """Validates system architecture and performance claims"""
    
    def __init__(self):
        self.results = []
        
    def validate_multi_cube_architecture(self) -> ValidationResult:
        """Verify that multi-cube architecture is properly instantiated"""
        try:
            from topological_cartesian.multi_cube_orchestrator import create_multi_cube_orchestrator
            
            # Create orchestrator
            orchestrator = create_multi_cube_orchestrator(enable_dnn_optimization=True)
            
            # Check cube count
            cube_count = len(orchestrator.cubes)
            expected_min_cubes = 5
            
            if cube_count >= expected_min_cubes:
                details = f"✅ Found {cube_count} cubes (expected ≥{expected_min_cubes})"
                passed = True
            else:
                details = f"❌ Found only {cube_count} cubes (expected ≥{expected_min_cubes})"
                passed = False
            
            # Check DNN optimization
            dnn_enabled = orchestrator.enable_dnn_optimization
            if dnn_enabled:
                details += "\n✅ DNN optimization enabled"
            else:
                details += "\n❌ DNN optimization disabled"
                passed = False
            
            # Check orchestration strategies
            strategies = list(orchestrator.orchestration_engine.orchestration_strategies.keys())
            required_strategies = ['adaptive', 'parallel']
            
            missing_strategies = [s for s in required_strategies if s not in strategies]
            if not missing_strategies:
                details += f"\n✅ All required orchestration strategies available: {strategies}"
            else:
                details += f"\n❌ Missing strategies: {missing_strategies}"
                passed = False
            
            return ValidationResult(
                test_name="multi_cube_architecture",
                passed=passed,
                details=details,
                measured_value=cube_count,
                expected_value=expected_min_cubes
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="multi_cube_architecture",
                passed=False,
                details=f"❌ Failed to instantiate multi-cube system: {str(e)}"
            )
    
    def validate_benchmark_reproducibility(self, num_trials: int = 3) -> ValidationResult:
        """Verify that benchmark results are reproducible"""
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'benchmarks'))
            from verses_comparison_suite import VERSESBenchmarkSuite
            
            # Run same test multiple times with same seed
            results = []
            
            for trial in range(num_trials):
                # Set consistent random seed
                random.seed(42)
                np.random.seed(42)
                
                # Run single mastermind trial
                suite = VERSESBenchmarkSuite()
                trial_results = suite.run_mastermind_benchmark(num_trials=1)
                
                if trial_results:
                    results.append({
                        'success': trial_results[0].success,
                        'execution_time': trial_results[0].execution_time,
                        'iterations': trial_results[0].iterations_required
                    })
            
            # Check if results are consistent
            if len(results) == num_trials:
                # Check success consistency
                success_rates = [r['success'] for r in results]
                success_consistent = len(set(success_rates)) == 1
                
                # Check timing consistency (within 10% variance)
                times = [r['execution_time'] for r in results]
                avg_time = sum(times) / len(times)
                max_variance = max(abs(t - avg_time) / avg_time for t in times) if avg_time > 0 else 0
                timing_consistent = max_variance < 0.1
                
                if success_consistent and timing_consistent:
                    details = f"✅ Reproducible results across {num_trials} trials\n"
                    details += f"   Success rate: {success_rates[0]}\n"
                    details += f"   Avg time: {avg_time:.4f}s (variance: {max_variance:.2%})"
                    passed = True
                else:
                    details = f"❌ Results not consistent across {num_trials} trials\n"
                    details += f"   Success rates: {success_rates}\n"
                    details += f"   Times: {times}\n"
                    details += f"   Max variance: {max_variance:.2%}"
                    passed = False
                
                return ValidationResult(
                    test_name="benchmark_reproducibility",
                    passed=passed,
                    details=details,
                    measured_value=max_variance,
                    expected_value=0.1,
                    variance=max_variance
                )
            else:
                return ValidationResult(
                    test_name="benchmark_reproducibility",
                    passed=False,
                    details=f"❌ Could not complete {num_trials} trials (got {len(results)})"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="benchmark_reproducibility",
                passed=False,
                details=f"❌ Reproducibility test failed: {str(e)}"
            )
    
    def validate_timing_accuracy(self) -> ValidationResult:
        """Verify that timing measurements are accurate"""
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'benchmarks'))
            from verses_comparison_suite import VERSESBenchmarkSuite
            
            suite = VERSESBenchmarkSuite()
            
            # Manual timing measurement
            start_time = time.time()
            trial_results = suite.run_mastermind_benchmark(num_trials=1)
            manual_time = time.time() - start_time
            
            if trial_results:
                reported_time = trial_results[0].execution_time
                
                # Allow for some overhead in manual timing
                time_diff = abs(manual_time - reported_time)
                acceptable_overhead = 0.1  # 100ms overhead acceptable
                
                if time_diff <= acceptable_overhead:
                    details = f"✅ Timing accuracy verified\n"
                    details += f"   Reported: {reported_time:.4f}s\n"
                    details += f"   Manual: {manual_time:.4f}s\n"
                    details += f"   Difference: {time_diff:.4f}s"
                    passed = True
                else:
                    details = f"❌ Timing measurements inaccurate\n"
                    details += f"   Reported: {reported_time:.4f}s\n"
                    details += f"   Manual: {manual_time:.4f}s\n"
                    details += f"   Difference: {time_diff:.4f}s (>{acceptable_overhead}s)"
                    passed = False
                
                return ValidationResult(
                    test_name="timing_accuracy",
                    passed=passed,
                    details=details,
                    measured_value=time_diff,
                    expected_value=acceptable_overhead
                )
            else:
                return ValidationResult(
                    test_name="timing_accuracy",
                    passed=False,
                    details="❌ No benchmark results to validate timing"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="timing_accuracy",
                passed=False,
                details=f"❌ Timing validation failed: {str(e)}"
            )
    
    def validate_performance_claims(self) -> ValidationResult:
        """Validate that performance claims are within reasonable bounds"""
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'benchmarks'))
            from verses_comparison_suite import VERSESBenchmarkSuite
            
            suite = VERSESBenchmarkSuite()
            
            # Run mastermind benchmark
            mastermind_results = suite.run_mastermind_benchmark(num_trials=5)
            
            # Calculate performance metrics
            successful_trials = [r for r in mastermind_results if r.success]
            success_rate = len(successful_trials) / len(mastermind_results)
            
            if successful_trials:
                avg_time = sum(r.execution_time for r in successful_trials) / len(successful_trials)
                
                # Expected performance bounds
                expected_success_rate = 0.8  # At least 80%
                expected_max_time = 1.0      # Should be under 1 second
                
                success_ok = success_rate >= expected_success_rate
                timing_ok = avg_time <= expected_max_time
                
                if success_ok and timing_ok:
                    details = f"✅ Performance within expected bounds\n"
                    details += f"   Success rate: {success_rate:.1%} (≥{expected_success_rate:.1%})\n"
                    details += f"   Avg time: {avg_time:.4f}s (≤{expected_max_time}s)"
                    passed = True
                else:
                    details = f"❌ Performance below expectations\n"
                    details += f"   Success rate: {success_rate:.1%} (expected ≥{expected_success_rate:.1%})\n"
                    details += f"   Avg time: {avg_time:.4f}s (expected ≤{expected_max_time}s)"
                    passed = False
                
                return ValidationResult(
                    test_name="performance_claims",
                    passed=passed,
                    details=details,
                    measured_value={'success_rate': success_rate, 'avg_time': avg_time},
                    expected_value={'success_rate': expected_success_rate, 'max_time': expected_max_time}
                )
            else:
                return ValidationResult(
                    test_name="performance_claims",
                    passed=False,
                    details="❌ No successful trials to evaluate performance"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="performance_claims",
                passed=False,
                details=f"❌ Performance validation failed: {str(e)}"
            )
    
    def run_full_validation(self) -> Dict[str, ValidationResult]:
        """Run complete validation suite"""
        print("🔍 Starting Third-Party Validation Suite")
        print("=" * 50)
        
        validation_tests = [
            ("Multi-Cube Architecture", self.validate_multi_cube_architecture),
            ("Benchmark Reproducibility", self.validate_benchmark_reproducibility),
            ("Timing Accuracy", self.validate_timing_accuracy),
            ("Performance Claims", self.validate_performance_claims)
        ]
        
        results = {}
        
        for test_name, test_func in validation_tests:
            print(f"\n🧪 Running: {test_name}")
            result = test_func()
            results[result.test_name] = result
            
            if result.passed:
                print(f"✅ PASSED: {test_name}")
            else:
                print(f"❌ FAILED: {test_name}")
            
            print(f"Details: {result.details}")
        
        # Summary
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        
        print(f"\n📊 Validation Summary")
        print("=" * 30)
        print(f"Passed: {passed_count}/{total_count}")
        print(f"Success Rate: {passed_count/total_count:.1%}")
        
        if passed_count == total_count:
            print("🎉 ALL VALIDATIONS PASSED")
        else:
            print("⚠️  SOME VALIDATIONS FAILED")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult]) -> str:
        """Generate a formatted validation report"""
        report = []
        report.append("# Third-Party Validation Report")
        report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Validator**: Independent Third Party")
        report.append("")
        
        # Summary
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        
        report.append("## Executive Summary")
        report.append(f"- **Tests Run**: {total_count}")
        report.append(f"- **Tests Passed**: {passed_count}")
        report.append(f"- **Success Rate**: {passed_count/total_count:.1%}")
        report.append("")
        
        if passed_count == total_count:
            report.append("**Overall Result**: ✅ ALL VALIDATIONS PASSED")
        else:
            report.append("**Overall Result**: ❌ SOME VALIDATIONS FAILED")
        
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            report.append(f"### {result.test_name.replace('_', ' ').title()}")
            report.append(f"**Status**: {status}")
            report.append("")
            report.append("**Details**:")
            report.append("```")
            report.append(result.details)
            report.append("```")
            report.append("")
            
            if result.measured_value is not None:
                report.append(f"**Measured Value**: {result.measured_value}")
            if result.expected_value is not None:
                report.append(f"**Expected Value**: {result.expected_value}")
            if result.variance > 0:
                report.append(f"**Variance**: {result.variance:.2%}")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        failed_tests = [r for r in results.values() if not r.passed]
        if not failed_tests:
            report.append("- All validation tests passed successfully")
            report.append("- Performance claims appear to be accurate and reproducible")
            report.append("- Multi-cube architecture is properly implemented and functional")
            report.append("- System is ready for production validation")
        else:
            report.append("- The following issues were identified:")
            for result in failed_tests:
                report.append(f"  - {result.test_name}: {result.details.split('❌')[1].split('\\n')[0] if '❌' in result.details else 'Failed'}")
            report.append("- Recommend addressing these issues before proceeding with validation")
        
        return "\n".join(report)

def main():
    """Main validation entry point"""
    validator = SystemValidator()
    results = validator.run_full_validation()
    
    # Generate report
    report = validator.generate_validation_report(results)
    
    # Save report
    report_filename = f"validation_report_{int(time.time())}.md"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Validation report saved to: {report_filename}")
    
    return results

if __name__ == "__main__":
    main()