#!/usr/bin/env python3
"""
Two-Phase TCDB Benchmark with Fast Ollama Model (llama3.2:1b)

This version uses a smaller, faster model for quicker testing.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path to import the main benchmark
sys.path.append(str(Path(__file__).parent))

from test_two_phase_ollama import TwoPhaseBenchmarkWithOllama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_fast_benchmark():
    """Run benchmark with fast model"""
    
    print("🚀 Two-Phase TCDB Benchmark with Fast Ollama Model")
    print("=" * 60)
    print("Using llama3.2:1b for faster generation")
    print("=" * 60)
    
    # Configuration for fast testing
    ollama_model = "llama3.2:1b"  # Smaller, faster model
    dataset_name = "ms_marco_synthetic"
    dataset_size = 500  # Smaller dataset for faster testing
    
    print(f"\n📊 Fast Test Configuration:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Size: {dataset_size} documents")
    print(f"   Ollama Model: {ollama_model}")
    
    benchmark = TwoPhaseBenchmarkWithOllama(ollama_model)
    
    try:
        print(f"   Systems: {list(benchmark.systems.keys())}")
        print(f"   Ollama Available: {benchmark.ollama_llm.available}")
        
        # Phase 1: Pure Database Performance
        print(f"\n🔄 Running Phase 1 (Fast)...")
        phase1_results = await benchmark.run_phase1(dataset_name, dataset_size)
        
        # Phase 2: End-to-End RAG Performance
        print(f"\n🔄 Running Phase 2 (Fast)...")
        phase2_result = await benchmark.run_phase2(dataset_name, dataset_size)
        
        # Generate report
        print(f"\n📄 Generating fast benchmark report...")
        report = benchmark.generate_report(phase1_results, phase2_result)
        
        # Save report
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"two_phase_fast_benchmark_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 FAST BENCHMARK RESULTS")
        print("=" * 60)
        print(report)
        
        print(f"\n📁 Fast benchmark report saved: {report_file}")
        
        if benchmark.ollama_llm.available:
            print("\n🎉 Fast Two-Phase Benchmark Completed Successfully!")
            print("⚡ Faster model provides quicker responses!")
        else:
            print("\n⚠️ Phase 2 incomplete due to Ollama unavailability")
        
    except Exception as e:
        logger.error(f"❌ Fast benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_fast_benchmark())