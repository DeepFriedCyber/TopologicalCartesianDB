#!/usr/bin/env python3
"""
Environment Test Script
=======================

This script tests that the virtual environment is properly set up
for the benchmarking system.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas:", pd.__version__)
    except ImportError as e:
        print("❌ pandas:", e)
        return False
    
    try:
        import numpy as np
        print("✅ numpy:", np.__version__)
    except ImportError as e:
        print("❌ numpy:", e)
        return False
    
    try:
        import matplotlib
        print("✅ matplotlib:", matplotlib.__version__)
    except ImportError as e:
        print("❌ matplotlib:", e)
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn:", sns.__version__)
    except ImportError as e:
        print("❌ seaborn:", e)
        return False
    
    try:
        import psutil
        print("✅ psutil:", psutil.__version__)
    except ImportError as e:
        print("❌ psutil:", e)
        return False
    
    try:
        import requests
        print("✅ requests:", requests.__version__)
    except ImportError as e:
        print("❌ requests:", e)
        return False
    
    try:
        import plotly
        print("✅ plotly:", plotly.__version__)
    except ImportError as e:
        print("❌ plotly:", e)
        return False
    
    try:
        import scipy
        print("✅ scipy:", scipy.__version__)
    except ImportError as e:
        print("❌ scipy:", e)
        return False
    
    try:
        import tqdm
        print("✅ tqdm:", tqdm.__version__)
    except ImportError as e:
        print("❌ tqdm:", e)
        return False
    
    try:
        import pytest
        print("✅ pytest:", pytest.__version__)
    except ImportError as e:
        print("❌ pytest:", e)
        return False
    
    try:
        import yaml
        print("✅ pyyaml:", yaml.__version__)
    except ImportError as e:
        print("❌ pyyaml:", e)
        return False
    
    try:
        import jsonschema
        print("✅ jsonschema:", jsonschema.__version__)
    except ImportError as e:
        print("❌ jsonschema:", e)
        return False
    
    # Optional packages
    try:
        import torch
        print("✅ torch:", torch.__version__)
    except ImportError as e:
        print("⚠️  torch (optional):", e)
    
    try:
        import kaggle
        print("✅ kaggle:", kaggle.__version__)
    except ImportError as e:
        print("⚠️  kaggle (optional):", e)
    
    return True

def test_dataset():
    """Test that the synthetic dataset is available"""
    print("\n📊 Testing dataset availability...")
    
    dataset_paths = [
        "kaggle_data/Open_LLM_Perf_Leaderboard.csv",
        "../kaggle_data/Open_LLM_Perf_Leaderboard.csv",
        "c:/Users/aps33/Projects/topological-cartesian-db/kaggle_data/Open_LLM_Perf_Leaderboard.csv"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                import pandas as pd
                df = pd.read_csv(path)
                print(f"✅ Dataset found at: {path}")
                print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                print(f"   Models: {df['Model'].nunique()} unique models")
                return True
            except Exception as e:
                print(f"❌ Dataset found but couldn't load: {e}")
    
    print("⚠️  Dataset not found, but can be created with setup_kaggle_data.py")
    return False

def test_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    print(f"   Current working directory: {os.getcwd()}")
    
    try:
        import psutil
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        print("   System info unavailable (psutil not installed)")

def test_benchmark_scripts():
    """Test that benchmark scripts can be imported"""
    print("\n🔧 Testing benchmark script imports...")
    
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        from setup_kaggle_data import KaggleDatasetSetup
        print("✅ setup_kaggle_data.py imports successfully")
    except ImportError as e:
        print(f"❌ setup_kaggle_data.py import failed: {e}")
        return False
    
    try:
        from kaggle_llm_benchmark import KaggleLLMBenchmark
        print("✅ kaggle_llm_benchmark.py imports successfully")
    except ImportError as e:
        print(f"❌ kaggle_llm_benchmark.py import failed: {e}")
        return False
    
    # Test VERSES comparison (might fail due to missing torch/main system)
    try:
        from verses_comparison_suite import VERSESBenchmarkSuite
        print("✅ verses_comparison_suite.py imports successfully")
    except ImportError as e:
        print(f"⚠️  verses_comparison_suite.py import failed (expected): {e}")
    
    try:
        from comprehensive_benchmark_suite import ComprehensiveBenchmarkSuite
        print("✅ comprehensive_benchmark_suite.py imports successfully")
    except ImportError as e:
        print(f"⚠️  comprehensive_benchmark_suite.py import failed (expected): {e}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Benchmarking Environment Setup")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test dataset
    dataset_ok = test_dataset()
    
    # Test system info
    test_system_info()
    
    # Test benchmark scripts
    scripts_ok = test_benchmark_scripts()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"   Core imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Dataset: {'✅ AVAILABLE' if dataset_ok else '⚠️  MISSING'}")
    print(f"   Benchmark scripts: {'✅ PASS' if scripts_ok else '❌ FAIL'}")
    
    if imports_ok and scripts_ok:
        print("\n🎉 Environment is ready for benchmarking!")
        print("\nNext steps:")
        print("1. Run: python setup_kaggle_data.py")
        print("2. Run: python kaggle_llm_benchmark.py")
        print("3. Install torch if needed: pip install torch")
        print("4. Run full suite: python comprehensive_benchmark_suite.py")
    else:
        print("\n⚠️  Environment needs attention. Check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()