#!/usr/bin/env python3
"""
TCDB Repository Cleanup and Organization Script
Reorganizes benchmark files into a proper directory structure
"""

import os
import shutil
from pathlib import Path

def reorganize_tcdb_repository():
    """
    Reorganize TCDB repository with proper directory structure
    """
    
    print("🏗️ TCDB Repository Cleanup and Organization")
    print("="*60)
    
    # Define the new structure
    structure = {
        # Official benchmarks (working, production-ready)
        "benchmarks/official/": [
            "optimized_public_benchmark.py",  # ✅ Working - main official benchmark
            "benchmark_visualization.py",     # ✅ Working - chart generation
        ],
        
        # LLM Integration benchmarks  
        "benchmarks/llm/": [
            "tcdb_ollama_integration_framework.py",  # ✅ Working framework
            "ollama_embedding_demo.py",              # ✅ Working demo
        ],
        
        # Development/testing benchmarks (needs fixes)
        "benchmarks/development/": [
            "tcdb_ollama_benchmark.py",      # ❌ Has red line errors
            "tcdb_ollama_real_benchmark.py", # ❓ Status unknown
            "public_dataset_benchmark.py",   # ❌ Has indentation errors  
            "local_public_benchmark.py",     # ❓ Local testing version
        ],
        
        # Legacy/archived benchmarks
        "benchmarks/archive/": [
            "vectordbbench_integration.py",  # Old VectorDBBench attempt
            "run_all_benchmarks.py",         # Legacy runner
        ],
        
        # Results and data
        "benchmarks/results/": [],  # For all benchmark output files
        "benchmarks/charts/": [],   # For generated visualizations
        
        # Configuration
        "benchmarks/config/": [
            "ollama_config.json",
            "vector_db_config.example.json",
        ],
        
        # Documentation
        "benchmarks/docs/": [
            "OLLAMA_INTEGRATION_GUIDE.md",
            "FINAL_OPTIMIZATION_SUMMARY.md",
            "COMPREHENSIVE_BENCHMARK_VALIDATION.md",
        ]
    }
    
    # Create directories
    for directory in structure.keys():
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Move files (simulation - would actually move in real script)
    for directory, files in structure.items():
        for file in files:
            if Path(file).exists():
                print(f"  📦 Would move: {file} → {directory}")
            else:
                print(f"  ⚠️ Not found: {file}")
    
    # Additional cleanup recommendations
    cleanup_recommendations = {
        "Remove duplicate result files": [
            "*benchmark_20*.json",
            "*benchmark_20*.csv", 
            "*benchmark_20*.txt",
        ],
        "Consolidate logs": [
            "Move all logs/ content to benchmarks/results/logs/",
        ],
        "Clean up root directory": [
            "Keep only: src/, tests/, docs/, README.md, requirements.txt"
        ]
    }
    
    print("\n🧹 Additional Cleanup Recommendations:")
    for category, items in cleanup_recommendations.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    return structure

if __name__ == "__main__":
    structure = reorganize_tcdb_repository()
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDED FINAL STRUCTURE:")
    print("="*60)
    
    recommended_structure = """
    📁 topological-cartesian-db/
    ├── 📂 src/                          # Core TCDB source code
    ├── 📂 tests/                        # Unit tests
    ├── 📂 docs/                         # Documentation
    ├── 📂 benchmarks/                   # 🆕 ORGANIZED BENCHMARKS
    │   ├── 📂 official/                 # ✅ Production-ready benchmarks
    │   │   ├── optimized_public_benchmark.py
    │   │   └── benchmark_visualization.py
    │   ├── 📂 llm/                      # LLM integration benchmarks
    │   │   ├── tcdb_ollama_integration_framework.py
    │   │   └── ollama_embedding_demo.py
    │   ├── 📂 development/              # Work-in-progress benchmarks
    │   │   ├── tcdb_ollama_benchmark.py (needs fixes)
    │   │   └── public_dataset_benchmark.py (needs fixes)
    │   ├── 📂 archive/                  # Legacy benchmarks
    │   ├── 📂 results/                  # All benchmark outputs
    │   ├── 📂 charts/                   # Generated visualizations
    │   ├── 📂 config/                   # Configuration files
    │   └── 📂 docs/                     # Benchmark documentation
    ├── 📂 data/                         # Dataset cache
    ├── 📂 logs/                         # System logs
    ├── 📄 README.md                     # Main documentation
    ├── 📄 requirements.txt              # Dependencies
    └── 📄 .gitignore                    # Git configuration
    """
    
    print(recommended_structure)
    
    print("\n🏆 BENEFITS OF THIS STRUCTURE:")
    print("  ✅ Clear separation of working vs broken benchmarks")
    print("  ✅ Easy to find official benchmarks for performance statements")  
    print("  ✅ Development benchmarks isolated for fixing")
    print("  ✅ All results organized in one place")
    print("  ✅ Clean root directory")
    print("  ✅ Scalable for future benchmark additions")
