#!/usr/bin/env python3
"""
TCDB Repository Cleanup Implementation
Actually performs the file moves to organize the repository
"""

import os
import shutil
from pathlib import Path
import glob

def create_benchmark_structure():
    """Create the organized benchmark directory structure"""
    
    directories = [
        "benchmarks/official",
        "benchmarks/llm", 
        "benchmarks/development",
        "benchmarks/archive",
        "benchmarks/results",
        "benchmarks/charts",
        "benchmarks/config",
        "benchmarks/docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {directory}")

def move_files():
    """Move files to their proper locations"""
    
    moves = {
        # Official production benchmarks
        "benchmarks/official/": [
            "optimized_public_benchmark.py",
            "benchmark_visualization.py"
        ],
        
        # LLM integration
        "benchmarks/llm/": [
            "tcdb_ollama_integration_framework.py",
            "ollama_embedding_demo.py"
        ],
        
        # Development (needs fixes)
        "benchmarks/development/": [
            "tcdb_ollama_benchmark.py",
            "tcdb_ollama_real_benchmark.py", 
            "public_dataset_benchmark.py",
            "local_public_benchmark.py"
        ],
        
        # Archive/legacy
        "benchmarks/archive/": [
            "vectordbbench_integration.py",
            "run_all_benchmarks.py"
        ],
        
        # Configuration
        "benchmarks/config/": [
            "ollama_config.json",
            "vector_db_config.example.json"
        ],
        
        # Documentation
        "benchmarks/docs/": [
            "OLLAMA_INTEGRATION_GUIDE.md",
            "FINAL_OPTIMIZATION_SUMMARY.md", 
            "COMPREHENSIVE_BENCHMARK_VALIDATION.md",
            "BEIR_BENCHMARK_VALIDATION_RESULTS.md",
            "ENHANCED_BENCHMARK_SUMMARY.md",
            "OFFICIAL_BENCHMARK_RESULTS.md"
        ]
    }
    
    for target_dir, files in moves.items():
        for file in files:
            if Path(file).exists():
                try:
                    shutil.move(file, target_dir)
                    print(f"✅ Moved: {file} → {target_dir}")
                except Exception as e:
                    print(f"❌ Error moving {file}: {e}")
            else:
                print(f"⚠️ Not found: {file}")

def move_result_files():
    """Move all benchmark result files to results directory"""
    
    # Move benchmark result files
    result_patterns = [
        "*benchmark_2025*.json",
        "*benchmark_2025*.csv",
        "*benchmark_2025*.txt",
        "*benchmark_2025*.md",
        "optimized_public_benchmark_*.json",
        "tcdb_ollama_*.json",
        "*_report_*.md",
        "*_report_*.json"
    ]
    
    for pattern in result_patterns:
        for file in glob.glob(pattern):
            try:
                shutil.move(file, "benchmarks/results/")
                print(f"📊 Moved result: {file} → benchmarks/results/")
            except Exception as e:
                print(f"❌ Error moving result {file}: {e}")

def move_chart_files():
    """Move existing charts to charts directory"""
    if Path("benchmark_charts").exists():
        try:
            shutil.move("benchmark_charts", "benchmarks/charts/generated")
            print("📈 Moved: benchmark_charts → benchmarks/charts/generated")
        except Exception as e:
            print(f"❌ Error moving charts: {e}")

def create_readme_files():
    """Create README files for each benchmark directory"""
    
    readmes = {
        "benchmarks/README.md": """# TCDB Benchmarks

This directory contains all TCDB benchmark tests organized by category.

## Directory Structure

- **official/**: Production-ready benchmarks for performance statements
- **llm/**: LLM integration benchmarks using Ollama
- **development/**: Work-in-progress benchmarks (may have issues)  
- **archive/**: Legacy benchmarks kept for reference
- **results/**: All benchmark output files and reports
- **charts/**: Generated performance visualizations
- **config/**: Configuration files for benchmarks
- **docs/**: Benchmark documentation and guides

## Quick Start

For official performance testing, use:
```bash
python benchmarks/official/optimized_public_benchmark.py
```

For generating charts:
```bash
python benchmarks/official/benchmark_visualization.py
```
""",
        
        "benchmarks/official/README.md": """# Official TCDB Benchmarks

✅ **Production-ready benchmarks for official performance statements**

## Available Benchmarks

- `optimized_public_benchmark.py` - Main benchmark comparing TCDB vs competitors
- `benchmark_visualization.py` - Generate professional performance charts

## Usage

```bash
# Run official benchmark
python optimized_public_benchmark.py

# Generate charts
python benchmark_visualization.py
```

These benchmarks produce the **1.5-1.9× performance advantages** used in official statements.
""",
        
        "benchmarks/development/README.md": """# Development Benchmarks

⚠️ **Work-in-progress benchmarks that may have issues**

## Status

- `tcdb_ollama_benchmark.py` - ❌ Has red line errors (needs fixing)
- `public_dataset_benchmark.py` - ❌ Has indentation errors (needs fixing)  
- `tcdb_ollama_real_benchmark.py` - ❓ Status unknown
- `local_public_benchmark.py` - ❓ Local testing version

## Notes

These benchmarks are not ready for production use. Use `../official/` benchmarks instead.
"""
    }
    
    for file_path, content in readmes.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"📝 Created: {file_path}")

def main():
    """Perform the complete repository reorganization"""
    
    print("🏗️ TCDB Repository Cleanup - Implementation")
    print("=" * 60)
    
    print("\n1️⃣ Creating directory structure...")
    create_benchmark_structure()
    
    print("\n2️⃣ Moving benchmark files...")
    move_files()
    
    print("\n3️⃣ Moving result files...")
    move_result_files()
    
    print("\n4️⃣ Moving chart files...")
    move_chart_files()
    
    print("\n5️⃣ Creating README files...")
    create_readme_files()
    
    print("\n" + "=" * 60)
    print("✅ REPOSITORY CLEANUP COMPLETE!")
    print("=" * 60)
    
    print("\n🎯 NEXT STEPS:")
    print("1. Use benchmarks/official/ for production testing")
    print("2. Fix files in benchmarks/development/ before using")
    print("3. Check benchmarks/results/ for all output files")
    print("4. Generate charts from benchmarks/official/benchmark_visualization.py")

if __name__ == "__main__":
    main()
