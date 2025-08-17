#!/usr/bin/env python3
"""
Show Mathematical Models Discovery in Action

This test demonstrates exactly which mathematical models are discovered
and how many are active in different scenarios.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.multi_cube_math_lab import (
    MultiCubeMathLaboratory, MathModelType
)

def show_available_models():
    """Show all available mathematical models"""
    
    print("🧮 AVAILABLE MATHEMATICAL MODELS")
    print("=" * 50)
    
    models = list(MathModelType)
    print(f"📊 Total Available Models: {len(models)}")
    print()
    
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model.value}")
    
    return models

def show_model_discovery():
    """Show mathematical model discovery process"""
    
    print("\n🔬 MATHEMATICAL MODEL DISCOVERY PROCESS")
    print("=" * 50)
    
    # Create math lab
    math_lab = MultiCubeMathLaboratory()
    
    # Show initial cube specializations
    print("🧊 Initial Cube Specializations:")
    for cube_name, spec in math_lab.cube_specializations.items():
        print(f"\n   {cube_name.upper()}:")
        print(f"      Primary: {spec.primary_model.value}")
        print(f"      Secondary: {[m.value for m in spec.secondary_models]}")
        
        total_models = 1 + len(spec.secondary_models)
        print(f"      Total Models: {total_models}")
    
    # Count unique models across all cubes
    all_models = set()
    for spec in math_lab.cube_specializations.values():
        all_models.add(spec.primary_model.value)
        all_models.update(m.value for m in spec.secondary_models)
    
    print(f"\n🎯 UNIQUE MODELS ACROSS ALL CUBES: {len(all_models)}")
    for i, model in enumerate(sorted(all_models), 1):
        print(f"   {i}. {model}")
    
    return all_models

def simulate_evolution():
    """Simulate evolution and show model changes"""
    
    print("\n🧬 EVOLUTION SIMULATION")
    print("=" * 50)
    
    # Create test coordinates
    coordinates = {
        'code_cube': np.random.rand(20, 5),
        'data_cube': np.random.rand(20, 5),
        'user_cube': np.random.rand(20, 5)
    }
    
    math_lab = MultiCubeMathLaboratory()
    
    print("🔬 Running mathematical experiments...")
    
    # Run experiments
    experiment_results = math_lab.run_parallel_experiments(coordinates, max_workers=2)
    
    # Count successful experiments by model
    model_success_count = {}
    model_total_count = {}
    
    for cube_name, experiments in experiment_results.items():
        for exp in experiments:
            model_name = exp.model_type.value
            
            if model_name not in model_total_count:
                model_total_count[model_name] = 0
                model_success_count[model_name] = 0
            
            model_total_count[model_name] += 1
            if exp.success:
                model_success_count[model_name] += 1
    
    print(f"\n📊 EXPERIMENT RESULTS:")
    print(f"   Models Tested: {len(model_total_count)}")
    print(f"   Total Experiments: {sum(model_total_count.values())}")
    print(f"   Successful Experiments: {sum(model_success_count.values())}")
    
    print(f"\n🏆 MODEL PERFORMANCE:")
    for model_name in sorted(model_total_count.keys()):
        success_rate = (model_success_count[model_name] / model_total_count[model_name]) * 100
        print(f"   {model_name}: {model_success_count[model_name]}/{model_total_count[model_name]} ({success_rate:.1f}%)")
    
    # Show models that were actually used (successful)
    successful_models = [model for model, count in model_success_count.items() if count > 0]
    
    print(f"\n✅ SUCCESSFULLY DISCOVERED MODELS: {len(successful_models)}")
    for i, model in enumerate(sorted(successful_models), 1):
        print(f"   {i}. {model}")
    
    return successful_models

if __name__ == "__main__":
    print("🧮 MATHEMATICAL MODEL DISCOVERY DEMONSTRATION")
    print("=" * 60)
    
    # Show available models
    available_models = show_available_models()
    
    # Show initial discovery
    initial_models = show_model_discovery()
    
    # Simulate evolution
    evolved_models = simulate_evolution()
    
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"🧮 Total Available Models: {len(available_models)}")
    print(f"🧊 Models in Initial Cubes: {len(initial_models)}")
    print(f"🧬 Models Discovered in Evolution: {len(evolved_models)}")
    
    print(f"\n🎯 EXPLANATION:")
    print(f"   • System has {len(available_models)} mathematical models available")
    print(f"   • Each cube starts with 3-4 models (primary + secondary)")
    print(f"   • Evolution tests models and discovers which ones work best")
    print(f"   • Different tests may discover different numbers of successful models")
    print(f"   • The '5 models' or '7 models' refers to successfully discovered models")
    
    print(f"\n🚀 The discrepancy you noticed is because:")
    print(f"   • Different domains favor different mathematical approaches")
    print(f"   • Evolution filters out poor-performing models")
    print(f"   • Some models fail due to implementation issues (like entropy bug)")
    print(f"   • Final count depends on which models prove successful in testing")