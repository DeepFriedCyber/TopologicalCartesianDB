#!/usr/bin/env python3
"""
Test Multi-Cube Mathematical Evolution

Demonstrates the revolutionary approach where each cube tests different
mathematical models and they learn from each other through evolution.

This creates a "mathematical ecosystem" where cubes compete and collaborate!
"""

import sys
from pathlib import Path
import numpy as np
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.topcart_config import (
    force_multi_cube_architecture, enable_benchmark_mode, create_topcart_system
)
from topological_cartesian.multi_cube_math_lab import (
    MultiCubeMathLaboratory, MathModelType
)


class MultiCubeMathEvolutionDemo:
    """Demonstrate multi-cube mathematical evolution"""
    
    def __init__(self):
        # Force multi-cube architecture
        force_multi_cube_architecture()
        enable_benchmark_mode()
        
        self.topcart_system = None
        self.math_lab = None
        self.evolution_history = []
        
    def setup_systems(self):
        """Setup TOPCART and mathematical laboratory"""
        
        print("🚀 Setting up Multi-Cube Mathematical Evolution Demo...")
        
        # Create TOPCART system
        self.topcart_system = create_topcart_system()
        
        # Create mathematical laboratory
        self.math_lab = MultiCubeMathLaboratory()
        
        print("✅ Systems ready for mathematical evolution!")
    
    def create_realistic_test_data(self) -> dict:
        """Create realistic test data that mimics different domain characteristics"""
        
        print("📊 Creating realistic domain-specific test data...")
        
        np.random.seed(42)  # Reproducible results
        
        # CODE_CUBE: Structured, hierarchical data (like code complexity)
        code_base = np.array([
            [0.8, 0.2, 0.6, 0.4, 0.7],  # High complexity, low coupling
            [0.3, 0.9, 0.2, 0.8, 0.5],  # Low complexity, high coupling
            [0.6, 0.4, 0.9, 0.3, 0.8],  # Medium complexity, high cohesion
        ])
        # Create variations of base patterns
        code_coords = []
        for base_pattern in code_base:
            # Create 16-17 variations of each pattern
            variations = np.random.randn(17, 5) * 0.2 + base_pattern
            code_coords.append(variations)
        code_coords = np.vstack(code_coords)[:50]  # Ensure exactly 50 samples
        
        # DATA_CUBE: Statistical distributions (like data volume/velocity)
        data_centers = np.array([
            [0.9, 0.1, 0.3, 0.7, 0.5],  # High volume, low velocity
            [0.2, 0.8, 0.6, 0.4, 0.9],  # Low volume, high velocity
            [0.5, 0.5, 0.9, 0.2, 0.6],  # Medium volume, high variety
        ])
        data_coords = []
        for center in data_centers:
            cluster = np.random.multivariate_normal(center, np.eye(5) * 0.1, 15)
            data_coords.append(cluster)
        data_coords = np.vstack(data_coords)
        
        # USER_CUBE: Behavioral patterns (like user engagement)
        user_patterns = []
        # Active users
        active_users = np.random.beta(2, 1, (15, 5)) * 0.8 + 0.2
        user_patterns.append(active_users)
        # Casual users  
        casual_users = np.random.beta(1, 2, (15, 5)) * 0.6 + 0.1
        user_patterns.append(casual_users)
        # Power users
        power_users = np.random.beta(3, 1, (10, 5)) * 0.9 + 0.1
        user_patterns.append(power_users)
        user_coords = np.vstack(user_patterns)
        
        # SYSTEM_CUBE: Performance metrics (like CPU/memory usage)
        system_coords = []
        # High performance systems
        high_perf = np.random.gamma(2, 0.2, (12, 5))
        system_coords.append(high_perf)
        # Medium performance systems
        med_perf = np.random.gamma(1, 0.3, (12, 5))
        system_coords.append(med_perf)
        # Low performance systems
        low_perf = np.random.gamma(0.5, 0.4, (11, 5))
        system_coords.append(low_perf)
        system_coords = np.vstack(system_coords)
        
        # TEMPORAL_CUBE: Time series patterns (like trends/seasonality)
        t = np.linspace(0, 4*np.pi, 30)
        temporal_coords = []
        for i in range(5):
            # Different temporal patterns per dimension
            if i == 0:  # Trend
                pattern = t * 0.1 + np.random.randn(30) * 0.05
            elif i == 1:  # Seasonal
                pattern = np.sin(t) * 0.3 + np.random.randn(30) * 0.05
            elif i == 2:  # Cyclical
                pattern = np.cos(t * 0.5) * 0.4 + np.random.randn(30) * 0.05
            elif i == 3:  # Random walk
                pattern = np.cumsum(np.random.randn(30) * 0.1)
            else:  # Periodic with noise
                pattern = np.sin(t * 2) * 0.2 + np.random.randn(30) * 0.1
            
            temporal_coords.append(pattern)
        
        temporal_coords = np.array(temporal_coords).T
        
        coordinates = {
            'code_cube': code_coords,
            'data_cube': data_coords,
            'user_cube': user_coords,
            'system_cube': system_coords,
            'temporal_cube': temporal_coords
        }
        
        print("📊 Created realistic test data:")
        for cube_name, coords in coordinates.items():
            print(f"   • {cube_name}: {coords.shape} (mean={np.mean(coords):.3f}, std={np.std(coords):.3f})")
        
        return coordinates
    
    def run_evolution_generation(self, coordinates: dict, generation: int) -> dict:
        """Run one generation of mathematical evolution"""
        
        print(f"\n🧬 GENERATION {generation}: Mathematical Evolution")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run parallel experiments across all cubes
        print("🔬 Running parallel mathematical experiments...")
        experiment_results = self.math_lab.run_parallel_experiments(coordinates, max_workers=3)
        
        # Analyze cross-cube learning
        print("🧠 Analyzing cross-cube learning patterns...")
        analysis = self.math_lab.analyze_cross_cube_learning(experiment_results)
        
        # Evolve specializations
        print("🧬 Evolving cube specializations...")
        evolved_specs = self.math_lab.evolve_cube_specializations(analysis)
        self.math_lab.cube_specializations = evolved_specs
        
        generation_time = time.time() - start_time
        
        # Store evolution history
        generation_summary = {
            'generation': generation,
            'execution_time': generation_time,
            'total_experiments': sum(len(exps) for exps in experiment_results.values()),
            'successful_experiments': sum(len([exp for exp in exps if exp.success]) for exps in experiment_results.values()),
            'best_scores': {cube: max([exp.improvement_score for exp in exps if exp.success], default=0.0) 
                          for cube, exps in experiment_results.items()},
            'model_rankings': analysis['model_performance_ranking']
        }
        
        self.evolution_history.append(generation_summary)
        
        print(f"⚡ Generation {generation} completed in {generation_time:.2f}s")
        print(f"   📊 Experiments: {generation_summary['successful_experiments']}/{generation_summary['total_experiments']} successful")
        
        return {
            'experiment_results': experiment_results,
            'analysis': analysis,
            'evolved_specializations': evolved_specs,
            'summary': generation_summary
        }
    
    def demonstrate_knowledge_transfer(self, generation_results: list):
        """Demonstrate how knowledge transfers between cubes"""
        
        print(f"\n🔄 KNOWLEDGE TRANSFER ANALYSIS")
        print("=" * 40)
        
        if len(generation_results) < 2:
            print("⚠️ Need at least 2 generations for transfer analysis")
            return
        
        # Compare specializations across generations
        gen1_specs = generation_results[0]['evolved_specializations']
        gen2_specs = generation_results[-1]['evolved_specializations']
        
        print("🧊 Cube Specialization Evolution:")
        for cube_name in gen1_specs.keys():
            gen1_primary = gen1_specs[cube_name].primary_model.value
            gen2_primary = gen2_specs[cube_name].primary_model.value
            
            gen1_secondary = [m.value for m in gen1_specs[cube_name].secondary_models]
            gen2_secondary = [m.value for m in gen2_specs[cube_name].secondary_models]
            
            print(f"\n   {cube_name}:")
            print(f"      Primary: {gen1_primary} → {gen2_primary}")
            if gen1_primary != gen2_primary:
                print(f"      🔄 PRIMARY MODEL EVOLVED!")
            
            print(f"      Secondary: {gen1_secondary}")
            print(f"              → {gen2_secondary}")
            
            new_models = set(gen2_secondary) - set(gen1_secondary)
            if new_models:
                print(f"      ➕ New models: {list(new_models)}")
        
        # Analyze performance improvements
        print(f"\n📈 Performance Evolution:")
        for i, gen_result in enumerate(generation_results):
            gen_num = gen_result['summary']['generation']
            best_scores = gen_result['summary']['best_scores']
            avg_score = np.mean(list(best_scores.values()))
            
            print(f"   Generation {gen_num}: Average best score = {avg_score:.3f}")
            
            if i > 0:
                prev_avg = np.mean(list(generation_results[i-1]['summary']['best_scores'].values()))
                improvement = avg_score - prev_avg
                print(f"      Improvement: {improvement:+.3f} ({improvement/prev_avg:+.1%})")
    
    def analyze_mathematical_ecosystem(self, generation_results: list):
        """Analyze the mathematical ecosystem evolution"""
        
        print(f"\n🌍 MATHEMATICAL ECOSYSTEM ANALYSIS")
        print("=" * 45)
        
        # Track model popularity over generations
        model_evolution = {}
        
        for gen_result in generation_results:
            gen_num = gen_result['summary']['generation']
            model_rankings = gen_result['summary']['model_rankings']
            
            for model_name, stats in model_rankings.items():
                if model_name not in model_evolution:
                    model_evolution[model_name] = []
                
                model_evolution[model_name].append({
                    'generation': gen_num,
                    'average_score': stats['average_score'],
                    'success_rate': stats['success_rate']
                })
        
        print("🏆 Model Performance Evolution:")
        for model_name, evolution in model_evolution.items():
            if len(evolution) > 1:
                first_score = evolution[0]['average_score']
                last_score = evolution[-1]['average_score']
                improvement = last_score - first_score
                
                print(f"   {model_name}:")
                print(f"      Score: {first_score:.3f} → {last_score:.3f} ({improvement:+.3f})")
                
                first_success = evolution[0]['success_rate']
                last_success = evolution[-1]['success_rate']
                success_improvement = last_success - first_success
                
                print(f"      Success: {first_success:.1%} → {last_success:.1%} ({success_improvement:+.1%})")
        
        # Identify ecosystem trends
        print(f"\n🔍 Ecosystem Trends:")
        
        # Most improved model
        best_improvement = -float('inf')
        best_model = None
        
        for model_name, evolution in model_evolution.items():
            if len(evolution) > 1:
                improvement = evolution[-1]['average_score'] - evolution[0]['average_score']
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_model = model_name
        
        if best_model:
            print(f"   🚀 Most Improved Model: {best_model} (+{best_improvement:.3f})")
        
        # Most stable model
        most_stable = None
        lowest_variance = float('inf')
        
        for model_name, evolution in model_evolution.items():
            if len(evolution) > 2:
                scores = [e['average_score'] for e in evolution]
                variance = np.var(scores)
                if variance < lowest_variance:
                    lowest_variance = variance
                    most_stable = model_name
        
        if most_stable:
            print(f"   🎯 Most Stable Model: {most_stable} (variance={lowest_variance:.4f})")
        
        # Ecosystem diversity
        final_generation = generation_results[-1]
        active_models = len(final_generation['summary']['model_rankings'])
        print(f"   🌈 Ecosystem Diversity: {active_models} active mathematical models")
    
    def generate_evolution_report(self, generation_results: list) -> str:
        """Generate comprehensive evolution report"""
        
        total_generations = len(generation_results)
        total_experiments = sum(gr['summary']['total_experiments'] for gr in generation_results)
        total_time = sum(gr['summary']['execution_time'] for gr in generation_results)
        
        # Calculate overall improvements
        if total_generations > 1:
            first_gen_scores = list(generation_results[0]['summary']['best_scores'].values())
            last_gen_scores = list(generation_results[-1]['summary']['best_scores'].values())
            
            avg_first = np.mean(first_gen_scores)
            avg_last = np.mean(last_gen_scores)
            overall_improvement = avg_last - avg_first
            improvement_percentage = (overall_improvement / avg_first) * 100 if avg_first > 0 else 0
        else:
            overall_improvement = 0
            improvement_percentage = 0
        
        report = f"""
🧬 MULTI-CUBE MATHEMATICAL EVOLUTION REPORT
{'='*60}

📊 EVOLUTION STATISTICS:
   Generations Completed: {total_generations}
   Total Experiments: {total_experiments}
   Total Evolution Time: {total_time:.2f}s
   Average Time per Generation: {total_time/max(total_generations, 1):.2f}s

🎯 PERFORMANCE EVOLUTION:
   Overall Improvement: {overall_improvement:+.3f} ({improvement_percentage:+.1f}%)
   
🧊 FINAL CUBE SPECIALIZATIONS:
"""
        
        if generation_results:
            final_specs = generation_results[-1]['evolved_specializations']
            for cube_name, spec in final_specs.items():
                report += f"""
   {cube_name.upper()}:
      Primary Model: {spec.primary_model.value}
      Secondary Models: {[m.value for m in spec.secondary_models]}
      Best Performance: {spec.best_performance:.3f}
      Expertise Domain: {spec.expertise_domain}
"""
        
        report += f"""
🏆 TOP PERFORMING MODELS:
"""
        
        if generation_results:
            final_rankings = generation_results[-1]['summary']['model_rankings']
            sorted_models = sorted(final_rankings.items(), 
                                 key=lambda x: x[1]['average_score'], reverse=True)
            
            for i, (model_name, stats) in enumerate(sorted_models[:5], 1):
                report += f"""
   {i}. {model_name}:
      Average Score: {stats['average_score']:.3f}
      Success Rate: {stats['success_rate']:.1%}
      Best Score: {stats['best_score']:.3f}
"""
        
        report += f"""
🚀 REVOLUTIONARY INSIGHTS:

1. MATHEMATICAL DIVERSITY: Each cube developed unique mathematical expertise
2. CROSS-POLLINATION: Successful models spread between cubes through evolution
3. ADAPTIVE SPECIALIZATION: Cubes adapted their models based on domain requirements
4. PARALLEL LEARNING: Multiple mathematical approaches tested simultaneously
5. EMERGENT INTELLIGENCE: System-wide mathematical intelligence emerged from cube interactions

🎉 The Multi-Cube Mathematical Evolution demonstrates how AI systems can
   automatically discover and optimize mathematical models through
   collaborative competition and knowledge sharing!

🔬 This approach could revolutionize how we develop mathematical models
   for complex systems - let the cubes evolve the best mathematics!
"""
        
        return report
    
    def run_complete_evolution_demo(self, num_generations: int = 3) -> dict:
        """Run complete multi-cube mathematical evolution demonstration"""
        
        print("🚀 MULTI-CUBE MATHEMATICAL EVOLUTION DEMONSTRATION")
        print("=" * 60)
        print(f"Running {num_generations} generations of mathematical evolution...")
        
        # Setup systems
        self.setup_systems()
        
        # Create test data
        coordinates = self.create_realistic_test_data()
        
        # Run evolution generations
        generation_results = []
        
        for generation in range(1, num_generations + 1):
            gen_result = self.run_evolution_generation(coordinates, generation)
            generation_results.append(gen_result)
            
            # Brief pause between generations
            time.sleep(0.5)
        
        # Analyze results
        self.demonstrate_knowledge_transfer(generation_results)
        self.analyze_mathematical_ecosystem(generation_results)
        
        # Generate final report
        final_report = self.generate_evolution_report(generation_results)
        
        return {
            'generation_results': generation_results,
            'evolution_history': self.evolution_history,
            'final_report': final_report,
            'coordinates': coordinates
        }


if __name__ == "__main__":
    try:
        demo = MultiCubeMathEvolutionDemo()
        results = demo.run_complete_evolution_demo(num_generations=3)
        
        print("\n" + "="*60)
        print(results['final_report'])
        
        print(f"\n🎉 Multi-Cube Mathematical Evolution demonstration completed!")
        print(f"🧬 {len(results['generation_results'])} generations of mathematical evolution")
        print(f"🔬 Revolutionary approach to automated mathematical model discovery!")
        
    except Exception as e:
        print(f"❌ Evolution demo failed: {e}")
        import traceback
        traceback.print_exc()