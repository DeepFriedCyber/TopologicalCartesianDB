#!/usr/bin/env python3
"""
Multi-Cube Swarm Optimizer - Advanced Cube Selection and Coordination

Implements swarm optimization techniques for optimal cube selection and
coordination strategies, improving resource utilization and query processing
efficiency through particle swarm optimization (PSO) algorithms.

Based on DNN optimization research from:
https://www.datasciencecentral.com/how-to-build-and-optimize-high-performance-deep-neural-networks-from-scratch/
"""

import numpy as np
import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for swarm optimization"""
    MINIMIZE_PROCESSING_TIME = "minimize_processing_time"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    OPTIMIZE_RESOURCE_USAGE = "optimize_resource_usage"
    BALANCE_LOAD = "balance_load"
    MAXIMIZE_COHERENCE = "maximize_coherence"


@dataclass
class SwarmParticle:
    """Represents a particle in the swarm optimization space"""
    particle_id: str
    position: np.ndarray  # Current cube selection/coordination strategy
    velocity: np.ndarray  # Velocity in optimization space
    personal_best_position: np.ndarray
    personal_best_fitness: float
    current_fitness: float
    cube_selection: Dict[str, float]  # Cube selection weights
    coordination_strategy: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class SwarmConfiguration:
    """Configuration for swarm optimization"""
    num_particles: int = 20
    max_iterations: int = 50
    inertia_weight: float = 0.7
    cognitive_coefficient: float = 1.5
    social_coefficient: float = 1.5
    convergence_threshold: float = 0.001
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_ACCURACY


@dataclass
class OptimizationResult:
    """Result of swarm optimization"""
    best_cube_selection: Dict[str, float]
    best_coordination_strategy: Dict[str, Any]
    best_fitness: float
    optimization_time: float
    iterations_completed: int
    convergence_achieved: bool
    particle_history: List[Dict[str, Any]]
    performance_improvement: float


class CubeSelectionFitnessEvaluator:
    """Evaluates fitness of cube selection strategies"""
    
    def __init__(self, cube_stats: Dict[str, Any]):
        self.cube_stats = cube_stats
        self.historical_performance = defaultdict(list)
        self.query_patterns = defaultdict(list)
    
    def evaluate_fitness(self, cube_selection: Dict[str, float], 
                        coordination_strategy: Dict[str, Any],
                        query_context: Dict[str, Any],
                        objective: OptimizationObjective) -> float:
        """Evaluate fitness of a cube selection strategy"""
        
        try:
            if objective == OptimizationObjective.MINIMIZE_PROCESSING_TIME:
                return self._evaluate_processing_time_fitness(cube_selection, query_context)
            elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
                return self._evaluate_accuracy_fitness(cube_selection, query_context)
            elif objective == OptimizationObjective.OPTIMIZE_RESOURCE_USAGE:
                return self._evaluate_resource_fitness(cube_selection, coordination_strategy)
            elif objective == OptimizationObjective.BALANCE_LOAD:
                return self._evaluate_load_balance_fitness(cube_selection)
            elif objective == OptimizationObjective.MAXIMIZE_COHERENCE:
                return self._evaluate_coherence_fitness(cube_selection, coordination_strategy)
            else:
                return self._evaluate_composite_fitness(cube_selection, coordination_strategy, query_context)
                
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def _evaluate_processing_time_fitness(self, cube_selection: Dict[str, float], 
                                        query_context: Dict[str, Any]) -> float:
        """Evaluate fitness based on expected processing time"""
        
        total_expected_time = 0.0
        total_weight = 0.0
        
        for cube_name, weight in cube_selection.items():
            if weight > 0.1:  # Only consider significantly weighted cubes
                cube_stats = self.cube_stats.get(cube_name, {})
                avg_time = cube_stats.get('performance_stats', {}).get('avg_processing_time', 1.0)
                current_load = cube_stats.get('current_load', 0)
                capacity = cube_stats.get('processing_capacity', 1000)
                
                # Adjust for current load
                load_factor = 1.0 + (current_load / capacity)
                expected_time = avg_time * load_factor * weight
                
                total_expected_time += expected_time
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Fitness is inverse of expected time (lower time = higher fitness)
        avg_expected_time = total_expected_time / total_weight
        fitness = 1.0 / (1.0 + avg_expected_time)
        
        return fitness
    
    def _evaluate_accuracy_fitness(self, cube_selection: Dict[str, float], 
                                 query_context: Dict[str, Any]) -> float:
        """Evaluate fitness based on expected accuracy"""
        
        query_type = query_context.get('query_type', 'general')
        query_complexity = query_context.get('complexity', 0.5)
        
        total_accuracy = 0.0
        total_weight = 0.0
        
        for cube_name, weight in cube_selection.items():
            if weight > 0.1:
                cube_stats = self.cube_stats.get(cube_name, {})
                base_accuracy = cube_stats.get('performance_stats', {}).get('accuracy_score', 0.8)
                
                # Adjust accuracy based on cube specialization match
                specialization_match = self._calculate_specialization_match(cube_name, query_context)
                adjusted_accuracy = base_accuracy * (0.5 + 0.5 * specialization_match)
                
                # Weight by complexity handling capability
                complexity_factor = self._get_complexity_handling_factor(cube_name, query_complexity)
                final_accuracy = adjusted_accuracy * complexity_factor * weight
                
                total_accuracy += final_accuracy
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_accuracy / total_weight
    
    def _evaluate_resource_fitness(self, cube_selection: Dict[str, float], 
                                 coordination_strategy: Dict[str, Any]) -> float:
        """Evaluate fitness based on resource utilization efficiency"""
        
        total_resource_efficiency = 0.0
        total_weight = 0.0
        
        for cube_name, weight in cube_selection.items():
            if weight > 0.1:
                cube_stats = self.cube_stats.get(cube_name, {})
                current_load = cube_stats.get('current_load', 0)
                capacity = cube_stats.get('processing_capacity', 1000)
                
                # Calculate resource efficiency (avoid overloading and underutilization)
                utilization = current_load / capacity if capacity > 0 else 0
                
                # Optimal utilization is around 70-80%
                if utilization < 0.3:
                    efficiency = utilization / 0.3  # Underutilized
                elif utilization > 0.9:
                    efficiency = (1.0 - utilization) / 0.1  # Overloaded
                else:
                    efficiency = 1.0  # Good utilization
                
                total_resource_efficiency += efficiency * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_resource_efficiency / total_weight
    
    def _evaluate_load_balance_fitness(self, cube_selection: Dict[str, float]) -> float:
        """Evaluate fitness based on load balancing across cubes"""
        
        loads = []
        weights = []
        
        for cube_name, weight in cube_selection.items():
            if weight > 0.1:
                cube_stats = self.cube_stats.get(cube_name, {})
                current_load = cube_stats.get('current_load', 0)
                capacity = cube_stats.get('processing_capacity', 1000)
                
                utilization = current_load / capacity if capacity > 0 else 0
                loads.append(utilization)
                weights.append(weight)
        
        if not loads:
            return 0.0
        
        # Calculate load balance (lower variance = better balance)
        if len(loads) > 1:
            load_variance = np.var(loads)
            balance_fitness = 1.0 / (1.0 + load_variance)
        else:
            balance_fitness = 1.0
        
        # Weight by selection diversity
        weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights if w > 0)
        diversity_bonus = min(1.0, weight_entropy / np.log(len(weights)))
        
        return balance_fitness * (0.7 + 0.3 * diversity_bonus)
    
    def _evaluate_coherence_fitness(self, cube_selection: Dict[str, float], 
                                  coordination_strategy: Dict[str, Any]) -> float:
        """Evaluate fitness based on expected cross-cube coherence"""
        
        selected_cubes = [name for name, weight in cube_selection.items() if weight > 0.1]
        
        if len(selected_cubes) < 2:
            return 1.0  # Single cube has perfect coherence
        
        # Calculate expected coherence based on cube domain similarity
        coherence_scores = []
        
        for i, cube1 in enumerate(selected_cubes):
            for cube2 in selected_cubes[i+1:]:
                similarity = self._get_cube_domain_similarity(cube1, cube2)
                weight1 = cube_selection[cube1]
                weight2 = cube_selection[cube2]
                
                # Weighted coherence contribution
                coherence_contribution = similarity * weight1 * weight2
                coherence_scores.append(coherence_contribution)
        
        if not coherence_scores:
            return 1.0
        
        return np.mean(coherence_scores)
    
    def _evaluate_composite_fitness(self, cube_selection: Dict[str, float], 
                                  coordination_strategy: Dict[str, Any],
                                  query_context: Dict[str, Any]) -> float:
        """Evaluate composite fitness combining multiple objectives"""
        
        # Calculate individual fitness components
        time_fitness = self._evaluate_processing_time_fitness(cube_selection, query_context)
        accuracy_fitness = self._evaluate_accuracy_fitness(cube_selection, query_context)
        resource_fitness = self._evaluate_resource_fitness(cube_selection, coordination_strategy)
        balance_fitness = self._evaluate_load_balance_fitness(cube_selection)
        coherence_fitness = self._evaluate_coherence_fitness(cube_selection, coordination_strategy)
        
        # Weighted combination (can be adjusted based on priorities)
        composite_fitness = (
            0.3 * accuracy_fitness +
            0.25 * time_fitness +
            0.2 * resource_fitness +
            0.15 * balance_fitness +
            0.1 * coherence_fitness
        )
        
        return composite_fitness
    
    def _calculate_specialization_match(self, cube_name: str, query_context: Dict[str, Any]) -> float:
        """Calculate how well a cube's specialization matches the query"""
        
        cube_stats = self.cube_stats.get(cube_name, {})
        expertise_domains = cube_stats.get('expertise_domains', [])
        
        query_keywords = query_context.get('keywords', [])
        query_type = query_context.get('query_type', 'general')
        
        # Simple keyword matching
        if not query_keywords or not expertise_domains:
            return 0.5
        
        matches = sum(1 for keyword in query_keywords if any(domain in keyword.lower() for domain in expertise_domains))
        match_ratio = matches / len(query_keywords)
        
        return match_ratio
    
    def _get_complexity_handling_factor(self, cube_name: str, complexity: float) -> float:
        """Get cube's capability to handle query complexity"""
        
        # Different cubes have different complexity handling capabilities
        complexity_capabilities = {
            'code_cube': 0.9,      # High complexity handling
            'data_cube': 0.8,      # Good with complex data
            'system_cube': 0.7,    # Moderate complexity
            'temporal_cube': 0.6,  # Specialized complexity
            'user_cube': 0.5       # Lower complexity focus
        }
        
        base_capability = complexity_capabilities.get(cube_name, 0.6)
        
        # Adjust based on complexity level
        if complexity > 0.8:
            return base_capability
        elif complexity < 0.3:
            return min(1.0, base_capability + 0.2)  # Bonus for simple queries
        else:
            return base_capability + (1.0 - base_capability) * (1.0 - complexity)
    
    def _get_cube_domain_similarity(self, cube1: str, cube2: str) -> float:
        """Get domain similarity between two cubes"""
        
        similarity_matrix = {
            ('code_cube', 'data_cube'): 0.7,
            ('code_cube', 'system_cube'): 0.8,
            ('code_cube', 'user_cube'): 0.4,
            ('code_cube', 'temporal_cube'): 0.5,
            ('data_cube', 'system_cube'): 0.6,
            ('data_cube', 'user_cube'): 0.5,
            ('data_cube', 'temporal_cube'): 0.7,
            ('user_cube', 'temporal_cube'): 0.6,
            ('user_cube', 'system_cube'): 0.4,
            ('temporal_cube', 'system_cube'): 0.5
        }
        
        key = (cube1, cube2) if cube1 < cube2 else (cube2, cube1)
        return similarity_matrix.get(key, 0.5)


class SwarmOptimizationEngine:
    """Core swarm optimization engine for cube selection"""
    
    def __init__(self, config: SwarmConfiguration):
        self.config = config
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.iteration_history = []
        self.convergence_history = []
        
    def optimize_cube_selection(self, cube_stats: Dict[str, Any], 
                              query_context: Dict[str, Any]) -> OptimizationResult:
        """Optimize cube selection using particle swarm optimization"""
        
        start_time = time.time()
        
        # Initialize fitness evaluator
        fitness_evaluator = CubeSelectionFitnessEvaluator(cube_stats)
        
        # Initialize swarm
        self._initialize_swarm(cube_stats, query_context)
        
        # Optimization loop
        iteration = 0
        converged = False
        
        while iteration < self.config.max_iterations and not converged:
            # Evaluate fitness for all particles
            self._evaluate_particle_fitness(fitness_evaluator, query_context)
            
            # Update global best
            self._update_global_best()
            
            # Update particle velocities and positions
            self._update_particles()
            
            # Check convergence
            converged = self._check_convergence()
            
            # Record iteration
            self._record_iteration(iteration)
            
            iteration += 1
        
        optimization_time = time.time() - start_time
        
        # Create result
        best_particle = self._get_best_particle()
        
        result = OptimizationResult(
            best_cube_selection=best_particle.cube_selection,
            best_coordination_strategy=best_particle.coordination_strategy,
            best_fitness=self.global_best_fitness,
            optimization_time=optimization_time,
            iterations_completed=iteration,
            convergence_achieved=converged,
            particle_history=self.iteration_history,
            performance_improvement=self._calculate_performance_improvement()
        )
        
        logger.info(f"🔍 Swarm optimization completed: {iteration} iterations, "
                   f"fitness={self.global_best_fitness:.3f}, time={optimization_time:.3f}s")
        
        return result
    
    def _initialize_swarm(self, cube_stats: Dict[str, Any], query_context: Dict[str, Any]):
        """Initialize the particle swarm"""
        
        cube_names = list(cube_stats.keys())
        num_cubes = len(cube_names)
        
        self.particles = []
        
        for i in range(self.config.num_particles):
            # Random cube selection weights (normalized)
            weights = np.random.random(num_cubes)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            
            cube_selection = {cube_names[j]: weights[j] for j in range(num_cubes)}
            
            # Random coordination strategy
            coordination_strategy = {
                'strategy_type': random.choice(['parallel', 'sequential', 'adaptive', 'topological']),
                'timeout': random.uniform(10, 60),
                'max_results': random.randint(3, 10),
                'coherence_threshold': random.uniform(0.5, 0.9)
            }
            
            # Create particle
            position = np.concatenate([weights, [coordination_strategy['timeout'], 
                                               coordination_strategy['max_results'],
                                               coordination_strategy['coherence_threshold']]])
            
            particle = SwarmParticle(
                particle_id=f"particle_{i}_{int(time.time())}",
                position=position.copy(),
                velocity=np.random.uniform(-0.1, 0.1, len(position)),
                personal_best_position=position.copy(),
                personal_best_fitness=float('-inf'),
                current_fitness=0.0,
                cube_selection=cube_selection,
                coordination_strategy=coordination_strategy
            )
            
            self.particles.append(particle)
    
    def _evaluate_particle_fitness(self, fitness_evaluator: CubeSelectionFitnessEvaluator, 
                                 query_context: Dict[str, Any]):
        """Evaluate fitness for all particles"""
        
        for particle in self.particles:
            fitness = fitness_evaluator.evaluate_fitness(
                particle.cube_selection,
                particle.coordination_strategy,
                query_context,
                self.config.objective
            )
            
            particle.current_fitness = fitness
            particle.performance_history.append(fitness)
            
            # Update personal best
            if fitness > particle.personal_best_fitness:
                particle.personal_best_fitness = fitness
                particle.personal_best_position = particle.position.copy()
    
    def _update_global_best(self):
        """Update global best position and fitness"""
        
        for particle in self.particles:
            if particle.current_fitness > self.global_best_fitness:
                self.global_best_fitness = particle.current_fitness
                self.global_best_position = particle.position.copy()
    
    def _update_particles(self):
        """Update particle velocities and positions"""
        
        for particle in self.particles:
            # PSO velocity update
            r1, r2 = np.random.random(2)
            
            cognitive_component = (self.config.cognitive_coefficient * r1 * 
                                 (particle.personal_best_position - particle.position))
            
            social_component = (self.config.social_coefficient * r2 * 
                              (self.global_best_position - particle.position))
            
            # Update velocity
            particle.velocity = (self.config.inertia_weight * particle.velocity + 
                               cognitive_component + social_component)
            
            # Clamp velocity
            particle.velocity = np.clip(particle.velocity, -0.5, 0.5)
            
            # Update position
            particle.position += particle.velocity
            
            # Update cube selection and coordination strategy from position
            self._update_particle_strategy(particle)
    
    def _update_particle_strategy(self, particle: SwarmParticle):
        """Update particle's cube selection and coordination strategy from position"""
        
        num_cubes = len(particle.cube_selection)
        
        # Extract cube weights from position
        cube_weights = particle.position[:num_cubes]
        cube_weights = np.abs(cube_weights)  # Ensure positive
        cube_weights = cube_weights / (np.sum(cube_weights) + 1e-10)  # Normalize
        
        # Update cube selection
        cube_names = list(particle.cube_selection.keys())
        particle.cube_selection = {cube_names[i]: cube_weights[i] for i in range(num_cubes)}
        
        # Extract coordination parameters
        if len(particle.position) > num_cubes:
            timeout = max(10, min(60, particle.position[num_cubes]))
            max_results = max(3, min(10, int(abs(particle.position[num_cubes + 1]))))
            coherence_threshold = max(0.1, min(0.9, abs(particle.position[num_cubes + 2])))
            
            particle.coordination_strategy.update({
                'timeout': timeout,
                'max_results': max_results,
                'coherence_threshold': coherence_threshold
            })
    
    def _check_convergence(self) -> bool:
        """Check if the swarm has converged"""
        
        if len(self.convergence_history) < 5:
            self.convergence_history.append(self.global_best_fitness)
            return False
        
        # Check if improvement is below threshold for recent iterations
        recent_improvements = []
        for i in range(1, min(6, len(self.convergence_history))):
            improvement = self.convergence_history[-i] - self.convergence_history[-i-1]
            recent_improvements.append(improvement)
        
        avg_improvement = np.mean(recent_improvements)
        converged = avg_improvement < self.config.convergence_threshold
        
        self.convergence_history.append(self.global_best_fitness)
        
        return converged
    
    def _record_iteration(self, iteration: int):
        """Record iteration statistics"""
        
        fitness_values = [p.current_fitness for p in self.particles]
        
        iteration_stats = {
            'iteration': iteration,
            'global_best_fitness': self.global_best_fitness,
            'avg_fitness': np.mean(fitness_values),
            'fitness_std': np.std(fitness_values),
            'best_particle_id': self._get_best_particle().particle_id
        }
        
        self.iteration_history.append(iteration_stats)
    
    def _get_best_particle(self) -> SwarmParticle:
        """Get the particle with the best fitness"""
        return max(self.particles, key=lambda p: p.current_fitness)
    
    def _calculate_performance_improvement(self) -> float:
        """Calculate performance improvement from optimization"""
        
        if len(self.iteration_history) < 2:
            return 0.0
        
        initial_fitness = self.iteration_history[0]['avg_fitness']
        final_fitness = self.global_best_fitness
        
        if initial_fitness > 0:
            improvement = (final_fitness - initial_fitness) / initial_fitness
        else:
            improvement = final_fitness
        
        return improvement


class MultiCubeSwarmOptimizer:
    """Main interface for multi-cube swarm optimization"""
    
    def __init__(self, config: Optional[SwarmConfiguration] = None):
        self.config = config or SwarmConfiguration()
        self.optimization_history = []
        self.performance_stats = defaultdict(list)
        
        logger.info(f"🔍 MultiCubeSwarmOptimizer initialized with {self.config.num_particles} particles, "
                   f"objective: {self.config.objective.value}")
    
    def optimize_cube_coordination(self, query: str, cube_stats: Dict[str, Any], 
                                 query_context: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Use swarm optimization for optimal cube coordination
        
        Args:
            query: The query to optimize for
            cube_stats: Current statistics for all cubes
            query_context: Additional context about the query
            
        Returns:
            OptimizationResult with best cube selection and coordination strategy
        """
        
        # Prepare query context
        if query_context is None:
            query_context = self._analyze_query_context(query)
        
        # Create optimization engine
        engine = SwarmOptimizationEngine(self.config)
        
        # Run optimization
        result = engine.optimize_cube_selection(cube_stats, query_context)
        
        # Record optimization
        self.optimization_history.append({
            'timestamp': time.time(),
            'query': query,
            'result': result,
            'query_context': query_context
        })
        
        # Update performance stats
        self.performance_stats['fitness_improvements'].append(result.performance_improvement)
        self.performance_stats['optimization_times'].append(result.optimization_time)
        self.performance_stats['iterations'].append(result.iterations_completed)
        
        logger.info(f"🔍 Optimization completed: fitness={result.best_fitness:.3f}, "
                   f"improvement={result.performance_improvement:+.1%}")
        
        return result
    
    def _analyze_query_context(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract context for optimization"""
        
        query_lower = query.lower()
        
        # Extract keywords
        keywords = query_lower.split()
        
        # Determine query type
        query_type = 'general'
        if any(word in query_lower for word in ['code', 'function', 'class', 'algorithm']):
            query_type = 'code'
        elif any(word in query_lower for word in ['data', 'analysis', 'processing']):
            query_type = 'data'
        elif any(word in query_lower for word in ['user', 'behavior', 'interaction']):
            query_type = 'user'
        elif any(word in query_lower for word in ['time', 'temporal', 'trend']):
            query_type = 'temporal'
        elif any(word in query_lower for word in ['system', 'performance', 'resource']):
            query_type = 'system'
        
        # Estimate complexity
        complexity = min(1.0, len(query.split()) / 50.0)  # Normalize by word count
        
        return {
            'keywords': keywords,
            'query_type': query_type,
            'complexity': complexity,
            'length': len(query),
            'word_count': len(keywords)
        }
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimizer statistics"""
        
        recent_optimizations = self.optimization_history[-20:] if self.optimization_history else []
        
        return {
            'total_optimizations': len(self.optimization_history),
            'configuration': {
                'num_particles': self.config.num_particles,
                'max_iterations': self.config.max_iterations,
                'objective': self.config.objective.value
            },
            'recent_performance': {
                'avg_fitness_improvement': np.mean(self.performance_stats['fitness_improvements'][-20:]) if self.performance_stats['fitness_improvements'] else 0.0,
                'avg_optimization_time': np.mean(self.performance_stats['optimization_times'][-20:]) if self.performance_stats['optimization_times'] else 0.0,
                'avg_iterations': np.mean(self.performance_stats['iterations'][-20:]) if self.performance_stats['iterations'] else 0.0
            },
            'optimization_trends': {
                'fitness_trend': self.performance_stats['fitness_improvements'][-10:] if self.performance_stats['fitness_improvements'] else [],
                'time_trend': self.performance_stats['optimization_times'][-10:] if self.performance_stats['optimization_times'] else []
            },
            'recent_optimizations': [
                {
                    'timestamp': opt['timestamp'],
                    'query_type': opt['query_context'].get('query_type', 'unknown'),
                    'fitness': opt['result'].best_fitness,
                    'improvement': opt['result'].performance_improvement
                }
                for opt in recent_optimizations
            ]
        }


# Factory function for easy creation
def create_multi_cube_swarm_optimizer(num_particles: int = 20, 
                                    max_iterations: int = 50,
                                    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_ACCURACY) -> MultiCubeSwarmOptimizer:
    """Create and initialize a multi-cube swarm optimizer"""
    
    config = SwarmConfiguration(
        num_particles=num_particles,
        max_iterations=max_iterations,
        objective=objective
    )
    
    return MultiCubeSwarmOptimizer(config)