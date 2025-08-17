#!/usr/bin/env python3
"""
Advanced Query Planning & Orchestration for TCDB

Addresses feedback on orchestrator complexity and query planning.
Implements sophisticated cross-domain query execution with cost optimization.
"""

import logging
import asyncio
import time
import json
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
import uuid
import heapq
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries supported"""
    VECTOR_SEARCH = "vector_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    TOPOLOGICAL_QUERY = "topological_query"
    HYBRID_SEARCH = "hybrid_search"
    CROSS_DOMAIN = "cross_domain"

class CubeType(Enum):
    """Cube types for query planning"""
    CODE = "code"
    DATA = "data"
    USER = "user"
    TEMPORAL = "temporal"
    SYSTEM = "system"
    ORCHESTRATOR = "orchestrator"

class ExecutionStrategy(Enum):
    """Query execution strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"

@dataclass
class QueryFragment:
    """Individual query fragment for a specific cube"""
    fragment_id: str
    cube_type: CubeType
    query_type: QueryType
    query_text: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_time: float = 0.0
    priority: int = 1

@dataclass
class QueryPlan:
    """Complete query execution plan"""
    plan_id: str
    original_query: str
    fragments: List[QueryFragment]
    execution_strategy: ExecutionStrategy
    estimated_total_cost: float
    estimated_total_time: float
    parallelization_factor: float
    topology_complexity: float
    created_at: datetime

@dataclass
class ExecutionResult:
    """Result from query fragment execution"""
    fragment_id: str
    cube_type: CubeType
    results: Any
    execution_time: float
    records_processed: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TopologicalRelationship:
    """Represents topological relationship between data points"""
    source_cube: CubeType
    target_cube: CubeType
    relationship_type: str
    strength: float
    geometric_distance: float
    semantic_similarity: float
    temporal_correlation: float

class QueryParser:
    """Parses natural language queries into structured query plans"""
    
    def __init__(self):
        # Query pattern recognition
        self.cube_keywords = {
            CubeType.CODE: ['code', 'function', 'class', 'method', 'repository', 'commit'],
            CubeType.DATA: ['data', 'dataset', 'table', 'record', 'database'],
            CubeType.USER: ['user', 'customer', 'person', 'profile', 'account'],
            CubeType.TEMPORAL: ['time', 'date', 'temporal', 'history', 'trend'],
            CubeType.SYSTEM: ['system', 'log', 'metric', 'performance', 'error']
        }
        
        self.query_type_patterns = {
            QueryType.VECTOR_SEARCH: ['similar', 'like', 'related', 'find'],
            QueryType.GRAPH_TRAVERSAL: ['connected', 'relationship', 'linked', 'path'],
            QueryType.TEMPORAL_ANALYSIS: ['trend', 'over time', 'history', 'timeline'],
            QueryType.TOPOLOGICAL_QUERY: ['topology', 'structure', 'shape', 'pattern'],
            QueryType.HYBRID_SEARCH: ['combine', 'merge', 'integrate', 'cross'],
            QueryType.CROSS_DOMAIN: ['across', 'between', 'correlate', 'relate']
        }
    
    def parse_query(self, query: str) -> List[QueryFragment]:
        """Parse natural language query into fragments"""
        
        query_lower = query.lower()
        fragments = []
        
        # Identify target cubes
        target_cubes = self._identify_target_cubes(query_lower)
        
        # Identify query types
        query_types = self._identify_query_types(query_lower)
        
        # Create fragments for each cube-query type combination
        for cube in target_cubes:
            for query_type in query_types:
                fragment = QueryFragment(
                    fragment_id=str(uuid.uuid4()),
                    cube_type=cube,
                    query_type=query_type,
                    query_text=query,
                    parameters=self._extract_parameters(query, cube, query_type)
                )
                fragments.append(fragment)
        
        # Add orchestrator fragment for cross-domain queries
        if len(target_cubes) > 1:
            orchestrator_fragment = QueryFragment(
                fragment_id=str(uuid.uuid4()),
                cube_type=CubeType.ORCHESTRATOR,
                query_type=QueryType.CROSS_DOMAIN,
                query_text=query,
                parameters={'target_cubes': [c.value for c in target_cubes]},
                dependencies=[f.fragment_id for f in fragments]
            )
            fragments.append(orchestrator_fragment)
        
        return fragments
    
    def _identify_target_cubes(self, query: str) -> List[CubeType]:
        """Identify which cubes are relevant to the query"""
        target_cubes = []
        
        for cube_type, keywords in self.cube_keywords.items():
            if any(keyword in query for keyword in keywords):
                target_cubes.append(cube_type)
        
        # Default to data cube if no specific cube identified
        if not target_cubes:
            target_cubes.append(CubeType.DATA)
        
        return target_cubes
    
    def _identify_query_types(self, query: str) -> List[QueryType]:
        """Identify query types from natural language"""
        query_types = []
        
        for query_type, patterns in self.query_type_patterns.items():
            if any(pattern in query for pattern in patterns):
                query_types.append(query_type)
        
        # Default to vector search if no specific type identified
        if not query_types:
            query_types.append(QueryType.VECTOR_SEARCH)
        
        return query_types
    
    def _extract_parameters(self, query: str, cube: CubeType, 
                          query_type: QueryType) -> Dict[str, Any]:
        """Extract parameters for specific cube and query type"""
        
        parameters = {
            'original_query': query,
            'cube_specific_terms': [],
            'filters': {},
            'limits': 10
        }
        
        # Extract cube-specific terms
        if cube in self.cube_keywords:
            for keyword in self.cube_keywords[cube]:
                if keyword in query.lower():
                    parameters['cube_specific_terms'].append(keyword)
        
        # Extract numerical parameters
        import re
        numbers = re.findall(r'\d+', query)
        if numbers:
            parameters['limits'] = min(int(numbers[0]), 100)
        
        return parameters

class CostEstimator:
    """Estimates execution costs for query fragments"""
    
    def __init__(self):
        # Base costs for different operations (in arbitrary units)
        self.base_costs = {
            QueryType.VECTOR_SEARCH: 1.0,
            QueryType.GRAPH_TRAVERSAL: 2.0,
            QueryType.TEMPORAL_ANALYSIS: 1.5,
            QueryType.TOPOLOGICAL_QUERY: 3.0,
            QueryType.HYBRID_SEARCH: 2.5,
            QueryType.CROSS_DOMAIN: 4.0
        }
        
        # Cube complexity factors
        self.cube_complexity = {
            CubeType.CODE: 1.2,
            CubeType.DATA: 1.0,
            CubeType.USER: 0.8,
            CubeType.TEMPORAL: 1.5,
            CubeType.SYSTEM: 1.1,
            CubeType.ORCHESTRATOR: 2.0
        }
        
        # Historical performance data
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
    
    def estimate_fragment_cost(self, fragment: QueryFragment) -> Tuple[float, float]:
        """Estimate cost and time for a query fragment"""
        
        # Base cost calculation
        base_cost = self.base_costs.get(fragment.query_type, 1.0)
        cube_factor = self.cube_complexity.get(fragment.cube_type, 1.0)
        
        # Parameter-based adjustments
        param_factor = 1.0
        if 'limits' in fragment.parameters:
            param_factor *= np.log10(fragment.parameters['limits'] + 1)
        
        # Historical performance adjustment
        history_key = f"{fragment.cube_type.value}_{fragment.query_type.value}"
        if history_key in self.performance_history:
            avg_time = np.mean(self.performance_history[history_key])
            param_factor *= (avg_time / 1.0)  # Normalize to 1.0 baseline
        
        estimated_cost = base_cost * cube_factor * param_factor
        estimated_time = estimated_cost * 0.5  # Rough time estimate
        
        return estimated_cost, estimated_time
    
    def update_performance_history(self, fragment: QueryFragment, 
                                 actual_time: float):
        """Update performance history with actual execution time"""
        
        history_key = f"{fragment.cube_type.value}_{fragment.query_type.value}"
        self.performance_history[history_key].append(actual_time)
        
        # Keep only recent history (last 100 executions)
        if len(self.performance_history[history_key]) > 100:
            self.performance_history[history_key] = \
                self.performance_history[history_key][-100:]

class TopologyAnalyzer:
    """Analyzes topological relationships between cubes"""
    
    def __init__(self):
        self.relationship_cache: Dict[str, TopologicalRelationship] = {}
        self.topology_metrics: Dict[str, float] = {}
    
    def analyze_cross_cube_topology(self, cubes: List[CubeType]) -> Dict[str, Any]:
        """Analyze topological relationships between cubes"""
        
        relationships = []
        complexity_score = 0.0
        
        # Analyze pairwise relationships
        for i, cube1 in enumerate(cubes):
            for cube2 in cubes[i+1:]:
                relationship = self._compute_relationship(cube1, cube2)
                relationships.append(relationship)
                complexity_score += relationship.geometric_distance
        
        # Calculate overall topology metrics
        if relationships:
            avg_strength = np.mean([r.strength for r in relationships])
            avg_distance = np.mean([r.geometric_distance for r in relationships])
            avg_similarity = np.mean([r.semantic_similarity for r in relationships])
            
            topology_complexity = (
                avg_distance * 0.4 + 
                (1 - avg_similarity) * 0.3 + 
                (1 - avg_strength) * 0.3
            )
        else:
            topology_complexity = 0.0
        
        return {
            'relationships': relationships,
            'complexity_score': topology_complexity,
            'num_relationships': len(relationships),
            'avg_strength': avg_strength if relationships else 0.0,
            'optimization_potential': max(0, 1 - topology_complexity)
        }
    
    def _compute_relationship(self, cube1: CubeType, 
                            cube2: CubeType) -> TopologicalRelationship:
        """Compute topological relationship between two cubes"""
        
        # Check cache first
        cache_key = f"{cube1.value}_{cube2.value}"
        if cache_key in self.relationship_cache:
            return self.relationship_cache[cache_key]
        
        # Define cube relationships (simplified model)
        relationship_matrix = {
            (CubeType.CODE, CubeType.DATA): (0.8, 0.3, 0.7, 0.2),
            (CubeType.CODE, CubeType.USER): (0.4, 0.7, 0.3, 0.1),
            (CubeType.CODE, CubeType.TEMPORAL): (0.6, 0.5, 0.5, 0.8),
            (CubeType.CODE, CubeType.SYSTEM): (0.9, 0.2, 0.8, 0.3),
            (CubeType.DATA, CubeType.USER): (0.7, 0.4, 0.6, 0.2),
            (CubeType.DATA, CubeType.TEMPORAL): (0.8, 0.3, 0.7, 0.9),
            (CubeType.DATA, CubeType.SYSTEM): (0.6, 0.5, 0.5, 0.4),
            (CubeType.USER, CubeType.TEMPORAL): (0.5, 0.6, 0.4, 0.7),
            (CubeType.USER, CubeType.SYSTEM): (0.4, 0.7, 0.3, 0.2),
            (CubeType.TEMPORAL, CubeType.SYSTEM): (0.7, 0.4, 0.6, 0.5)
        }
        
        # Get relationship metrics (strength, distance, similarity, correlation)
        key = (cube1, cube2) if (cube1, cube2) in relationship_matrix else (cube2, cube1)
        if key in relationship_matrix:
            strength, distance, similarity, correlation = relationship_matrix[key]
        else:
            # Default values for unknown relationships
            strength, distance, similarity, correlation = (0.3, 0.8, 0.2, 0.1)
        
        relationship = TopologicalRelationship(
            source_cube=cube1,
            target_cube=cube2,
            relationship_type="geometric",
            strength=strength,
            geometric_distance=distance,
            semantic_similarity=similarity,
            temporal_correlation=correlation
        )
        
        # Cache the relationship
        self.relationship_cache[cache_key] = relationship
        
        return relationship

class QueryOptimizer:
    """Optimizes query execution plans"""
    
    def __init__(self):
        self.cost_estimator = CostEstimator()
        self.topology_analyzer = TopologyAnalyzer()
    
    def optimize_query_plan(self, fragments: List[QueryFragment]) -> QueryPlan:
        """Optimize query execution plan"""
        
        # Estimate costs for all fragments
        for fragment in fragments:
            cost, time = self.cost_estimator.estimate_fragment_cost(fragment)
            fragment.estimated_cost = cost
            fragment.estimated_time = time
        
        # Analyze topology if cross-domain query
        cube_types = list(set(f.cube_type for f in fragments))
        topology_analysis = self.topology_analyzer.analyze_cross_cube_topology(cube_types)
        
        # Determine optimal execution strategy
        execution_strategy = self._determine_execution_strategy(fragments, topology_analysis)
        
        # Optimize fragment order
        optimized_fragments = self._optimize_fragment_order(fragments, execution_strategy)
        
        # Calculate total estimates
        total_cost = sum(f.estimated_cost for f in optimized_fragments)
        total_time = self._calculate_total_time(optimized_fragments, execution_strategy)
        
        # Calculate parallelization factor
        parallelization_factor = self._calculate_parallelization_factor(
            optimized_fragments, execution_strategy
        )
        
        plan = QueryPlan(
            plan_id=str(uuid.uuid4()),
            original_query=fragments[0].query_text if fragments else "",
            fragments=optimized_fragments,
            execution_strategy=execution_strategy,
            estimated_total_cost=total_cost,
            estimated_total_time=total_time,
            parallelization_factor=parallelization_factor,
            topology_complexity=topology_analysis['complexity_score'],
            created_at=datetime.now()
        )
        
        return plan
    
    def _determine_execution_strategy(self, fragments: List[QueryFragment],
                                    topology_analysis: Dict[str, Any]) -> ExecutionStrategy:
        """Determine optimal execution strategy"""
        
        num_fragments = len(fragments)
        complexity = topology_analysis['complexity_score']
        has_dependencies = any(f.dependencies for f in fragments)
        
        # Decision logic for execution strategy
        if num_fragments == 1:
            return ExecutionStrategy.SEQUENTIAL
        elif has_dependencies:
            return ExecutionStrategy.PIPELINE
        elif complexity > 0.7:
            return ExecutionStrategy.ADAPTIVE
        elif num_fragments <= 3:
            return ExecutionStrategy.PARALLEL
        else:
            return ExecutionStrategy.PIPELINE
    
    def _optimize_fragment_order(self, fragments: List[QueryFragment],
                                strategy: ExecutionStrategy) -> List[QueryFragment]:
        """Optimize the order of fragment execution"""
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            # Sort by cost (cheapest first)
            return sorted(fragments, key=lambda f: f.estimated_cost)
        
        elif strategy == ExecutionStrategy.PARALLEL:
            # Sort by priority and cost
            return sorted(fragments, key=lambda f: (f.priority, f.estimated_cost))
        
        elif strategy == ExecutionStrategy.PIPELINE:
            # Topological sort based on dependencies
            return self._topological_sort(fragments)
        
        elif strategy == ExecutionStrategy.ADAPTIVE:
            # Complex optimization considering multiple factors
            return self._adaptive_optimization(fragments)
        
        return fragments
    
    def _topological_sort(self, fragments: List[QueryFragment]) -> List[QueryFragment]:
        """Perform topological sort of fragments based on dependencies"""
        
        # Build dependency graph
        in_degree = {f.fragment_id: 0 for f in fragments}
        graph = {f.fragment_id: [] for f in fragments}
        fragment_map = {f.fragment_id: f for f in fragments}
        
        for fragment in fragments:
            for dep in fragment.dependencies:
                if dep in graph:
                    graph[dep].append(fragment.fragment_id)
                    in_degree[fragment.fragment_id] += 1
        
        # Kahn's algorithm
        queue = [fid for fid, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(fragment_map[current])
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _adaptive_optimization(self, fragments: List[QueryFragment]) -> List[QueryFragment]:
        """Adaptive optimization considering multiple factors"""
        
        # Score each fragment based on multiple criteria
        scored_fragments = []
        
        for fragment in fragments:
            score = (
                fragment.estimated_cost * 0.3 +
                fragment.estimated_time * 0.3 +
                fragment.priority * 0.2 +
                len(fragment.dependencies) * 0.2
            )
            scored_fragments.append((score, fragment))
        
        # Sort by score
        scored_fragments.sort(key=lambda x: x[0])
        
        return [fragment for _, fragment in scored_fragments]
    
    def _calculate_total_time(self, fragments: List[QueryFragment],
                            strategy: ExecutionStrategy) -> float:
        """Calculate total execution time based on strategy"""
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            return sum(f.estimated_time for f in fragments)
        
        elif strategy == ExecutionStrategy.PARALLEL:
            return max(f.estimated_time for f in fragments) if fragments else 0.0
        
        elif strategy in [ExecutionStrategy.PIPELINE, ExecutionStrategy.ADAPTIVE]:
            # Simplified pipeline calculation
            return sum(f.estimated_time for f in fragments) * 0.7
        
        return sum(f.estimated_time for f in fragments)
    
    def _calculate_parallelization_factor(self, fragments: List[QueryFragment],
                                        strategy: ExecutionStrategy) -> float:
        """Calculate parallelization factor"""
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            return 1.0
        elif strategy == ExecutionStrategy.PARALLEL:
            return min(len(fragments), 4.0)  # Assume max 4 parallel workers
        elif strategy == ExecutionStrategy.PIPELINE:
            return min(len(fragments) * 0.6, 3.0)
        else:  # ADAPTIVE
            return min(len(fragments) * 0.8, 3.5)

class QueryExecutor:
    """Executes optimized query plans"""
    
    def __init__(self):
        self.cost_estimator = CostEstimator()
        self.active_executions: Dict[str, Any] = {}
    
    async def execute_query_plan(self, plan: QueryPlan) -> Dict[str, Any]:
        """Execute a complete query plan"""
        
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        self.active_executions[execution_id] = {
            'plan_id': plan.plan_id,
            'status': 'running',
            'start_time': start_time,
            'fragments_completed': 0,
            'total_fragments': len(plan.fragments)
        }
        
        try:
            # Execute based on strategy
            if plan.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                results = await self._execute_sequential(plan.fragments)
            elif plan.execution_strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_parallel(plan.fragments)
            elif plan.execution_strategy == ExecutionStrategy.PIPELINE:
                results = await self._execute_pipeline(plan.fragments)
            else:  # ADAPTIVE
                results = await self._execute_adaptive(plan.fragments)
            
            # Combine results
            combined_results = self._combine_results(results, plan)
            
            execution_time = time.time() - start_time
            
            # Update performance history
            for fragment, result in zip(plan.fragments, results):
                self.cost_estimator.update_performance_history(
                    fragment, result.execution_time
                )
            
            self.active_executions[execution_id]['status'] = 'completed'
            
            return {
                'execution_id': execution_id,
                'plan_id': plan.plan_id,
                'results': combined_results,
                'execution_time': execution_time,
                'fragments_executed': len(results),
                'strategy_used': plan.execution_strategy.value,
                'performance_metrics': {
                    'estimated_time': plan.estimated_total_time,
                    'actual_time': execution_time,
                    'accuracy': abs(plan.estimated_total_time - execution_time) / max(execution_time, 0.1),
                    'parallelization_achieved': plan.parallelization_factor
                }
            }
            
        except Exception as e:
            self.active_executions[execution_id]['status'] = 'failed'
            logger.error(f"❌ Query execution failed: {e}")
            raise
        
        finally:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_sequential(self, fragments: List[QueryFragment]) -> List[ExecutionResult]:
        """Execute fragments sequentially"""
        
        results = []
        for fragment in fragments:
            result = await self._execute_fragment(fragment)
            results.append(result)
        
        return results
    
    async def _execute_parallel(self, fragments: List[QueryFragment]) -> List[ExecutionResult]:
        """Execute fragments in parallel"""
        
        tasks = [self._execute_fragment(fragment) for fragment in fragments]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _execute_pipeline(self, fragments: List[QueryFragment]) -> List[ExecutionResult]:
        """Execute fragments in pipeline fashion"""
        
        results = []
        dependency_results = {}
        
        for fragment in fragments:
            # Wait for dependencies
            for dep_id in fragment.dependencies:
                if dep_id not in dependency_results:
                    # Find and execute dependency
                    dep_fragment = next(f for f in fragments if f.fragment_id == dep_id)
                    dep_result = await self._execute_fragment(dep_fragment)
                    dependency_results[dep_id] = dep_result
            
            # Execute current fragment
            result = await self._execute_fragment(fragment, dependency_results)
            results.append(result)
            dependency_results[fragment.fragment_id] = result
        
        return results
    
    async def _execute_adaptive(self, fragments: List[QueryFragment]) -> List[ExecutionResult]:
        """Execute fragments with adaptive strategy"""
        
        # Start with parallel execution for independent fragments
        independent_fragments = [f for f in fragments if not f.dependencies]
        dependent_fragments = [f for f in fragments if f.dependencies]
        
        results = []
        
        # Execute independent fragments in parallel
        if independent_fragments:
            parallel_results = await self._execute_parallel(independent_fragments)
            results.extend(parallel_results)
        
        # Execute dependent fragments in pipeline
        if dependent_fragments:
            pipeline_results = await self._execute_pipeline(dependent_fragments)
            results.extend(pipeline_results)
        
        return results
    
    async def _execute_fragment(self, fragment: QueryFragment,
                              dependency_results: Optional[Dict[str, ExecutionResult]] = None) -> ExecutionResult:
        """Execute a single query fragment"""
        
        start_time = time.time()
        
        # Simulate fragment execution (in production, this would route to actual cubes)
        await asyncio.sleep(fragment.estimated_time * 0.1)  # Simulate work
        
        # Generate mock results
        results = {
            'fragment_id': fragment.fragment_id,
            'cube_type': fragment.cube_type.value,
            'query_type': fragment.query_type.value,
            'matches': np.random.randint(1, 100),
            'data': f"Results for {fragment.query_text}"
        }
        
        execution_time = time.time() - start_time
        
        return ExecutionResult(
            fragment_id=fragment.fragment_id,
            cube_type=fragment.cube_type,
            results=results,
            execution_time=execution_time,
            records_processed=results['matches'],
            metadata={'dependencies_used': len(fragment.dependencies)}
        )
    
    def _combine_results(self, results: List[ExecutionResult], 
                        plan: QueryPlan) -> Dict[str, Any]:
        """Combine results from all fragments"""
        
        combined = {
            'query': plan.original_query,
            'total_matches': sum(r.records_processed for r in results),
            'cube_results': {},
            'cross_domain_insights': {},
            'execution_summary': {
                'fragments_executed': len(results),
                'total_records_processed': sum(r.records_processed for r in results),
                'execution_strategy': plan.execution_strategy.value,
                'topology_complexity': plan.topology_complexity
            }
        }
        
        # Group results by cube
        for result in results:
            cube_name = result.cube_type.value
            if cube_name not in combined['cube_results']:
                combined['cube_results'][cube_name] = []
            combined['cube_results'][cube_name].append(result.results)
        
        # Generate cross-domain insights for multi-cube queries
        if len(set(r.cube_type for r in results)) > 1:
            combined['cross_domain_insights'] = self._generate_cross_domain_insights(results)
        
        return combined
    
    def _generate_cross_domain_insights(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Generate insights from cross-domain query results"""
        
        cube_types = [r.cube_type.value for r in results]
        total_matches = sum(r.records_processed for r in results)
        
        insights = {
            'cubes_involved': cube_types,
            'total_cross_domain_matches': total_matches,
            'cube_contribution': {
                r.cube_type.value: r.records_processed / total_matches 
                for r in results if total_matches > 0
            },
            'topological_patterns': {
                'connectivity': len(cube_types) / 5.0,  # Normalized by max cubes
                'complexity': np.mean([r.execution_time for r in results]),
                'coherence': 1.0 - (np.std([r.records_processed for r in results]) / 
                                  (np.mean([r.records_processed for r in results]) + 1))
            }
        }
        
        return insights

class AdvancedQueryPlanner:
    """Main query planning and execution system"""
    
    def __init__(self):
        self.parser = QueryParser()
        self.optimizer = QueryOptimizer()
        self.executor = QueryExecutor()
        
        logger.info("🧠 Advanced Query Planner initialized")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a complete query from parsing to execution"""
        
        start_time = time.time()
        
        # Parse query into fragments
        fragments = self.parser.parse_query(query)
        logger.info(f"📝 Parsed query into {len(fragments)} fragments")
        
        # Optimize execution plan
        plan = self.optimizer.optimize_query_plan(fragments)
        logger.info(f"🎯 Created execution plan: {plan.execution_strategy.value}")
        
        # Execute plan
        results = await self.executor.execute_query_plan(plan)
        
        total_time = time.time() - start_time
        
        return {
            'query': query,
            'planning_time': results['execution_time'],
            'total_time': total_time,
            'plan': {
                'plan_id': plan.plan_id,
                'strategy': plan.execution_strategy.value,
                'fragments': len(plan.fragments),
                'estimated_cost': plan.estimated_total_cost,
                'topology_complexity': plan.topology_complexity
            },
            'results': results['results'],
            'performance': results['performance_metrics']
        }
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query processing statistics"""
        
        return {
            'active_executions': len(self.executor.active_executions),
            'performance_history_size': sum(
                len(history) for history in 
                self.optimizer.cost_estimator.performance_history.values()
            ),
            'cached_relationships': len(
                self.optimizer.topology_analyzer.relationship_cache
            ),
            'system_status': 'operational'
        }

async def test_advanced_query_planner():
    """Test the advanced query planning system"""
    
    print("🧠 Testing Advanced Query Planner")
    print("=" * 50)
    
    planner = AdvancedQueryPlanner()
    
    # Test queries
    test_queries = [
        "Find code related to user authentication",
        "Show data trends over time for customer interactions",
        "Analyze system performance metrics and correlate with code changes",
        "Find relationships between user behavior and system errors",
        "Search for similar data patterns across all domains"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        
        try:
            result = await planner.process_query(query)
            
            print(f"✅ Query processed successfully")
            print(f"   Strategy: {result['plan']['strategy']}")
            print(f"   Fragments: {result['plan']['fragments']}")
            print(f"   Total time: {result['total_time']:.3f}s")
            print(f"   Matches: {result['results']['total_matches']}")
            print(f"   Cubes involved: {list(result['results']['cube_results'].keys())}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Get statistics
    stats = planner.get_query_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Active executions: {stats['active_executions']}")
    print(f"   Performance history: {stats['performance_history_size']} records")
    print(f"   Cached relationships: {stats['cached_relationships']}")
    
    print("\n🧠 Advanced Query Planner Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_advanced_query_planner())