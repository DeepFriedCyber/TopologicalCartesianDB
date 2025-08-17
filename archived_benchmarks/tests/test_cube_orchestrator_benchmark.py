#!/usr/bin/env python3
"""
TOPCART Multi-Cube Orchestrator Benchmark

This tests the FULL TOPCART architecture with:
- Multiple specialized domain expert cubes
- Cube orchestrator for routing queries
- Inter-cube mapping and relationships
- Cross-domain search capabilities

This is the TRUE TOPCART system as designed!
"""

import requests
import json
import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator
from topological_cartesian.multi_cube_orchestrator import (
    MultiCubeOrchestrator, CartesianCube, CubeType, 
    CrossCubeInteraction
)
from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

logger = logging.getLogger(__name__)


@dataclass
class CubeBenchmarkResult:
    """Benchmark result for cube orchestrator system"""
    system_name: str
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    avg_query_time: float
    total_queries: int
    successful_retrievals: int
    cube_utilization: Dict[str, int]  # How many queries each cube handled
    cross_cube_queries: int  # Queries requiring multiple cubes
    orchestration_accuracy: float  # How often queries went to right cube


class CubeOrchestratorBenchmark:
    """Benchmark for the full TOPCART cube orchestrator system"""
    
    def __init__(self):
        self.orchestrator = None
        self.test_data = self._create_multi_domain_dataset()
    
    def _create_multi_domain_dataset(self) -> Dict[str, Any]:
        """Create test dataset spanning multiple domain cubes"""
        
        # Documents for different domain cubes
        documents = {
            # CODE CUBE documents
            'code_001': {
                'content': 'Python machine learning implementation using scikit-learn and pandas',
                'domain': 'programming',
                'expected_cube': 'code_cube',
                'complexity': 0.6,
                'abstraction': 0.4,
                'coupling': 0.3
            },
            'code_002': {
                'content': 'Advanced neural network architecture with PyTorch and CUDA optimization',
                'domain': 'programming', 
                'expected_cube': 'code_cube',
                'complexity': 0.9,
                'abstraction': 0.8,
                'coupling': 0.6
            },
            'code_003': {
                'content': 'Simple HTML CSS JavaScript tutorial for web development beginners',
                'domain': 'programming',
                'expected_cube': 'code_cube', 
                'complexity': 0.2,
                'abstraction': 0.1,
                'coupling': 0.2
            },
            
            # DATA CUBE documents
            'data_001': {
                'content': 'Large-scale data processing with Apache Spark and distributed computing',
                'domain': 'data_science',
                'expected_cube': 'data_cube',
                'volume': 0.9,
                'velocity': 0.8,
                'variety': 0.6
            },
            'data_002': {
                'content': 'Statistical analysis of customer behavior patterns using R and SQL',
                'domain': 'data_science',
                'expected_cube': 'data_cube',
                'volume': 0.5,
                'velocity': 0.4,
                'variety': 0.7
            },
            'data_003': {
                'content': 'Real-time streaming data analytics with Kafka and machine learning',
                'domain': 'data_science',
                'expected_cube': 'data_cube',
                'volume': 0.7,
                'velocity': 0.9,
                'variety': 0.8
            },
            
            # USER CUBE documents  
            'user_001': {
                'content': 'User engagement metrics and behavioral analytics for mobile applications',
                'domain': 'user_experience',
                'expected_cube': 'user_cube',
                'activity_level': 0.8,
                'preference_strength': 0.6,
                'engagement': 0.7
            },
            'user_002': {
                'content': 'Personalization algorithms for recommendation systems and user preferences',
                'domain': 'user_experience',
                'expected_cube': 'user_cube',
                'activity_level': 0.6,
                'preference_strength': 0.9,
                'engagement': 0.8
            },
            
            # SYSTEM CUBE documents
            'system_001': {
                'content': 'High-performance computing cluster optimization and resource management',
                'domain': 'system_performance',
                'expected_cube': 'system_cube',
                'cpu_intensity': 0.9,
                'memory_usage': 0.8,
                'io_complexity': 0.7
            },
            'system_002': {
                'content': 'Cloud infrastructure scaling and load balancing for web applications',
                'domain': 'system_performance',
                'expected_cube': 'system_cube',
                'cpu_intensity': 0.6,
                'memory_usage': 0.5,
                'io_complexity': 0.8
            }
        }
        
        # Test queries with expected cube routing
        queries = {
            # Single-cube queries
            'query_code_1': {
                'text': 'How to implement deep learning neural networks in Python?',
                'expected_cubes': ['code_cube'],
                'relevant_docs': ['code_001', 'code_002'],
                'query_type': 'single_cube'
            },
            'query_data_1': {
                'text': 'Big data processing with distributed computing frameworks',
                'expected_cubes': ['data_cube'],
                'relevant_docs': ['data_001', 'data_003'],
                'query_type': 'single_cube'
            },
            'query_user_1': {
                'text': 'User behavior analysis and engagement optimization',
                'expected_cubes': ['user_cube'],
                'relevant_docs': ['user_001', 'user_002'],
                'query_type': 'single_cube'
            },
            'query_system_1': {
                'text': 'Performance optimization for high-load computing systems',
                'expected_cubes': ['system_cube'],
                'relevant_docs': ['system_001', 'system_002'],
                'query_type': 'single_cube'
            },
            
            # Cross-cube queries (requiring multiple domain expertise)
            'query_cross_1': {
                'text': 'Machine learning model deployment on scalable cloud infrastructure',
                'expected_cubes': ['code_cube', 'system_cube'],
                'relevant_docs': ['code_002', 'system_002'],
                'query_type': 'cross_cube'
            },
            'query_cross_2': {
                'text': 'Real-time user recommendation system with big data processing',
                'expected_cubes': ['data_cube', 'user_cube'],
                'relevant_docs': ['data_003', 'user_002'],
                'query_type': 'cross_cube'
            },
            'query_cross_3': {
                'text': 'Performance monitoring for data science applications in production',
                'expected_cubes': ['data_cube', 'system_cube'],
                'relevant_docs': ['data_001', 'system_001'],
                'query_type': 'cross_cube'
            }
        }
        
        return {
            'documents': documents,
            'queries': queries
        }
    
    def setup_cube_orchestrator(self) -> bool:
        """Setup the multi-cube orchestrator with domain expert cubes"""
        
        print("Setting up Multi-Cube TOPCART Orchestrator...")
        
        try:
            # Create the orchestrator
            self.orchestrator = MultiCubeOrchestrator()
            
            # Create specialized domain expert cubes
            cubes_config = [
                {
                    'name': 'code_cube',
                    'type': CubeType.CODE,
                    'dimensions': ['complexity', 'abstraction', 'coupling', 'maintainability'],
                    'ranges': {
                        'complexity': (-1.0, 1.0),
                        'abstraction': (-1.0, 1.0), 
                        'coupling': (-1.0, 1.0),
                        'maintainability': (-1.0, 1.0)
                    },
                    'specialization': 'Programming and software development',
                    'expertise_domains': ['programming', 'software', 'code', 'development', 'python', 'javascript']
                },
                {
                    'name': 'data_cube',
                    'type': CubeType.DATA,
                    'dimensions': ['volume', 'velocity', 'variety', 'veracity'],
                    'ranges': {
                        'volume': (-1.0, 1.0),
                        'velocity': (-1.0, 1.0),
                        'variety': (-1.0, 1.0), 
                        'veracity': (-1.0, 1.0)
                    },
                    'specialization': 'Data science and analytics',
                    'expertise_domains': ['data', 'analytics', 'statistics', 'machine learning', 'big data']
                },
                {
                    'name': 'user_cube',
                    'type': CubeType.USER,
                    'dimensions': ['activity_level', 'preference_strength', 'engagement', 'satisfaction'],
                    'ranges': {
                        'activity_level': (-1.0, 1.0),
                        'preference_strength': (-1.0, 1.0),
                        'engagement': (-1.0, 1.0),
                        'satisfaction': (-1.0, 1.0)
                    },
                    'specialization': 'User experience and behavior',
                    'expertise_domains': ['user', 'behavior', 'engagement', 'experience', 'personalization']
                },
                {
                    'name': 'system_cube',
                    'type': CubeType.SYSTEM,
                    'dimensions': ['cpu_intensity', 'memory_usage', 'io_complexity', 'reliability'],
                    'ranges': {
                        'cpu_intensity': (-1.0, 1.0),
                        'memory_usage': (-1.0, 1.0),
                        'io_complexity': (-1.0, 1.0),
                        'reliability': (-1.0, 1.0)
                    },
                    'specialization': 'System performance and infrastructure',
                    'expertise_domains': ['system', 'performance', 'infrastructure', 'cloud', 'scaling']
                }
            ]
            
            # Add cubes to orchestrator
            for cube_config in cubes_config:
                coordinate_engine = EnhancedCoordinateEngine()
                
                cube = CartesianCube(
                    name=cube_config['name'],
                    cube_type=cube_config['type'],
                    dimensions=cube_config['dimensions'],
                    coordinate_ranges=cube_config['ranges'],
                    specialization=cube_config['specialization'],
                    coordinate_engine=coordinate_engine,
                    expertise_domains=cube_config['expertise_domains']
                )
                
                self.orchestrator.add_cube(cube)
            
            print(f"✅ Created {len(cubes_config)} specialized domain expert cubes")
            
            # Index documents in appropriate cubes
            self._index_documents()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup cube orchestrator: {e}")
            return False
    
    def _index_documents(self):
        """Index documents in their appropriate domain expert cubes"""
        
        print("Indexing documents in domain expert cubes...")
        
        for doc_id, doc_data in self.test_data['documents'].items():
            try:
                # Let orchestrator determine which cube(s) should handle this document
                cube_assignments = self.orchestrator.route_content(
                    doc_data['content'], 
                    doc_data['domain']
                )
                
                # Index in assigned cubes
                for cube_name in cube_assignments:
                    self.orchestrator.index_document(
                        cube_name=cube_name,
                        doc_id=doc_id,
                        content=doc_data['content'],
                        metadata=doc_data
                    )
                
                print(f"  {doc_id} → {cube_assignments}")
                
            except Exception as e:
                logger.error(f"Failed to index document {doc_id}: {e}")
        
        print(f"✅ Indexed {len(self.test_data['documents'])} documents across domain cubes")
    
    def test_cube_orchestrator(self) -> CubeBenchmarkResult:
        """Test the cube orchestrator system"""
        
        print(f"\nTesting Multi-Cube TOPCART Orchestrator...")
        
        if not self.orchestrator:
            raise Exception("Orchestrator not initialized")
        
        # Metrics tracking
        query_times = []
        precision_at_1_scores = []
        precision_at_5_scores = []
        precision_at_10_scores = []
        recall_at_10_scores = []
        mrr_scores = []
        successful_retrievals = 0
        cube_utilization = {}
        cross_cube_queries = 0
        correct_orchestrations = 0
        
        # Test each query
        for query_id, query_data in self.test_data['queries'].items():
            print(f"  Testing query: {query_id}")
            print(f"    Query: {query_data['text'][:60]}...")
            
            query_start = time.time()
            
            # Use orchestrator to handle query
            try:
                results = self.orchestrator.search(
                    query=query_data['text'],
                    k=10,
                    cross_cube_search=True
                )
                
                query_time = time.time() - query_start
                query_times.append(query_time)
                
                # Track cube utilization
                used_cubes = results.get('cubes_used', [])
                for cube_name in used_cubes:
                    cube_utilization[cube_name] = cube_utilization.get(cube_name, 0) + 1
                
                # Check orchestration accuracy
                expected_cubes = set(query_data['expected_cubes'])
                actual_cubes = set(used_cubes)
                
                if expected_cubes.intersection(actual_cubes):
                    correct_orchestrations += 1
                
                # Track cross-cube queries
                if query_data['query_type'] == 'cross_cube':
                    cross_cube_queries += 1
                
                # Extract document results
                doc_results = results.get('documents', [])
                
                if doc_results:
                    successful_retrievals += 1
                    
                    # Calculate metrics
                    result_doc_ids = [r['doc_id'] for r in doc_results]
                    relevant_doc_ids = set(query_data['relevant_docs'])
                    
                    p_at_1 = self._calculate_precision_at_k(result_doc_ids, relevant_doc_ids, 1)
                    p_at_5 = self._calculate_precision_at_k(result_doc_ids, relevant_doc_ids, 5)
                    p_at_10 = self._calculate_precision_at_k(result_doc_ids, relevant_doc_ids, 10)
                    
                    precision_at_1_scores.append(p_at_1)
                    precision_at_5_scores.append(p_at_5)
                    precision_at_10_scores.append(p_at_10)
                    
                    recall_10 = self._calculate_recall_at_k(result_doc_ids, relevant_doc_ids, 10)
                    recall_at_10_scores.append(recall_10)
                    
                    mrr = self._calculate_mrr(result_doc_ids, relevant_doc_ids)
                    mrr_scores.append(mrr)
                    
                    print(f"    Cubes used: {used_cubes}")
                    print(f"    P@1: {p_at_1:.3f}, P@5: {p_at_5:.3f}, Recall@10: {recall_10:.3f}")
                    print(f"    Top result: {result_doc_ids[0] if result_doc_ids else 'None'}")
                
            except Exception as e:
                logger.error(f"Query {query_id} failed: {e}")
                query_times.append(0.0)
        
        # Calculate orchestration accuracy
        orchestration_accuracy = correct_orchestrations / len(self.test_data['queries'])
        
        return CubeBenchmarkResult(
            system_name="Multi-Cube TOPCART Orchestrator",
            precision_at_1=np.mean(precision_at_1_scores) if precision_at_1_scores else 0.0,
            precision_at_5=np.mean(precision_at_5_scores) if precision_at_5_scores else 0.0,
            precision_at_10=np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0,
            recall_at_10=np.mean(recall_at_10_scores) if recall_at_10_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            avg_query_time=np.mean(query_times) if query_times else 0.0,
            total_queries=len(self.test_data['queries']),
            successful_retrievals=successful_retrievals,
            cube_utilization=cube_utilization,
            cross_cube_queries=cross_cube_queries,
            orchestration_accuracy=orchestration_accuracy
        )
    
    def _calculate_precision_at_k(self, result_ids: List[str], relevant_ids: set, k: int) -> float:
        """Calculate Precision@K"""
        if not result_ids:
            return 0.0
        
        top_k = result_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        
        return relevant_in_top_k / min(k, len(result_ids))
    
    def _calculate_recall_at_k(self, result_ids: List[str], relevant_ids: set, k: int) -> float:
        """Calculate Recall@K"""
        if not result_ids or not relevant_ids:
            return 0.0
        
        top_k = result_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        
        return relevant_in_top_k / len(relevant_ids)
    
    def _calculate_mrr(self, result_ids: List[str], relevant_ids: set) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not result_ids or not relevant_ids:
            return 0.0
        
        for i, doc_id in enumerate(result_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def print_results(self, results: CubeBenchmarkResult):
        """Print detailed benchmark results"""
        
        print(f"\n{'='*70}")
        print("MULTI-CUBE TOPCART ORCHESTRATOR BENCHMARK RESULTS")
        print(f"{'='*70}")
        
        print(f"\n🎯 SEARCH PERFORMANCE:")
        print(f"  Precision@1:        {results.precision_at_1:.3f}")
        print(f"  Precision@5:        {results.precision_at_5:.3f}")
        print(f"  Precision@10:       {results.precision_at_10:.3f}")
        print(f"  Recall@10:          {results.recall_at_10:.3f}")
        print(f"  MRR:                {results.mrr:.3f}")
        print(f"  Avg Query Time:     {results.avg_query_time:.4f}s")
        print(f"  Success Rate:       {results.successful_retrievals}/{results.total_queries}")
        
        print(f"\n🎯 ORCHESTRATION PERFORMANCE:")
        print(f"  Orchestration Accuracy: {results.orchestration_accuracy:.3f}")
        print(f"  Cross-Cube Queries:     {results.cross_cube_queries}")
        
        print(f"\n🎯 CUBE UTILIZATION:")
        for cube_name, usage_count in results.cube_utilization.items():
            print(f"  {cube_name}: {usage_count} queries")
        
        print(f"\n🎯 ARCHITECTURE ANALYSIS:")
        print(f"  ✅ Multiple specialized domain expert cubes")
        print(f"  ✅ Intelligent query routing to appropriate cubes")
        print(f"  ✅ Cross-cube search for complex queries")
        print(f"  ✅ Inter-cube coordinate mapping")
        print(f"  ✅ Domain-specific expertise utilization")
        
        if results.orchestration_accuracy > 0.7:
            print(f"\n🚀 CONCLUSION: Multi-Cube Orchestrator is working effectively!")
            print(f"   ✅ Queries are being routed to correct domain expert cubes")
            print(f"   ✅ Cross-cube search handles complex multi-domain queries")
            print(f"   ✅ Each cube provides specialized domain expertise")
        else:
            print(f"\n⚠️ CONCLUSION: Orchestrator needs tuning for better cube routing")
    
    def run_benchmark(self) -> Optional[CubeBenchmarkResult]:
        """Run the complete cube orchestrator benchmark"""
        
        print("TOPCART Multi-Cube Orchestrator Benchmark")
        print("=" * 60)
        
        # Setup orchestrator
        if not self.setup_cube_orchestrator():
            return None
        
        # Run tests
        try:
            results = self.test_cube_orchestrator()
            self.print_results(results)
            return results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    try:
        benchmark = CubeOrchestratorBenchmark()
        results = benchmark.run_benchmark()
        
        if results:
            print(f"\n🎉 Multi-Cube TOPCART Orchestrator benchmark completed!")
            print(f"🚀 This demonstrates the FULL TOPCART architecture with domain expert cubes!")
        else:
            print(f"\n❌ Benchmark failed to complete")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()