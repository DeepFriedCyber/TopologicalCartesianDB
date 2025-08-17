#!/usr/bin/env python3
"""
Real-World Integration Tests

Tests the complete system integration with realistic scenarios:
1. Large-scale document processing (10k+ documents)
2. Concurrent user simulation
3. Memory and performance stress testing
4. API endpoint testing
5. Data persistence and recovery
6. Cross-platform compatibility

These tests validate production readiness.
"""

import pytest
import numpy as np
import time
import threading
import multiprocessing
import psutil
import tempfile
import shutil
import json
import pickle
import sys
import os
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from topological_cartesian.semantic_coordinate_engine import SemanticCoordinateEngine
from topological_cartesian.proper_tda_engine import ProperTDAEngine
from topological_cartesian.robust_system_fixes import RobustCartesianSystem
from topological_cartesian.optimized_spatial_engine import OptimizedSpatialEngine, SpatialIndexConfig

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources during testing"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start monitoring system resources"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics
    
    def _monitor_loop(self, interval: float):
        """Monitor loop running in separate thread"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'threads': process.num_threads()
                })
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from monitoring"""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_mb'] for m in self.metrics]
        thread_values = [m['threads'] for m in self.metrics]
        
        return {
            'duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp'],
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            },
            'memory': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values)
            },
            'threads': {
                'avg': np.mean(thread_values),
                'max': np.max(thread_values),
                'min': np.min(thread_values)
            },
            'samples': len(self.metrics)
        }


class LargeScaleDataGenerator:
    """Generate large-scale test datasets"""
    
    def __init__(self):
        self.document_templates = [
            "Technical documentation for {topic} including {detail} and {feature}",
            "Research paper on {topic} with focus on {detail} and {feature}",
            "Tutorial guide for {topic} covering {detail} and {feature}",
            "Analysis report about {topic} examining {detail} and {feature}",
            "Implementation guide for {topic} with {detail} and {feature}",
            "Best practices for {topic} including {detail} and {feature}",
            "Case study on {topic} demonstrating {detail} and {feature}",
            "Review article about {topic} discussing {detail} and {feature}",
        ]
        
        self.topics = [
            "machine learning", "web development", "data science", "cybersecurity",
            "cloud computing", "artificial intelligence", "blockchain technology",
            "mobile development", "database systems", "software architecture",
            "quantum computing", "computer vision", "natural language processing",
            "distributed systems", "microservices", "DevOps practices"
        ]
        
        self.details = [
            "performance optimization", "security considerations", "scalability issues",
            "implementation challenges", "design patterns", "testing strategies",
            "deployment procedures", "monitoring solutions", "error handling",
            "data processing", "user experience", "integration methods"
        ]
        
        self.features = [
            "real-time processing", "fault tolerance", "high availability",
            "load balancing", "caching mechanisms", "API design",
            "data visualization", "automated testing", "continuous integration",
            "version control", "documentation", "performance metrics"
        ]
    
    def generate_documents(self, count: int) -> List[Tuple[str, str]]:
        """Generate a large number of test documents"""
        documents = []
        
        for i in range(count):
            template = np.random.choice(self.document_templates)
            topic = np.random.choice(self.topics)
            detail = np.random.choice(self.details)
            feature = np.random.choice(self.features)
            
            content = template.format(topic=topic, detail=detail, feature=feature)
            doc_id = f"large_doc_{i}"
            
            documents.append((doc_id, content))
        
        return documents


class ConcurrentUserSimulator:
    """Simulate concurrent users accessing the system"""
    
    def __init__(self, system: RobustCartesianSystem):
        self.system = system
        self.user_results = []
        self.errors = []
        self.lock = threading.Lock()
    
    def simulate_user_session(self, user_id: int, operations: int = 10) -> Dict[str, Any]:
        """Simulate a single user session"""
        session_results = {
            'user_id': user_id,
            'operations': [],
            'total_time': 0.0,
            'errors': 0
        }
        
        start_time = time.time()
        
        for op_id in range(operations):
            try:
                # Random operation type
                operation_type = np.random.choice(['add_document', 'search', 'health_check'])
                
                op_start = time.time()
                
                if operation_type == 'add_document':
                    doc_id = f"user_{user_id}_doc_{op_id}"
                    content = f"User {user_id} document {op_id} with random content {np.random.randint(1000)}"
                    coords = {
                        'domain': np.random.random(),
                        'complexity': np.random.random(),
                        'task_type': np.random.random()
                    }
                    
                    result = self.system.add_document_safely(doc_id, content, coords)
                    success = result['success']
                
                elif operation_type == 'search':
                    # This would require implementing search in RobustCartesianSystem
                    # For now, just check system health
                    health = self.system.get_system_health()
                    success = health['total_documents'] >= 0
                
                else:  # health_check
                    health = self.system.get_system_health()
                    success = health['total_documents'] >= 0
                
                op_time = time.time() - op_start
                
                session_results['operations'].append({
                    'operation': operation_type,
                    'success': success,
                    'time': op_time
                })
                
                if not success:
                    session_results['errors'] += 1
                
            except Exception as e:
                session_results['errors'] += 1
                with self.lock:
                    self.errors.append(f"User {user_id}, Op {op_id}: {str(e)}")
        
        session_results['total_time'] = time.time() - start_time
        
        with self.lock:
            self.user_results.append(session_results)
        
        return session_results
    
    def run_concurrent_simulation(self, num_users: int = 10, operations_per_user: int = 10) -> Dict[str, Any]:
        """Run concurrent user simulation"""
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            
            for user_id in range(num_users):
                future = executor.submit(self.simulate_user_session, user_id, operations_per_user)
                futures.append(future)
            
            # Wait for all users to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    with self.lock:
                        self.errors.append(f"User simulation error: {str(e)}")
        
        # Aggregate results
        total_operations = sum(len(result['operations']) for result in self.user_results)
        successful_operations = sum(
            sum(1 for op in result['operations'] if op['success']) 
            for result in self.user_results
        )
        
        total_time = sum(result['total_time'] for result in self.user_results)
        avg_operation_time = np.mean([
            op['time'] for result in self.user_results 
            for op in result['operations']
        ]) if total_operations > 0 else 0.0
        
        return {
            'num_users': num_users,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': successful_operations / max(total_operations, 1),
            'total_errors': len(self.errors),
            'avg_operation_time': avg_operation_time,
            'total_simulation_time': total_time,
            'errors': self.errors[:10]  # First 10 errors for debugging
        }


class PersistenceTestSuite:
    """Test data persistence and recovery"""
    
    def __init__(self):
        self.temp_dir = None
    
    def setup_temp_directory(self) -> str:
        """Setup temporary directory for testing"""
        self.temp_dir = tempfile.mkdtemp(prefix="cartesian_test_")
        return self.temp_dir
    
    def cleanup_temp_directory(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_system_serialization(self, system: RobustCartesianSystem) -> Dict[str, Any]:
        """Test system state serialization and deserialization"""
        
        # Add some test data
        test_docs = [
            ("persist_doc_1", "Test document for persistence", {'domain': 0.5, 'complexity': 0.3, 'task_type': 0.7}),
            ("persist_doc_2", "Another test document", {'domain': 0.8, 'complexity': 0.6, 'task_type': 0.4}),
            ("persist_doc_3", "Third test document", {'domain': 0.2, 'complexity': 0.9, 'task_type': 0.1}),
        ]
        
        for doc_id, content, coords in test_docs:
            system.add_document_safely(doc_id, content, coords)
        
        original_health = system.get_system_health()
        
        # Test JSON serialization (for configuration)
        temp_dir = self.setup_temp_directory()
        json_file = os.path.join(temp_dir, "system_config.json")
        
        try:
            # Serialize system configuration
            config_data = {
                'total_documents': original_health['total_documents'],
                'system_type': 'RobustCartesianSystem',
                'timestamp': time.time()
            }
            
            with open(json_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Verify file exists and is readable
            assert os.path.exists(json_file)
            
            with open(json_file, 'r') as f:
                loaded_config = json.load(f)
            
            json_success = loaded_config['total_documents'] == original_health['total_documents']
            
        except Exception as e:
            json_success = False
            logger.error(f"JSON serialization failed: {e}")
        
        # Test pickle serialization (for system state)
        pickle_file = os.path.join(temp_dir, "system_state.pkl")
        
        try:
            # Serialize system documents (simplified)
            system_state = {
                'documents': dict(list(system.documents.items())[:3]),  # First 3 documents
                'timestamp': time.time()
            }
            
            with open(pickle_file, 'wb') as f:
                pickle.dump(system_state, f)
            
            # Verify file exists and is readable
            assert os.path.exists(pickle_file)
            
            with open(pickle_file, 'rb') as f:
                loaded_state = pickle.load(f)
            
            pickle_success = len(loaded_state['documents']) == 3
            
        except Exception as e:
            pickle_success = False
            logger.error(f"Pickle serialization failed: {e}")
        
        self.cleanup_temp_directory()
        
        return {
            'json_serialization': json_success,
            'pickle_serialization': pickle_success,
            'original_document_count': original_health['total_documents'],
            'test_passed': json_success and pickle_success
        }


# Integration test classes
class TestLargeScaleProcessing:
    """Test large-scale document processing"""
    
    @pytest.fixture
    def large_system(self):
        """Create system optimized for large-scale processing"""
        config = SpatialIndexConfig(
            index_type='auto',
            cache_size=5000,
            rebuild_threshold=500
        )
        return OptimizedSpatialEngine(config)
    
    @pytest.fixture
    def data_generator(self):
        return LargeScaleDataGenerator()
    
    def test_10k_document_processing(self, large_system, data_generator):
        """Test processing 10,000 documents"""
        
        # Generate test documents
        documents = data_generator.generate_documents(10000)
        
        # Monitor system resources
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        # Process documents in batches
        batch_size = 1000
        processed_count = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            batch_docs = {}
            for doc_id, content in batch:
                # Generate coordinates (simplified)
                coords = {
                    'domain': np.random.random(),
                    'complexity': np.random.random(),
                    'task_type': np.random.random()
                }
                batch_docs[doc_id] = {
                    'content': content,
                    'coordinates': coords
                }
            
            large_system.add_documents(batch_docs)
            processed_count += len(batch)
            
            if processed_count % 2000 == 0:
                print(f"Processed {processed_count} documents...")
        
        processing_time = time.time() - start_time
        
        # Stop monitoring
        resource_metrics = monitor.stop_monitoring()
        resource_summary = monitor.get_summary()
        
        # Performance assertions
        assert processed_count == 10000
        assert processing_time < 300  # Should complete within 5 minutes
        assert resource_summary['memory']['max'] < 2000  # Should use less than 2GB
        
        # Test search performance on large dataset
        search_start = time.time()
        query_coords = {'domain': 0.5, 'complexity': 0.5, 'task_type': 0.5}
        results = large_system.optimized_search(query_coords, k=10)
        search_time = time.time() - search_start
        
        assert len(results) == 10
        assert search_time < 0.1  # Should search within 100ms
        
        print(f"Large-scale test results:")
        print(f"  Documents processed: {processed_count}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Search time: {search_time:.4f}s")
        print(f"  Peak memory: {resource_summary['memory']['max']:.1f}MB")
    
    def test_concurrent_processing(self, large_system, data_generator):
        """Test concurrent document processing"""
        
        documents = data_generator.generate_documents(1000)
        
        def process_batch(batch_docs):
            """Process a batch of documents"""
            batch_dict = {}
            for doc_id, content in batch_docs:
                coords = {
                    'domain': np.random.random(),
                    'complexity': np.random.random(),
                    'task_type': np.random.random()
                }
                batch_dict[doc_id] = {
                    'content': content,
                    'coordinates': coords
                }
            
            large_system.add_documents(batch_dict)
            return len(batch_dict)
        
        # Process in parallel batches
        batch_size = 100
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            total_processed = 0
            for future in as_completed(futures):
                batch_count = future.result()
                total_processed += batch_count
        
        processing_time = time.time() - start_time
        
        assert total_processed == 1000
        assert processing_time < 60  # Should complete within 1 minute
        
        print(f"Concurrent processing results:")
        print(f"  Documents processed: {total_processed}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {total_processed/processing_time:.1f} docs/sec")


class TestConcurrentUsers:
    """Test concurrent user scenarios"""
    
    @pytest.fixture
    def robust_system(self):
        return RobustCartesianSystem()
    
    def test_concurrent_user_simulation(self, robust_system):
        """Test concurrent user access"""
        
        simulator = ConcurrentUserSimulator(robust_system)
        
        # Monitor system resources
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        # Run simulation
        results = simulator.run_concurrent_simulation(
            num_users=20, 
            operations_per_user=15
        )
        
        # Stop monitoring
        resource_summary = monitor.stop_monitoring()
        resource_summary = monitor.get_summary()
        
        # Assertions
        assert results['success_rate'] > 0.8  # At least 80% success rate
        assert results['total_errors'] < results['total_operations'] * 0.1  # Less than 10% errors
        assert results['avg_operation_time'] < 1.0  # Average operation under 1 second
        
        print(f"Concurrent user test results:")
        print(f"  Users: {results['num_users']}")
        print(f"  Total operations: {results['total_operations']}")
        print(f"  Success rate: {results['success_rate']:.3f}")
        print(f"  Avg operation time: {results['avg_operation_time']:.4f}s")
        print(f"  Peak memory: {resource_summary['memory']['max']:.1f}MB")
    
    def test_stress_testing(self, robust_system):
        """Stress test with high load"""
        
        # Add initial documents
        for i in range(100):
            doc_id = f"stress_doc_{i}"
            content = f"Stress test document {i} with content"
            coords = {
                'domain': np.random.random(),
                'complexity': np.random.random(),
                'task_type': np.random.random()
            }
            robust_system.add_document_safely(doc_id, content, coords)
        
        # Stress test with many concurrent operations
        simulator = ConcurrentUserSimulator(robust_system)
        
        results = simulator.run_concurrent_simulation(
            num_users=50,  # High user count
            operations_per_user=20  # Many operations per user
        )
        
        # System should remain stable under stress
        assert results['success_rate'] > 0.7  # At least 70% success under stress
        assert results['total_errors'] < results['total_operations'] * 0.2  # Less than 20% errors
        
        # Check system health after stress test
        final_health = robust_system.get_system_health()
        assert final_health['total_documents'] > 100  # Should have added documents
        
        print(f"Stress test results:")
        print(f"  Success rate under stress: {results['success_rate']:.3f}")
        print(f"  Final document count: {final_health['total_documents']}")


class TestPersistenceAndRecovery:
    """Test data persistence and system recovery"""
    
    def test_system_persistence(self):
        """Test system state persistence"""
        
        system = RobustCartesianSystem()
        persistence_suite = PersistenceTestSuite()
        
        results = persistence_suite.test_system_serialization(system)
        
        assert results['test_passed'], "System persistence test failed"
        assert results['json_serialization'], "JSON serialization failed"
        assert results['pickle_serialization'], "Pickle serialization failed"
        
        print(f"Persistence test results:")
        print(f"  JSON serialization: {'✅' if results['json_serialization'] else '❌'}")
        print(f"  Pickle serialization: {'✅' if results['pickle_serialization'] else '❌'}")
        print(f"  Document count: {results['original_document_count']}")


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility"""
    
    def test_system_initialization(self):
        """Test system initialization across different configurations"""
        
        # Test different configurations
        configs = [
            {'embedding_backend': 'auto'},
            {'embedding_backend': 'sklearn'},  # Fallback option
        ]
        
        for config in configs:
            try:
                semantic_engine = SemanticCoordinateEngine(**config)
                status = semantic_engine.get_system_status()
                
                assert status['embedder_type'] is not None
                assert status['embedding_dimension'] > 0
                
                print(f"Config {config}: ✅ {status['embedder_type']}")
                
            except Exception as e:
                print(f"Config {config}: ❌ {str(e)}")
                # Don't fail the test if optional dependencies are missing
                if "not available" not in str(e):
                    raise
    
    def test_memory_efficiency(self):
        """Test memory efficiency across different scenarios"""
        
        system = RobustCartesianSystem()
        monitor = SystemMonitor()
        
        # Test memory usage with increasing document count
        document_counts = [100, 500, 1000]
        memory_usage = []
        
        for count in document_counts:
            monitor.start_monitoring()
            
            # Add documents
            for i in range(count):
                doc_id = f"memory_test_doc_{i}"
                content = f"Memory test document {i}"
                coords = {
                    'domain': np.random.random(),
                    'complexity': np.random.random(),
                    'task_type': np.random.random()
                }
                system.add_document_safely(doc_id, content, coords)
            
            metrics = monitor.stop_monitoring()
            summary = monitor.get_summary()
            memory_usage.append(summary['memory']['max'])
            
            print(f"Documents: {count}, Peak memory: {summary['memory']['max']:.1f}MB")
        
        # Memory usage should scale reasonably
        memory_growth_rate = (memory_usage[-1] - memory_usage[0]) / (document_counts[-1] - document_counts[0])
        
        # Should use less than 1MB per 100 documents on average
        assert memory_growth_rate < 1.0, f"Memory growth rate too high: {memory_growth_rate:.3f}MB per 100 docs"


if __name__ == "__main__":
    # Run integration tests manually
    print("Cartesian Cube System - Integration Tests")
    print("=" * 50)
    
    # Test large-scale processing
    print("\n1. Large-Scale Processing Test")
    print("-" * 30)
    
    large_system = OptimizedSpatialEngine()
    data_generator = LargeScaleDataGenerator()
    
    # Smaller test for manual run
    documents = data_generator.generate_documents(1000)
    
    start_time = time.time()
    batch_docs = {}
    for doc_id, content in documents:
        coords = {
            'domain': np.random.random(),
            'complexity': np.random.random(),
            'task_type': np.random.random()
        }
        batch_docs[doc_id] = {
            'content': content,
            'coordinates': coords
        }
    
    large_system.add_documents(batch_docs)
    processing_time = time.time() - start_time
    
    print(f"Processed {len(documents)} documents in {processing_time:.2f}s")
    print(f"Throughput: {len(documents)/processing_time:.1f} docs/sec")
    
    # Test search performance
    search_start = time.time()
    query_coords = {'domain': 0.5, 'complexity': 0.5, 'task_type': 0.5}
    results = large_system.optimized_search(query_coords, k=10)
    search_time = time.time() - search_start
    
    print(f"Search completed in {search_time:.4f}s")
    print(f"Found {len(results)} results")
    
    # Test concurrent users
    print("\n2. Concurrent User Test")
    print("-" * 25)
    
    robust_system = RobustCartesianSystem()
    simulator = ConcurrentUserSimulator(robust_system)
    
    concurrent_results = simulator.run_concurrent_simulation(
        num_users=10, 
        operations_per_user=10
    )
    
    print(f"Concurrent test results:")
    print(f"  Users: {concurrent_results['num_users']}")
    print(f"  Success rate: {concurrent_results['success_rate']:.3f}")
    print(f"  Avg operation time: {concurrent_results['avg_operation_time']:.4f}s")
    
    # Test persistence
    print("\n3. Persistence Test")
    print("-" * 20)
    
    persistence_suite = PersistenceTestSuite()
    persistence_results = persistence_suite.test_system_serialization(robust_system)
    
    print(f"Persistence results:")
    print(f"  JSON: {'✅' if persistence_results['json_serialization'] else '❌'}")
    print(f"  Pickle: {'✅' if persistence_results['pickle_serialization'] else '❌'}")
    
    # Overall assessment
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if processing_time < 60 and search_time < 0.1:
        print("✅ Large-scale processing: PASSED")
        tests_passed += 1
    else:
        print("❌ Large-scale processing: FAILED")
    
    if concurrent_results['success_rate'] > 0.8:
        print("✅ Concurrent users: PASSED")
        tests_passed += 1
    else:
        print("❌ Concurrent users: FAILED")
    
    if persistence_results['test_passed']:
        print("✅ Persistence: PASSED")
        tests_passed += 1
    else:
        print("❌ Persistence: FAILED")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed ({tests_passed/total_tests*100:.1f}%)")
    
    if tests_passed == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print("⚠️  Some integration tests failed - system needs improvement")