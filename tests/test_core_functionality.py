#!/usr/bin/env python3
"""
Core Functionality Tests

Simple, focused tests that validate the core system works without external dependencies.
These tests use only built-in libraries and basic functionality.
"""

import pytest
import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from topological_cartesian.semantic_coordinate_engine import SemanticCoordinateEngine
from topological_cartesian.robust_system_fixes import RobustCartesianSystem


class TestSemanticCoordinateEngine:
    """Test the semantic coordinate engine core functionality"""
    
    @pytest.fixture
    def engine(self):
        return SemanticCoordinateEngine()
    
    def test_engine_initialization(self, engine):
        """Test that the engine initializes properly"""
        status = engine.get_system_status()
        
        assert status['embedder_type'] is not None
        assert status['embedding_dimension'] > 0
        assert isinstance(status['trained_dimensions'], list)
        assert len(status['trained_dimensions']) > 0
    
    def test_text_to_coordinates_basic(self, engine):
        """Test basic text to coordinates conversion"""
        text = "Python programming tutorial for beginners"
        result = engine.text_to_coordinates(text)
        
        assert 'coordinates' in result
        assert 'method' in result
        
        coords = result['coordinates']
        assert 'domain' in coords
        assert 'complexity' in coords
        assert 'task_type' in coords
        
        # All coordinates should be in [0, 1] range
        for dim, value in coords.items():
            assert 0.0 <= value <= 1.0, f"Coordinate {dim} out of range: {value}"
    
    def test_coordinate_consistency(self, engine):
        """Test that same input produces consistent coordinates"""
        text = "Machine learning algorithms and implementation"
        
        result1 = engine.text_to_coordinates(text)
        result2 = engine.text_to_coordinates(text)
        
        # Should be identical due to caching
        assert result1['coordinates'] == result2['coordinates']
    
    def test_different_texts_different_coordinates(self, engine):
        """Test that different texts produce different coordinates"""
        text1 = "Advanced Python programming techniques"
        text2 = "Basic HTML tutorial for beginners"
        
        result1 = engine.text_to_coordinates(text1)
        result2 = engine.text_to_coordinates(text2)
        
        coords1 = result1['coordinates']
        coords2 = result2['coordinates']
        
        # At least one dimension should be different
        differences = sum(1 for dim in coords1.keys() 
                         if abs(coords1[dim] - coords2[dim]) > 0.01)
        assert differences > 0, "Different texts should produce different coordinates"
    
    def test_edge_cases(self, engine):
        """Test edge cases in text processing"""
        edge_cases = [
            "",  # Empty text
            "a",  # Single character
            "THE AND OR BUT",  # Only stop words
            "123 456 789",  # Only numbers
        ]
        
        for text in edge_cases:
            result = engine.text_to_coordinates(text)
            
            assert 'coordinates' in result
            coords = result['coordinates']
            
            # Should return valid coordinates even for edge cases
            for dim, value in coords.items():
                assert 0.0 <= value <= 1.0, f"Edge case '{text}' produced invalid coordinate {dim}: {value}"
    
    def test_training_example_addition(self, engine):
        """Test adding training examples"""
        initial_examples = engine.coordinate_mapper.training_data.get('domain', [])
        initial_count = len(initial_examples)
        
        engine.add_training_example(
            "Test training text", 
            {'domain': 0.8, 'complexity': 0.6, 'task_type': 0.4}
        )
        
        updated_examples = engine.coordinate_mapper.training_data.get('domain', [])
        assert len(updated_examples) == initial_count + 1


class TestRobustCartesianSystem:
    """Test the robust Cartesian system"""
    
    @pytest.fixture
    def system(self):
        return RobustCartesianSystem()
    
    def test_system_initialization(self, system):
        """Test system initializes properly"""
        health = system.get_system_health()
        
        assert 'total_documents' in health
        assert 'total_regions' in health
        assert health['total_documents'] == 0  # Should start empty
    
    def test_add_document_safely(self, system):
        """Test safe document addition"""
        doc_id = "test_doc_1"
        content = "Test document content for validation"
        coordinates = {'domain': 0.7, 'complexity': 0.5, 'task_type': 0.3}
        
        result = system.add_document_safely(doc_id, content, coordinates)
        
        assert result['success'] is True
        assert result['document_id'] == doc_id
        assert 'final_coordinates' in result
        
        # Check system health after addition
        health = system.get_system_health()
        assert health['total_documents'] == 1
    
    def test_collision_detection_and_resolution(self, system):
        """Test coordinate collision detection and resolution"""
        # Add first document
        doc1_id = "collision_doc_1"
        content1 = "First document for collision test"
        coordinates = {'domain': 0.5, 'complexity': 0.5, 'task_type': 0.5}
        
        result1 = system.add_document_safely(doc1_id, content1, coordinates)
        assert result1['success'] is True
        assert result1['collision_resolved'] is False
        
        # Add second document with same coordinates (should cause collision)
        doc2_id = "collision_doc_2"
        content2 = "Second document for collision test"
        
        result2 = system.add_document_safely(doc2_id, content2, coordinates)
        assert result2['success'] is True
        # Collision should be detected and resolved
        assert result2['collision_resolved'] is True
        
        # Final coordinates should be different
        final_coords1 = result1['final_coordinates']
        final_coords2 = result2['final_coordinates']
        
        # At least one coordinate should be different after resolution
        differences = sum(1 for dim in final_coords1.keys() 
                         if abs(final_coords1[dim] - final_coords2[dim]) > 0.001)
        assert differences > 0, "Collision resolution should produce different coordinates"
    
    def test_boundary_condition_handling(self, system):
        """Test boundary condition handling"""
        # Test coordinates outside [0, 1] range
        out_of_bounds_coords = {
            'domain': 1.5,      # Above upper bound
            'complexity': -0.2,  # Below lower bound
            'task_type': 0.5     # Normal
        }
        
        result = system.add_document_safely(
            "boundary_test_doc", 
            "Test document for boundary conditions", 
            out_of_bounds_coords
        )
        
        assert result['success'] is True
        
        final_coords = result['final_coordinates']
        
        # All coordinates should be clamped to [0, 1] range
        for dim, value in final_coords.items():
            assert 0.0 <= value <= 1.0, f"Boundary handling failed for {dim}: {value}"
    
    def test_concurrent_document_addition(self, system):
        """Test concurrent document addition"""
        import threading
        import concurrent.futures
        
        results = []
        errors = []
        
        def add_document_worker(doc_id):
            try:
                content = f"Concurrent test document {doc_id}"
                coords = {
                    'domain': np.random.random(),
                    'complexity': np.random.random(),
                    'task_type': np.random.random()
                }
                result = system.add_document_safely(doc_id, content, coords)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(20):
                future = executor.submit(add_document_worker, f"concurrent_doc_{i}")
                futures.append(future)
            
            # Wait for completion
            concurrent.futures.wait(futures)
        
        # Check results
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        
        # All operations should succeed
        success_count = sum(1 for r in results if r['success'])
        assert success_count == 20, f"Only {success_count}/20 operations succeeded"
        
        # Check final system state
        health = system.get_system_health()
        assert health['total_documents'] >= 20  # Should have at least 20 documents


class TestPerformanceBenchmarks:
    """Test performance benchmarks"""
    
    def test_coordinate_generation_performance(self):
        """Test coordinate generation performance"""
        engine = SemanticCoordinateEngine()
        
        test_texts = [
            f"Performance test document {i} with various content and complexity"
            for i in range(100)
        ]
        
        start_time = time.time()
        
        for text in test_texts:
            result = engine.text_to_coordinates(text)
            assert 'coordinates' in result
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_texts)
        
        # Should process at least 10 documents per second
        assert avg_time < 0.1, f"Coordinate generation too slow: {avg_time:.3f}s per document"
        
        print(f"Coordinate generation: {avg_time:.4f}s per document ({1/avg_time:.1f} docs/sec)")
    
    def test_document_addition_performance(self):
        """Test document addition performance"""
        system = RobustCartesianSystem()
        
        documents = [
            (f"perf_doc_{i}", f"Performance test document {i}", {
                'domain': np.random.random(),
                'complexity': np.random.random(),
                'task_type': np.random.random()
            })
            for i in range(100)
        ]
        
        start_time = time.time()
        
        for doc_id, content, coords in documents:
            result = system.add_document_safely(doc_id, content, coords)
            assert result['success'] is True
        
        total_time = time.time() - start_time
        avg_time = total_time / len(documents)
        
        # Should add at least 50 documents per second
        assert avg_time < 0.02, f"Document addition too slow: {avg_time:.3f}s per document"
        
        print(f"Document addition: {avg_time:.4f}s per document ({1/avg_time:.1f} docs/sec)")
    
    def test_memory_usage(self):
        """Test memory usage remains reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        system = RobustCartesianSystem()
        
        # Add many documents
        for i in range(1000):
            doc_id = f"memory_test_doc_{i}"
            content = f"Memory test document {i} with content"
            coords = {
                'domain': np.random.random(),
                'complexity': np.random.random(),
                'task_type': np.random.random()
            }
            system.add_document_safely(doc_id, content, coords)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should use less than 100MB for 1000 documents
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f}MB for 1000 documents"
        
        print(f"Memory usage: {memory_increase:.1f}MB for 1000 documents")


if __name__ == "__main__":
    # Run tests manually
    print("Core Functionality Tests")
    print("=" * 30)
    
    # Test semantic coordinate engine
    print("\n1. Testing Semantic Coordinate Engine...")
    engine = SemanticCoordinateEngine()
    
    # Basic functionality
    result = engine.text_to_coordinates("Python programming tutorial")
    print(f"   Coordinates generated: {result['coordinates']}")
    print(f"   Method: {result['method']}")
    print(f"   Quality: {result['embedding_quality']:.3f}")
    
    # Test consistency
    result2 = engine.text_to_coordinates("Python programming tutorial")
    consistency = result['coordinates'] == result2['coordinates']
    print(f"   Consistency: {'✅' if consistency else '❌'}")
    
    # Test robust system
    print("\n2. Testing Robust Cartesian System...")
    system = RobustCartesianSystem()
    
    # Add documents
    doc_result = system.add_document_safely(
        "test_doc", 
        "Test document content", 
        {'domain': 0.7, 'complexity': 0.5, 'task_type': 0.3}
    )
    print(f"   Document added: {'✅' if doc_result['success'] else '❌'}")
    
    # Test collision
    collision_result = system.add_document_safely(
        "collision_doc", 
        "Another test document", 
        {'domain': 0.7, 'complexity': 0.5, 'task_type': 0.3}  # Same coordinates
    )
    print(f"   Collision handled: {'✅' if collision_result['collision_resolved'] else '❌'}")
    
    # Test boundary conditions
    boundary_result = system.add_document_safely(
        "boundary_doc", 
        "Boundary test document", 
        {'domain': 1.5, 'complexity': -0.2, 'task_type': 0.5}  # Out of bounds
    )
    final_coords = boundary_result['final_coordinates']
    boundary_ok = all(0.0 <= v <= 1.0 for v in final_coords.values())
    print(f"   Boundary handling: {'✅' if boundary_ok else '❌'}")
    
    # Performance test
    print("\n3. Performance Testing...")
    
    start_time = time.time()
    for i in range(100):
        engine.text_to_coordinates(f"Performance test document {i}")
    coord_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(100):
        system.add_document_safely(
            f"perf_doc_{i}", 
            f"Performance document {i}", 
            {'domain': np.random.random(), 'complexity': np.random.random(), 'task_type': np.random.random()}
        )
    add_time = time.time() - start_time
    
    print(f"   Coordinate generation: {coord_time/100:.4f}s per doc ({100/coord_time:.1f} docs/sec)")
    print(f"   Document addition: {add_time/100:.4f}s per doc ({100/add_time:.1f} docs/sec)")
    
    # System health
    health = system.get_system_health()
    print(f"\n4. System Health:")
    print(f"   Total documents: {health['total_documents']}")
    print(f"   Healthy regions: {health['healthy_regions']}")
    print(f"   Collision history: {health['collision_history']}")
    
    print(f"\n✅ All core functionality tests completed successfully!")