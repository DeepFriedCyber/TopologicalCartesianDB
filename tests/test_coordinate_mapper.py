#!/usr/bin/env python3
"""
Test Suite for Coordinate Mapping Logic

Implements TDD approach for core Cartesian Cube concepts as recommended in feedback.
Tests coordinate mapping, geometric routing, and boundary cases.
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from topological_cartesian.coordinate_quantification import (
    EnhancedCoordinateQuantificationSystem,
    DomainQuantifier,
    ComplexityQuantifier,
    TaskTypeQuantifier
)
from topological_cartesian.enhanced_routing import (
    EnhancedRoutingEngine,
    RoutingConfig,
    DistanceMetric,
    RoutingStrategy
)


class TestCoordinateMapping:
    """Test suite for coordinate mapping logic"""
    
    @pytest.fixture
    def coord_system(self):
        """Fixture for coordinate quantification system"""
        return EnhancedCoordinateQuantificationSystem()
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return {
            'doc_python_tutorial': {
                'content': 'Python programming tutorial for beginners learning basic syntax',
                'expected_domain': 0.9,  # Programming domain
                'expected_complexity': 0.2,  # Beginner level
                'expected_task_type': 0.2   # Tutorial type
            },
            'doc_ml_advanced': {
                'content': 'Advanced machine learning optimization techniques for expert practitioners',
                'expected_domain': 0.9,  # Programming/Science domain
                'expected_complexity': 0.9,  # Advanced level
                'expected_task_type': 0.8   # Analysis type
            },
            'doc_business_strategy': {
                'content': 'Business strategy analysis and market research methodology',
                'expected_domain': 0.3,  # Business domain
                'expected_complexity': 0.5,  # Intermediate level
                'expected_task_type': 0.8   # Analysis type
            }
        }
    
    def test_domain_assignment_programming(self, coord_system):
        """Test domain assignment for programming content"""
        task = "Python code optimization and debugging techniques"
        result = coord_system.text_to_coordinates(task)
        
        assert result['all_valid'], "Coordinate validation should pass"
        assert result['coordinates']['domain'] >= 0.7, "Programming content should map to high domain values"
        assert 'programming' in result['explanations']['domain'].lower()
    
    def test_domain_assignment_business(self, coord_system):
        """Test domain assignment for business content"""
        task = "Business strategy and market analysis for revenue growth"
        result = coord_system.text_to_coordinates(task)
        
        assert result['all_valid'], "Coordinate validation should pass"
        assert result['coordinates']['domain'] <= 0.6, "Business content should map to lower domain values"
        assert 'business' in result['explanations']['domain'].lower()
    
    def test_complexity_assignment_beginner(self, coord_system):
        """Test complexity assignment for beginner content"""
        task = "Basic introduction to programming tutorial for beginners"
        result = coord_system.text_to_coordinates(task)
        
        assert result['coordinates']['complexity'] <= 0.4, "Beginner content should have low complexity"
        assert 'beginner' in result['explanations']['complexity'].lower()
    
    def test_complexity_assignment_advanced(self, coord_system):
        """Test complexity assignment for advanced content"""
        task = "Advanced optimization algorithms and complex architecture patterns"
        result = coord_system.text_to_coordinates(task)
        
        assert result['coordinates']['complexity'] >= 0.7, "Advanced content should have high complexity"
        assert 'advanced' in result['explanations']['complexity'].lower()
    
    def test_task_type_assignment_tutorial(self, coord_system):
        """Test task type assignment for tutorial content"""
        task = "Step by step tutorial guide with examples and walkthrough"
        result = coord_system.text_to_coordinates(task)
        
        assert result['coordinates']['task_type'] <= 0.4, "Tutorial content should have low task_type values"
        assert 'tutorial' in result['explanations']['task_type'].lower()
    
    def test_task_type_assignment_analysis(self, coord_system):
        """Test task type assignment for analysis content"""
        task = "Comparative analysis and evaluation of different approaches"
        result = coord_system.text_to_coordinates(task)
        
        assert result['coordinates']['task_type'] >= 0.6, "Analysis content should have high task_type values"
        assert 'analysis' in result['explanations']['task_type'].lower()
    
    def test_coordinate_consistency(self, coord_system):
        """Test that same input produces consistent coordinates"""
        task = "Python programming tutorial"
        
        result1 = coord_system.text_to_coordinates(task)
        result2 = coord_system.text_to_coordinates(task)
        
        assert result1['coordinates'] == result2['coordinates'], "Same input should produce identical coordinates"
    
    def test_coordinate_validation(self, coord_system):
        """Test coordinate validation against schemas"""
        task = "Test content for validation"
        result = coord_system.text_to_coordinates(task)
        
        # All coordinates should be in [0, 1] range
        for dim, value in result['coordinates'].items():
            assert 0.0 <= value <= 1.0, f"Coordinate {dim} should be in [0, 1] range"
        
        # All validations should pass
        assert all(result['validation_results'].values()), "All coordinate validations should pass"
    
    def test_empty_input_handling(self, coord_system):
        """Test handling of empty or minimal input"""
        result = coord_system.text_to_coordinates("")
        
        # Should return default values without crashing
        assert result['coordinates']['domain'] == 0.5, "Empty input should return default domain"
        assert result['coordinates']['complexity'] == 0.6, "Empty input should return default complexity"
        assert result['coordinates']['task_type'] == 0.5, "Empty input should return default task_type"
    
    def test_coordinate_collision_detection(self, coord_system, sample_documents):
        """Test detection and handling of coordinate collisions"""
        coordinates_list = []
        
        for doc_id, doc_data in sample_documents.items():
            result = coord_system.text_to_coordinates(doc_data['content'])
            coordinates_list.append(result['coordinates'])
        
        # Check for exact collisions (should be rare with good quantification)
        unique_coords = set()
        collisions = []
        
        for i, coords in enumerate(coordinates_list):
            coord_tuple = tuple(sorted(coords.items()))
            if coord_tuple in unique_coords:
                collisions.append(i)
            unique_coords.add(coord_tuple)
        
        # Log collisions for analysis (shouldn't fail test unless excessive)
        if collisions:
            print(f"Coordinate collisions detected at indices: {collisions}")
        
        # Should have mostly unique coordinates
        assert len(unique_coords) >= len(coordinates_list) * 0.8, "Should have mostly unique coordinates"


class TestGeometricRouting:
    """Test suite for geometric routing algorithms"""
    
    @pytest.fixture
    def routing_engine(self):
        """Fixture for routing engine"""
        config = RoutingConfig(
            distance_metric=DistanceMetric.WEIGHTED_EUCLIDEAN,
            routing_strategy=RoutingStrategy.ADAPTIVE,
            dimension_weights={'domain': 1.2, 'complexity': 1.0, 'task_type': 0.8},
            fallback_threshold=0.3
        )
        return EnhancedRoutingEngine(config)
    
    @pytest.fixture
    def sample_documents_with_coords(self):
        """Sample documents with known coordinates"""
        return {
            'doc_python_basic': {
                'content': 'Python basics tutorial',
                'coordinates': {'domain': 0.9, 'complexity': 0.2, 'task_type': 0.2}
            },
            'doc_python_advanced': {
                'content': 'Advanced Python optimization',
                'coordinates': {'domain': 0.9, 'complexity': 0.9, 'task_type': 0.7}
            },
            'doc_business_analysis': {
                'content': 'Business strategy analysis',
                'coordinates': {'domain': 0.3, 'complexity': 0.6, 'task_type': 0.8}
            },
            'doc_creative_design': {
                'content': 'Creative design principles',
                'coordinates': {'domain': 0.1, 'complexity': 0.4, 'task_type': 0.3}
            }
        }
    
    def test_nearest_neighbor_routing(self, routing_engine, sample_documents_with_coords):
        """Test nearest neighbor routing functionality"""
        routing_engine.add_documents(sample_documents_with_coords)
        
        # Query similar to Python basic tutorial
        query_coords = {'domain': 0.85, 'complexity': 0.25, 'task_type': 0.25}
        results = routing_engine.route_query(query_coords, max_results=3)
        
        assert len(results) > 0, "Should return routing results"
        assert results[0].document_id == 'doc_python_basic', "Should route to most similar document"
        assert results[0].similarity_score > 0.8, "Should have high similarity for close match"
    
    def test_routing_with_weights(self, routing_engine, sample_documents_with_coords):
        """Test that dimension weights affect routing decisions"""
        routing_engine.add_documents(sample_documents_with_coords)
        
        # Query that should prioritize domain due to higher weight
        query_coords = {'domain': 0.9, 'complexity': 0.5, 'task_type': 0.5}
        results = routing_engine.route_query(query_coords, max_results=2)
        
        # Both Python documents should rank higher than business document
        python_docs = [r for r in results if 'python' in r.document_id.lower()]
        assert len(python_docs) >= 1, "Should prioritize documents with matching domain"
    
    def test_fallback_mechanism(self, routing_engine, sample_documents_with_coords):
        """Test fallback mechanism for poor matches"""
        routing_engine.add_documents(sample_documents_with_coords)
        
        # Query very different from all documents
        query_coords = {'domain': 0.5, 'complexity': 1.0, 'task_type': 1.0}
        results = routing_engine.route_query(query_coords, max_results=3)
        
        # Should still return results but with fallback indicators
        assert len(results) > 0, "Should return results even for poor matches"
        
        # At least some results should indicate fallback was used
        fallback_used = any(r.fallback_used for r in results)
        low_similarity = any(r.similarity_score < 0.3 for r in results)
        
        assert fallback_used or low_similarity, "Should indicate when matches are poor"
    
    def test_empty_cube_region_handling(self, routing_engine):
        """Test handling of queries to empty cube regions"""
        # Add only one document in a specific region
        single_doc = {
            'doc_isolated': {
                'content': 'Isolated document',
                'coordinates': {'domain': 0.1, 'complexity': 0.1, 'task_type': 0.1}
            }
        }
        routing_engine.add_documents(single_doc)
        
        # Query in a completely different region
        query_coords = {'domain': 0.9, 'complexity': 0.9, 'task_type': 0.9}
        results = routing_engine.route_query(query_coords, max_results=3)
        
        # Should still return the isolated document with appropriate confidence
        assert len(results) == 1, "Should return available documents even from distant regions"
        assert results[0].confidence < 0.5, "Should have low confidence for distant matches"
    
    def test_boundary_condition_handling(self, routing_engine):
        """Test handling of boundary conditions in coordinate space"""
        # Documents at coordinate space boundaries
        boundary_docs = {
            'doc_min_boundary': {
                'content': 'Minimum boundary document',
                'coordinates': {'domain': 0.0, 'complexity': 0.0, 'task_type': 0.0}
            },
            'doc_max_boundary': {
                'content': 'Maximum boundary document',
                'coordinates': {'domain': 1.0, 'complexity': 1.0, 'task_type': 1.0}
            },
            'doc_mixed_boundary': {
                'content': 'Mixed boundary document',
                'coordinates': {'domain': 0.0, 'complexity': 1.0, 'task_type': 0.5}
            }
        }
        routing_engine.add_documents(boundary_docs)
        
        # Test queries at boundaries
        boundary_queries = [
            {'domain': 0.0, 'complexity': 0.0, 'task_type': 0.0},
            {'domain': 1.0, 'complexity': 1.0, 'task_type': 1.0},
            {'domain': 0.5, 'complexity': 0.5, 'task_type': 0.5}
        ]
        
        for query_coords in boundary_queries:
            results = routing_engine.route_query(query_coords, max_results=3)
            assert len(results) > 0, f"Should handle boundary query {query_coords}"
            assert all(0.0 <= r.similarity_score <= 1.0 for r in results), "Similarity scores should be valid"
    
    def test_routing_performance_stress(self, routing_engine):
        """Stress test routing performance with many documents"""
        # Generate many documents with random coordinates
        import random
        random.seed(42)  # For reproducible tests
        
        large_doc_set = {}
        for i in range(1000):  # Smaller number for CI/CD compatibility
            large_doc_set[f'doc_{i}'] = {
                'content': f'Document {i} content',
                'coordinates': {
                    'domain': random.random(),
                    'complexity': random.random(),
                    'task_type': random.random()
                }
            }
        
        routing_engine.add_documents(large_doc_set)
        
        # Test multiple queries
        query_coords = {'domain': 0.5, 'complexity': 0.5, 'task_type': 0.5}
        
        import time
        start_time = time.time()
        results = routing_engine.route_query(query_coords, max_results=10)
        routing_time = time.time() - start_time
        
        assert len(results) == 10, "Should return requested number of results"
        assert routing_time < 1.0, "Routing should complete within reasonable time"
        
        # Results should be properly sorted by similarity
        similarities = [r.similarity_score for r in results]
        assert similarities == sorted(similarities, reverse=True), "Results should be sorted by similarity"


class TestCubeRegionBoundaries:
    """Test suite for cube region boundary cases"""
    
    def test_coordinate_space_coverage(self):
        """Test that coordinate quantification covers the full space"""
        coord_system = EnhancedCoordinateQuantificationSystem()
        
        # Test various content types to ensure full space coverage
        test_cases = [
            "creative writing and artistic expression",  # Should map to low domain
            "advanced quantum computing algorithms",      # Should map to high domain, high complexity
            "basic introduction to business concepts",   # Should map to medium domain, low complexity
            "troubleshooting network connectivity issues" # Should map to high domain, medium complexity
        ]
        
        coordinates_list = []
        for content in test_cases:
            result = coord_system.text_to_coordinates(content)
            coordinates_list.append(result['coordinates'])
        
        # Check that we're using different regions of the coordinate space
        domains = [c['domain'] for c in coordinates_list]
        complexities = [c['complexity'] for c in coordinates_list]
        task_types = [c['task_type'] for c in coordinates_list]
        
        # Should have reasonable spread across dimensions
        assert max(domains) - min(domains) > 0.3, "Should use diverse domain values"
        assert max(complexities) - min(complexities) > 0.3, "Should use diverse complexity values"
        assert max(task_types) - min(task_types) > 0.2, "Should use diverse task_type values"
    
    def test_cube_region_transitions(self):
        """Test smooth transitions between cube regions"""
        coord_system = EnhancedCoordinateQuantificationSystem()
        
        # Test content that should create smooth transitions
        transition_content = [
            "basic programming concepts",
            "intermediate programming techniques", 
            "advanced programming optimization",
            "expert-level programming architecture"
        ]
        
        results = []
        for content in transition_content:
            result = coord_system.text_to_coordinates(content)
            results.append(result['coordinates'])
        
        # Complexity should increase smoothly
        complexities = [r['complexity'] for r in results]
        
        # Check for monotonic increase (allowing for some noise)
        for i in range(len(complexities) - 1):
            # Allow for small decreases due to quantification noise
            assert complexities[i+1] >= complexities[i] - 0.1, f"Complexity should generally increase: {complexities}"
    
    def test_edge_case_content(self):
        """Test edge cases in content processing"""
        coord_system = EnhancedCoordinateQuantificationSystem()
        
        edge_cases = [
            "",  # Empty content
            "a",  # Single character
            "the and or but",  # Only stop words
            "🚀 💻 🎨",  # Only emojis
            "123 456 789",  # Only numbers
            "UPPERCASE CONTENT ONLY",  # All caps
            "mixed CASE content With Numbers123 and symbols!@#"  # Mixed content
        ]
        
        for content in edge_cases:
            result = coord_system.text_to_coordinates(content)
            
            # Should not crash and should return valid coordinates
            assert result['all_valid'], f"Should handle edge case: '{content}'"
            
            coords = result['coordinates']
            for dim, value in coords.items():
                assert 0.0 <= value <= 1.0, f"Coordinate {dim} should be valid for content: '{content}'"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])