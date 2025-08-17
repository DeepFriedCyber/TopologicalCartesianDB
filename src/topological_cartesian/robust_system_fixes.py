#!/usr/bin/env python3
"""
Robust System Fixes

Addresses critical bugs and edge cases identified in the feedback:
1. Coordinate collisions and resolution
2. Empty cube region handling
3. Boundary condition management
4. Race condition fixes
5. Performance optimizations
"""

import numpy as np
import threading
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CoordinateCollision:
    """Represents a coordinate collision between multiple documents"""
    coordinate_key: str
    colliding_documents: List[str]
    collision_timestamp: float
    resolution_strategy: str
    resolved: bool = False
    resolution_details: Optional[Dict[str, Any]] = None


@dataclass
class CubeRegionHealth:
    """Health status of a cube region"""
    region_id: str
    coordinate_bounds: Dict[str, Tuple[float, float]]
    document_count: int
    density: float
    last_access_time: float
    health_score: float
    issues: List[str] = field(default_factory=list)


class CoordinateCollisionResolver:
    """Handles coordinate collisions with multiple resolution strategies"""
    
    def __init__(self):
        self.collision_history = {}
        self.resolution_strategies = {
            'jitter': self._jitter_resolution,
            'hierarchical': self._hierarchical_resolution,
            'semantic_separation': self._semantic_separation,
            'temporal_offset': self._temporal_offset
        }
        self.collision_lock = threading.Lock()
    
    def detect_collision(self, new_doc_id: str, coordinates: Dict[str, float], 
                        existing_documents: Dict[str, Dict[str, Any]]) -> Optional[CoordinateCollision]:
        """Detect if new coordinates collide with existing documents"""
        
        coordinate_key = self._generate_coordinate_key(coordinates)
        colliding_docs = []
        
        for doc_id, doc_data in existing_documents.items():
            if doc_id == new_doc_id:
                continue
            
            existing_coords = doc_data.get('coordinates', {})
            existing_key = self._generate_coordinate_key(existing_coords)
            
            if self._coordinates_collide(coordinates, existing_coords):
                colliding_docs.append(doc_id)
        
        if colliding_docs:
            collision = CoordinateCollision(
                coordinate_key=coordinate_key,
                colliding_documents=[new_doc_id] + colliding_docs,
                collision_timestamp=time.time(),
                resolution_strategy='auto'
            )
            return collision
        
        return None
    
    def resolve_collision(self, collision: CoordinateCollision, 
                         documents: Dict[str, Dict[str, Any]], 
                         strategy: str = 'auto') -> Dict[str, Dict[str, float]]:
        """Resolve coordinate collision using specified strategy"""
        
        with self.collision_lock:
            if strategy == 'auto':
                strategy = self._select_best_strategy(collision, documents)
            
            if strategy not in self.resolution_strategies:
                logger.error(f"Unknown resolution strategy: {strategy}")
                strategy = 'jitter'
            
            resolution_func = self.resolution_strategies[strategy]
            resolved_coordinates = resolution_func(collision, documents)
            
            # Update collision record
            collision.resolved = True
            collision.resolution_strategy = strategy
            collision.resolution_details = {
                'resolved_at': time.time(),
                'new_coordinates': resolved_coordinates
            }
            
            self.collision_history[collision.coordinate_key] = collision
            
            logger.info(f"Resolved collision for {len(collision.colliding_documents)} documents using {strategy}")
            
            return resolved_coordinates
    
    def _generate_coordinate_key(self, coordinates: Dict[str, float], precision: int = 3) -> str:
        """Generate a key for coordinate comparison"""
        rounded_coords = {k: round(v, precision) for k, v in coordinates.items()}
        coord_str = json.dumps(rounded_coords, sort_keys=True)
        return hashlib.md5(coord_str.encode()).hexdigest()[:8]
    
    def _coordinates_collide(self, coords1: Dict[str, float], coords2: Dict[str, float], 
                           threshold: float = 0.05) -> bool:
        """Check if two coordinate sets collide within threshold"""
        common_dims = set(coords1.keys()) & set(coords2.keys())
        if not common_dims:
            return False
        
        for dim in common_dims:
            if abs(coords1[dim] - coords2[dim]) > threshold:
                return False
        
        return True
    
    def _select_best_strategy(self, collision: CoordinateCollision, 
                            documents: Dict[str, Dict[str, Any]]) -> str:
        """Select the best resolution strategy based on collision characteristics"""
        
        num_colliding = len(collision.colliding_documents)
        
        # Get document content lengths to assess semantic similarity
        content_lengths = []
        for doc_id in collision.colliding_documents:
            if doc_id in documents:
                content = documents[doc_id].get('content', '')
                content_lengths.append(len(content))
        
        avg_content_length = np.mean(content_lengths) if content_lengths else 0
        
        # Decision logic
        if num_colliding <= 2 and avg_content_length < 100:
            return 'jitter'  # Simple case
        elif num_colliding > 5:
            return 'hierarchical'  # Complex case needs structure
        elif avg_content_length > 500:
            return 'semantic_separation'  # Rich content needs semantic analysis
        else:
            return 'temporal_offset'  # Default case
    
    def _jitter_resolution(self, collision: CoordinateCollision, 
                          documents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Resolve collision by adding small random offsets"""
        
        resolved_coords = {}
        jitter_magnitude = 0.15  # Increased from 0.05 to 0.15 for better separation
        
        for i, doc_id in enumerate(collision.colliding_documents):
            if doc_id not in documents:
                continue
            
            original_coords = documents[doc_id].get('coordinates', {})
            new_coords = {}
            
            for dim, value in original_coords.items():
                # Add random offset with increasing magnitude for each document
                document_index = collision.colliding_documents.index(doc_id)
                offset_magnitude = jitter_magnitude * (1 + document_index * 0.5)
                offset = np.random.uniform(-offset_magnitude, offset_magnitude)
                new_value = max(0.0, min(1.0, value + offset))
                new_coords[dim] = round(new_value, 3)
            
            resolved_coords[doc_id] = new_coords
        
        return resolved_coords
    
    def _hierarchical_resolution(self, collision: CoordinateCollision, 
                               documents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Resolve collision by creating hierarchical coordinate structure"""
        
        resolved_coords = {}
        base_coords = None
        
        # Find base coordinates from first document
        first_doc = collision.colliding_documents[0]
        if first_doc in documents:
            base_coords = documents[first_doc].get('coordinates', {})
        
        if not base_coords:
            return self._jitter_resolution(collision, documents)
        
        # Create hierarchical offsets
        hierarchy_step = 0.02
        
        for i, doc_id in enumerate(collision.colliding_documents):
            if doc_id not in documents:
                continue
            
            new_coords = {}
            for dim, base_value in base_coords.items():
                # Create hierarchical offset based on position
                hierarchy_offset = (i * hierarchy_step) % 0.1  # Wrap around
                new_value = max(0.0, min(1.0, base_value + hierarchy_offset))
                new_coords[dim] = round(new_value, 3)
            
            resolved_coords[doc_id] = new_coords
        
        return resolved_coords
    
    def _semantic_separation(self, collision: CoordinateCollision, 
                           documents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Resolve collision using semantic analysis to separate documents"""
        
        # This would use the semantic coordinate engine in a real implementation
        # For now, use content-based heuristics
        
        resolved_coords = {}
        
        for doc_id in collision.colliding_documents:
            if doc_id not in documents:
                continue
            
            doc_content = documents[doc_id].get('content', '')
            original_coords = documents[doc_id].get('coordinates', {})
            
            # Analyze content characteristics
            word_count = len(doc_content.split())
            unique_words = len(set(doc_content.lower().split()))
            
            # Adjust coordinates based on content analysis
            new_coords = {}
            for dim, original_value in original_coords.items():
                if dim == 'complexity':
                    # Adjust complexity based on vocabulary diversity
                    complexity_adjustment = (unique_words / max(1, word_count)) * 0.1
                    new_value = max(0.0, min(1.0, original_value + complexity_adjustment))
                elif dim == 'domain':
                    # Keep domain relatively stable
                    new_value = original_value
                else:
                    # Small adjustment for other dimensions
                    adjustment = hash(doc_content) % 100 / 1000.0  # Deterministic but varied
                    new_value = max(0.0, min(1.0, original_value + adjustment))
                
                new_coords[dim] = round(new_value, 3)
            
            resolved_coords[doc_id] = new_coords
        
        return resolved_coords
    
    def _temporal_offset(self, collision: CoordinateCollision, 
                        documents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Resolve collision using temporal information"""
        
        resolved_coords = {}
        
        # Sort documents by some temporal factor (using doc_id as proxy)
        sorted_docs = sorted(collision.colliding_documents)
        
        for i, doc_id in enumerate(sorted_docs):
            if doc_id not in documents:
                continue
            
            original_coords = documents[doc_id].get('coordinates', {})
            new_coords = {}
            
            # Apply temporal offset
            temporal_offset = i * 0.01  # Small incremental offset
            
            for dim, value in original_coords.items():
                new_value = max(0.0, min(1.0, value + temporal_offset))
                new_coords[dim] = round(new_value, 3)
            
            resolved_coords[doc_id] = new_coords
        
        return resolved_coords


class CubeRegionHealthMonitor:
    """Monitors health of cube regions and handles empty regions"""
    
    def __init__(self):
        self.region_health = {}
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = 0
        self.health_lock = threading.Lock()
    
    def check_region_health(self, documents: Dict[str, Dict[str, Any]], 
                          coordinate_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, CubeRegionHealth]:
        """Check health of all cube regions"""
        
        with self.health_lock:
            current_time = time.time()
            
            if current_time - self.last_health_check < self.health_check_interval:
                return self.region_health
            
            # Analyze regions
            region_analysis = self._analyze_regions(documents, coordinate_bounds)
            
            # Update health records
            for region_id, analysis in region_analysis.items():
                health = CubeRegionHealth(
                    region_id=region_id,
                    coordinate_bounds=analysis['bounds'],
                    document_count=analysis['doc_count'],
                    density=analysis['density'],
                    last_access_time=analysis['last_access'],
                    health_score=analysis['health_score'],
                    issues=analysis['issues']
                )
                
                self.region_health[region_id] = health
            
            self.last_health_check = current_time
            
            return self.region_health
    
    def _analyze_regions(self, documents: Dict[str, Dict[str, Any]], 
                        coordinate_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, Any]]:
        """Analyze cube regions for health metrics"""
        
        # Create grid of regions
        grid_size = 10  # 10x10x10 grid
        regions = {}
        
        # Initialize regions
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    region_id = f"region_{i}_{j}_{k}"
                    regions[region_id] = {
                        'bounds': {
                            'domain': (i/grid_size, (i+1)/grid_size),
                            'complexity': (j/grid_size, (j+1)/grid_size),
                            'task_type': (k/grid_size, (k+1)/grid_size)
                        },
                        'documents': [],
                        'doc_count': 0,
                        'density': 0.0,
                        'last_access': 0.0,
                        'health_score': 0.0,
                        'issues': []
                    }
        
        # Assign documents to regions
        for doc_id, doc_data in documents.items():
            coords = doc_data.get('coordinates', {})
            region_id = self._get_region_for_coordinates(coords, grid_size)
            
            if region_id in regions:
                regions[region_id]['documents'].append(doc_id)
                regions[region_id]['doc_count'] += 1
        
        # Calculate health metrics
        for region_id, region_data in regions.items():
            doc_count = region_data['doc_count']
            
            # Calculate density
            region_volume = 1.0 / (grid_size ** 3)  # Volume of each region
            density = doc_count / region_volume if region_volume > 0 else 0
            
            # Calculate health score
            health_score = self._calculate_region_health_score(region_data)
            
            # Identify issues
            issues = self._identify_region_issues(region_data)
            
            regions[region_id].update({
                'density': density,
                'health_score': health_score,
                'issues': issues,
                'last_access': time.time()  # Simplified
            })
        
        return regions
    
    def _get_region_for_coordinates(self, coordinates: Dict[str, float], grid_size: int) -> str:
        """Get region ID for given coordinates"""
        domain = coordinates.get('domain', 0.5)
        complexity = coordinates.get('complexity', 0.5)
        task_type = coordinates.get('task_type', 0.5)
        
        i = min(int(domain * grid_size), grid_size - 1)
        j = min(int(complexity * grid_size), grid_size - 1)
        k = min(int(task_type * grid_size), grid_size - 1)
        
        return f"region_{i}_{j}_{k}"
    
    def _calculate_region_health_score(self, region_data: Dict[str, Any]) -> float:
        """Calculate health score for a region"""
        doc_count = region_data['doc_count']
        density = region_data['density']
        
        # Health factors
        population_score = min(1.0, doc_count / 10.0)  # Ideal: 10 docs per region
        density_score = min(1.0, density / 100.0)      # Reasonable density
        
        # Penalize empty regions
        if doc_count == 0:
            return 0.0
        
        # Penalize overcrowded regions
        if doc_count > 100:
            overcrowding_penalty = 0.5
        else:
            overcrowding_penalty = 0.0
        
        health_score = (population_score * 0.6 + density_score * 0.4) - overcrowding_penalty
        
        return max(0.0, min(1.0, health_score))
    
    def _identify_region_issues(self, region_data: Dict[str, Any]) -> List[str]:
        """Identify issues with a region"""
        issues = []
        
        doc_count = region_data['doc_count']
        density = region_data['density']
        
        if doc_count == 0:
            issues.append("empty_region")
        elif doc_count == 1:
            issues.append("isolated_document")
        elif doc_count > 100:
            issues.append("overcrowded")
        
        if density > 1000:
            issues.append("high_density")
        elif density < 1 and doc_count > 0:
            issues.append("low_density")
        
        return issues
    
    def get_empty_regions(self) -> List[str]:
        """Get list of empty regions"""
        return [region_id for region_id, health in self.region_health.items() 
                if health.document_count == 0]
    
    def get_problematic_regions(self) -> List[str]:
        """Get list of regions with health issues"""
        return [region_id for region_id, health in self.region_health.items() 
                if health.health_score < 0.5 or health.issues]


class ThreadSafeMultiCubeOrchestrator:
    """Thread-safe version of multi-cube orchestrator to fix race conditions"""
    
    def __init__(self):
        self.cubes = {}
        self.cube_locks = {}
        self.global_lock = threading.RLock()  # Reentrant lock
        self.operation_queue = deque()
        self.queue_lock = threading.Lock()
        self.active_operations = {}
        
    def add_cube(self, cube_name: str, cube_instance: Any):
        """Thread-safe cube addition"""
        with self.global_lock:
            self.cubes[cube_name] = cube_instance
            self.cube_locks[cube_name] = threading.RLock()
            logger.info(f"Added cube: {cube_name}")
    
    def execute_operation(self, operation_id: str, cube_name: str, 
                         operation_func: callable, *args, **kwargs) -> Any:
        """Execute operation with proper locking"""
        
        if cube_name not in self.cubes:
            raise ValueError(f"Cube {cube_name} not found")
        
        # Record operation start
        with self.queue_lock:
            self.active_operations[operation_id] = {
                'cube_name': cube_name,
                'start_time': time.time(),
                'thread_id': threading.current_thread().ident
            }
        
        try:
            # Acquire cube-specific lock
            with self.cube_locks[cube_name]:
                cube_instance = self.cubes[cube_name]
                result = operation_func(cube_instance, *args, **kwargs)
                
                return result
                
        finally:
            # Clean up operation record
            with self.queue_lock:
                if operation_id in self.active_operations:
                    del self.active_operations[operation_id]
    
    def parallel_operation(self, operations: List[Tuple[str, str, callable]], 
                          max_workers: int = 4) -> Dict[str, Any]:
        """Execute multiple operations in parallel safely"""
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all operations
            future_to_op = {}
            
            for op_id, cube_name, operation_func in operations:
                future = executor.submit(
                    self.execute_operation, 
                    op_id, cube_name, operation_func
                )
                future_to_op[future] = op_id
            
            # Collect results
            for future in as_completed(future_to_op):
                op_id = future_to_op[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results[op_id] = {'success': True, 'result': result}
                except Exception as e:
                    logger.error(f"Operation {op_id} failed: {e}")
                    results[op_id] = {'success': False, 'error': str(e)}
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get thread-safe system status"""
        with self.global_lock:
            with self.queue_lock:
                return {
                    'total_cubes': len(self.cubes),
                    'cube_names': list(self.cubes.keys()),
                    'active_operations': len(self.active_operations),
                    'operation_details': dict(self.active_operations)
                }


class BoundaryConditionHandler:
    """Handles boundary conditions in coordinate space"""
    
    def __init__(self):
        self.boundary_tolerance = 0.01
        self.boundary_handlers = {
            'clamp': self._clamp_to_boundaries,
            'wrap': self._wrap_around_boundaries,
            'reflect': self._reflect_at_boundaries,
            'extend': self._extend_coordinate_space
        }
    
    def handle_boundary_condition(self, coordinates: Dict[str, float], 
                                 method: str = 'clamp') -> Dict[str, float]:
        """Handle coordinates that are at or beyond boundaries"""
        
        if method not in self.boundary_handlers:
            logger.warning(f"Unknown boundary method: {method}, using 'clamp'")
            method = 'clamp'
        
        handler = self.boundary_handlers[method]
        return handler(coordinates)
    
    def _clamp_to_boundaries(self, coordinates: Dict[str, float]) -> Dict[str, float]:
        """Clamp coordinates to [0, 1] range"""
        return {dim: max(0.0, min(1.0, value)) for dim, value in coordinates.items()}
    
    def _wrap_around_boundaries(self, coordinates: Dict[str, float]) -> Dict[str, float]:
        """Wrap coordinates around boundaries"""
        wrapped = {}
        for dim, value in coordinates.items():
            if value < 0:
                wrapped[dim] = 1.0 + (value % 1.0)
            elif value > 1:
                wrapped[dim] = value % 1.0
            else:
                wrapped[dim] = value
        return wrapped
    
    def _reflect_at_boundaries(self, coordinates: Dict[str, float]) -> Dict[str, float]:
        """Reflect coordinates at boundaries"""
        reflected = {}
        for dim, value in coordinates.items():
            if value < 0:
                reflected[dim] = abs(value)
            elif value > 1:
                reflected[dim] = 2.0 - value
            else:
                reflected[dim] = value
            
            # Ensure still in bounds after reflection
            reflected[dim] = max(0.0, min(1.0, reflected[dim]))
        
        return reflected
    
    def _extend_coordinate_space(self, coordinates: Dict[str, float]) -> Dict[str, float]:
        """Extend coordinate space to accommodate out-of-bounds values"""
        # This would require updating the entire coordinate system
        # For now, just clamp
        logger.info("Coordinate space extension requested - using clamp for now")
        return self._clamp_to_boundaries(coordinates)
    
    def detect_boundary_issues(self, coordinates: Dict[str, float]) -> List[str]:
        """Detect potential boundary-related issues"""
        issues = []
        
        for dim, value in coordinates.items():
            if value < self.boundary_tolerance:
                issues.append(f"{dim}_at_lower_boundary")
            elif value > (1.0 - self.boundary_tolerance):
                issues.append(f"{dim}_at_upper_boundary")
            elif value < 0 or value > 1:
                issues.append(f"{dim}_out_of_bounds")
        
        return issues


# Integration class that brings all fixes together
class RobustCartesianSystem:
    """
    Robust Cartesian system that integrates all the fixes for critical issues.
    """
    
    def __init__(self):
        self.collision_resolver = CoordinateCollisionResolver()
        self.health_monitor = CubeRegionHealthMonitor()
        self.orchestrator = ThreadSafeMultiCubeOrchestrator()
        self.boundary_handler = BoundaryConditionHandler()
        
        self.documents = {}
        self.system_lock = threading.RLock()
        
        logger.info("RobustCartesianSystem initialized with all fixes")
    
    def add_document_safely(self, doc_id: str, content: str, 
                           coordinates: Dict[str, float]) -> Dict[str, Any]:
        """Add document with collision detection and resolution"""
        
        with self.system_lock:
            # Handle boundary conditions
            safe_coordinates = self.boundary_handler.handle_boundary_condition(coordinates)
            
            # Detect collisions
            collision = self.collision_resolver.detect_collision(
                doc_id, safe_coordinates, self.documents
            )
            
            if collision:
                logger.info(f"Collision detected for document {doc_id}")
                
                # Add the new document temporarily for collision resolution
                temp_doc = {
                    'content': content,
                    'coordinates': safe_coordinates,
                    'added_at': time.time()
                }
                temp_documents = self.documents.copy()
                temp_documents[doc_id] = temp_doc
                
                resolved_coords = self.collision_resolver.resolve_collision(
                    collision, temp_documents
                )
                
                # Update all affected documents
                for affected_doc_id, new_coords in resolved_coords.items():
                    if affected_doc_id in self.documents:
                        self.documents[affected_doc_id]['coordinates'] = new_coords
                    elif affected_doc_id == doc_id:
                        safe_coordinates = new_coords
            
            # Add document
            self.documents[doc_id] = {
                'content': content,
                'coordinates': safe_coordinates,
                'added_at': time.time()
            }
            
            # Check region health
            coordinate_bounds = {'domain': (0, 1), 'complexity': (0, 1), 'task_type': (0, 1)}
            health_status = self.health_monitor.check_region_health(
                self.documents, coordinate_bounds
            )
            
            return {
                'success': True,
                'document_id': doc_id,
                'final_coordinates': safe_coordinates,
                'collision_resolved': collision is not None,
                'region_health': len([h for h in health_status.values() if h.health_score > 0.5])
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        
        with self.system_lock:
            coordinate_bounds = {'domain': (0, 1), 'complexity': (0, 1), 'task_type': (0, 1)}
            region_health = self.health_monitor.check_region_health(
                self.documents, coordinate_bounds
            )
            
            empty_regions = self.health_monitor.get_empty_regions()
            problematic_regions = self.health_monitor.get_problematic_regions()
            
            return {
                'total_documents': len(self.documents),
                'total_regions': len(region_health),
                'healthy_regions': len([h for h in region_health.values() if h.health_score > 0.7]),
                'empty_regions': len(empty_regions),
                'problematic_regions': len(problematic_regions),
                'collision_history': len(self.collision_resolver.collision_history),
                'orchestrator_status': self.orchestrator.get_system_status(),
                'boundary_issues': sum(
                    len(self.boundary_handler.detect_boundary_issues(doc['coordinates']))
                    for doc in self.documents.values()
                )
            }


# Example usage and testing
if __name__ == "__main__":
    # Test the robust system
    robust_system = RobustCartesianSystem()
    
    print("Robust Cartesian System Demo")
    print("=" * 40)
    
    # Test collision detection and resolution
    test_documents = [
        ("doc1", "Python programming tutorial", {'domain': 0.9, 'complexity': 0.3, 'task_type': 0.2}),
        ("doc2", "Python coding guide", {'domain': 0.9, 'complexity': 0.3, 'task_type': 0.2}),  # Collision!
        ("doc3", "Advanced ML algorithms", {'domain': 0.9, 'complexity': 0.9, 'task_type': 0.8}),
        ("doc4", "Business strategy", {'domain': 0.3, 'complexity': 0.6, 'task_type': 0.7}),
    ]
    
    for doc_id, content, coords in test_documents:
        result = robust_system.add_document_safely(doc_id, content, coords)
        print(f"\nAdded {doc_id}:")
        print(f"  Success: {result['success']}")
        print(f"  Collision resolved: {result['collision_resolved']}")
        print(f"  Final coordinates: {result['final_coordinates']}")
    
    # Test boundary conditions
    boundary_test_coords = {'domain': 1.5, 'complexity': -0.2, 'task_type': 0.5}
    safe_coords = robust_system.boundary_handler.handle_boundary_condition(boundary_test_coords)
    print(f"\nBoundary test:")
    print(f"  Original: {boundary_test_coords}")
    print(f"  Safe: {safe_coords}")
    
    # Get system health
    health = robust_system.get_system_health()
    print(f"\nSystem Health:")
    for key, value in health.items():
        print(f"  {key}: {value}")