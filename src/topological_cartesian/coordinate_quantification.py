#!/usr/bin/env python3
"""
Enhanced Coordinate Quantification System

Addresses the critical feedback about coordinate system design and quantification.
Implements formal mapping schemas, normalization, and validation frameworks.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class DimensionType(Enum):
    """Types of coordinate dimensions"""
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    HYBRID = "hybrid"


@dataclass
class DimensionSchema:
    """Formal schema for a coordinate dimension"""
    name: str
    dimension_type: DimensionType
    range_min: float
    range_max: float
    categories: Optional[List[str]] = None
    normalization_method: str = "min_max"
    validation_rules: List[str] = field(default_factory=list)
    weight: float = 1.0
    description: str = ""
    
    def validate_value(self, value: float) -> bool:
        """Validate a coordinate value against this dimension schema"""
        if not (self.range_min <= value <= self.range_max):
            return False
        
        for rule in self.validation_rules:
            if not self._apply_validation_rule(rule, value):
                return False
        
        return True
    
    def _apply_validation_rule(self, rule: str, value: float) -> bool:
        """Apply a validation rule to a value"""
        # Simple rule engine - can be extended
        if rule.startswith("not_zero"):
            return value != 0.0
        elif rule.startswith("positive"):
            return value > 0.0
        elif rule.startswith("discrete"):
            # Check if value is close to discrete levels
            levels = [0.0, 0.25, 0.5, 0.75, 1.0]
            return any(abs(value - level) < 0.05 for level in levels)
        
        return True


class CoordinateQuantifier(ABC):
    """Abstract base class for coordinate quantification strategies"""
    
    @abstractmethod
    def quantify(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Quantify a text along this dimension"""
        pass
    
    @abstractmethod
    def get_explanation(self, text: str, value: float) -> str:
        """Explain how the quantification was derived"""
        pass


class KeywordBasedQuantifier(CoordinateQuantifier):
    """Quantifies coordinates based on keyword analysis with TF-IDF weighting"""
    
    def __init__(self, dimension_name: str, keyword_categories: Dict[str, List[str]], 
                 weights: Optional[Dict[str, float]] = None):
        self.dimension_name = dimension_name
        self.keyword_categories = keyword_categories
        self.weights = weights or {cat: 1.0 for cat in keyword_categories.keys()}
        self.category_scores = {}
    
    def quantify(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Quantify using weighted keyword analysis"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words)
        
        if word_count == 0:
            return 0.5  # Default neutral value
        
        category_scores = {}
        
        for category, keywords in self.keyword_categories.items():
            # Calculate TF-IDF style scoring
            matches = sum(1 for word in words if word in keywords)
            tf = matches / word_count if word_count > 0 else 0
            
            # Simple IDF approximation (can be enhanced with corpus statistics)
            idf = math.log(len(self.keyword_categories) / (1 + len(keywords)))
            
            category_scores[category] = tf * idf * self.weights.get(category, 1.0)
        
        self.category_scores = category_scores
        
        # Convert category scores to coordinate value
        return self._scores_to_coordinate(category_scores)
    
    def _scores_to_coordinate(self, scores: Dict[str, float]) -> float:
        """Convert category scores to a coordinate value"""
        if not scores:
            return 0.5
        
        # Weighted average approach
        total_weight = sum(scores.values())
        if total_weight == 0:
            return 0.5
        
        # Map categories to coordinate positions
        category_positions = self._get_category_positions()
        
        weighted_position = sum(
            scores[cat] * category_positions.get(cat, 0.5) 
            for cat in scores.keys()
        ) / total_weight
        
        return max(0.0, min(1.0, weighted_position))
    
    def _get_category_positions(self) -> Dict[str, float]:
        """Get coordinate positions for categories (override in subclasses)"""
        categories = list(self.keyword_categories.keys())
        if len(categories) == 1:
            return {categories[0]: 0.5}
        
        # Distribute categories evenly across coordinate space
        positions = {}
        for i, category in enumerate(categories):
            positions[category] = i / (len(categories) - 1)
        
        return positions
    
    def get_explanation(self, text: str, value: float) -> str:
        """Explain the quantification"""
        if not self.category_scores:
            return f"{self.dimension_name}: Default value (no keywords matched)"
        
        explanations = []
        for category, score in self.category_scores.items():
            if score > 0:
                explanations.append(f"{category}: {score:.3f}")
        
        return f"{self.dimension_name} ({value:.3f}): " + ", ".join(explanations)


class DomainQuantifier(KeywordBasedQuantifier):
    """Specialized quantifier for domain dimension"""
    
    def __init__(self):
        keyword_categories = {
            'programming': [
                'python', 'javascript', 'java', 'code', 'programming', 'algorithm', 
                'function', 'class', 'method', 'variable', 'api', 'framework',
                'library', 'debugging', 'testing', 'deployment', 'git', 'database'
            ],
            'business': [
                'business', 'strategy', 'market', 'marketing', 'sales', 'revenue',
                'customer', 'client', 'profit', 'investment', 'roi', 'kpi',
                'management', 'leadership', 'team', 'project', 'budget'
            ],
            'science': [
                'research', 'study', 'analysis', 'data', 'statistics', 'experiment',
                'hypothesis', 'theory', 'model', 'simulation', 'measurement',
                'observation', 'methodology', 'results', 'conclusion'
            ],
            'creative': [
                'design', 'art', 'creative', 'visual', 'aesthetic', 'style',
                'color', 'layout', 'typography', 'brand', 'content', 'writing',
                'storytelling', 'narrative', 'imagery', 'inspiration'
            ]
        }
        
        weights = {
            'programming': 1.2,  # Slightly higher weight for technical content
            'business': 1.0,
            'science': 1.1,
            'creative': 0.9
        }
        
        super().__init__('domain', keyword_categories, weights)
    
    def _get_category_positions(self) -> Dict[str, float]:
        """Map domain categories to coordinate positions"""
        return {
            'programming': 0.9,   # High technical domain
            'science': 0.7,       # Scientific domain
            'business': 0.3,      # Business domain
            'creative': 0.1       # Creative domain
        }


class ComplexityQuantifier(KeywordBasedQuantifier):
    """Specialized quantifier for complexity dimension"""
    
    def __init__(self):
        keyword_categories = {
            'beginner': [
                'beginner', 'basic', 'introduction', 'tutorial', 'guide', 'learn',
                'start', 'simple', 'easy', 'fundamental', 'overview', 'primer',
                'getting started', 'first steps', 'basics'
            ],
            'intermediate': [
                'intermediate', 'moderate', 'standard', 'typical', 'common',
                'practical', 'applied', 'implementation', 'example', 'case study'
            ],
            'advanced': [
                'advanced', 'expert', 'complex', 'sophisticated', 'optimization',
                'architecture', 'enterprise', 'scalable', 'performance',
                'deep dive', 'comprehensive', 'mastery', 'professional'
            ]
        }
        
        super().__init__('complexity', keyword_categories)
    
    def _get_category_positions(self) -> Dict[str, float]:
        """Map complexity categories to coordinate positions"""
        return {
            'beginner': 0.2,      # Low complexity
            'intermediate': 0.5,   # Medium complexity
            'advanced': 0.9       # High complexity
        }


class TaskTypeQuantifier(KeywordBasedQuantifier):
    """Specialized quantifier for task type dimension"""
    
    def __init__(self):
        keyword_categories = {
            'tutorial': [
                'tutorial', 'guide', 'how to', 'step by step', 'walkthrough',
                'instructions', 'learn', 'teach', 'example', 'demo'
            ],
            'reference': [
                'reference', 'documentation', 'manual', 'specification', 'api',
                'definition', 'syntax', 'parameters', 'options', 'configuration'
            ],
            'analysis': [
                'analysis', 'compare', 'comparison', 'evaluate', 'assessment',
                'review', 'study', 'research', 'investigation', 'examination'
            ],
            'troubleshooting': [
                'troubleshooting', 'debug', 'error', 'problem', 'issue', 'fix',
                'solution', 'resolve', 'diagnose', 'repair'
            ]
        }
        
        super().__init__('task_type', keyword_categories)
    
    def _get_category_positions(self) -> Dict[str, float]:
        """Map task type categories to coordinate positions"""
        return {
            'tutorial': 0.2,          # Learning-oriented
            'reference': 0.4,         # Information-oriented
            'troubleshooting': 0.6,   # Problem-oriented
            'analysis': 0.8           # Understanding-oriented
        }


class EnhancedCoordinateQuantificationSystem:
    """
    Enhanced coordinate quantification system addressing the technical feedback.
    
    Features:
    - Formal dimension schemas with validation
    - Multiple quantification strategies
    - Normalization and weighting
    - Explainable coordinate generation
    - Extensible architecture
    """
    
    def __init__(self):
        self.dimension_schemas = self._initialize_dimension_schemas()
        self.quantifiers = self._initialize_quantifiers()
        self.normalization_stats = {}
        self.validation_enabled = True
    
    def _initialize_dimension_schemas(self) -> Dict[str, DimensionSchema]:
        """Initialize formal dimension schemas"""
        return {
            'domain': DimensionSchema(
                name='domain',
                dimension_type=DimensionType.CONTINUOUS,
                range_min=0.0,
                range_max=1.0,
                normalization_method='min_max',
                validation_rules=['positive'],
                weight=1.2,  # Higher weight for domain specialization
                description='Domain specialization: 0=creative, 0.3=business, 0.7=science, 0.9=technical'
            ),
            'complexity': DimensionSchema(
                name='complexity',
                dimension_type=DimensionType.CONTINUOUS,
                range_min=0.0,
                range_max=1.0,
                normalization_method='min_max',
                validation_rules=['positive'],
                weight=1.0,
                description='Content complexity: 0.2=beginner, 0.5=intermediate, 0.9=advanced'
            ),
            'task_type': DimensionSchema(
                name='task_type',
                dimension_type=DimensionType.CONTINUOUS,
                range_min=0.0,
                range_max=1.0,
                normalization_method='min_max',
                validation_rules=['positive'],
                weight=0.8,
                description='Task orientation: 0.2=tutorial, 0.4=reference, 0.6=troubleshooting, 0.8=analysis'
            )
        }
    
    def _initialize_quantifiers(self) -> Dict[str, CoordinateQuantifier]:
        """Initialize coordinate quantifiers"""
        return {
            'domain': DomainQuantifier(),
            'complexity': ComplexityQuantifier(),
            'task_type': TaskTypeQuantifier()
        }
    
    def text_to_coordinates(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert text to formal coordinates with validation and explanation.
        
        Returns:
            Dict containing coordinates, validation status, and explanations
        """
        coordinates = {}
        explanations = {}
        validation_results = {}
        
        for dimension_name, quantifier in self.quantifiers.items():
            # Quantify the dimension
            raw_value = quantifier.quantify(text, context)
            
            # Apply normalization
            normalized_value = self._normalize_value(dimension_name, raw_value)
            
            # Validate against schema
            schema = self.dimension_schemas[dimension_name]
            is_valid = schema.validate_value(normalized_value) if self.validation_enabled else True
            
            # Store results
            coordinates[dimension_name] = round(normalized_value, 3)
            explanations[dimension_name] = quantifier.get_explanation(text, normalized_value)
            validation_results[dimension_name] = is_valid
        
        return {
            'coordinates': coordinates,
            'explanations': explanations,
            'validation_results': validation_results,
            'all_valid': all(validation_results.values()),
            'schemas_used': {name: schema.description for name, schema in self.dimension_schemas.items()}
        }
    
    def _normalize_value(self, dimension_name: str, value: float) -> float:
        """Normalize a coordinate value according to its schema"""
        schema = self.dimension_schemas[dimension_name]
        
        if schema.normalization_method == 'min_max':
            # Ensure value is within range
            return max(schema.range_min, min(schema.range_max, value))
        elif schema.normalization_method == 'z_score':
            # Z-score normalization (requires statistics)
            if dimension_name in self.normalization_stats:
                stats = self.normalization_stats[dimension_name]
                normalized = (value - stats['mean']) / stats['std']
                # Convert to 0-1 range
                return max(0.0, min(1.0, (normalized + 3) / 6))  # Assume ±3 std range
            else:
                return value  # Fall back to raw value
        
        return value
    
    def add_dimension(self, schema: DimensionSchema, quantifier: CoordinateQuantifier):
        """Add a new dimension to the system"""
        self.dimension_schemas[schema.name] = schema
        self.quantifiers[schema.name] = quantifier
        logger.info(f"Added new dimension: {schema.name}")
    
    def update_normalization_stats(self, dimension_name: str, values: List[float]):
        """Update normalization statistics for a dimension"""
        if not values:
            return
        
        self.normalization_stats[dimension_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def get_dimension_info(self) -> Dict[str, Any]:
        """Get information about all dimensions"""
        return {
            'dimensions': {
                name: {
                    'schema': {
                        'type': schema.dimension_type.value,
                        'range': [schema.range_min, schema.range_max],
                        'weight': schema.weight,
                        'description': schema.description
                    },
                    'quantifier': type(self.quantifiers[name]).__name__
                }
                for name, schema in self.dimension_schemas.items()
            },
            'total_dimensions': len(self.dimension_schemas),
            'validation_enabled': self.validation_enabled
        }
    
    def validate_coordinate_set(self, coordinates: Dict[str, float]) -> Dict[str, Any]:
        """Validate a complete coordinate set"""
        results = {}
        
        for dim_name, value in coordinates.items():
            if dim_name in self.dimension_schemas:
                schema = self.dimension_schemas[dim_name]
                results[dim_name] = {
                    'valid': schema.validate_value(value),
                    'in_range': schema.range_min <= value <= schema.range_max,
                    'schema_type': schema.dimension_type.value
                }
            else:
                results[dim_name] = {
                    'valid': False,
                    'error': 'Unknown dimension'
                }
        
        return {
            'dimension_results': results,
            'all_valid': all(r.get('valid', False) for r in results.values()),
            'missing_dimensions': set(self.dimension_schemas.keys()) - set(coordinates.keys())
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the enhanced system
    coord_system = EnhancedCoordinateQuantificationSystem()
    
    # Test with sample texts
    test_texts = [
        "Python programming tutorial for beginners",
        "Advanced machine learning optimization techniques",
        "Business strategy analysis and market research",
        "Creative design principles and visual aesthetics"
    ]
    
    print("Enhanced Coordinate Quantification System Demo")
    print("=" * 50)
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = coord_system.text_to_coordinates(text)
        
        print(f"Coordinates: {result['coordinates']}")
        print(f"Valid: {result['all_valid']}")
        
        for dim, explanation in result['explanations'].items():
            print(f"  {explanation}")
    
    print(f"\nDimension Info:")
    info = coord_system.get_dimension_info()
    for dim_name, dim_info in info['dimensions'].items():
        print(f"  {dim_name}: {dim_info['schema']['description']}")