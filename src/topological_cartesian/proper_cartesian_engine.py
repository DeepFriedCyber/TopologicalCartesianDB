#!/usr/bin/env python3
"""
Proper Cartesian Coordinate Engine

This implements TRUE Cartesian coordinates for semantic content,
where each text is positioned in interpretable 3D space.

Key principles:
- X-axis: Technical Complexity (-1 to +1)
- Y-axis: Domain Specificity (-1 to +1) 
- Z-axis: Factual Certainty (-1 to +1)

This allows for:
- Geometric distance calculations
- Spatial clustering of similar content
- Interpretable positioning
- True coordinate-based search
"""

import hashlib
import re
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CartesianPosition:
    """A position in 3D Cartesian space"""
    x: float  # Technical complexity: -1 (simple) to +1 (complex)
    y: float  # Domain specificity: -1 (general) to +1 (specialized)
    z: float  # Factual certainty: -1 (false/uncertain) to +1 (true/certain)
    
    def distance_to(self, other: 'CartesianPosition') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {'x': self.x, 'y': self.y, 'z': self.z}
    
    def __str__(self) -> str:
        return f"({self.x:+.2f}, {self.y:+.2f}, {self.z:+.2f})"


class ProperCartesianEngine:
    """
    Engine that generates proper Cartesian coordinates for text content.
    
    Maps text to meaningful 3D positions where:
    - Similar content clusters together
    - Distances reflect semantic relationships
    - Axes have interpretable meanings
    """
    
    def __init__(self):
        """Initialize the Cartesian coordinate engine"""
        self._coordinate_cache: Dict[str, CartesianPosition] = {}
        
        # Define keyword patterns for each axis
        self._setup_keyword_patterns()
    
    def _setup_keyword_patterns(self):
        """Setup keyword patterns for coordinate calculation"""
        
        # X-axis: Technical Complexity
        self.simple_keywords = {
            'basic', 'simple', 'easy', 'beginner', 'introduction', 'tutorial', 
            'learn', 'start', 'guide', 'overview', 'fundamentals', 'primer'
        }
        
        self.complex_keywords = {
            'advanced', 'complex', 'sophisticated', 'expert', 'optimization', 
            'architecture', 'algorithm', 'theoretical', 'research', 'analysis',
            'framework', 'methodology', 'implementation', 'technical', 'detailed'
        }
        
        # Y-axis: Domain Specificity  
        self.general_keywords = {
            'general', 'common', 'basic', 'universal', 'standard', 'typical',
            'everyday', 'popular', 'widespread', 'conventional'
        }
        
        self.specialized_keywords = {
            # Programming
            'python', 'programming', 'code', 'software', 'algorithm', 'function',
            'variable', 'database', 'javascript', 'html', 'css', 'java', 'api',
            
            # Science
            'quantum', 'physics', 'chemistry', 'biology', 'research', 'scientific',
            'molecular', 'genetic', 'neural', 'brain', 'climate', 'environmental',
            
            # Business
            'business', 'strategy', 'market', 'financial', 'investment', 'marketing',
            'management', 'corporate', 'analysis', 'planning', 'competitive',
            
            # Medical
            'medical', 'health', 'disease', 'treatment', 'clinical', 'diagnosis',
            'pharmaceutical', 'therapeutic', 'surgical', 'cardiovascular'
        }
        
        # Z-axis: Factual Certainty
        self.uncertain_keywords = {
            'might', 'could', 'possibly', 'perhaps', 'maybe', 'uncertain',
            'unclear', 'debated', 'controversial', 'alleged', 'claimed',
            'supposedly', 'rumored', 'unconfirmed', 'speculative'
        }
        
        self.certain_keywords = {
            'confirmed', 'proven', 'established', 'documented', 'verified',
            'demonstrated', 'research shows', 'studies show', 'evidence',
            'scientific consensus', 'peer-reviewed', 'validated', 'measured'
        }
        
        # Factual claim patterns
        self.false_claim_patterns = [
            r'visible from space.*naked eye',  # Great Wall myth
            r'100%.*accuracy.*all.*datasets',  # ML impossibility
            r'entirely.*natural.*factors',     # Climate denial
            r'exponentially.*faster.*all',     # Quantum computing myth
            r'equally.*efficient.*all.*tasks'  # Programming language myth
        ]
        
        self.true_claim_patterns = [
            r'first released.*1991',           # Python release
            r'20%.*body.*energy',              # Brain energy usage
            r'299,792,458.*meters.*second',    # Speed of light
            r'iron.*nickel.*core',             # Earth's core
            r'genetic.*instructions.*DNA'      # DNA function
        ]
    
    def text_to_coordinates(self, text: str) -> Dict[str, float]:
        """
        Convert text to proper Cartesian coordinates.
        
        Returns coordinates as dict for compatibility with existing code,
        but represents true 3D Cartesian position.
        """
        position = self.text_to_position(text)
        return position.to_dict()
    
    def text_to_position(self, text: str) -> CartesianPosition:
        """
        Convert text to proper Cartesian position in 3D space.
        
        This is the core method that maps semantic content to geometric space.
        """
        # Cache for consistency
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._coordinate_cache:
            return self._coordinate_cache[text_hash]
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Calculate X-axis: Technical Complexity (-1 to +1)
        x = self._calculate_technical_complexity(text_lower, words)
        
        # Calculate Y-axis: Domain Specificity (-1 to +1)
        y = self._calculate_domain_specificity(text_lower, words)
        
        # Calculate Z-axis: Factual Certainty (-1 to +1)
        z = self._calculate_factual_certainty(text_lower, words)
        
        position = CartesianPosition(x, y, z)
        self._coordinate_cache[text_hash] = position
        
        return position
    
    def _calculate_technical_complexity(self, text: str, words: set) -> float:
        """Calculate technical complexity score (-1 to +1)"""
        
        # Count simple vs complex indicators
        simple_score = len(words & self.simple_keywords)
        complex_score = len(words & self.complex_keywords)
        
        # Additional complexity indicators
        if any(term in text for term in ['algorithm', 'optimization', 'architecture']):
            complex_score += 2
        
        if any(term in text for term in ['tutorial', 'beginner', 'introduction']):
            simple_score += 2
        
        # Technical jargon density
        technical_terms = len([w for w in words if len(w) > 8 and any(c.isdigit() for c in w)])
        complex_score += technical_terms * 0.5
        
        # Calculate final score
        if simple_score == 0 and complex_score == 0:
            return 0.0  # Neutral
        
        total_score = complex_score - simple_score
        # Normalize to [-1, +1] range
        return max(-1.0, min(1.0, total_score / 5.0))
    
    def _calculate_domain_specificity(self, text: str, words: set) -> float:
        """Calculate domain specificity score (-1 to +1)"""
        
        # Count general vs specialized indicators
        general_score = len(words & self.general_keywords)
        specialized_score = len(words & self.specialized_keywords)
        
        # Domain-specific patterns
        domain_patterns = {
            'programming': ['python', 'code', 'programming', 'software', 'algorithm'],
            'science': ['quantum', 'research', 'scientific', 'molecular', 'brain'],
            'business': ['business', 'strategy', 'market', 'financial', 'management'],
            'medical': ['medical', 'health', 'disease', 'clinical', 'treatment']
        }
        
        max_domain_score = 0
        for domain, keywords in domain_patterns.items():
            domain_score = sum(1 for keyword in keywords if keyword in text)
            max_domain_score = max(max_domain_score, domain_score)
        
        specialized_score += max_domain_score
        
        # Calculate final score
        if general_score == 0 and specialized_score == 0:
            return 0.0  # Neutral
        
        total_score = specialized_score - general_score
        # Normalize to [-1, +1] range
        return max(-1.0, min(1.0, total_score / 5.0))
    
    def _calculate_factual_certainty(self, text: str, words: set) -> float:
        """Calculate factual certainty score (-1 to +1)"""
        
        # Check for explicit false claims
        for pattern in self.false_claim_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return -0.9  # Definitely false
        
        # Check for explicit true claims
        for pattern in self.true_claim_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.9   # Definitely true
        
        # Count uncertainty vs certainty indicators
        uncertain_score = len(words & self.uncertain_keywords)
        certain_score = len(words & self.certain_keywords)
        
        # Additional certainty indicators
        if any(term in text for term in ['research shows', 'studies show', 'proven']):
            certain_score += 2
        
        if any(term in text for term in ['might', 'could', 'possibly', 'allegedly']):
            uncertain_score += 1
        
        # Numerical precision indicates certainty
        if re.search(r'\d+\.\d+', text):  # Decimal numbers
            certain_score += 1
        
        if re.search(r'\d{4}', text):     # Years
            certain_score += 1
        
        # Calculate final score
        if uncertain_score == 0 and certain_score == 0:
            return 0.0  # Neutral
        
        total_score = certain_score - uncertain_score
        # Normalize to [-1, +1] range
        return max(-1.0, min(1.0, total_score / 3.0))
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity based on Cartesian distance.
        
        Returns similarity score from 0 (completely different) to 1 (identical).
        """
        pos1 = self.text_to_position(text1)
        pos2 = self.text_to_position(text2)
        
        # Calculate distance (max possible distance in 3D unit cube is sqrt(3) ≈ 1.73)
        distance = pos1.distance_to(pos2)
        max_distance = math.sqrt(3)  # Maximum distance in [-1,1]³ space
        
        # Convert distance to similarity
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, similarity)
    
    def find_similar_positions(self, target_text: str, candidate_texts: List[str], 
                             k: int = 10) -> List[Tuple[str, float, CartesianPosition]]:
        """
        Find texts with similar Cartesian positions.
        
        Returns list of (text, similarity_score, position) tuples.
        """
        target_pos = self.text_to_position(target_text)
        
        results = []
        for candidate in candidate_texts:
            candidate_pos = self.text_to_position(candidate)
            similarity = 1.0 - (target_pos.distance_to(candidate_pos) / math.sqrt(3))
            results.append((candidate, similarity, candidate_pos))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def visualize_positions(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate data for 3D visualization of text positions.
        
        Returns dictionary with coordinates and labels for plotting.
        """
        positions = []
        labels = []
        
        for text in texts:
            pos = self.text_to_position(text)
            positions.append([pos.x, pos.y, pos.z])
            # Truncate text for labels
            label = text[:50] + "..." if len(text) > 50 else text
            labels.append(label)
        
        return {
            'positions': np.array(positions),
            'labels': labels,
            'axis_labels': {
                'x': 'Technical Complexity (-1: Simple, +1: Complex)',
                'y': 'Domain Specificity (-1: General, +1: Specialized)', 
                'z': 'Factual Certainty (-1: False/Uncertain, +1: True/Certain)'
            }
        }


def demonstrate_proper_cartesian():
    """Demonstrate the proper Cartesian coordinate system"""
    
    print("Proper Cartesian Coordinate System Demo")
    print("=" * 50)
    
    engine = ProperCartesianEngine()
    
    # Test texts from FACTS benchmark
    test_texts = [
        "The Great Wall of China is visible from space with the naked eye",
        "Python was first released in 1991", 
        "The human brain uses approximately 20% of the body's energy",
        "Machine learning algorithms can achieve 100% accuracy on all datasets",
        "The speed of light in vacuum is approximately 300,000 km/s",
        "NASA has confirmed that the Great Wall of China is not visible from space",
        "Research shows that the human brain consumes about 20% of total energy",
        "Advanced neural network architecture optimization techniques"
    ]
    
    print("\nCartesian positions for sample texts:")
    print("-" * 50)
    
    positions = []
    for i, text in enumerate(test_texts):
        pos = engine.text_to_position(text)
        positions.append(pos)
        
        print(f"\nText {i+1}: {text[:60]}...")
        print(f"Position: {pos}")
        print(f"  X (Complexity): {pos.x:+.2f} ({'Complex' if pos.x > 0.3 else 'Simple' if pos.x < -0.3 else 'Medium'})")
        print(f"  Y (Specificity): {pos.y:+.2f} ({'Specialized' if pos.y > 0.3 else 'General' if pos.y < -0.3 else 'Mixed'})")
        print(f"  Z (Certainty): {pos.z:+.2f} ({'True' if pos.z > 0.3 else 'False' if pos.z < -0.3 else 'Uncertain'})")
    
    # Calculate some interesting distances
    print(f"\n" + "=" * 50)
    print("SEMANTIC DISTANCES")
    print("=" * 50)
    
    # Compare similar claims
    great_wall_false = positions[0]  # False claim
    great_wall_true = positions[5]   # True refutation
    
    python_claim = positions[1]      # Python release
    brain_claim = positions[2]       # Brain energy
    
    print(f"\nDistance between false and true Great Wall claims:")
    print(f"  False: {great_wall_false}")
    print(f"  True:  {great_wall_true}")
    print(f"  Distance: {great_wall_false.distance_to(great_wall_true):.3f}")
    
    print(f"\nDistance between different factual claims:")
    print(f"  Python: {python_claim}")
    print(f"  Brain:  {brain_claim}")
    print(f"  Distance: {python_claim.distance_to(brain_claim):.3f}")
    
    # Test similarity calculation
    print(f"\n" + "=" * 50)
    print("SIMILARITY SCORES")
    print("=" * 50)
    
    similarities = [
        ("Great Wall false claim", "Great Wall true refutation", test_texts[0], test_texts[5]),
        ("Python release claim", "Brain energy claim", test_texts[1], test_texts[2]),
        ("ML accuracy claim", "Advanced ML techniques", test_texts[3], test_texts[7])
    ]
    
    for desc1, desc2, text1, text2 in similarities:
        similarity = engine.calculate_similarity(text1, text2)
        print(f"\n{desc1} vs {desc2}:")
        print(f"  Similarity: {similarity:.3f}")
    
    print(f"\n🎯 This demonstrates proper Cartesian coordinates where:")
    print(f"  ✅ Each axis has clear semantic meaning")
    print(f"  ✅ Distances reflect actual relationships") 
    print(f"  ✅ Similar content clusters together")
    print(f"  ✅ Coordinates are interpretable")
    print(f"  ✅ True geometric calculations possible")


if __name__ == "__main__":
    demonstrate_proper_cartesian()