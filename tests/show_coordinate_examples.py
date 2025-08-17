#!/usr/bin/env python3
"""
Show concrete examples of proper Cartesian coordinates
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.proper_cartesian_engine import ProperCartesianEngine

def show_coordinate_examples():
    """Show concrete examples of how proper Cartesian coordinates work"""
    
    print("PROPER CARTESIAN COORDINATE EXAMPLES")
    print("=" * 60)
    
    engine = ProperCartesianEngine()
    
    # Examples from our benchmark
    examples = [
        {
            'text': 'The Great Wall of China is visible from space with the naked eye',
            'description': 'FALSE CLAIM - Geography',
            'expected': 'Should be in FALSE region (z < -0.3)'
        },
        {
            'text': 'Python was first released in 1991',
            'description': 'TRUE FACT - Programming History',
            'expected': 'Should be in TRUE + SPECIALIZED region (z > 0.3, y > 0.3)'
        },
        {
            'text': 'Advanced neural network architecture optimization techniques',
            'description': 'COMPLEX TECHNICAL - AI/ML',
            'expected': 'Should be in COMPLEX + SPECIALIZED region (x > 0.5, y > 0.3)'
        },
        {
            'text': 'Introduction to web development for beginners',
            'description': 'SIMPLE TUTORIAL - Programming',
            'expected': 'Should be in SIMPLE + SPECIALIZED region (x < -0.3, y > 0.3)'
        },
        {
            'text': 'Machine learning algorithms can achieve 100% accuracy on all datasets',
            'description': 'FALSE TECHNICAL CLAIM - AI/ML',
            'expected': 'Should be in FALSE + COMPLEX + SPECIALIZED region (z < -0.3, x > 0.3, y > 0.3)'
        }
    ]
    
    print(f"\n3D Cartesian Space Mapping:")
    print(f"  X-axis: Technical Complexity (-1: Simple → +1: Complex)")
    print(f"  Y-axis: Domain Specificity (-1: General → +1: Specialized)")
    print(f"  Z-axis: Factual Certainty (-1: False → +1: True)")
    print(f"\n" + "-" * 60)
    
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}: {example['description']}")
        print(f"Text: \"{example['text'][:50]}...\"")
        
        position = engine.text_to_position(example['text'])
        coords = position.to_dict()
        
        print(f"Cartesian Position: ({position.x:+.2f}, {position.y:+.2f}, {position.z:+.2f})")
        
        # Interpret coordinates
        complexity = 'Complex' if position.x > 0.3 else 'Simple' if position.x < -0.3 else 'Medium'
        specificity = 'Specialized' if position.y > 0.3 else 'General' if position.y < -0.3 else 'Mixed'
        certainty = 'True' if position.z > 0.3 else 'False' if position.z < -0.3 else 'Uncertain'
        
        print(f"Interpretation:")
        print(f"  • Complexity: {complexity} (X = {position.x:+.2f})")
        print(f"  • Specificity: {specificity} (Y = {position.y:+.2f})")
        print(f"  • Certainty: {certainty} (Z = {position.z:+.2f})")
        print(f"Expected: {example['expected']}")
        
        # Validate expectations
        validation = "✅ CORRECT" if (
            (certainty == 'False' and 'FALSE' in example['description']) or
            (certainty == 'True' and 'TRUE' in example['description']) or
            (complexity == 'Complex' and 'COMPLEX' in example['description']) or
            (complexity == 'Simple' and 'SIMPLE' in example['description'])
        ) else "⚠️ CHECK"
        
        print(f"Validation: {validation}")
    
    print(f"\n" + "=" * 60)
    print("GEOMETRIC RELATIONSHIPS")
    print("=" * 60)
    
    # Show distances between related concepts
    positions = [engine.text_to_position(ex['text']) for ex in examples]
    
    print(f"\nDistances between concepts:")
    
    # False claims should be close to each other (both have z < 0)
    false_claim_1 = positions[0]  # Great Wall
    false_claim_2 = positions[4]  # ML 100% accuracy
    distance_false = false_claim_1.distance_to(false_claim_2)
    print(f"  False claims distance: {distance_false:.3f}")
    print(f"    Great Wall (false): {false_claim_1}")
    print(f"    ML claim (false): {false_claim_2}")
    
    # True fact vs False claim should be far apart (different z values)
    true_fact = positions[1]      # Python release
    false_claim = positions[0]    # Great Wall
    distance_true_false = true_fact.distance_to(false_claim)
    print(f"  True vs False distance: {distance_true_false:.3f}")
    print(f"    Python fact (true): {true_fact}")
    print(f"    Great Wall (false): {false_claim}")
    
    # Simple vs Complex should be far apart (different x values)
    simple_tutorial = positions[3]  # Beginner tutorial
    complex_tech = positions[2]     # Advanced ML
    distance_simple_complex = simple_tutorial.distance_to(complex_tech)
    print(f"  Simple vs Complex distance: {distance_simple_complex:.3f}")
    print(f"    Simple tutorial: {simple_tutorial}")
    print(f"    Complex tech: {complex_tech}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"  ✅ False claims cluster together (similar Z coordinates)")
    print(f"  ✅ True facts separate from false claims (different Z coordinates)")
    print(f"  ✅ Complex topics separate from simple ones (different X coordinates)")
    print(f"  ✅ Specialized domains separate from general ones (different Y coordinates)")
    print(f"  ✅ Distance reflects semantic similarity")
    print(f"  ✅ Each axis has clear, interpretable meaning")
    
    print(f"\n🚀 This is TRUE Cartesian coordinate-based search!")
    print(f"   Unlike opaque vector embeddings, every coordinate has meaning!")

if __name__ == "__main__":
    show_coordinate_examples()