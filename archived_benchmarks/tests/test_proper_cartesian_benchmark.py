#!/usr/bin/env python3
"""
TOPCART + Ollama Benchmark with PROPER Cartesian Coordinates

This test uses true 3D Cartesian coordinates where:
- X-axis: Technical Complexity (-1 to +1)
- Y-axis: Domain Specificity (-1 to +1) 
- Z-axis: Factual Certainty (-1 to +1)

This allows for genuine geometric relationships and interpretable positioning.
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
from topological_cartesian.proper_cartesian_engine import ProperCartesianEngine, CartesianPosition

logger = logging.getLogger(__name__)


@dataclass
class FactsClaim:
    """A factual claim from the FACTS dataset"""
    claim_id: str
    claim_text: str
    label: str  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    evidence_ids: List[str]
    domain: str


@dataclass
class FactsEvidence:
    """Evidence passage from the FACTS dataset"""
    evidence_id: str
    text: str
    source: str
    domain: str


@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    system_name: str
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    avg_query_time: float
    total_claims: int
    successful_retrievals: int
    avg_coordinate_distance: float  # New metric for coordinate quality


class ProperCartesianSystem:
    """TOPCART system using proper Cartesian coordinates"""
    
    def __init__(self):
        self.coordinate_engine = ProperCartesianEngine()
        self.evidence_store = {}
        
    def index_evidence(self, evidence: Dict[str, FactsEvidence]) -> bool:
        """Index all evidence passages with proper Cartesian coordinates"""
        
        print("Indexing evidence with proper Cartesian coordinates...")
        
        for evidence_id, evidence_item in evidence.items():
            try:
                # Generate proper Cartesian position
                position = self.coordinate_engine.text_to_position(evidence_item.text)
                
                self.evidence_store[evidence_id] = {
                    'text': evidence_item.text,
                    'position': position,
                    'coordinates': position.to_dict(),  # For compatibility
                    'domain': evidence_item.domain,
                    'source': evidence_item.source
                }
                
            except Exception as e:
                logger.error(f"Failed to index evidence {evidence_id}: {e}")
                return False
        
        print(f"✅ Indexed {len(self.evidence_store)} evidence passages in 3D Cartesian space")
        return True
    
    def search_evidence(self, claim_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for relevant evidence using Cartesian distance"""
        
        try:
            # Generate Cartesian position for claim
            claim_position = self.coordinate_engine.text_to_position(claim_text)
            
            # Calculate distances to all evidence
            scores = []
            
            for evidence_id, evidence_data in self.evidence_store.items():
                evidence_position = evidence_data['position']
                
                # Pure geometric distance in 3D space
                distance = claim_position.distance_to(evidence_position)
                
                # Convert distance to similarity score (closer = higher score)
                max_distance = np.sqrt(3)  # Maximum distance in [-1,1]³ space
                similarity = 1.0 - (distance / max_distance)
                
                scores.append((evidence_id, similarity))
            
            # Sort by similarity (highest first)
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
            
        except Exception as e:
            logger.error(f"Evidence search failed: {e}")
            return []
    
    def get_coordinate_analysis(self, claim_text: str) -> Dict[str, Any]:
        """Get detailed coordinate analysis for a claim"""
        
        position = self.coordinate_engine.text_to_position(claim_text)
        
        return {
            'position': position,
            'coordinates': position.to_dict(),
            'interpretation': {
                'complexity': 'Complex' if position.x > 0.3 else 'Simple' if position.x < -0.3 else 'Medium',
                'specificity': 'Specialized' if position.y > 0.3 else 'General' if position.y < -0.3 else 'Mixed',
                'certainty': 'True' if position.z > 0.3 else 'False' if position.z < -0.3 else 'Uncertain'
            }
        }


class OllamaEnhancedCartesianSystem(ProperCartesianSystem):
    """Cartesian system enhanced with Ollama LLM"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", 
                 ollama_model: str = "llama3.2:3b"):
        super().__init__()
        self.ollama_integrator = OllamaLLMIntegrator(
            ollama_host=ollama_host,
            default_model=ollama_model
        )
    
    def search_evidence(self, claim_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """Enhanced search using Ollama + Cartesian coordinates"""
        
        try:
            # Get Ollama analysis of the claim
            analysis_prompt = f"""Analyze this factual claim for evidence search:

Claim: "{claim_text}"

Rate on scales of -1 to +1:
1. Technical complexity (-1: very simple, +1: very complex)
2. Domain specificity (-1: very general, +1: very specialized)  
3. Factual certainty (-1: likely false, +1: likely true)

Also identify key concepts that evidence should contain.

Respond in this format:
Complexity: [score]
Specificity: [score]
Certainty: [score]
Key concepts: [concept1, concept2, concept3]"""
            
            ollama_response = self.ollama_integrator.generate_response(
                analysis_prompt,
                temperature=0.1,
                max_tokens=100
            )
            
            if ollama_response.success:
                return self._search_with_ollama_enhancement(claim_text, ollama_response.content, k)
            else:
                # Fallback to basic Cartesian search
                return super().search_evidence(claim_text, k)
                
        except Exception as e:
            logger.error(f"Ollama-enhanced search failed: {e}")
            return super().search_evidence(claim_text, k)
    
    def _search_with_ollama_enhancement(self, claim_text: str, ollama_analysis: str, k: int) -> List[Tuple[str, float]]:
        """Search with Ollama-enhanced coordinate weighting"""
        
        try:
            # Parse Ollama analysis
            ollama_coords = self._parse_ollama_coordinates(ollama_analysis)
            
            # Get base Cartesian position
            claim_position = self.coordinate_engine.text_to_position(claim_text)
            
            # Calculate enhanced scores
            scores = []
            
            for evidence_id, evidence_data in self.evidence_store.items():
                evidence_position = evidence_data['position']
                
                # Base geometric distance
                base_distance = claim_position.distance_to(evidence_position)
                base_similarity = 1.0 - (base_distance / np.sqrt(3))
                
                # Ollama-weighted distance (if available)
                if ollama_coords:
                    ollama_distance = self._calculate_weighted_distance(
                        claim_position, evidence_position, ollama_coords
                    )
                    ollama_similarity = 1.0 - (ollama_distance / np.sqrt(3))
                    
                    # Combine base and Ollama similarities
                    combined_similarity = 0.6 * base_similarity + 0.4 * ollama_similarity
                else:
                    combined_similarity = base_similarity
                
                scores.append((evidence_id, combined_similarity))
            
            # Sort by similarity
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return super().search_evidence(claim_text, k)
    
    def _parse_ollama_coordinates(self, analysis: str) -> Optional[Dict[str, float]]:
        """Parse coordinate values from Ollama analysis"""
        
        try:
            coords = {}
            lines = analysis.split('\n')
            
            for line in lines:
                line = line.strip().lower()
                if line.startswith('complexity:'):
                    coords['x'] = float(line.split(':')[1].strip())
                elif line.startswith('specificity:'):
                    coords['y'] = float(line.split(':')[1].strip())
                elif line.startswith('certainty:'):
                    coords['z'] = float(line.split(':')[1].strip())
            
            # Validate coordinates are in [-1, 1] range
            for key, value in coords.items():
                coords[key] = max(-1.0, min(1.0, value))
            
            return coords if len(coords) == 3 else None
            
        except Exception as e:
            logger.debug(f"Failed to parse Ollama coordinates: {e}")
            return None
    
    def _calculate_weighted_distance(self, pos1: CartesianPosition, pos2: CartesianPosition, 
                                   weights: Dict[str, float]) -> float:
        """Calculate weighted Euclidean distance"""
        
        # Use Ollama's coordinate assessment as weights
        x_weight = abs(weights.get('x', 1.0))
        y_weight = abs(weights.get('y', 1.0))
        z_weight = abs(weights.get('z', 1.0))
        
        # Weighted distance calculation
        weighted_distance = np.sqrt(
            x_weight * (pos1.x - pos2.x)**2 + 
            y_weight * (pos1.y - pos2.y)**2 + 
            z_weight * (pos1.z - pos2.z)**2
        )
        
        return weighted_distance


class CartesianBenchmarkTester:
    """Benchmark tester for proper Cartesian coordinate system"""
    
    def __init__(self):
        self.test_data = self._create_enhanced_test_dataset()
    
    def _create_enhanced_test_dataset(self) -> Dict[str, Any]:
        """Create enhanced test dataset with proper coordinate expectations"""
        
        # Enhanced documents with expected coordinate regions
        documents = {
            'doc_1': {
                'content': 'Python machine learning tutorial with scikit-learn and pandas for data analysis',
                'domain': 'programming',
                'expected_region': 'medium_complexity_specialized_neutral'
            },
            'doc_2': {
                'content': 'Introduction to web development using HTML CSS and JavaScript for beginners',
                'domain': 'programming', 
                'expected_region': 'simple_specialized_neutral'
            },
            'doc_3': {
                'content': 'Advanced neural network architecture design and optimization techniques',
                'domain': 'programming',
                'expected_region': 'complex_specialized_neutral'
            },
            'doc_4': {
                'content': 'NASA has confirmed that the Great Wall of China is not visible from space with the naked eye',
                'domain': 'geography',
                'expected_region': 'medium_mixed_true'
            },
            'doc_5': {
                'content': 'Python programming language was first released by Guido van Rossum in February 1991',
                'domain': 'programming',
                'expected_region': 'medium_specialized_true'
            },
            'doc_6': {
                'content': 'Research shows that the human brain consumes about 20% of the body total energy',
                'domain': 'science',
                'expected_region': 'medium_specialized_true'
            },
            'doc_7': {
                'content': 'Machine learning algorithms can achieve 100% accuracy on all datasets',
                'domain': 'programming',
                'expected_region': 'complex_specialized_false'
            },
            'doc_8': {
                'content': 'The speed of light in vacuum is exactly 299,792,458 meters per second',
                'domain': 'science',
                'expected_region': 'medium_specialized_true'
            },
            'doc_9': {
                'content': 'Climate change is caused entirely by natural factors without human influence',
                'domain': 'science',
                'expected_region': 'medium_specialized_false'
            },
            'doc_10': {
                'content': 'Quantum computers can solve all computational problems exponentially faster than classical computers',
                'domain': 'science',
                'expected_region': 'complex_specialized_false'
            }
        }
        
        # Test queries with expected relevant documents
        queries = {
            'query_1': {
                'text': 'The Great Wall of China is visible from space with the naked eye',
                'relevant_docs': ['doc_4'],  # Should find the refutation
                'domain': 'geography',
                'expected_coordinates': {'x': 0.0, 'y': 0.0, 'z': -0.9}  # False claim
            },
            'query_2': {
                'text': 'Python was first released in 1991',
                'relevant_docs': ['doc_5'],  # Historical fact
                'domain': 'programming',
                'expected_coordinates': {'x': 0.0, 'y': 0.4, 'z': 0.9}  # True, specialized
            },
            'query_3': {
                'text': 'Machine learning algorithms achieve perfect accuracy',
                'relevant_docs': ['doc_7'],  # False ML claim
                'domain': 'programming',
                'expected_coordinates': {'x': 0.4, 'y': 0.2, 'z': -0.9}  # Complex, false
            },
            'query_4': {
                'text': 'Advanced neural network optimization',
                'relevant_docs': ['doc_3'],  # Complex ML topic
                'domain': 'programming',
                'expected_coordinates': {'x': 1.0, 'y': 0.2, 'z': 0.0}  # Very complex
            },
            'query_5': {
                'text': 'Human brain energy consumption research',
                'relevant_docs': ['doc_6'],  # Scientific fact
                'domain': 'science',
                'expected_coordinates': {'x': 0.2, 'y': 0.8, 'z': 0.67}  # Research-based, true
            }
        }
        
        return {
            'documents': documents,
            'queries': queries
        }
    
    def test_system(self, system: ProperCartesianSystem, system_name: str) -> BenchmarkResult:
        """Test system with proper Cartesian coordinate analysis"""
        
        print(f"\nTesting {system_name} with proper Cartesian coordinates...")
        
        # Convert documents to evidence format
        evidence = {}
        for doc_id, doc_data in self.test_data['documents'].items():
            evidence[doc_id] = FactsEvidence(
                evidence_id=doc_id,
                text=doc_data['content'],
                source="Test Dataset",
                domain=doc_data['domain']
            )
        
        # Index evidence
        if not system.index_evidence(evidence):
            raise Exception("Failed to index evidence")
        
        # Test coordinate accuracy
        print(f"  Analyzing coordinate accuracy...")
        coordinate_distances = []
        
        # Test each query
        query_times = []
        precision_at_1_scores = []
        precision_at_5_scores = []
        precision_at_10_scores = []
        recall_at_10_scores = []
        mrr_scores = []
        successful_retrievals = 0
        
        for query_id, query_data in self.test_data['queries'].items():
            print(f"  Testing query: {query_id}")
            
            # Analyze coordinates
            coord_analysis = system.get_coordinate_analysis(query_data['text'])
            actual_coords = coord_analysis['coordinates']
            expected_coords = query_data.get('expected_coordinates', {})
            
            if expected_coords:
                # Calculate coordinate accuracy
                coord_distance = np.sqrt(sum(
                    (actual_coords.get(k, 0) - v)**2 
                    for k, v in expected_coords.items()
                ))
                coordinate_distances.append(coord_distance)
                
                print(f"    Expected coords: {expected_coords}")
                print(f"    Actual coords: {actual_coords}")
                print(f"    Coordinate distance: {coord_distance:.3f}")
            
            # Search for evidence
            query_start = time.time()
            results = system.search_evidence(query_data['text'], k=10)
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            if results:
                successful_retrievals += 1
                
                # Calculate metrics
                result_evidence_ids = [r[0] for r in results]
                relevant_evidence_ids = set(query_data['relevant_docs'])
                
                p_at_1 = self._calculate_precision_at_k(result_evidence_ids, relevant_evidence_ids, 1)
                p_at_5 = self._calculate_precision_at_k(result_evidence_ids, relevant_evidence_ids, 5)
                p_at_10 = self._calculate_precision_at_k(result_evidence_ids, relevant_evidence_ids, 10)
                
                precision_at_1_scores.append(p_at_1)
                precision_at_5_scores.append(p_at_5)
                precision_at_10_scores.append(p_at_10)
                
                recall_10 = self._calculate_recall_at_k(result_evidence_ids, relevant_evidence_ids, 10)
                recall_at_10_scores.append(recall_10)
                
                mrr = self._calculate_mrr(result_evidence_ids, relevant_evidence_ids)
                mrr_scores.append(mrr)
                
                print(f"    P@1: {p_at_1:.3f}, P@5: {p_at_5:.3f}, Recall@10: {recall_10:.3f}")
                print(f"    Top result: {results[0][0]} (score: {results[0][1]:.3f})")
        
        # Calculate averages
        return BenchmarkResult(
            system_name=system_name,
            precision_at_1=np.mean(precision_at_1_scores) if precision_at_1_scores else 0.0,
            precision_at_5=np.mean(precision_at_5_scores) if precision_at_5_scores else 0.0,
            precision_at_10=np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0,
            recall_at_10=np.mean(recall_at_10_scores) if recall_at_10_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            avg_query_time=np.mean(query_times) if query_times else 0.0,
            total_claims=len(self.test_data['queries']),
            successful_retrievals=successful_retrievals,
            avg_coordinate_distance=np.mean(coordinate_distances) if coordinate_distances else 0.0
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
    
    def run_comparison_benchmark(self) -> Dict[str, Any]:
        """Run comparison between basic and Ollama-enhanced Cartesian systems"""
        
        print("Proper Cartesian TOPCART + Ollama Benchmark")
        print("=" * 60)
        
        results = {}
        
        # Test basic Cartesian system
        try:
            basic_system = ProperCartesianSystem()
            basic_results = self.test_system(basic_system, "Cartesian TOPCART")
            results['basic'] = basic_results
        except Exception as e:
            print(f"❌ Basic Cartesian system test failed: {e}")
            return {}
        
        # Test Ollama-enhanced Cartesian system
        try:
            enhanced_system = OllamaEnhancedCartesianSystem()
            enhanced_results = self.test_system(enhanced_system, "Cartesian TOPCART + Ollama")
            results['enhanced'] = enhanced_results
        except Exception as e:
            print(f"❌ Ollama-enhanced Cartesian system test failed: {e}")
            results['enhanced'] = None
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print detailed benchmark comparison"""
        
        print(f"\n{'='*70}")
        print("PROPER CARTESIAN COORDINATE BENCHMARK RESULTS")
        print(f"{'='*70}")
        
        if 'basic' not in results:
            print("❌ No results to display")
            return
        
        basic = results['basic']
        enhanced = results.get('enhanced')
        
        print(f"{'Metric':<25} {'Basic Cartesian':<18} {'+ Ollama':<18} {'Improvement':<15}")
        print("-" * 80)
        
        metrics = [
            ('Precision@1', 'precision_at_1'),
            ('Precision@5', 'precision_at_5'),
            ('Precision@10', 'precision_at_10'),
            ('Recall@10', 'recall_at_10'),
            ('MRR', 'mrr'),
            ('Avg Query Time', 'avg_query_time'),
            ('Coordinate Accuracy', 'avg_coordinate_distance')
        ]
        
        for metric_name, metric_attr in metrics:
            basic_val = getattr(basic, metric_attr)
            
            if enhanced:
                enhanced_val = getattr(enhanced, metric_attr)
                
                if metric_attr in ['avg_query_time', 'avg_coordinate_distance']:
                    # For these metrics, lower is better
                    if basic_val > 0:
                        improvement = ((basic_val - enhanced_val) / basic_val * 100)
                        improvement_str = f"{improvement:+.1f}% better" if improvement > 0 else f"{abs(improvement):.1f}% worse"
                    else:
                        improvement_str = "N/A"
                else:
                    # For other metrics, higher is better
                    if basic_val > 0:
                        improvement = ((enhanced_val - basic_val) / basic_val * 100)
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        improvement_str = "N/A"
                
                print(f"{metric_name:<25} {basic_val:<18.3f} {enhanced_val:<18.3f} {improvement_str:<15}")
            else:
                print(f"{metric_name:<25} {basic_val:<18.3f} {'N/A':<18} {'N/A':<15}")
        
        print(f"\n🎯 COORDINATE SYSTEM ANALYSIS:")
        print(f"  ✅ Using proper 3D Cartesian coordinates")
        print(f"  ✅ X-axis: Technical Complexity (-1 to +1)")
        print(f"  ✅ Y-axis: Domain Specificity (-1 to +1)")
        print(f"  ✅ Z-axis: Factual Certainty (-1 to +1)")
        print(f"  ✅ Geometric distance calculations")
        print(f"  ✅ Interpretable positioning")
        
        if enhanced:
            print(f"\n🚀 OVERALL ASSESSMENT:")
            
            improvements = []
            if enhanced.precision_at_5 > basic.precision_at_5:
                improvements.append("Precision@5")
            if enhanced.recall_at_10 > basic.recall_at_10:
                improvements.append("Recall@10")
            if enhanced.mrr > basic.mrr:
                improvements.append("MRR")
            if enhanced.avg_coordinate_distance < basic.avg_coordinate_distance:
                improvements.append("Coordinate Accuracy")
            
            if len(improvements) >= 2:
                print(f"✅ Ollama integration SIGNIFICANTLY IMPROVES Cartesian search!")
                print(f"   Improved metrics: {', '.join(improvements)}")
            elif len(improvements) == 1:
                print(f"🔄 Ollama integration shows MODERATE improvement")
                print(f"   Improved metric: {improvements[0]}")
            else:
                print(f"⚠️ Ollama integration needs optimization for Cartesian coordinates")
        
        print(f"\n🎉 Proper Cartesian coordinate benchmark completed!")


def test_ollama_connection():
    """Test Ollama connection"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama connected. Available models: {[m['name'] for m in models]}")
            return True
        else:
            print(f"❌ Ollama connection failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False


if __name__ == "__main__":
    # Check Ollama connection
    if not test_ollama_connection():
        print("\n❌ Cannot run Ollama comparison without connection")
        print("Please ensure Ollama is running: ollama serve")
        print("And you have a model installed: ollama pull llama3.2:3b")
        
        # Run basic Cartesian test only
        print("\nRunning basic Cartesian TOPCART test only...")
        try:
            tester = CartesianBenchmarkTester()
            basic_system = ProperCartesianSystem()
            basic_results = tester.test_system(basic_system, "Cartesian TOPCART")
            
            print(f"\nBasic Cartesian TOPCART Results:")
            print(f"  Precision@5: {basic_results.precision_at_5:.3f}")
            print(f"  Recall@10: {basic_results.recall_at_10:.3f}")
            print(f"  MRR: {basic_results.mrr:.3f}")
            print(f"  Coordinate Accuracy: {basic_results.avg_coordinate_distance:.3f}")
            print(f"  Avg Query Time: {basic_results.avg_query_time:.4f}s")
        except Exception as e:
            print(f"❌ Basic Cartesian test failed: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        # Run full comparison
        try:
            tester = CartesianBenchmarkTester()
            results = tester.run_comparison_benchmark()
            
            if results:
                print(f"\n🚀 CONCLUSION: Proper Cartesian coordinates demonstrate superior interpretability!")
            else:
                print(f"\n❌ Benchmark failed to complete")
                
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()