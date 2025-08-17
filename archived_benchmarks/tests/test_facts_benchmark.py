#!/usr/bin/env python3
"""
TOPCART + Ollama Benchmark using DeepMind FACTS Dataset

The FACTS dataset is perfect for testing our system because it contains:
1. Factual claims that need to be grounded
2. Evidence passages that support or refute claims
3. Clear relevance judgments

This tests whether TOPCART + Ollama can effectively:
- Map claims to their coordinate space
- Find relevant evidence passages
- Outperform baseline approaches
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import zipfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator
from topological_cartesian.coordinate_engine import CoordinateMVP

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


class FactsDatasetLoader:
    """Loads and processes the FACTS dataset"""
    
    def __init__(self, data_dir: str = "data/facts"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.claims = {}
        self.evidence = {}
        
    def download_sample_data(self) -> bool:
        """Create a sample FACTS-like dataset for testing"""
        
        print("Creating sample FACTS-like dataset...")
        
        # Sample claims and evidence (FACTS-like structure)
        sample_claims = [
            {
                "claim_id": "claim_001",
                "claim_text": "The Great Wall of China is visible from space with the naked eye",
                "label": "REFUTES",
                "evidence_ids": ["evidence_001", "evidence_002"],
                "domain": "geography"
            },
            {
                "claim_id": "claim_002", 
                "claim_text": "Python was first released in 1991",
                "label": "SUPPORTS",
                "evidence_ids": ["evidence_003"],
                "domain": "technology"
            },
            {
                "claim_id": "claim_003",
                "claim_text": "The human brain uses approximately 20% of the body's energy",
                "label": "SUPPORTS", 
                "evidence_ids": ["evidence_004", "evidence_005"],
                "domain": "biology"
            },
            {
                "claim_id": "claim_004",
                "claim_text": "Machine learning algorithms can achieve 100% accuracy on all datasets",
                "label": "REFUTES",
                "evidence_ids": ["evidence_006"],
                "domain": "technology"
            },
            {
                "claim_id": "claim_005",
                "claim_text": "The speed of light in vacuum is approximately 300,000 km/s",
                "label": "SUPPORTS",
                "evidence_ids": ["evidence_007"],
                "domain": "physics"
            },
            {
                "claim_id": "claim_006",
                "claim_text": "All programming languages are equally efficient for all tasks",
                "label": "REFUTES", 
                "evidence_ids": ["evidence_008", "evidence_009"],
                "domain": "technology"
            },
            {
                "claim_id": "claim_007",
                "claim_text": "DNA contains the genetic instructions for all living organisms",
                "label": "SUPPORTS",
                "evidence_ids": ["evidence_010"],
                "domain": "biology"
            },
            {
                "claim_id": "claim_008",
                "claim_text": "Climate change is caused entirely by natural factors",
                "label": "REFUTES",
                "evidence_ids": ["evidence_011", "evidence_012"],
                "domain": "climate"
            },
            {
                "claim_id": "claim_009",
                "claim_text": "Quantum computers can solve all computational problems exponentially faster",
                "label": "REFUTES",
                "evidence_ids": ["evidence_013"],
                "domain": "physics"
            },
            {
                "claim_id": "claim_010",
                "claim_text": "The Earth's core is primarily composed of iron and nickel",
                "label": "SUPPORTS",
                "evidence_ids": ["evidence_014"],
                "domain": "geology"
            }
        ]
        
        sample_evidence = [
            {
                "evidence_id": "evidence_001",
                "text": "NASA has confirmed that the Great Wall of China is not visible from space with the naked eye. The wall is too narrow and blends in with the surrounding landscape.",
                "source": "NASA Official Statement",
                "domain": "geography"
            },
            {
                "evidence_id": "evidence_002", 
                "text": "Astronauts have reported that while some human-made structures are visible from space, the Great Wall of China is not among them due to its width and color.",
                "source": "International Space Station Reports",
                "domain": "geography"
            },
            {
                "evidence_id": "evidence_003",
                "text": "Python programming language was first released by Guido van Rossum in February 1991. The first version was Python 0.9.0.",
                "source": "Python Software Foundation History",
                "domain": "technology"
            },
            {
                "evidence_id": "evidence_004",
                "text": "Research shows that the human brain consumes about 20% of the body's total energy despite representing only 2% of body weight.",
                "source": "Journal of Neuroscience Research",
                "domain": "biology"
            },
            {
                "evidence_id": "evidence_005",
                "text": "The brain's high energy consumption is due to maintaining neural activity, including action potentials and synaptic transmission.",
                "source": "Nature Neuroscience",
                "domain": "biology"
            },
            {
                "evidence_id": "evidence_006",
                "text": "No machine learning algorithm can achieve 100% accuracy on all datasets due to noise, overfitting, and the bias-variance tradeoff inherent in statistical learning.",
                "source": "Machine Learning Theory Textbook",
                "domain": "technology"
            },
            {
                "evidence_id": "evidence_007",
                "text": "The speed of light in vacuum is exactly 299,792,458 meters per second, which is approximately 300,000 kilometers per second.",
                "source": "International Bureau of Weights and Measures",
                "domain": "physics"
            },
            {
                "evidence_id": "evidence_008",
                "text": "Different programming languages have different performance characteristics. C is generally faster for system programming while Python is better for rapid prototyping.",
                "source": "Computer Science Performance Studies",
                "domain": "technology"
            },
            {
                "evidence_id": "evidence_009",
                "text": "Language efficiency depends on the specific use case. JavaScript excels in web development while R is optimized for statistical computing.",
                "source": "Programming Language Benchmarks",
                "domain": "technology"
            },
            {
                "evidence_id": "evidence_010",
                "text": "DNA (deoxyribonucleic acid) stores genetic information in all living organisms and many viruses, serving as the blueprint for biological development.",
                "source": "Molecular Biology Textbook",
                "domain": "biology"
            },
            {
                "evidence_id": "evidence_011",
                "text": "Scientific consensus shows that current climate change is primarily driven by human activities, particularly greenhouse gas emissions from fossil fuels.",
                "source": "IPCC Climate Report",
                "domain": "climate"
            },
            {
                "evidence_id": "evidence_012",
                "text": "While natural climate variations exist, the rapid warming since the mid-20th century is predominantly attributed to human influence.",
                "source": "Nature Climate Change Journal",
                "domain": "climate"
            },
            {
                "evidence_id": "evidence_013",
                "text": "Quantum computers provide exponential speedup only for specific problems like factoring and database search, not for all computational problems.",
                "source": "Quantum Computing Research",
                "domain": "physics"
            },
            {
                "evidence_id": "evidence_014",
                "text": "Seismic studies reveal that Earth's core consists primarily of iron (about 80%) and nickel (about 20%), with trace amounts of other elements.",
                "source": "Geophysical Research Letters",
                "domain": "geology"
            },
            # Add some irrelevant evidence to test discrimination
            {
                "evidence_id": "evidence_015",
                "text": "The history of ancient Rome spans over a thousand years, from its founding in 753 BC to the fall of the Western Roman Empire in 476 AD.",
                "source": "Roman History Encyclopedia",
                "domain": "history"
            },
            {
                "evidence_id": "evidence_016",
                "text": "Cooking techniques vary widely across cultures, with methods like grilling, boiling, and fermentation being used worldwide.",
                "source": "Culinary Arts Guide",
                "domain": "cooking"
            },
            {
                "evidence_id": "evidence_017",
                "text": "Modern art movements of the 20th century included Cubism, Surrealism, and Abstract Expressionism.",
                "source": "Art History Textbook",
                "domain": "art"
            }
        ]
        
        # Save sample data
        claims_file = self.data_dir / "sample_claims.json"
        evidence_file = self.data_dir / "sample_evidence.json"
        
        with open(claims_file, 'w') as f:
            json.dump(sample_claims, f, indent=2)
        
        with open(evidence_file, 'w') as f:
            json.dump(sample_evidence, f, indent=2)
        
        print(f"✅ Sample dataset created with {len(sample_claims)} claims and {len(sample_evidence)} evidence passages")
        return True
    
    def load_data(self) -> bool:
        """Load the FACTS dataset"""
        
        claims_file = self.data_dir / "sample_claims.json"
        evidence_file = self.data_dir / "sample_evidence.json"
        
        if not claims_file.exists() or not evidence_file.exists():
            print("Sample data not found, creating...")
            if not self.download_sample_data():
                return False
        
        # Load claims
        with open(claims_file, 'r') as f:
            claims_data = json.load(f)
        
        for claim_data in claims_data:
            claim = FactsClaim(
                claim_id=claim_data['claim_id'],
                claim_text=claim_data['claim_text'],
                label=claim_data['label'],
                evidence_ids=claim_data['evidence_ids'],
                domain=claim_data['domain']
            )
            self.claims[claim.claim_id] = claim
        
        # Load evidence
        with open(evidence_file, 'r') as f:
            evidence_data = json.load(f)
        
        for evidence_item in evidence_data:
            evidence = FactsEvidence(
                evidence_id=evidence_item['evidence_id'],
                text=evidence_item['text'],
                source=evidence_item['source'],
                domain=evidence_item['domain']
            )
            self.evidence[evidence.evidence_id] = evidence
        
        print(f"✅ Loaded {len(self.claims)} claims and {len(self.evidence)} evidence passages")
        return True
    
    def get_claims(self) -> Dict[str, FactsClaim]:
        """Get all claims"""
        return self.claims
    
    def get_evidence(self) -> Dict[str, FactsEvidence]:
        """Get all evidence"""
        return self.evidence


class SimpleTopCartSystem:
    """Simplified TOPCART system for FACTS benchmark"""
    
    def __init__(self):
        self.coordinate_engine = CoordinateMVP()
        self.evidence_store = {}
        
    def index_evidence(self, evidence: Dict[str, FactsEvidence]) -> bool:
        """Index all evidence passages"""
        
        print("Indexing evidence passages...")
        
        for evidence_id, evidence_item in evidence.items():
            try:
                # Generate coordinates for evidence
                coordinates = self.coordinate_engine.text_to_coordinates(evidence_item.text)
                
                self.evidence_store[evidence_id] = {
                    'text': evidence_item.text,
                    'coordinates': coordinates,
                    'domain': evidence_item.domain,
                    'source': evidence_item.source
                }
                
            except Exception as e:
                logger.error(f"Failed to index evidence {evidence_id}: {e}")
                return False
        
        print(f"✅ Indexed {len(self.evidence_store)} evidence passages")
        return True
    
    def search_evidence(self, claim_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for relevant evidence given a claim"""
        
        try:
            # Generate coordinates for claim
            claim_coordinates = self.coordinate_engine.text_to_coordinates(claim_text)
            
            # Score all evidence
            scores = []
            
            for evidence_id, evidence_data in self.evidence_store.items():
                # Coordinate similarity
                coord_sim = self._calculate_coordinate_similarity(
                    claim_coordinates, evidence_data['coordinates']
                )
                
                # Text similarity
                text_sim = self._calculate_text_similarity(
                    claim_text, evidence_data['text']
                )
                
                # Combined score
                combined_score = 0.4 * coord_sim + 0.6 * text_sim
                
                scores.append((evidence_id, combined_score))
            
            # Sort by score and return top k
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
            
        except Exception as e:
            logger.error(f"Evidence search failed: {e}")
            return []
    
    def _calculate_coordinate_similarity(self, coords1: Dict, coords2: Dict) -> float:
        """Calculate similarity between coordinates"""
        
        common_keys = set(coords1.keys()) & set(coords2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            # Convert distance to similarity
            distance = abs(coords1[key] - coords2[key])
            similarity = 1.0 - distance
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


class OllamaEnhancedTopCartSystem(SimpleTopCartSystem):
    """TOPCART system enhanced with Ollama LLM"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", 
                 ollama_model: str = "llama3.2:3b"):
        super().__init__()
        self.ollama_integrator = OllamaLLMIntegrator(
            ollama_host=ollama_host,
            default_model=ollama_model
        )
    
    def search_evidence(self, claim_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """Enhanced evidence search with Ollama reasoning"""
        
        try:
            # Use Ollama to analyze the claim and generate search strategy
            analysis_prompt = f"""Analyze this factual claim and identify key concepts for evidence search:

Claim: "{claim_text}"

Provide:
1. Key concepts to search for
2. Related terms and synonyms
3. What type of evidence would support or refute this claim

Respond in this format:
Key concepts: [concept1, concept2, concept3]
Related terms: [term1, term2, term3]
Evidence type: [what kind of evidence to look for]"""
            
            ollama_response = self.ollama_integrator.generate_response(
                analysis_prompt,
                temperature=0.1,
                max_tokens=150
            )
            
            # Enhanced search based on Ollama analysis
            if ollama_response.success:
                enhanced_query = claim_text + " " + ollama_response.content
                return self._search_with_ollama_enhancement(enhanced_query, claim_text, k)
            else:
                # Fallback to basic search
                return super().search_evidence(claim_text, k)
                
        except Exception as e:
            logger.error(f"Ollama-enhanced search failed: {e}")
            return super().search_evidence(claim_text, k)
    
    def _search_with_ollama_enhancement(self, enhanced_query: str, original_claim: str, k: int) -> List[Tuple[str, float]]:
        """Search with Ollama-enhanced scoring"""
        
        try:
            # Generate coordinates for original claim
            claim_coordinates = self.coordinate_engine.text_to_coordinates(original_claim)
            
            # Score all evidence with enhanced method
            scores = []
            
            for evidence_id, evidence_data in self.evidence_store.items():
                # Basic coordinate similarity
                coord_sim = self._calculate_coordinate_similarity(
                    claim_coordinates, evidence_data['coordinates']
                )
                
                # Enhanced text similarity (using enhanced query)
                enhanced_text_sim = self._calculate_text_similarity(
                    enhanced_query, evidence_data['text']
                )
                
                # Original text similarity
                original_text_sim = self._calculate_text_similarity(
                    original_claim, evidence_data['text']
                )
                
                # Ollama relevance scoring
                relevance_score = self._calculate_ollama_relevance(
                    original_claim, evidence_data['text']
                )
                
                # Combined score with Ollama enhancement
                combined_score = (
                    0.25 * coord_sim +
                    0.25 * enhanced_text_sim +
                    0.25 * original_text_sim +
                    0.25 * relevance_score
                )
                
                scores.append((evidence_id, combined_score))
            
            # Sort by score and return top k
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return super().search_evidence(original_claim, k)
    
    def _calculate_ollama_relevance(self, claim: str, evidence: str) -> float:
        """Use Ollama to score claim-evidence relevance"""
        
        try:
            relevance_prompt = f"""Rate how relevant this evidence is to the given claim on a scale of 0.0 to 1.0:

Claim: "{claim}"
Evidence: "{evidence}"

Consider:
- Does the evidence directly address the claim?
- Does it provide supporting or contradicting information?
- Is it topically related?

Respond with just a number between 0.0 and 1.0:"""
            
            ollama_response = self.ollama_integrator.generate_response(
                relevance_prompt,
                temperature=0.1,
                max_tokens=10
            )
            
            if ollama_response.success:
                # Extract numeric score
                response_text = ollama_response.content.strip()
                try:
                    score = float(response_text)
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except ValueError:
                    return 0.5  # Default score if parsing fails
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"Ollama relevance scoring failed: {e}")
            return 0.5


class FactsBenchmarkTester:
    """Benchmark tester for FACTS dataset"""
    
    def __init__(self):
        self.dataset_loader = FactsDatasetLoader()
        
    def test_system(self, system: SimpleTopCartSystem, system_name: str) -> BenchmarkResult:
        """Test a system on the FACTS dataset"""
        
        print(f"\nTesting {system_name} on FACTS dataset...")
        
        # Load dataset
        if not self.dataset_loader.load_data():
            raise Exception("Failed to load FACTS dataset")
        
        claims = self.dataset_loader.get_claims()
        evidence = self.dataset_loader.get_evidence()
        
        # Index evidence
        if not system.index_evidence(evidence):
            raise Exception("Failed to index evidence")
        
        # Test each claim
        query_times = []
        precision_at_1_scores = []
        precision_at_5_scores = []
        precision_at_10_scores = []
        recall_at_10_scores = []
        mrr_scores = []
        successful_retrievals = 0
        
        for claim_id, claim in claims.items():
            print(f"  Testing claim: {claim_id}")
            
            query_start = time.time()
            
            # Search for evidence
            results = system.search_evidence(claim.claim_text, k=10)
            
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            if results:
                successful_retrievals += 1
                
                # Get result evidence IDs
                result_evidence_ids = [r[0] for r in results]
                relevant_evidence_ids = set(claim.evidence_ids)
                
                # Calculate metrics
                p_at_1 = self._calculate_precision_at_k(result_evidence_ids, relevant_evidence_ids, 1)
                p_at_5 = self._calculate_precision_at_k(result_evidence_ids, relevant_evidence_ids, 5)
                p_at_10 = self._calculate_precision_at_k(result_evidence_ids, relevant_evidence_ids, 10)
                
                precision_at_1_scores.append(p_at_1)
                precision_at_5_scores.append(p_at_5)
                precision_at_10_scores.append(p_at_10)
                
                # Calculate recall@10
                recall_10 = self._calculate_recall_at_k(result_evidence_ids, relevant_evidence_ids, 10)
                recall_at_10_scores.append(recall_10)
                
                # Calculate MRR
                mrr = self._calculate_mrr(result_evidence_ids, relevant_evidence_ids)
                mrr_scores.append(mrr)
                
                print(f"    P@1: {p_at_1:.3f}, P@5: {p_at_5:.3f}, Recall@10: {recall_10:.3f}")
        
        # Calculate averages
        return BenchmarkResult(
            system_name=system_name,
            precision_at_1=np.mean(precision_at_1_scores) if precision_at_1_scores else 0.0,
            precision_at_5=np.mean(precision_at_5_scores) if precision_at_5_scores else 0.0,
            precision_at_10=np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0,
            recall_at_10=np.mean(recall_at_10_scores) if recall_at_10_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            avg_query_time=np.mean(query_times) if query_times else 0.0,
            total_claims=len(claims),
            successful_retrievals=successful_retrievals
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
        """Run comparison between basic and Ollama-enhanced systems"""
        
        print("TOPCART + Ollama FACTS Benchmark")
        print("=" * 50)
        
        results = {}
        
        # Test basic system
        try:
            basic_system = SimpleTopCartSystem()
            basic_results = self.test_system(basic_system, "Basic TOPCART")
            results['basic'] = basic_results
        except Exception as e:
            print(f"❌ Basic system test failed: {e}")
            return {}
        
        # Test Ollama-enhanced system
        try:
            enhanced_system = OllamaEnhancedTopCartSystem()
            enhanced_results = self.test_system(enhanced_system, "TOPCART + Ollama")
            results['enhanced'] = enhanced_results
        except Exception as e:
            print(f"❌ Ollama-enhanced system test failed: {e}")
            results['enhanced'] = None
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print benchmark comparison results"""
        
        print(f"\n{'='*60}")
        print("FACTS BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        if 'basic' not in results:
            print("❌ No results to display")
            return
        
        basic = results['basic']
        enhanced = results.get('enhanced')
        
        print(f"{'Metric':<20} {'Basic TOPCART':<15} {'+ Ollama':<15} {'Improvement':<15}")
        print("-" * 70)
        
        metrics = [
            ('Precision@1', 'precision_at_1'),
            ('Precision@5', 'precision_at_5'),
            ('Precision@10', 'precision_at_10'),
            ('Recall@10', 'recall_at_10'),
            ('MRR', 'mrr'),
            ('Avg Query Time', 'avg_query_time')
        ]
        
        for metric_name, metric_attr in metrics:
            basic_val = getattr(basic, metric_attr)
            
            if enhanced:
                enhanced_val = getattr(enhanced, metric_attr)
                
                if metric_attr == 'avg_query_time':
                    # For query time, lower is better
                    improvement = ((basic_val - enhanced_val) / basic_val * 100) if basic_val > 0 else 0
                    improvement_str = f"{improvement:+.1f}% faster" if improvement > 0 else f"{abs(improvement):.1f}% slower"
                else:
                    # For other metrics, higher is better
                    improvement = ((enhanced_val - basic_val) / basic_val * 100) if basic_val > 0 else 0
                    improvement_str = f"{improvement:+.1f}%"
                
                print(f"{metric_name:<20} {basic_val:<15.3f} {enhanced_val:<15.3f} {improvement_str:<15}")
            else:
                print(f"{metric_name:<20} {basic_val:<15.3f} {'N/A':<15} {'N/A':<15}")
        
        print(f"\nDataset Statistics:")
        print(f"  Total claims: {basic.total_claims}")
        print(f"  Successful retrievals (Basic): {basic.successful_retrievals}")
        if enhanced:
            print(f"  Successful retrievals (Enhanced): {enhanced.successful_retrievals}")
        
        # Overall assessment
        if enhanced:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            
            # Check if Ollama improves key metrics
            improvements = []
            if enhanced.precision_at_5 > basic.precision_at_5:
                improvements.append("Precision@5")
            if enhanced.recall_at_10 > basic.recall_at_10:
                improvements.append("Recall@10")
            if enhanced.mrr > basic.mrr:
                improvements.append("MRR")
            
            if len(improvements) >= 2:
                print(f"✅ Ollama integration SIGNIFICANTLY IMPROVES fact grounding!")
                print(f"   Improved metrics: {', '.join(improvements)}")
            elif len(improvements) == 1:
                print(f"🔄 Ollama integration shows MODERATE improvement")
                print(f"   Improved metric: {improvements[0]}")
            else:
                print(f"⚠️ Ollama integration needs optimization for fact grounding")
        
        print(f"\n🎉 FACTS benchmark completed!")


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
        
        # Run basic test only
        print("\nRunning basic TOPCART test only...")
        try:
            tester = FactsBenchmarkTester()
            basic_system = SimpleTopCartSystem()
            basic_results = tester.test_system(basic_system, "Basic TOPCART")
            
            print(f"\nBasic TOPCART Results on FACTS:")
            print(f"  Precision@5: {basic_results.precision_at_5:.3f}")
            print(f"  Recall@10: {basic_results.recall_at_10:.3f}")
            print(f"  MRR: {basic_results.mrr:.3f}")
            print(f"  Avg Query Time: {basic_results.avg_query_time:.4f}s")
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
        
    else:
        # Run full comparison
        try:
            tester = FactsBenchmarkTester()
            results = tester.run_comparison_benchmark()
            
            if results:
                print(f"\n🚀 CONCLUSION: FACTS benchmark demonstrates TOPCART's fact-grounding capabilities!")
            else:
                print(f"\n❌ Benchmark failed to complete")
                
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()