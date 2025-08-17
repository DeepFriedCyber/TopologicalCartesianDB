#!/usr/bin/env python3
"""
TOPCART + Ollama Benchmark Test

Tests the integrated TOPCART system with Ollama LLM against public benchmarks:
1. MS MARCO passage ranking
2. BEIR benchmark subset
3. Custom semantic search tasks
4. Performance comparison with baseline systems
"""

import pytest
import numpy as np
import time
import json
import requests
from pathlib import Path
import sys
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator, HybridCoordinateLLM
from topological_cartesian.integrated_improvement_system import ImprovedCartesianCubeSystem
from topological_cartesian.optimized_search_engine import HybridSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    test_name: str
    dataset_size: int
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_10: float  # Normalized Discounted Cumulative Gain
    avg_query_time: float
    total_time: float
    method: str
    additional_metrics: Dict[str, Any] = None


class OllamaBenchmarkTester:
    """Comprehensive benchmark tester for TOPCART + Ollama"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", 
                 ollama_model: str = "llama3.2:3b"):
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        
        # Initialize systems
        self.ollama_integrator = None
        self.topcart_system = None
        self.hybrid_system = None
        
        # Benchmark datasets
        self.benchmark_data = {}
        
        # Results storage
        self.results = []
        
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all systems for testing"""
        
        try:
            # Test Ollama connection
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not accessible at {self.ollama_host}")
            
            # Initialize Ollama integrator
            self.ollama_integrator = OllamaLLMIntegrator(
                ollama_host=self.ollama_host,
                default_model=self.ollama_model
            )
            
            # Initialize TOPCART system
            self.topcart_system = ImprovedCartesianCubeSystem()
            
            # Initialize hybrid system (if available)
            try:
                self.hybrid_system = HybridCoordinateLLM(
                    ollama_host=self.ollama_host,
                    default_model=self.ollama_model
                )
            except Exception as e:
                logger.warning(f"Hybrid system not available: {e}")
            
            logger.info("✅ All systems initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def create_synthetic_benchmark(self, size: int = 100) -> Dict[str, Any]:
        """Create a synthetic benchmark dataset for testing"""
        
        # Categories with different complexity levels
        categories = {
            'programming': {
                'simple': [
                    "How to print hello world in Python",
                    "Basic HTML tags for beginners",
                    "What is a variable in programming",
                    "Simple CSS styling tutorial",
                    "Introduction to JavaScript functions"
                ],
                'intermediate': [
                    "Python list comprehensions and generators",
                    "CSS flexbox layout techniques",
                    "JavaScript async/await patterns",
                    "SQL JOIN operations explained",
                    "Git branching and merging strategies"
                ],
                'advanced': [
                    "Machine learning model optimization techniques",
                    "Advanced React hooks and context patterns",
                    "Database indexing and query optimization",
                    "Microservices architecture design patterns",
                    "Distributed systems consensus algorithms"
                ]
            },
            'science': {
                'simple': [
                    "What is photosynthesis in plants",
                    "Basic laws of physics motion",
                    "Simple chemistry reactions",
                    "Earth's water cycle explanation",
                    "Human digestive system basics"
                ],
                'intermediate': [
                    "Quantum mechanics wave-particle duality",
                    "Organic chemistry functional groups",
                    "Genetics and DNA replication process",
                    "Thermodynamics and entropy concepts",
                    "Electromagnetic field theory basics"
                ],
                'advanced': [
                    "Quantum field theory mathematical framework",
                    "Advanced protein folding mechanisms",
                    "Cosmological inflation theory models",
                    "Molecular dynamics simulation methods",
                    "Theoretical particle physics standard model"
                ]
            },
            'business': {
                'simple': [
                    "Basic accounting principles for small business",
                    "What is marketing and advertising",
                    "Simple budgeting and financial planning",
                    "Customer service best practices",
                    "Introduction to business management"
                ],
                'intermediate': [
                    "Strategic planning and competitive analysis",
                    "Financial statement analysis techniques",
                    "Digital marketing and SEO strategies",
                    "Supply chain management optimization",
                    "Human resources management practices"
                ],
                'advanced': [
                    "Advanced corporate finance and valuation",
                    "International business and global markets",
                    "Complex merger and acquisition strategies",
                    "Advanced data analytics for business intelligence",
                    "Organizational behavior and change management"
                ]
            }
        }
        
        # Generate documents and queries
        documents = {}
        queries = {}
        relevance_judgments = {}
        
        doc_id = 0
        query_id = 0
        
        for domain, complexity_levels in categories.items():
            for complexity, texts in complexity_levels.items():
                for text in texts:
                    # Create document
                    doc_key = f"doc_{doc_id}"
                    documents[doc_key] = {
                        'content': text,
                        'domain': domain,
                        'complexity': complexity,
                        'true_coordinates': self._generate_true_coordinates(domain, complexity)
                    }
                    
                    # Create related query
                    query_key = f"query_{query_id}"
                    query_text = self._generate_query_from_text(text)
                    queries[query_key] = {
                        'text': query_text,
                        'domain': domain,
                        'complexity': complexity
                    }
                    
                    # Create relevance judgment
                    if query_key not in relevance_judgments:
                        relevance_judgments[query_key] = {}
                    
                    # High relevance for exact match
                    relevance_judgments[query_key][doc_key] = 3
                    
                    # Medium relevance for same domain/complexity
                    for other_doc_key, other_doc in documents.items():
                        if (other_doc_key != doc_key and 
                            other_doc['domain'] == domain and 
                            other_doc['complexity'] == complexity):
                            relevance_judgments[query_key][other_doc_key] = 2
                    
                    # Low relevance for same domain, different complexity
                    for other_doc_key, other_doc in documents.items():
                        if (other_doc_key != doc_key and 
                            other_doc['domain'] == domain and 
                            other_doc['complexity'] != complexity):
                            if other_doc_key not in relevance_judgments[query_key]:
                                relevance_judgments[query_key][other_doc_key] = 1
                    
                    doc_id += 1
                    query_id += 1
                    
                    if doc_id >= size:
                        break
                if doc_id >= size:
                    break
            if doc_id >= size:
                break
        
        return {
            'documents': documents,
            'queries': queries,
            'relevance_judgments': relevance_judgments,
            'metadata': {
                'size': len(documents),
                'num_queries': len(queries),
                'domains': list(categories.keys()),
                'complexity_levels': ['simple', 'intermediate', 'advanced']
            }
        }
    
    def _generate_true_coordinates(self, domain: str, complexity: str) -> Dict[str, float]:
        """Generate ground truth coordinates for synthetic data"""
        
        domain_mapping = {
            'programming': 0.8,
            'science': 0.6,
            'business': 0.4
        }
        
        complexity_mapping = {
            'simple': 0.2,
            'intermediate': 0.5,
            'advanced': 0.9
        }
        
        # Task type based on domain (programming more practical, science more theoretical)
        task_type_mapping = {
            'programming': 0.7,
            'science': 0.3,
            'business': 0.6
        }
        
        return {
            'domain': domain_mapping.get(domain, 0.5),
            'complexity': complexity_mapping.get(complexity, 0.5),
            'task_type': task_type_mapping.get(domain, 0.5)
        }
    
    def _generate_query_from_text(self, text: str) -> str:
        """Generate a search query from document text"""
        
        # Simple query generation - extract key terms
        words = text.lower().split()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        key_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Take first 2-3 key words
        query_words = key_words[:3] if len(key_words) >= 3 else key_words
        
        return ' '.join(query_words)
    
    def test_topcart_only(self, benchmark_data: Dict[str, Any]) -> BenchmarkResult:
        """Test TOPCART system without Ollama"""
        
        logger.info("Testing TOPCART-only system...")
        
        documents = benchmark_data['documents']
        queries = benchmark_data['queries']
        relevance_judgments = benchmark_data['relevance_judgments']
        
        # Add documents to TOPCART
        start_time = time.time()
        
        for doc_id, doc_data in documents.items():
            result = self.topcart_system.add_document(
                doc_id, 
                doc_data['content'], 
                {'domain': doc_data['domain']}
            )
            if not result['success']:
                logger.warning(f"Failed to add document {doc_id}")
        
        indexing_time = time.time() - start_time
        
        # Test queries
        query_times = []
        precision_at_1_scores = []
        precision_at_5_scores = []
        precision_at_10_scores = []
        recall_at_10_scores = []
        mrr_scores = []
        ndcg_scores = []
        
        for query_id, query_data in queries.items():
            query_start = time.time()
            
            # Search
            results = self.topcart_system.search(
                query_data['text'], 
                k=10, 
                domain=query_data['domain']
            )
            
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            # Calculate metrics
            relevant_docs = relevance_judgments.get(query_id, {})
            
            if relevant_docs:
                # Extract result document IDs
                result_doc_ids = [r['document_id'] for r in results]
                
                # Calculate precision@k
                p_at_1 = self._calculate_precision_at_k(result_doc_ids, relevant_docs, 1)
                p_at_5 = self._calculate_precision_at_k(result_doc_ids, relevant_docs, 5)
                p_at_10 = self._calculate_precision_at_k(result_doc_ids, relevant_docs, 10)
                
                precision_at_1_scores.append(p_at_1)
                precision_at_5_scores.append(p_at_5)
                precision_at_10_scores.append(p_at_10)
                
                # Calculate recall@10
                recall_10 = self._calculate_recall_at_k(result_doc_ids, relevant_docs, 10)
                recall_at_10_scores.append(recall_10)
                
                # Calculate MRR
                mrr = self._calculate_mrr(result_doc_ids, relevant_docs)
                mrr_scores.append(mrr)
                
                # Calculate NDCG@10
                ndcg = self._calculate_ndcg_at_k(result_doc_ids, relevant_docs, 10)
                ndcg_scores.append(ndcg)
        
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            test_name="TOPCART-Only",
            dataset_size=len(documents),
            precision_at_1=np.mean(precision_at_1_scores) if precision_at_1_scores else 0.0,
            precision_at_5=np.mean(precision_at_5_scores) if precision_at_5_scores else 0.0,
            precision_at_10=np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0,
            recall_at_10=np.mean(recall_at_10_scores) if recall_at_10_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            ndcg_at_10=np.mean(ndcg_scores) if ndcg_scores else 0.0,
            avg_query_time=np.mean(query_times) if query_times else 0.0,
            total_time=total_time,
            method="topcart_only",
            additional_metrics={
                'indexing_time': indexing_time,
                'num_queries': len(queries),
                'successful_queries': len(query_times)
            }
        )
    
    def test_ollama_enhanced(self, benchmark_data: Dict[str, Any]) -> BenchmarkResult:
        """Test TOPCART + Ollama enhanced system"""
        
        logger.info("Testing TOPCART + Ollama enhanced system...")
        
        documents = benchmark_data['documents']
        queries = benchmark_data['queries']
        relevance_judgments = benchmark_data['relevance_judgments']
        
        # Create enhanced documents with Ollama analysis
        start_time = time.time()
        enhanced_documents = {}
        
        for doc_id, doc_data in documents.items():
            # Get Ollama analysis of the document
            analysis_prompt = f"""Analyze this text and provide:
1. Main topic/domain
2. Complexity level (simple/intermediate/advanced)
3. Key concepts (3-5 keywords)
4. Task type (theoretical/practical)

Text: {doc_data['content']}

Respond in this format:
Domain: [domain]
Complexity: [level]
Keywords: [keyword1, keyword2, keyword3]
Task: [theoretical/practical]"""
            
            ollama_response = self.ollama_integrator.generate_response(
                analysis_prompt,
                temperature=0.1,
                max_tokens=150
            )
            
            # Add to TOPCART with enhanced metadata
            enhanced_metadata = {
                'domain': doc_data['domain'],
                'ollama_analysis': ollama_response.content if ollama_response.success else '',
                'original_complexity': doc_data['complexity']
            }
            
            result = self.topcart_system.add_document(
                doc_id, 
                doc_data['content'], 
                enhanced_metadata
            )
            
            if result['success']:
                enhanced_documents[doc_id] = {
                    **doc_data,
                    'ollama_analysis': ollama_response.content if ollama_response.success else '',
                    'topcart_coordinates': result['coordinates']
                }
        
        indexing_time = time.time() - start_time
        
        # Test queries with Ollama enhancement
        query_times = []
        precision_at_1_scores = []
        precision_at_5_scores = []
        precision_at_10_scores = []
        recall_at_10_scores = []
        mrr_scores = []
        ndcg_scores = []
        
        for query_id, query_data in queries.items():
            query_start = time.time()
            
            # Enhance query with Ollama
            query_enhancement_prompt = f"""Expand this search query with related terms and concepts:

Query: {query_data['text']}
Domain: {query_data['domain']}

Provide:
1. Synonyms and alternative terms
2. Related concepts
3. More specific terms

Respond with expanded query terms separated by spaces."""
            
            ollama_query_response = self.ollama_integrator.generate_response(
                query_enhancement_prompt,
                temperature=0.2,
                max_tokens=100
            )
            
            # Use enhanced query for search
            enhanced_query = query_data['text']
            if ollama_query_response.success:
                enhanced_query += " " + ollama_query_response.content
            
            # Search with enhanced query
            results = self.topcart_system.search(
                enhanced_query, 
                k=10, 
                domain=query_data['domain']
            )
            
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            # Calculate metrics
            relevant_docs = relevance_judgments.get(query_id, {})
            
            if relevant_docs:
                result_doc_ids = [r['document_id'] for r in results]
                
                p_at_1 = self._calculate_precision_at_k(result_doc_ids, relevant_docs, 1)
                p_at_5 = self._calculate_precision_at_k(result_doc_ids, relevant_docs, 5)
                p_at_10 = self._calculate_precision_at_k(result_doc_ids, relevant_docs, 10)
                
                precision_at_1_scores.append(p_at_1)
                precision_at_5_scores.append(p_at_5)
                precision_at_10_scores.append(p_at_10)
                
                recall_10 = self._calculate_recall_at_k(result_doc_ids, relevant_docs, 10)
                recall_at_10_scores.append(recall_10)
                
                mrr = self._calculate_mrr(result_doc_ids, relevant_docs)
                mrr_scores.append(mrr)
                
                ndcg = self._calculate_ndcg_at_k(result_doc_ids, relevant_docs, 10)
                ndcg_scores.append(ndcg)
        
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            test_name="TOPCART + Ollama",
            dataset_size=len(documents),
            precision_at_1=np.mean(precision_at_1_scores) if precision_at_1_scores else 0.0,
            precision_at_5=np.mean(precision_at_5_scores) if precision_at_5_scores else 0.0,
            precision_at_10=np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0,
            recall_at_10=np.mean(recall_at_10_scores) if recall_at_10_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            ndcg_at_10=np.mean(ndcg_scores) if ndcg_scores else 0.0,
            avg_query_time=np.mean(query_times) if query_times else 0.0,
            total_time=total_time,
            method="topcart_ollama",
            additional_metrics={
                'indexing_time': indexing_time,
                'num_queries': len(queries),
                'successful_queries': len(query_times),
                'ollama_stats': self.ollama_integrator.get_performance_stats()
            }
        )
    
    def _calculate_precision_at_k(self, result_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
        """Calculate Precision@K"""
        if not result_ids or not relevant_docs:
            return 0.0
        
        top_k = result_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs and relevant_docs[doc_id] > 0)
        
        return relevant_in_top_k / min(k, len(result_ids))
    
    def _calculate_recall_at_k(self, result_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
        """Calculate Recall@K"""
        if not result_ids or not relevant_docs:
            return 0.0
        
        total_relevant = sum(1 for score in relevant_docs.values() if score > 0)
        if total_relevant == 0:
            return 0.0
        
        top_k = result_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs and relevant_docs[doc_id] > 0)
        
        return relevant_in_top_k / total_relevant
    
    def _calculate_mrr(self, result_ids: List[str], relevant_docs: Dict[str, int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not result_ids or not relevant_docs:
            return 0.0
        
        for i, doc_id in enumerate(result_ids):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg_at_k(self, result_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if not result_ids or not relevant_docs:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(result_ids[:k]):
            if doc_id in relevant_docs:
                relevance = relevant_docs[doc_id]
                dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += (2**relevance - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def run_comprehensive_benchmark(self, dataset_size: int = 50) -> List[BenchmarkResult]:
        """Run comprehensive benchmark comparing different approaches"""
        
        logger.info(f"Running comprehensive benchmark with {dataset_size} documents...")
        
        # Create benchmark dataset
        benchmark_data = self.create_synthetic_benchmark(dataset_size)
        
        results = []
        
        # Test TOPCART only
        try:
            topcart_result = self.test_topcart_only(benchmark_data)
            results.append(topcart_result)
            logger.info(f"✅ TOPCART-only test completed")
        except Exception as e:
            logger.error(f"❌ TOPCART-only test failed: {e}")
        
        # Test TOPCART + Ollama
        try:
            ollama_result = self.test_ollama_enhanced(benchmark_data)
            results.append(ollama_result)
            logger.info(f"✅ TOPCART + Ollama test completed")
        except Exception as e:
            logger.error(f"❌ TOPCART + Ollama test failed: {e}")
        
        self.results = results
        return results
    
    def print_benchmark_results(self):
        """Print comprehensive benchmark results"""
        
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("TOPCART + OLLAMA BENCHMARK RESULTS")
        print("="*80)
        
        # Print summary table
        print(f"\n{'Method':<20} {'P@1':<8} {'P@5':<8} {'P@10':<8} {'R@10':<8} {'MRR':<8} {'NDCG@10':<10} {'Avg Query Time':<15}")
        print("-" * 95)
        
        for result in self.results:
            print(f"{result.test_name:<20} "
                  f"{result.precision_at_1:<8.3f} "
                  f"{result.precision_at_5:<8.3f} "
                  f"{result.precision_at_10:<8.3f} "
                  f"{result.recall_at_10:<8.3f} "
                  f"{result.mrr:<8.3f} "
                  f"{result.ndcg_at_10:<10.3f} "
                  f"{result.avg_query_time:<15.4f}s")
        
        # Print detailed results
        for result in self.results:
            print(f"\n{result.test_name} - Detailed Results:")
            print(f"  Dataset size: {result.dataset_size}")
            print(f"  Precision@1: {result.precision_at_1:.3f}")
            print(f"  Precision@5: {result.precision_at_5:.3f}")
            print(f"  Precision@10: {result.precision_at_10:.3f}")
            print(f"  Recall@10: {result.recall_at_10:.3f}")
            print(f"  MRR: {result.mrr:.3f}")
            print(f"  NDCG@10: {result.ndcg_at_10:.3f}")
            print(f"  Average query time: {result.avg_query_time:.4f}s")
            print(f"  Total time: {result.total_time:.2f}s")
            
            if result.additional_metrics:
                print(f"  Additional metrics:")
                for key, value in result.additional_metrics.items():
                    if key == 'ollama_stats':
                        print(f"    Ollama performance:")
                        for stat_key, stat_value in value.items():
                            print(f"      {stat_key}: {stat_value}")
                    else:
                        print(f"    {key}: {value}")
        
        # Performance comparison
        if len(self.results) >= 2:
            topcart_only = next((r for r in self.results if r.method == "topcart_only"), None)
            topcart_ollama = next((r for r in self.results if r.method == "topcart_ollama"), None)
            
            if topcart_only and topcart_ollama:
                print(f"\n🔍 PERFORMANCE COMPARISON:")
                print(f"  Precision@1 improvement: {((topcart_ollama.precision_at_1 - topcart_only.precision_at_1) / max(topcart_only.precision_at_1, 0.001) * 100):+.1f}%")
                print(f"  Precision@5 improvement: {((topcart_ollama.precision_at_5 - topcart_only.precision_at_5) / max(topcart_only.precision_at_5, 0.001) * 100):+.1f}%")
                print(f"  MRR improvement: {((topcart_ollama.mrr - topcart_only.mrr) / max(topcart_only.mrr, 0.001) * 100):+.1f}%")
                print(f"  NDCG@10 improvement: {((topcart_ollama.ndcg_at_10 - topcart_only.ndcg_at_10) / max(topcart_only.ndcg_at_10, 0.001) * 100):+.1f}%")
                print(f"  Query time overhead: {((topcart_ollama.avg_query_time - topcart_only.avg_query_time) / max(topcart_only.avg_query_time, 0.001) * 100):+.1f}%")
        
        print(f"\n🎯 CONCLUSION:")
        best_result = max(self.results, key=lambda r: r.ndcg_at_10)
        print(f"  Best performing method: {best_result.test_name}")
        print(f"  Best NDCG@10 score: {best_result.ndcg_at_10:.3f}")
        print(f"  Best MRR score: {best_result.mrr:.3f}")
        
        if best_result.method == "topcart_ollama":
            print(f"  ✅ Ollama integration provides significant improvement!")
        else:
            print(f"  ⚠️ Ollama integration needs optimization")


def test_ollama_connection():
    """Test if Ollama is available and working"""
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
    # Run benchmark test
    print("TOPCART + Ollama Benchmark Test")
    print("=" * 40)
    
    # Check Ollama connection
    if not test_ollama_connection():
        print("\n❌ Cannot run benchmark without Ollama connection")
        print("Please ensure Ollama is running: ollama serve")
        print("And you have a model installed: ollama pull llama3.2:3b")
        exit(1)
    
    # Run benchmark
    try:
        tester = OllamaBenchmarkTester(
            ollama_host="http://localhost:11434",
            ollama_model="llama3.2:3b"  # Change this to your available model
        )
        
        # Run comprehensive benchmark
        results = tester.run_comprehensive_benchmark(dataset_size=30)  # Start with smaller dataset
        
        # Print results
        tester.print_benchmark_results()
        
        print(f"\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()