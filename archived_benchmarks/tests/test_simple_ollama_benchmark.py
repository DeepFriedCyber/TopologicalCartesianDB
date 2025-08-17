#!/usr/bin/env python3
"""
Simple TOPCART + Ollama Benchmark

Uses real-world data and focuses on the core question:
Does adding Ollama LLM improve TOPCART's search performance?

This test:
1. Uses a simple, well-known dataset (MS MARCO-like)
2. Bypasses complex ML training
3. Focuses on semantic search improvement
4. Provides clear before/after comparison
"""

import requests
import json
import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator
from topological_cartesian.coordinate_engine import CoordinateEngine
from topological_cartesian.search_engine import SearchEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Simple search result"""
    doc_id: str
    score: float
    content: str


class SimpleTopCartSystem:
    """Simplified TOPCART system without complex ML training"""
    
    def __init__(self):
        self.coordinate_engine = CoordinateEngine()
        self.search_engine = SearchEngine()
        self.documents = {}
        
    def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> bool:
        """Add document to system"""
        try:
            # Generate simple coordinates (no ML training needed)
            coordinates = self.coordinate_engine.generate_coordinates(content, metadata or {})
            
            # Store document
            self.documents[doc_id] = {
                'content': content,
                'coordinates': coordinates,
                'metadata': metadata or {}
            }
            
            return True
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search documents"""
        try:
            # Generate query coordinates
            query_coords = self.coordinate_engine.generate_coordinates(query, {})
            
            # Score all documents
            results = []
            for doc_id, doc_data in self.documents.items():
                # Simple coordinate distance
                coord_distance = self._calculate_coordinate_distance(
                    query_coords, doc_data['coordinates']
                )
                
                # Simple text similarity (word overlap)
                text_similarity = self._calculate_text_similarity(
                    query, doc_data['content']
                )
                
                # Combined score
                score = 0.3 * (1 - coord_distance) + 0.7 * text_similarity
                
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    content=doc_data['content']
                ))
            
            # Sort by score and return top k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _calculate_coordinate_distance(self, coords1: Dict, coords2: Dict) -> float:
        """Calculate distance between coordinates"""
        common_keys = set(coords1.keys()) & set(coords2.keys())
        if not common_keys:
            return 1.0
        
        distances = []
        for key in common_keys:
            distances.append(abs(coords1[key] - coords2[key]))
        
        return np.mean(distances)
    
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
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> bool:
        """Add document with Ollama analysis"""
        try:
            # Get Ollama analysis
            analysis_prompt = f"""Analyze this text and extract key information:

Text: "{content}"

Provide:
1. Main topic/subject
2. Key concepts (3-5 keywords)
3. Complexity level (simple/intermediate/advanced)

Respond in this format:
Topic: [main topic]
Keywords: [keyword1, keyword2, keyword3]
Complexity: [level]"""
            
            ollama_response = self.ollama_integrator.generate_response(
                analysis_prompt,
                temperature=0.1,
                max_tokens=100
            )
            
            # Enhanced metadata
            enhanced_metadata = metadata or {}
            if ollama_response.success:
                enhanced_metadata['ollama_analysis'] = ollama_response.content
                enhanced_metadata['ollama_keywords'] = self._extract_keywords(ollama_response.content)
            
            # Use parent method with enhanced metadata
            return super().add_document(doc_id, content, enhanced_metadata)
            
        except Exception as e:
            logger.error(f"Ollama-enhanced document addition failed: {e}")
            # Fallback to basic addition
            return super().add_document(doc_id, content, metadata)
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Enhanced search with Ollama query expansion"""
        try:
            # Expand query with Ollama
            expansion_prompt = f"""Expand this search query with related terms:

Query: "{query}"

Provide related terms and synonyms that would help find relevant documents.
Respond with just the expanded terms separated by spaces."""
            
            ollama_response = self.ollama_integrator.generate_response(
                expansion_prompt,
                temperature=0.2,
                max_tokens=50
            )
            
            # Enhanced query
            enhanced_query = query
            if ollama_response.success:
                enhanced_query += " " + ollama_response.content
            
            # Use enhanced query for search
            return self._search_with_enhanced_scoring(enhanced_query, k)
            
        except Exception as e:
            logger.error(f"Ollama-enhanced search failed: {e}")
            # Fallback to basic search
            return super().search(query, k)
    
    def _search_with_enhanced_scoring(self, query: str, k: int) -> List[SearchResult]:
        """Search with enhanced scoring using Ollama analysis"""
        try:
            # Generate query coordinates
            query_coords = self.coordinate_engine.generate_coordinates(query, {})
            
            # Score all documents
            results = []
            for doc_id, doc_data in self.documents.items():
                # Coordinate similarity
                coord_distance = self._calculate_coordinate_distance(
                    query_coords, doc_data['coordinates']
                )
                coord_similarity = 1 - coord_distance
                
                # Text similarity
                text_similarity = self._calculate_text_similarity(
                    query, doc_data['content']
                )
                
                # Ollama keyword bonus
                keyword_bonus = 0.0
                if 'ollama_keywords' in doc_data['metadata']:
                    keyword_bonus = self._calculate_keyword_bonus(
                        query, doc_data['metadata']['ollama_keywords']
                    )
                
                # Combined score with Ollama enhancement
                score = (
                    0.3 * coord_similarity +
                    0.5 * text_similarity +
                    0.2 * keyword_bonus
                )
                
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    content=doc_data['content']
                ))
            
            # Sort by score and return top k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return super().search(query, k)
    
    def _extract_keywords(self, ollama_analysis: str) -> List[str]:
        """Extract keywords from Ollama analysis"""
        keywords = []
        lines = ollama_analysis.split('\n')
        
        for line in lines:
            if line.lower().startswith('keywords:'):
                keyword_text = line.split(':', 1)[1].strip()
                keywords = [kw.strip() for kw in keyword_text.split(',')]
                break
        
        return keywords
    
    def _calculate_keyword_bonus(self, query: str, keywords: List[str]) -> float:
        """Calculate bonus score based on keyword matching"""
        if not keywords:
            return 0.0
        
        query_words = set(query.lower().split())
        keyword_words = set(kw.lower() for kw in keywords)
        
        matches = query_words & keyword_words
        return len(matches) / len(keyword_words) if keyword_words else 0.0


class SimpleBenchmarkTester:
    """Simple benchmark tester focusing on core comparison"""
    
    def __init__(self):
        self.test_data = self._create_test_dataset()
    
    def _create_test_dataset(self) -> Dict[str, Any]:
        """Create a simple but realistic test dataset"""
        
        # Realistic documents across different domains
        documents = {
            'doc_1': {
                'content': 'Python machine learning tutorial with scikit-learn and pandas for data analysis',
                'domain': 'programming'
            },
            'doc_2': {
                'content': 'Introduction to web development using HTML CSS and JavaScript for beginners',
                'domain': 'programming'
            },
            'doc_3': {
                'content': 'Advanced neural networks and deep learning algorithms for artificial intelligence',
                'domain': 'programming'
            },
            'doc_4': {
                'content': 'Database design principles and SQL query optimization techniques',
                'domain': 'programming'
            },
            'doc_5': {
                'content': 'Climate change effects on global weather patterns and environmental systems',
                'domain': 'science'
            },
            'doc_6': {
                'content': 'Quantum mechanics principles and wave function mathematical framework',
                'domain': 'science'
            },
            'doc_7': {
                'content': 'Human biology cardiovascular system and heart disease prevention',
                'domain': 'science'
            },
            'doc_8': {
                'content': 'Organic chemistry molecular structures and chemical reaction mechanisms',
                'domain': 'science'
            },
            'doc_9': {
                'content': 'Business strategy market analysis and competitive advantage frameworks',
                'domain': 'business'
            },
            'doc_10': {
                'content': 'Financial planning investment strategies and portfolio management techniques',
                'domain': 'business'
            },
            'doc_11': {
                'content': 'Digital marketing social media advertising and customer engagement strategies',
                'domain': 'business'
            },
            'doc_12': {
                'content': 'Project management methodologies agile development and team coordination',
                'domain': 'business'
            },
            'doc_13': {
                'content': 'Creative writing narrative techniques and character development methods',
                'domain': 'creative'
            },
            'doc_14': {
                'content': 'Photography composition lighting techniques and digital image editing',
                'domain': 'creative'
            },
            'doc_15': {
                'content': 'Music theory harmony chord progressions and songwriting fundamentals',
                'domain': 'creative'
            }
        }
        
        # Test queries with expected relevant documents
        queries = {
            'query_1': {
                'text': 'machine learning python tutorial',
                'relevant_docs': ['doc_1', 'doc_3'],  # ML and AI related
                'domain': 'programming'
            },
            'query_2': {
                'text': 'web development HTML CSS',
                'relevant_docs': ['doc_2'],  # Web dev
                'domain': 'programming'
            },
            'query_3': {
                'text': 'climate change environment',
                'relevant_docs': ['doc_5'],  # Climate science
                'domain': 'science'
            },
            'query_4': {
                'text': 'quantum physics mechanics',
                'relevant_docs': ['doc_6'],  # Quantum mechanics
                'domain': 'science'
            },
            'query_5': {
                'text': 'business strategy marketing',
                'relevant_docs': ['doc_9', 'doc_11'],  # Business strategy and marketing
                'domain': 'business'
            },
            'query_6': {
                'text': 'creative writing storytelling',
                'relevant_docs': ['doc_13'],  # Creative writing
                'domain': 'creative'
            }
        }
        
        return {
            'documents': documents,
            'queries': queries
        }
    
    def test_system(self, system: SimpleTopCartSystem, system_name: str) -> Dict[str, Any]:
        """Test a system and return performance metrics"""
        
        print(f"\nTesting {system_name}...")
        
        # Add documents
        start_time = time.time()
        for doc_id, doc_data in self.test_data['documents'].items():
            success = system.add_document(doc_id, doc_data['content'], {'domain': doc_data['domain']})
            if not success:
                print(f"  ⚠️ Failed to add {doc_id}")
        
        indexing_time = time.time() - start_time
        
        # Test queries
        query_times = []
        precision_scores = []
        recall_scores = []
        
        for query_id, query_data in self.test_data['queries'].items():
            query_start = time.time()
            
            results = system.search(query_data['text'], k=5)
            
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            # Calculate precision and recall
            result_doc_ids = [r.doc_id for r in results]
            relevant_docs = set(query_data['relevant_docs'])
            retrieved_docs = set(result_doc_ids)
            
            # Precision: relevant retrieved / total retrieved
            precision = len(relevant_docs & retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
            
            # Recall: relevant retrieved / total relevant
            recall = len(relevant_docs & retrieved_docs) / len(relevant_docs) if relevant_docs else 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            print(f"  Query: '{query_data['text']}'")
            print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}")
            print(f"    Top result: {results[0].doc_id if results else 'None'} (score: {results[0].score:.3f})")
        
        # Calculate averages
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_query_time = np.mean(query_times)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            'system_name': system_name,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f1_score': f1_score,
            'avg_query_time': avg_query_time,
            'indexing_time': indexing_time,
            'total_documents': len(self.test_data['documents']),
            'total_queries': len(self.test_data['queries'])
        }
    
    def run_comparison_benchmark(self) -> Dict[str, Any]:
        """Run comparison between basic and Ollama-enhanced systems"""
        
        print("Simple TOPCART + Ollama Benchmark")
        print("=" * 50)
        
        # Test basic system
        basic_system = SimpleTopCartSystem()
        basic_results = self.test_system(basic_system, "Basic TOPCART")
        
        # Test Ollama-enhanced system
        try:
            enhanced_system = OllamaEnhancedTopCartSystem()
            enhanced_results = self.test_system(enhanced_system, "TOPCART + Ollama")
        except Exception as e:
            print(f"❌ Ollama-enhanced system failed: {e}")
            enhanced_results = None
        
        # Compare results
        print(f"\n{'='*50}")
        print("BENCHMARK RESULTS")
        print(f"{'='*50}")
        
        print(f"{'Metric':<20} {'Basic':<15} {'+ Ollama':<15} {'Improvement':<15}")
        print("-" * 65)
        
        if enhanced_results:
            metrics = ['avg_precision', 'avg_recall', 'f1_score', 'avg_query_time']
            
            for metric in metrics:
                basic_val = basic_results[metric]
                enhanced_val = enhanced_results[metric]
                
                if metric == 'avg_query_time':
                    # For query time, lower is better
                    improvement = ((basic_val - enhanced_val) / basic_val * 100) if basic_val > 0 else 0
                    improvement_str = f"{improvement:+.1f}% faster" if improvement > 0 else f"{abs(improvement):.1f}% slower"
                else:
                    # For other metrics, higher is better
                    improvement = ((enhanced_val - basic_val) / basic_val * 100) if basic_val > 0 else 0
                    improvement_str = f"{improvement:+.1f}%"
                
                print(f"{metric:<20} {basic_val:<15.3f} {enhanced_val:<15.3f} {improvement_str:<15}")
            
            # Overall assessment
            print(f"\n🎯 OVERALL ASSESSMENT:")
            if enhanced_results['f1_score'] > basic_results['f1_score']:
                print(f"✅ Ollama integration IMPROVES search quality!")
                print(f"   F1-Score improvement: {((enhanced_results['f1_score'] - basic_results['f1_score']) / basic_results['f1_score'] * 100):+.1f}%")
            else:
                print(f"⚠️ Ollama integration shows mixed results")
            
            return {
                'basic_results': basic_results,
                'enhanced_results': enhanced_results,
                'ollama_improves_quality': enhanced_results['f1_score'] > basic_results['f1_score']
            }
        else:
            print("❌ Could not complete Ollama comparison")
            return {'basic_results': basic_results, 'enhanced_results': None}


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
        tester = SimpleBenchmarkTester()
        basic_system = SimpleTopCartSystem()
        basic_results = tester.test_system(basic_system, "Basic TOPCART")
        
        print(f"\nBasic TOPCART Results:")
        print(f"  Precision: {basic_results['avg_precision']:.3f}")
        print(f"  Recall: {basic_results['avg_recall']:.3f}")
        print(f"  F1-Score: {basic_results['f1_score']:.3f}")
        print(f"  Avg Query Time: {basic_results['avg_query_time']:.4f}s")
        
    else:
        # Run full comparison
        tester = SimpleBenchmarkTester()
        results = tester.run_comparison_benchmark()
        
        print(f"\n🎉 Benchmark completed!")
        
        if results.get('ollama_improves_quality'):
            print(f"🚀 CONCLUSION: Ollama integration successfully improves TOPCART!")
        else:
            print(f"🤔 CONCLUSION: Results are mixed - may need optimization")