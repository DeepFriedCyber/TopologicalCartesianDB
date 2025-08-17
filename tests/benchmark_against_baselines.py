#!/usr/bin/env python3
"""
Benchmark Against Established Baselines

Compares our Cartesian Cube system against well-established baselines:
1. TF-IDF + Cosine Similarity (classical IR)
2. BM25 (Elasticsearch/Lucene standard)
3. Sentence-BERT embeddings + FAISS
4. LSA/LSI (Latent Semantic Analysis)
5. Word2Vec/Doc2Vec embeddings

This provides objective validation against proven systems.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from topological_cartesian.semantic_coordinate_engine import SemanticCoordinateEngine
from topological_cartesian.optimized_spatial_engine import OptimizedSpatialEngine, SpatialIndexConfig

logger = logging.getLogger(__name__)

# Import baseline libraries with fallbacks
BASELINE_LIBS_AVAILABLE = {}

try:
    from sentence_transformers import SentenceTransformer
    BASELINE_LIBS_AVAILABLE['sentence_transformers'] = True
except ImportError:
    BASELINE_LIBS_AVAILABLE['sentence_transformers'] = False

try:
    import faiss
    BASELINE_LIBS_AVAILABLE['faiss'] = True
except ImportError:
    BASELINE_LIBS_AVAILABLE['faiss'] = False

try:
    from gensim.models import Word2Vec, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument
    BASELINE_LIBS_AVAILABLE['gensim'] = True
except ImportError:
    BASELINE_LIBS_AVAILABLE['gensim'] = False

try:
    from rank_bm25 import BM25Okapi
    BASELINE_LIBS_AVAILABLE['bm25'] = True
except ImportError:
    BASELINE_LIBS_AVAILABLE['bm25'] = False


class BaselineSystem:
    """Base class for baseline systems"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.build_time = 0.0
        self.index_size = 0
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build search index from documents"""
        raise NotImplementedError
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'build_time': self.build_time,
            'index_size': self.index_size
        }


class TFIDFBaseline(BaselineSystem):
    """TF-IDF + Cosine Similarity baseline"""
    
    def __init__(self, max_features: int = 10000):
        super().__init__("TF-IDF + Cosine")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.doc_vectors = None
        self.doc_ids = []
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build TF-IDF index"""
        start_time = time.time()
        
        self.doc_ids = doc_ids.copy()
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        
        self.build_time = time.time() - start_time
        self.index_size = self.doc_vectors.shape[0] * self.doc_vectors.shape[1]
        self.is_trained = True
        
        logger.info(f"Built TF-IDF index: {self.doc_vectors.shape} in {self.build_time:.3f}s")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using cosine similarity"""
        if not self.is_trained:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(similarities[idx])))
        
        return results


class BM25Baseline(BaselineSystem):
    """BM25 baseline (Elasticsearch/Lucene standard)"""
    
    def __init__(self):
        super().__init__("BM25")
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_docs = []
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build BM25 index"""
        if not BASELINE_LIBS_AVAILABLE['bm25']:
            logger.warning("BM25 library not available")
            return
        
        start_time = time.time()
        
        self.doc_ids = doc_ids.copy()
        
        # Simple tokenization
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        self.build_time = time.time() - start_time
        self.index_size = len(self.tokenized_docs) * np.mean([len(doc) for doc in self.tokenized_docs])
        self.is_trained = True
        
        logger.info(f"Built BM25 index: {len(documents)} docs in {self.build_time:.3f}s")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 scoring"""
        if not self.is_trained or not BASELINE_LIBS_AVAILABLE['bm25']:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results


class SentenceBERTBaseline(BaselineSystem):
    """Sentence-BERT + FAISS baseline"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__("Sentence-BERT + FAISS")
        self.model_name = model_name
        self.model = None
        self.index = None
        self.doc_ids = []
        self.embeddings = None
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build Sentence-BERT + FAISS index"""
        if not BASELINE_LIBS_AVAILABLE['sentence_transformers'] or not BASELINE_LIBS_AVAILABLE['faiss']:
            logger.warning("Sentence-BERT or FAISS not available")
            return
        
        start_time = time.time()
        
        # Load model
        self.model = SentenceTransformer(self.model_name)
        self.doc_ids = doc_ids.copy()
        
        # Generate embeddings
        self.embeddings = self.model.encode(documents)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype(np.float32))
        
        self.build_time = time.time() - start_time
        self.index_size = self.embeddings.shape[0] * self.embeddings.shape[1]
        self.is_trained = True
        
        logger.info(f"Built Sentence-BERT index: {self.embeddings.shape} in {self.build_time:.3f}s")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using Sentence-BERT embeddings"""
        if not self.is_trained or not BASELINE_LIBS_AVAILABLE['sentence_transformers']:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.doc_ids) and idx >= 0:
                results.append((self.doc_ids[idx], float(sim)))
        
        return results


class LSABaseline(BaselineSystem):
    """Latent Semantic Analysis baseline"""
    
    def __init__(self, n_components: int = 300):
        super().__init__("LSA/LSI")
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.svd = TruncatedSVD(n_components=n_components)
        self.doc_vectors = None
        self.doc_ids = []
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build LSA index"""
        start_time = time.time()
        
        self.doc_ids = doc_ids.copy()
        
        # TF-IDF vectorization
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # SVD dimensionality reduction
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)
        
        self.build_time = time.time() - start_time
        self.index_size = self.doc_vectors.shape[0] * self.doc_vectors.shape[1]
        self.is_trained = True
        
        logger.info(f"Built LSA index: {self.doc_vectors.shape} in {self.build_time:.3f}s")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using LSA vectors"""
        if not self.is_trained:
            return []
        
        # Transform query
        query_tfidf = self.vectorizer.transform([query])
        query_vector = self.svd.transform(query_tfidf)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(similarities[idx])))
        
        return results


class Doc2VecBaseline(BaselineSystem):
    """Doc2Vec baseline"""
    
    def __init__(self, vector_size: int = 100, epochs: int = 20):
        super().__init__("Doc2Vec")
        self.vector_size = vector_size
        self.epochs = epochs
        self.model = None
        self.doc_ids = []
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build Doc2Vec index"""
        if not BASELINE_LIBS_AVAILABLE['gensim']:
            logger.warning("Gensim not available for Doc2Vec")
            return
        
        start_time = time.time()
        
        self.doc_ids = doc_ids.copy()
        
        # Prepare tagged documents
        tagged_docs = [TaggedDocument(words=doc.lower().split(), tags=[doc_id]) 
                      for doc, doc_id in zip(documents, doc_ids)]
        
        # Train Doc2Vec model
        self.model = Doc2Vec(
            tagged_docs,
            vector_size=self.vector_size,
            window=2,
            min_count=1,
            workers=4,
            epochs=self.epochs
        )
        
        self.build_time = time.time() - start_time
        self.index_size = len(doc_ids) * self.vector_size
        self.is_trained = True
        
        logger.info(f"Built Doc2Vec index: {len(documents)} docs in {self.build_time:.3f}s")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using Doc2Vec similarity"""
        if not self.is_trained or not BASELINE_LIBS_AVAILABLE['gensim']:
            return []
        
        # Infer query vector
        query_vector = self.model.infer_vector(query.lower().split())
        
        # Find most similar documents
        similar_docs = self.model.dv.most_similar([query_vector], topn=k)
        
        results = []
        for doc_id, similarity in similar_docs:
            results.append((doc_id, float(similarity)))
        
        return results


class CartesianCubeSystem(BaselineSystem):
    """Our Cartesian Cube system for comparison"""
    
    def __init__(self):
        super().__init__("Cartesian Cube")
        self.semantic_engine = SemanticCoordinateEngine()
        self.spatial_engine = OptimizedSpatialEngine()
        self.documents = {}
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build Cartesian Cube index"""
        start_time = time.time()
        
        # Generate coordinates and add documents
        for doc_id, document in zip(doc_ids, documents):
            result = self.semantic_engine.text_to_coordinates(document)
            self.documents[doc_id] = {
                'content': document,
                'coordinates': result['coordinates']
            }
        
        # Add to spatial engine
        self.spatial_engine.add_documents(self.documents)
        
        self.build_time = time.time() - start_time
        self.index_size = len(documents)
        self.is_trained = True
        
        logger.info(f"Built Cartesian Cube index: {len(documents)} docs in {self.build_time:.3f}s")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using Cartesian Cube system"""
        if not self.is_trained:
            return []
        
        # Get query coordinates
        result = self.semantic_engine.text_to_coordinates(query)
        query_coords = result['coordinates']
        
        # Perform search
        search_results = self.spatial_engine.optimized_search(query_coords, k=k)
        
        # Convert to standard format
        results = []
        for result in search_results:
            doc_id = result['document_id']
            similarity = result['similarity_score']
            results.append((doc_id, similarity))
        
        return results


class BenchmarkEvaluator:
    """Evaluates and compares different systems"""
    
    def __init__(self):
        self.systems = {}
        self.evaluation_results = {}
    
    def add_system(self, system: BaselineSystem):
        """Add a system to benchmark"""
        self.systems[system.name] = system
    
    def run_benchmark(self, documents: List[str], doc_ids: List[str], 
                     queries: List[str], relevance_judgments: Dict[str, List[str]],
                     k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Run comprehensive benchmark"""
        
        results = {}
        
        for system_name, system in self.systems.items():
            print(f"Benchmarking {system_name}...")
            
            # Build index
            try:
                system.build_index(documents, doc_ids)
                if not system.is_trained:
                    print(f"  Skipping {system_name} - failed to build index")
                    continue
            except Exception as e:
                print(f"  Error building {system_name}: {e}")
                continue
            
            # Evaluate queries
            system_results = self._evaluate_system(
                system, queries, relevance_judgments, k_values
            )
            
            # Add system stats
            system_results['system_stats'] = system.get_stats()
            
            results[system_name] = system_results
            
            print(f"  Build time: {system.build_time:.3f}s")
            print(f"  Avg search time: {system_results['avg_search_time']:.4f}s")
            print(f"  P@1: {system_results['precision_at_k']['p@1']:.3f}")
            print(f"  P@5: {system_results['precision_at_k']['p@5']:.3f}")
        
        return results
    
    def _evaluate_system(self, system: BaselineSystem, queries: List[str],
                        relevance_judgments: Dict[str, List[str]], 
                        k_values: List[int]) -> Dict[str, Any]:
        """Evaluate a single system"""
        
        query_results = []
        search_times = []
        
        for i, query in enumerate(queries):
            query_id = f"query_{i}"
            relevant_docs = relevance_judgments.get(query_id, [])
            
            # Perform search
            start_time = time.time()
            search_results = system.search(query, k=max(k_values))
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            
            # Calculate metrics
            retrieved_docs = [doc_id for doc_id, _ in search_results]
            
            precision_at_k = {}
            recall_at_k = {}
            
            for k in k_values:
                retrieved_k = retrieved_docs[:k]
                
                if len(retrieved_k) > 0:
                    relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
                    precision_at_k[f'p@{k}'] = relevant_retrieved / len(retrieved_k)
                else:
                    precision_at_k[f'p@{k}'] = 0.0
                
                if len(relevant_docs) > 0:
                    recall_at_k[f'r@{k}'] = relevant_retrieved / len(relevant_docs)
                else:
                    recall_at_k[f'r@{k}'] = 0.0
            
            query_results.append({
                'query_id': query_id,
                'query': query,
                'relevant_docs': relevant_docs,
                'retrieved_docs': retrieved_docs,
                'precision_at_k': precision_at_k,
                'recall_at_k': recall_at_k,
                'search_time': search_time
            })
        
        # Aggregate results
        avg_precision_at_k = {}
        avg_recall_at_k = {}
        
        for k in k_values:
            precisions = [qr['precision_at_k'][f'p@{k}'] for qr in query_results]
            recalls = [qr['recall_at_k'][f'r@{k}'] for qr in query_results]
            
            avg_precision_at_k[f'p@{k}'] = np.mean(precisions)
            avg_recall_at_k[f'r@{k}'] = np.mean(recalls)
        
        return {
            'precision_at_k': avg_precision_at_k,
            'recall_at_k': avg_recall_at_k,
            'avg_search_time': np.mean(search_times),
            'total_queries': len(queries),
            'individual_results': query_results
        }
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive comparison report"""
        
        report = []
        report.append("BENCHMARK COMPARISON REPORT")
        report.append("=" * 50)
        
        # System overview
        report.append("\nSYSTEM OVERVIEW:")
        report.append("-" * 20)
        
        for system_name, result in results.items():
            stats = result['system_stats']
            report.append(f"{system_name}:")
            report.append(f"  Build Time: {stats['build_time']:.3f}s")
            report.append(f"  Index Size: {stats['index_size']:,}")
            report.append(f"  Avg Search Time: {result['avg_search_time']:.4f}s")
        
        # Performance comparison
        report.append("\nPERFORMANCE COMPARISON:")
        report.append("-" * 25)
        
        # Create comparison table
        metrics = ['p@1', 'p@3', 'p@5', 'p@10']
        
        # Header
        header = f"{'System':<20}"
        for metric in metrics:
            header += f"{metric:>8}"
        header += f"{'Time(ms)':>10}"
        report.append(header)
        report.append("-" * len(header))
        
        # Data rows
        for system_name, result in results.items():
            row = f"{system_name:<20}"
            for metric in metrics:
                value = result['precision_at_k'].get(metric, 0.0)
                row += f"{value:>8.3f}"
            
            search_time_ms = result['avg_search_time'] * 1000
            row += f"{search_time_ms:>10.2f}"
            report.append(row)
        
        # Rankings
        report.append("\nRANKINGS:")
        report.append("-" * 10)
        
        # Rank by P@1
        p1_ranking = sorted(results.items(), 
                           key=lambda x: x[1]['precision_at_k'].get('p@1', 0.0), 
                           reverse=True)
        
        report.append("By Precision@1:")
        for i, (system_name, result) in enumerate(p1_ranking, 1):
            p1_score = result['precision_at_k'].get('p@1', 0.0)
            report.append(f"  {i}. {system_name}: {p1_score:.3f}")
        
        # Rank by speed
        speed_ranking = sorted(results.items(), 
                              key=lambda x: x[1]['avg_search_time'])
        
        report.append("\nBy Search Speed:")
        for i, (system_name, result) in enumerate(speed_ranking, 1):
            speed_ms = result['avg_search_time'] * 1000
            report.append(f"  {i}. {system_name}: {speed_ms:.2f}ms")
        
        # Statistical significance (simplified)
        report.append("\nSTATISTICAL ANALYSIS:")
        report.append("-" * 20)
        
        if len(results) >= 2:
            # Find best and worst systems
            best_system = max(results.items(), 
                            key=lambda x: x[1]['precision_at_k'].get('p@1', 0.0))
            worst_system = min(results.items(), 
                             key=lambda x: x[1]['precision_at_k'].get('p@1', 0.0))
            
            best_p1 = best_system[1]['precision_at_k'].get('p@1', 0.0)
            worst_p1 = worst_system[1]['precision_at_k'].get('p@1', 0.0)
            
            if worst_p1 > 0:
                improvement = ((best_p1 - worst_p1) / worst_p1) * 100
                report.append(f"Best system ({best_system[0]}) is {improvement:.1f}% better than worst ({worst_system[0]})")
            
            # Check if our system is competitive
            if "Cartesian Cube" in results:
                our_p1 = results["Cartesian Cube"]['precision_at_k'].get('p@1', 0.0)
                our_rank = [i for i, (name, _) in enumerate(p1_ranking, 1) if name == "Cartesian Cube"][0]
                
                report.append(f"Cartesian Cube ranks #{our_rank} out of {len(results)} systems")
                
                if our_rank <= len(results) // 2:
                    report.append("✅ Cartesian Cube performs competitively!")
                else:
                    report.append("⚠️  Cartesian Cube needs improvement")
        
        return "\n".join(report)


def create_test_dataset() -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    """Create a test dataset with documents, queries, and relevance judgments"""
    
    # Documents (categorized for easier relevance judgments)
    documents = [
        # Programming documents
        "Python programming tutorial for beginners with examples and exercises",
        "Advanced JavaScript techniques for web development and optimization",
        "Machine learning algorithms implementation in Python using scikit-learn",
        "Database design principles and SQL query optimization techniques",
        "Web development with React and Node.js for modern applications",
        
        # Science documents  
        "Quantum mechanics principles and applications in modern physics",
        "Climate change effects on global weather patterns and ecosystems",
        "Genetic engineering techniques and their applications in medicine",
        "Artificial intelligence research trends and future developments",
        "Space exploration missions and discoveries in the solar system",
        
        # Business documents
        "Marketing strategies for digital transformation in modern businesses",
        "Financial planning and investment strategies for retirement savings",
        "Project management methodologies and best practices for teams",
        "Supply chain optimization and logistics management solutions",
        "Customer relationship management and sales process improvement",
        
        # Health documents
        "Nutrition guidelines and healthy eating habits for better wellness",
        "Exercise routines and fitness programs for different age groups",
        "Mental health awareness and stress management techniques",
        "Medical research advances in cancer treatment and prevention",
        "Public health policies and disease prevention strategies"
    ]
    
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Test queries
    queries = [
        "How to learn Python programming?",
        "What are the effects of climate change?",
        "Best marketing strategies for businesses",
        "Healthy eating and nutrition tips",
        "Machine learning with Python",
        "Space exploration and discoveries",
        "Financial planning for retirement",
        "Web development with JavaScript",
        "Mental health and stress management",
        "Database optimization techniques"
    ]
    
    # Relevance judgments (which documents are relevant for each query)
    relevance_judgments = {
        "query_0": ["doc_0", "doc_2"],  # Python programming
        "query_1": ["doc_6", "doc_19"],  # Climate change
        "query_2": ["doc_10", "doc_14"],  # Marketing strategies
        "query_3": ["doc_15", "doc_16"],  # Nutrition and health
        "query_4": ["doc_2", "doc_8"],   # Machine learning
        "query_5": ["doc_9"],            # Space exploration
        "query_6": ["doc_11"],           # Financial planning
        "query_7": ["doc_1", "doc_4"],   # Web development
        "query_8": ["doc_17"],           # Mental health
        "query_9": ["doc_3"]             # Database optimization
    }
    
    return documents, doc_ids, queries, relevance_judgments


if __name__ == "__main__":
    # Run comprehensive benchmark
    print("Cartesian Cube System - Baseline Comparison")
    print("=" * 50)
    
    # Create test dataset
    documents, doc_ids, queries, relevance_judgments = create_test_dataset()
    
    print(f"Dataset: {len(documents)} documents, {len(queries)} queries")
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator()
    
    # Add baseline systems
    evaluator.add_system(TFIDFBaseline())
    evaluator.add_system(LSABaseline())
    evaluator.add_system(CartesianCubeSystem())
    
    # Add optional systems if libraries are available
    if BASELINE_LIBS_AVAILABLE['bm25']:
        evaluator.add_system(BM25Baseline())
    
    if BASELINE_LIBS_AVAILABLE['sentence_transformers'] and BASELINE_LIBS_AVAILABLE['faiss']:
        evaluator.add_system(SentenceBERTBaseline())
    
    if BASELINE_LIBS_AVAILABLE['gensim']:
        evaluator.add_system(Doc2VecBaseline())
    
    print(f"Systems to benchmark: {list(evaluator.systems.keys())}")
    
    # Run benchmark
    results = evaluator.run_benchmark(
        documents, doc_ids, queries, relevance_judgments, 
        k_values=[1, 3, 5, 10]
    )
    
    # Generate and print report
    report = evaluator.generate_comparison_report(results)
    print("\n" + report)
    
    # Save results
    import json
    with open("benchmark_results.json", "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for system_name, result in results.items():
            json_result = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    json_result[key] = {k: float(v) if isinstance(v, np.floating) else v 
                                       for k, v in value.items()}
                elif isinstance(value, (np.floating, np.integer)):
                    json_result[key] = float(value)
                else:
                    json_result[key] = value
            json_results[system_name] = json_result
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to benchmark_results.json")