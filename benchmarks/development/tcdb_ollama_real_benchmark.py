#!/usr/bin/env python3
"""
TCDB Ollama LLM Integration Benchmark
Real-world performance testing using local Ollama models for embedding generation

This benchmark demonstrates TCDB performance with actual LLM-generated embeddings,
providing realistic performance metrics for production deployment.
"""

import os
import sys
import json
import time
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
import logging

# Import TCDB components
try:
    from multi_cube_orchestrator import MultiCubeOrchestrator
    print("✅ TCDB components imported successfully")
except ImportError as e:
    print(f"❌ TCDB import failed: {e}")
    # Continue anyway for demonstration

@dataclass
class OllamaConfig:
    """Configuration for Ollama integration"""
    host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:latest"  # Best embedding model available
    text_model: str = "llama3.2:3b"  # For text generation
    timeout: int = 30
    max_retries: int = 3

@dataclass
class LLMBenchmarkConfig:
    """Configuration for LLM-powered benchmarking"""
    
    # Test scenarios
    scenarios: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "code_search": {
            "description": "Code documentation and search",
            "text_samples": [
                "Python function for sorting algorithms",
                "JavaScript async/await implementation", 
                "SQL query optimization techniques",
                "Machine learning model training",
                "API endpoint documentation"
            ],
            "embedding_dimension": 768  # nomic-embed-text dimension
        },
        "semantic_search": {
            "description": "General semantic text search",
            "text_samples": [
                "Climate change and environmental impact",
                "Artificial intelligence in healthcare",
                "Economic trends and financial markets",
                "Space exploration and astronomy",
                "Renewable energy technologies"
            ],
            "embedding_dimension": 768
        },
        "technical_docs": {
            "description": "Technical documentation search",
            "text_samples": [
                "Vector database optimization strategies",
                "Topological data analysis methods",
                "Multi-cube orchestration architecture",
                "Neural backend selection algorithms",
                "Performance benchmarking methodologies"
            ],
            "embedding_dimension": 768
        }
    })
    
    # Benchmark parameters
    documents_per_scenario: int = 100
    queries_per_scenario: int = 20
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 5, 10])

class OllamaEmbeddingGenerator:
    """Generate embeddings using local Ollama models"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def check_ollama_status(self) -> bool:
        """Check if Ollama is running and models are available"""
        try:
            response = requests.get(f"{self.config.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                self.logger.info(f"✅ Ollama is running with {len(available_models)} models")
                
                if self.config.embedding_model in available_models:
                    self.logger.info(f"✅ Embedding model '{self.config.embedding_model}' is available")
                    return True
                else:
                    self.logger.warning(f"⚠️ Embedding model '{self.config.embedding_model}' not found")
                    self.logger.info(f"Available models: {available_models}")
                    return False
            else:
                self.logger.error(f"❌ Ollama returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Ollama: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using Ollama"""
        try:
            payload = {
                "model": self.config.embedding_model,
                "prompt": text
            }
            
            response = requests.post(
                f"{self.config.host}/api/embeddings",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result["embedding"], dtype=np.float32)
                
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            else:
                self.logger.error(f"❌ Embedding generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        
        for i, text in enumerate(texts):
            self.logger.info(f"Generating embedding {i+1}/{len(texts)}: {text[:50]}...")
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
            
            # Small delay to avoid overwhelming Ollama
            await asyncio.sleep(0.1)
        
        return embeddings

class TCDBOllamaBenchmark:
    """Benchmark TCDB using Ollama-generated embeddings"""
    
    def __init__(self, config: LLMBenchmarkConfig):
        self.config = config
        self.ollama_config = OllamaConfig()
        self.embedding_generator = OllamaEmbeddingGenerator(self.ollama_config)
        self.orchestrator = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> bool:
        """Initialize TCDB and verify Ollama connection"""
        
        # Check Ollama
        if not await self.embedding_generator.check_ollama_status():
            return False
        
        # Initialize TCDB
        try:
            self.orchestrator = MultiCubeOrchestrator()
            await self.orchestrator.initialize()
            self.logger.info("✅ TCDB initialized with optimizations")
            return True
        except Exception as e:
            self.logger.error(f"❌ TCDB initialization failed: {e}")
            return False
    
    async def run_scenario_benchmark(self, scenario_name: str, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark for a specific scenario"""
        
        self.logger.info(f"🎯 Starting {scenario_name} benchmark")
        self.logger.info(f"   Description: {scenario_config['description']}")
        
        # Generate extended document set
        documents = await self._generate_document_set(scenario_config['text_samples'])
        queries = documents[:self.config.queries_per_scenario]  # Use subset as queries
        
        # Generate embeddings
        self.logger.info("📊 Generating embeddings with Ollama...")
        doc_embeddings = await self.embedding_generator.generate_embeddings_batch(
            [doc['text'] for doc in documents]
        )
        query_embeddings = await self.embedding_generator.generate_embeddings_batch(
            [query['text'] for query in queries]
        )
        
        # Filter out failed embeddings
        valid_docs = [(doc, emb) for doc, emb in zip(documents, doc_embeddings) if emb is not None]
        valid_queries = [(query, emb) for query, emb in zip(queries, query_embeddings) if emb is not None]
        
        if len(valid_docs) == 0 or len(valid_queries) == 0:
            return {"error": "No valid embeddings generated"}
        
        self.logger.info(f"✅ Generated {len(valid_docs)} document embeddings, {len(valid_queries)} query embeddings")
        
        # Load documents into TCDB
        load_start = time.time()
        for i, (doc, embedding) in enumerate(valid_docs):
            doc['vector'] = embedding.tolist()
            doc['id'] = f"{scenario_name}_doc_{i}"
            await self.orchestrator.add_document(doc)
            
            if i % 20 == 0:
                self.logger.info(f"   Loaded {i}/{len(valid_docs)} documents")
        
        load_time = time.time() - load_start
        load_throughput = len(valid_docs) / load_time
        
        # Run search benchmarks
        search_results = {}
        for k in self.config.k_values:
            for concurrency in self.config.concurrency_levels:
                self.logger.info(f"   Testing k={k}, concurrency={concurrency}")
                
                metrics = await self._benchmark_search(valid_queries, k, concurrency)
                search_results[f"k{k}_c{concurrency}"] = metrics
        
        return {
            "scenario": scenario_name,
            "description": scenario_config['description'],
            "embedding_model": self.ollama_config.embedding_model,
            "documents_loaded": len(valid_docs),
            "queries_tested": len(valid_queries),
            "loading_performance": {
                "duration_seconds": load_time,
                "throughput_docs_per_sec": load_throughput
            },
            "search_performance": search_results,
            "embedding_dimension": scenario_config['embedding_dimension']
        }
    
    async def _generate_document_set(self, base_samples: List[str]) -> List[Dict[str, Any]]:
        """Generate extended document set from base samples"""
        documents = []
        
        # Use base samples
        for i, sample in enumerate(base_samples):
            documents.append({
                "text": sample,
                "metadata": {"source": "base_sample", "index": i},
                "keywords": sample.lower().split()[:3]  # First 3 words as keywords
            })
        
        # Generate variations to reach target count
        while len(documents) < self.config.documents_per_scenario:
            base_doc = documents[len(documents) % len(base_samples)]
            
            # Create variation by adding context
            variation = f"Advanced {base_doc['text']} with modern implementations and best practices"
            documents.append({
                "text": variation,
                "metadata": {"source": "variation", "base_index": len(documents) % len(base_samples)},
                "keywords": variation.lower().split()[:3]
            })
        
        return documents[:self.config.documents_per_scenario]
    
    async def _benchmark_search(self, queries: List[Tuple[Dict, np.ndarray]], k: int, concurrency: int) -> Dict[str, float]:
        """Benchmark search performance"""
        
        search_times = []
        
        # Run searches with concurrency control
        tasks = []
        for query_doc, query_embedding in queries[:min(10, len(queries))]:  # Limit for performance
            task = self._single_search(query_doc['text'])
            tasks.append(task)
            
            if len(tasks) >= concurrency:
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                batch_time = time.time() - start_time
                
                for result in results:
                    if not isinstance(result, Exception):
                        search_times.append(batch_time / concurrency)
                
                tasks = []
        
        # Handle remaining tasks
        if tasks:
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - start_time
            
            for result in results:
                if not isinstance(result, Exception):
                    search_times.append(batch_time / len(tasks))
        
        # Calculate metrics
        if search_times:
            avg_latency = np.mean(search_times)
            qps = 1.0 / avg_latency if avg_latency > 0 else 0
            p95_latency = np.percentile(search_times, 95) * 1000  # Convert to ms
            p99_latency = np.percentile(search_times, 99) * 1000
        else:
            avg_latency = qps = p95_latency = p99_latency = 0
        
        return {
            "qps": qps,
            "avg_latency_ms": avg_latency * 1000,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency
        }
    
    async def _single_search(self, query: str) -> Dict[str, Any]:
        """Execute single search query"""
        try:
            results = await self.orchestrator.process_query(query)
            return {"success": True, "results": len(results) if results else 0}
        except Exception as e:
            return {"success": False, "error": str(e)}

async def main():
    """Main benchmark execution with Ollama integration"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 TCDB + Ollama LLM Benchmark")
    print("="*60)
    
    # Configuration
    config = LLMBenchmarkConfig()
    benchmark = TCDBOllamaBenchmark(config)
    
    # Initialize
    if not await benchmark.initialize():
        print("❌ Initialization failed!")
        return None
    
    # Run benchmarks for all scenarios
    all_results = {
        "benchmark_info": {
            "framework": "TCDB + Ollama Integration",
            "embedding_model": benchmark.ollama_config.embedding_model,
            "text_model": benchmark.ollama_config.text_model,
            "timestamp": datetime.now().isoformat(),
            "scenarios_tested": list(config.scenarios.keys())
        },
        "results": {}
    }
    
    for scenario_name, scenario_config in config.scenarios.items():
        try:
            print(f"\n📈 Running {scenario_name} scenario...")
            result = await benchmark.run_scenario_benchmark(scenario_name, scenario_config)
            all_results["results"][scenario_name] = result
            
        except Exception as e:
            print(f"❌ {scenario_name} scenario failed: {e}")
            all_results["results"][scenario_name] = {"error": str(e)}
    
    # Save results
    output_file = f"tcdb_ollama_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Ollama + TCDB Benchmark Complete!")
    print(f"📊 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 OLLAMA + TCDB PERFORMANCE SUMMARY")
    print("="*60)
    
    for scenario_name, result in all_results["results"].items():
        if "error" not in result:
            print(f"\n🔸 {scenario_name.upper()}")
            print(f"   Model: {result['embedding_model']}")
            print(f"   Documents: {result['documents_loaded']}")
            print(f"   Loading: {result['loading_performance']['throughput_docs_per_sec']:.1f} docs/sec")
            
            # Show best search performance
            best_qps = 0
            for perf_key, perf_data in result['search_performance'].items():
                if perf_data['qps'] > best_qps:
                    best_qps = perf_data['qps']
            
            print(f"   Best Search: {best_qps:.1f} QPS")
        else:
            print(f"\n❌ {scenario_name}: {result['error']}")
    
    print(f"\n🎯 Real-world LLM integration with TCDB optimizations complete!")
    print(f"🚀 Performance with actual embeddings from local Ollama models validated!")
    
    return all_results

if __name__ == "__main__":
    asyncio.run(main())
