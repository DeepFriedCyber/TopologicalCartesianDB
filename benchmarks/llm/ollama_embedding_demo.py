#!/usr/bin/env python3
"""
Ollama LLM Embedding Generation Demo
Demonstrates real-world embedding generation for vector database benchmarking

This shows how to integrate local Ollama models for realistic benchmarking
scenarios with actual LLM-generated embeddings.
"""

import json
import time
import numpy as np
import requests
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
import logging

class OllamaEmbeddingDemo:
    """Demonstrate Ollama embedding generation for TCDB benchmarking"""
    
    def __init__(self):
        self.host = "http://localhost:11434"
        self.embedding_model = "all-minilm:latest"  # Alternative embedding model
        self.logger = logging.getLogger(__name__)
        
    async def check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                print(f"✅ Ollama is running with {len(available_models)} models")
                print(f"📋 Available models: {', '.join(available_models)}")
                
                # Check if our embedding model exists (case insensitive)
                embedding_found = False
                for model in available_models:
                    if "nomic-embed" in model.lower() or "all-minilm" in model.lower():
                        self.embedding_model = model
                        embedding_found = True
                        break
                
                if embedding_found:
                    print(f"✅ Using embedding model: '{self.embedding_model}'")
                    return True
                else:
                    print(f"❌ No embedding model found in available models")
                    print("📝 Try: ollama pull nomic-embed-text")
                    return False
            else:
                print(f"❌ Ollama returned status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using Ollama"""
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            response = requests.post(
                f"{self.host}/api/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result["embedding"], dtype=np.float32)
                
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            else:
                print(f"❌ Embedding generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    async def benchmark_embedding_generation(self) -> Dict[str, Any]:
        """Benchmark Ollama embedding generation performance"""
        
        # Test scenarios with different types of content
        test_scenarios = {
            "code_documentation": [
                "Python function for vector similarity search",
                "JavaScript async/await implementation patterns",
                "SQL query optimization for large datasets",
                "Machine learning model evaluation metrics",
                "REST API design best practices"
            ],
            "technical_content": [
                "Vector database performance optimization",
                "Topological data analysis in machine learning",
                "Multi-dimensional scaling algorithms",
                "Neural network architecture design",
                "Distributed computing frameworks"
            ],
            "general_queries": [
                "Climate change impact on agriculture",
                "Renewable energy adoption trends",
                "Artificial intelligence in healthcare",
                "Space exploration technologies",
                "Economic policy analysis"
            ]
        }
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "embedding_model": self.embedding_model,
            "scenarios": {}
        }
        
        for scenario_name, texts in test_scenarios.items():
            print(f"\n📊 Testing {scenario_name} scenario...")
            
            scenario_results = {
                "texts_processed": len(texts),
                "embeddings_generated": 0,
                "total_time_seconds": 0,
                "avg_time_per_embedding": 0,
                "embedding_dimension": 0,
                "embedding_samples": []
            }
            
            start_time = time.time()
            successful_embeddings = 0
            
            for i, text in enumerate(texts):
                print(f"   Generating embedding {i+1}/{len(texts)}: {text[:50]}...")
                
                embedding_start = time.time()
                embedding = await self.generate_embedding(text)
                embedding_time = time.time() - embedding_start
                
                if embedding is not None:
                    successful_embeddings += 1
                    if scenario_results["embedding_dimension"] == 0:
                        scenario_results["embedding_dimension"] = len(embedding)
                    
                    # Store sample for analysis
                    scenario_results["embedding_samples"].append({
                        "text": text,
                        "embedding_time_seconds": embedding_time,
                        "embedding_norm": float(np.linalg.norm(embedding)),
                        "embedding_mean": float(np.mean(embedding)),
                        "embedding_std": float(np.std(embedding))
                    })
                
                # Small delay to avoid overwhelming Ollama
                await asyncio.sleep(0.1)
            
            total_time = time.time() - start_time
            
            scenario_results.update({
                "embeddings_generated": successful_embeddings,
                "total_time_seconds": total_time,
                "avg_time_per_embedding": total_time / successful_embeddings if successful_embeddings > 0 else 0,
                "throughput_embeddings_per_sec": successful_embeddings / total_time if total_time > 0 else 0
            })
            
            results["scenarios"][scenario_name] = scenario_results
            
            print(f"   ✅ Generated {successful_embeddings}/{len(texts)} embeddings")
            print(f"   ⏱️ {scenario_results['avg_time_per_embedding']:.2f}s per embedding")
            print(f"   🚀 {scenario_results['throughput_embeddings_per_sec']:.1f} embeddings/sec")
        
        return results
    
    async def similarity_search_demo(self) -> Dict[str, Any]:
        """Demonstrate similarity search with generated embeddings"""
        
        print(f"\n🔍 Similarity Search Demo")
        
        # Generate embeddings for a document collection
        documents = [
            "Python vector database implementation",
            "Machine learning model optimization",
            "JavaScript frontend development",
            "Database query performance tuning",
            "Neural network architecture design"
        ]
        
        query = "How to optimize database performance"
        
        print(f"📚 Generating embeddings for {len(documents)} documents...")
        doc_embeddings = []
        
        for doc in documents:
            embedding = await self.generate_embedding(doc)
            if embedding is not None:
                doc_embeddings.append((doc, embedding))
        
        print(f"🔎 Generating embedding for query: '{query}'")
        query_embedding = await self.generate_embedding(query)
        
        if query_embedding is None:
            return {"error": "Failed to generate query embedding"}
        
        # Calculate similarities
        similarities = []
        for doc, doc_embedding in doc_embeddings:
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((doc, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📈 Similarity Search Results:")
        for i, (doc, sim) in enumerate(similarities):
            print(f"   {i+1}. {doc} (similarity: {sim:.4f})")
        
        return {
            "query": query,
            "documents": documents,
            "results": similarities,
            "embedding_dimension": len(query_embedding)
        }

async def main():
    """Main demo execution"""
    
    print("🚀 Ollama LLM Embedding Demo for TCDB")
    print("="*60)
    
    demo = OllamaEmbeddingDemo()
    
    # Check Ollama status
    if not await demo.check_ollama_status():
        print("❌ Please ensure Ollama is running and has the embedding model")
        return
    
    # Run embedding generation benchmark
    print(f"\n📊 Starting embedding generation benchmark...")
    benchmark_results = await demo.benchmark_embedding_generation()
    
    # Run similarity search demo
    similarity_results = await demo.similarity_search_demo()
    
    # Compile final results
    final_results = {
        "demo_info": {
            "framework": "Ollama + TCDB Integration Demo",
            "embedding_model": demo.embedding_model,
            "timestamp": datetime.now().isoformat()
        },
        "embedding_benchmark": benchmark_results,
        "similarity_search_demo": similarity_results
    }
    
    # Save results
    output_file = f"ollama_embedding_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Ollama Embedding Demo Complete!")
    print(f"📊 Results saved to: {output_file}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("📈 OLLAMA EMBEDDING PERFORMANCE SUMMARY")
    print("="*60)
    
    total_embeddings = 0
    total_time = 0
    
    for scenario_name, scenario_data in benchmark_results["scenarios"].items():
        print(f"\n🔸 {scenario_name.upper()}")
        print(f"   Embeddings: {scenario_data['embeddings_generated']}")
        print(f"   Dimension: {scenario_data['embedding_dimension']}")
        print(f"   Throughput: {scenario_data['throughput_embeddings_per_sec']:.1f} embeddings/sec")
        print(f"   Avg Time: {scenario_data['avg_time_per_embedding']:.2f}s per embedding")
        
        total_embeddings += scenario_data['embeddings_generated']
        total_time += scenario_data['total_time_seconds']
    
    overall_throughput = total_embeddings / total_time if total_time > 0 else 0
    print(f"\n🏆 OVERALL PERFORMANCE:")
    print(f"   Total Embeddings: {total_embeddings}")
    print(f"   Overall Throughput: {overall_throughput:.1f} embeddings/sec")
    print(f"   Model: {demo.embedding_model}")
    
    print(f"\n🎯 Ollama integration ready for TCDB realistic benchmarking!")
    print(f"🚀 Real-world embedding generation validated with local LLM!")
    
    return final_results

if __name__ == "__main__":
    asyncio.run(main())
