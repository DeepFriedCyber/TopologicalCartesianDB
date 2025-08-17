#!/usr/bin/env python3
"""
Ollama LLM Integration for Topological Cartesian DB
==================================================

Integrates local Ollama models with our coordinate system for hybrid AI capability.
This finally adds REAL language understanding to our fast coordinate engine!
"""

import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OllamaResponse:
    """Response from Ollama model"""
    content: str
    model: str
    processing_time: float
    success: bool
    error: Optional[str] = None

class OllamaLLMIntegrator:
    """
    Integrates Ollama local LLM with our coordinate system.
    
    This creates a hybrid system:
    - Fast coordinate-based context retrieval (our strength)
    - Real language understanding and reasoning (Ollama's strength)
    """
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 default_model: str = "llama3.2:3b",
                 timeout: int = 30):
        """
        Initialize Ollama integration.
        
        Args:
            ollama_host: Ollama server URL
            default_model: Default model to use (llama3.2:3b is fast and capable)
            timeout: Request timeout in seconds
        """
        self.ollama_host = ollama_host
        self.default_model = default_model
        self.timeout = timeout
        self.available_models = []
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        
        # Initialize connection
        self._check_ollama_connection()
        self._load_available_models()
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama connection successful")
                return True
            else:
                logger.error(f"❌ Ollama connection failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            return False
    
    def _load_available_models(self):
        """Load list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]
                logger.info(f"📋 Available models: {self.available_models}")
            else:
                logger.warning("⚠️ Could not load available models")
        except Exception as e:
            logger.warning(f"⚠️ Error loading models: {e}")
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a specific model is available (pull if needed)"""
        if model_name in self.available_models:
            return True
        
        logger.info(f"📥 Pulling model: {model_name}")
        try:
            # Pull model
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes for model download
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Model {model_name} pulled successfully")
                self._load_available_models()  # Refresh model list
                return True
            else:
                logger.error(f"❌ Failed to pull model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pulling model {model_name}: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         model: Optional[str] = None,
                         system_prompt: Optional[str] = None,
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> OllamaResponse:
        """
        Generate response using Ollama model.
        
        Args:
            prompt: User prompt/query
            model: Model to use (defaults to default_model)
            system_prompt: System prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        """
        start_time = time.time()
        model = model or self.default_model
        
        # Ensure model is available
        if not self.ensure_model_available(model):
            return OllamaResponse(
                content="",
                model=model,
                processing_time=0.0,
                success=False,
                error=f"Model {model} not available"
            )
        
        try:
            # Prepare request
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_data["system"] = system_prompt
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('response', '').strip()
                
                # Update statistics
                self.request_count += 1
                self.total_processing_time += processing_time
                self.success_count += 1
                
                logger.debug(f"✅ Ollama response generated in {processing_time:.2f}s")
                
                return OllamaResponse(
                    content=content,
                    model=model,
                    processing_time=processing_time,
                    success=True
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Ollama request failed: {error_msg}")
                
                return OllamaResponse(
                    content="",
                    model=model,
                    processing_time=processing_time,
                    success=False,
                    error=error_msg
                )
                
        except requests.exceptions.Timeout:
            processing_time = time.time() - start_time
            error_msg = f"Request timeout after {self.timeout}s"
            logger.error(f"❌ Ollama timeout: {error_msg}")
            
            return OllamaResponse(
                content="",
                model=model,
                processing_time=processing_time,
                success=False,
                error=error_msg
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"❌ Ollama error: {error_msg}")
            
            return OllamaResponse(
                content="",
                model=model,
                processing_time=processing_time,
                success=False,
                error=error_msg
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = (self.total_processing_time / self.request_count 
                   if self.request_count > 0 else 0.0)
        success_rate = (self.success_count / self.request_count 
                       if self.request_count > 0 else 0.0)
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "success_rate": success_rate,
            "average_processing_time": avg_time,
            "total_processing_time": self.total_processing_time,
            "available_models": self.available_models,
            "default_model": self.default_model
        }

class HybridCoordinateLLM:
    """
    Hybrid system combining our fast coordinates with Ollama LLM reasoning.
    
    This is the breakthrough integration that gives us:
    - Real AI reasoning capability (from Ollama)
    - Ultra-fast context retrieval (from coordinates)
    - Explainable results (from coordinate analysis)
    """
    
    def __init__(self, coordinate_engine, ollama_integrator: OllamaLLMIntegrator):
        """
        Initialize hybrid system.
        
        Args:
            coordinate_engine: Our existing coordinate engine
            ollama_integrator: Ollama LLM integrator
        """
        self.coordinate_engine = coordinate_engine
        self.ollama = ollama_integrator
        
        # Performance tracking
        self.hybrid_requests = 0
        self.coordinate_time = 0.0
        self.llm_time = 0.0
        
        logger.info("🚀 Hybrid Coordinate-LLM system initialized")
    
    def process_query(self, 
                     query: str, 
                     max_context_docs: int = 3,
                     model: Optional[str] = None,
                     temperature: float = 0.7) -> Dict[str, Any]:
        """
        Process query using hybrid coordinate + LLM approach.
        
        This is our breakthrough method:
        1. Fast coordinate-based context retrieval (our strength)
        2. Real LLM reasoning with context (Ollama's strength)
        3. Enhanced result with coordinate explanations
        """
        start_time = time.time()
        
        # Step 1: Fast coordinate-based context retrieval
        coord_start = time.time()
        context_docs = self.coordinate_engine.get_llm_context(query, max_docs=max_context_docs)
        coord_time = time.time() - coord_start
        
        # Step 2: Prepare enhanced prompt with coordinate context
        system_prompt = self._build_system_prompt(context_docs)
        enhanced_prompt = self._build_enhanced_prompt(query, context_docs)
        
        # Step 3: Real LLM reasoning
        llm_start = time.time()
        llm_response = self.ollama.generate_response(
            prompt=enhanced_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature
        )
        llm_time = time.time() - llm_start
        
        # Step 4: Combine results
        total_time = time.time() - start_time
        
        # Update statistics
        self.hybrid_requests += 1
        self.coordinate_time += coord_time
        self.llm_time += llm_time
        
        result = {
            "query": query,
            "llm_response": llm_response.content,
            "coordinate_context": context_docs,
            "processing_times": {
                "coordinate_retrieval": coord_time,
                "llm_generation": llm_time,
                "total": total_time
            },
            "model_used": llm_response.model,
            "success": llm_response.success,
            "error": llm_response.error,
            "hybrid_advantage": {
                "fast_context": f"{coord_time:.3f}s",
                "context_docs": len(context_docs),
                "coordinate_explanations": [doc.get('relevance_explanation', '') for doc in context_docs]
            }
        }
        
        logger.info(f"🎯 Hybrid query processed in {total_time:.2f}s "
                   f"(coord: {coord_time:.3f}s, llm: {llm_time:.2f}s)")
        
        return result
    
    def _build_system_prompt(self, context_docs: List[Dict[str, Any]]) -> str:
        """Build system prompt with coordinate context"""
        if not context_docs:
            return "You are a helpful AI assistant."
        
        context_info = []
        for i, doc in enumerate(context_docs, 1):
            coords = doc.get('coordinates', {})
            explanation = doc.get('relevance_explanation', '')
            context_info.append(f"Context {i}: {explanation}")
        
        return f"""You are a helpful AI assistant with access to relevant context.

COORDINATE-BASED CONTEXT:
{chr(10).join(context_info)}

Use this context to provide more accurate and relevant responses. The coordinate system has already identified the most relevant information for this query."""
    
    def _build_enhanced_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Build enhanced prompt with coordinate context"""
        if not context_docs:
            return query
        
        context_content = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get('content', '')
            if content:
                context_content.append(f"[Context {i}]: {content}")
        
        if not context_content:
            return query
        
        enhanced = f"""Based on the following relevant context:

{chr(10).join(context_content)}

Please answer this question: {query}"""
        
        return enhanced
    
    def get_hybrid_stats(self) -> Dict[str, Any]:
        """Get hybrid system performance statistics"""
        avg_coord_time = (self.coordinate_time / self.hybrid_requests 
                         if self.hybrid_requests > 0 else 0.0)
        avg_llm_time = (self.llm_time / self.hybrid_requests 
                       if self.hybrid_requests > 0 else 0.0)
        
        return {
            "hybrid_requests": self.hybrid_requests,
            "average_coordinate_time": avg_coord_time,
            "average_llm_time": avg_llm_time,
            "total_coordinate_time": self.coordinate_time,
            "total_llm_time": self.llm_time,
            "coordinate_engine_stats": self.coordinate_engine.get_performance_stats() if hasattr(self.coordinate_engine, 'get_performance_stats') else {},
            "ollama_stats": self.ollama.get_performance_stats()
        }

def test_ollama_integration():
    """Test Ollama integration"""
    print("🧪 Testing Ollama Integration")
    print("=" * 50)
    
    # Initialize Ollama integrator
    ollama = OllamaLLMIntegrator()
    
    # Test basic functionality
    test_prompt = "What is 2 + 2? Explain your reasoning."
    
    print(f"📝 Test prompt: {test_prompt}")
    print("⏳ Generating response...")
    
    response = ollama.generate_response(test_prompt)
    
    if response.success:
        print(f"✅ Success! Response in {response.processing_time:.2f}s")
        print(f"🤖 Model: {response.model}")
        print(f"💬 Response: {response.content}")
    else:
        print(f"❌ Failed: {response.error}")
    
    # Show statistics
    stats = ollama.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Requests: {stats['total_requests']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    print(f"   Avg Time: {stats['average_processing_time']:.2f}s")
    print(f"   Available Models: {stats['available_models']}")

if __name__ == "__main__":
    test_ollama_integration()