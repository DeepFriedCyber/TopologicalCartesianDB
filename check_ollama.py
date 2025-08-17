#!/usr/bin/env python3
"""
Check Ollama availability and installed models
"""

import requests
import json

def check_ollama():
    """Check if Ollama is running and list available models"""
    
    ollama_url = "http://localhost:11434"
    
    print("🤖 Checking Ollama Status")
    print("=" * 40)
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✅ Ollama is running!")
            
            models_data = response.json()
            models = models_data.get("models", [])
            
            if models:
                print(f"\n📋 Available Models ({len(models)}):")
                print("-" * 40)
                
                for model in models:
                    name = model.get("name", "Unknown")
                    size = model.get("size", 0)
                    modified = model.get("modified_at", "Unknown")
                    
                    # Convert size to human readable
                    if size > 1024**3:
                        size_str = f"{size / (1024**3):.1f} GB"
                    elif size > 1024**2:
                        size_str = f"{size / (1024**2):.1f} MB"
                    else:
                        size_str = f"{size} bytes"
                    
                    print(f"  • {name}")
                    print(f"    Size: {size_str}")
                    print(f"    Modified: {modified[:19] if len(modified) > 19 else modified}")
                    print()
                
                # Recommend models for testing
                print("🎯 Recommended Models for Testing:")
                print("-" * 40)
                
                model_names = [m["name"] for m in models]
                
                recommendations = [
                    ("llama2", "Good general purpose model"),
                    ("llama2:7b", "7B parameter version"),
                    ("mistral", "Fast and efficient"),
                    ("codellama", "Code-focused model"),
                    ("phi", "Small but capable model")
                ]
                
                available_recommendations = []
                for model_name, description in recommendations:
                    if any(model_name in name for name in model_names):
                        available_recommendations.append((model_name, description))
                        print(f"  ✅ {model_name}: {description}")
                
                if not available_recommendations:
                    print("  ⚠️ No recommended models found")
                    print("\n💡 To install a model, run:")
                    print("     ollama pull llama2")
                    print("     ollama pull mistral")
                
            else:
                print("⚠️ No models installed")
                print("\n💡 To install a model, run:")
                print("     ollama pull llama2")
                print("     ollama pull mistral")
            
            print("\n🚀 Ready to run benchmark with:")
            print("     python tests/test_two_phase_ollama.py")
            
        else:
            print(f"❌ Ollama responded with status: {response.status_code}")
            print("💡 Make sure Ollama is running: ollama serve")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("💡 Ollama might not be running or installed")
        print("\n📥 To install Ollama:")
        print("   1. Visit: https://ollama.ai/")
        print("   2. Download and install")
        print("   3. Run: ollama serve")
        print("   4. Install a model: ollama pull llama2")
        
    except requests.exceptions.Timeout:
        print("❌ Ollama connection timed out")
        print("💡 Ollama might be starting up, try again in a moment")
        
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")

def test_model_generation(model_name: str = "llama2"):
    """Test model generation with a simple prompt"""
    
    ollama_url = "http://localhost:11434"
    
    print(f"\n🧪 Testing Model Generation: {model_name}")
    print("=" * 50)
    
    try:
        test_prompt = "What is artificial intelligence? Please provide a brief explanation."
        
        print(f"📝 Prompt: {test_prompt}")
        print("🔄 Generating response...")
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "num_predict": 100,
                    "temperature": 0.7
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "No response")
            
            print("✅ Generation successful!")
            print(f"📄 Response: {generated_text}")
            
            # Show performance metrics
            if "eval_count" in result and "eval_duration" in result:
                tokens = result["eval_count"]
                duration_ns = result["eval_duration"]
                duration_s = duration_ns / 1e9
                tokens_per_second = tokens / duration_s if duration_s > 0 else 0
                
                print(f"\n📊 Performance:")
                print(f"   Tokens: {tokens}")
                print(f"   Duration: {duration_s:.2f}s")
                print(f"   Speed: {tokens_per_second:.1f} tokens/second")
            
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Generation timed out")
        print("💡 Model might be loading, try again")
        
    except Exception as e:
        print(f"❌ Error testing generation: {e}")

if __name__ == "__main__":
    check_ollama()
    
    # Ask if user wants to test generation
    try:
        test_gen = input("\n🧪 Test model generation? (y/n): ").lower().strip()
        if test_gen in ['y', 'yes']:
            model = input("Enter model name (default: llama2): ").strip()
            if not model:
                model = "llama2"
            test_model_generation(model)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")