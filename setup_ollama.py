#!/usr/bin/env python3
"""
Ollama Setup Script for Topological Cartesian DB
===============================================

Automates the installation and configuration of Ollama for local LLM integration.
This script will get us from "no AI" to "real AI capability" in minutes!
"""

import os
import sys
import subprocess
import requests
import time
import json
from pathlib import Path

def check_ollama_installed():
    """Check if Ollama is already installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama already installed: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ Ollama not found")
    return False

def install_ollama_windows():
    """Install Ollama on Windows"""
    print("📥 Installing Ollama for Windows...")
    
    # Download Ollama installer
    installer_url = "https://ollama.com/download/windows"
    installer_path = "ollama-windows-installer.exe"
    
    print(f"⬇️  Downloading from {installer_url}")
    
    try:
        response = requests.get(installer_url, stream=True)
        response.raise_for_status()
        
        with open(installer_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded: {installer_path}")
        
        # Run installer
        print("🚀 Running installer...")
        print("⚠️  Please follow the installation prompts")
        
        subprocess.run([installer_path], check=True)
        
        # Clean up
        os.remove(installer_path)
        
        print("✅ Ollama installation completed!")
        print("🔄 Please restart your terminal and run this script again")
        return True
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def wait_for_ollama_service():
    """Wait for Ollama service to start"""
    print("⏳ Waiting for Ollama service to start...")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✅ Ollama service is running!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print("❌ Ollama service failed to start")
    return False

def start_ollama_service():
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    
    try:
        # Try to start Ollama service
        if os.name == 'nt':  # Windows
            subprocess.Popen(['ollama', 'serve'], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Unix-like
            subprocess.Popen(['ollama', 'serve'])
        
        return wait_for_ollama_service()
        
    except Exception as e:
        print(f"❌ Failed to start Ollama service: {e}")
        return False

def pull_recommended_models():
    """Pull recommended models for our use case"""
    
    # Recommended models for different use cases
    models = [
        {
            "name": "llama3.2:3b",
            "description": "Fast, efficient model (3B parameters)",
            "use_case": "Quick responses, math problems",
            "size": "~2GB"
        },
        {
            "name": "llama3.2:1b", 
            "description": "Ultra-fast model (1B parameters)",
            "use_case": "Very fast responses, simple tasks",
            "size": "~1GB"
        },
        {
            "name": "codellama:7b-code",
            "description": "Code-specialized model (7B parameters)",
            "use_case": "Programming tasks, HumanEval benchmark",
            "size": "~4GB"
        }
    ]
    
    print("\n📋 Recommended Models:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model['name']} - {model['description']}")
        print(f"      Use case: {model['use_case']}")
        print(f"      Size: {model['size']}")
    
    print("\n🎯 We'll start with llama3.2:3b (best balance of speed and capability)")
    
    # Pull the primary model
    primary_model = "llama3.2:3b"
    if pull_model(primary_model):
        print(f"✅ Primary model {primary_model} ready!")
        
        # Ask if user wants additional models
        print(f"\n❓ Would you like to pull additional models?")
        print(f"   - llama3.2:1b (faster, smaller)")
        print(f"   - codellama:7b-code (better for programming)")
        
        response = input("Pull additional models? (y/n): ").lower().strip()
        if response == 'y':
            for model in ["llama3.2:1b", "codellama:7b-code"]:
                pull_model(model)
    
    return True

def pull_model(model_name: str):
    """Pull a specific model"""
    print(f"\n📥 Pulling model: {model_name}")
    print("⏳ This may take several minutes depending on your internet connection...")
    
    try:
        # Use ollama pull command
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ Model {model_name} pulled successfully!")
            return True
        else:
            print(f"❌ Failed to pull {model_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout pulling {model_name} (>10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error pulling {model_name}: {e}")
        return False

def test_ollama_functionality():
    """Test basic Ollama functionality"""
    print("\n🧪 Testing Ollama functionality...")
    
    try:
        # Test API endpoint
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama API not responding")
            return False
        
        models = response.json().get('models', [])
        if not models:
            print("❌ No models available")
            return False
        
        print(f"✅ Found {len(models)} available models:")
        for model in models:
            print(f"   - {model['name']}")
        
        # Test generation with first available model
        test_model = models[0]['name']
        print(f"\n🧪 Testing generation with {test_model}...")
        
        test_request = {
            "model": test_model,
            "prompt": "What is 2 + 2?",
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=test_request, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            print(f"✅ Test successful!")
            print(f"   Question: What is 2 + 2?")
            print(f"   Answer: {answer}")
            return True
        else:
            print(f"❌ Generation test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_ollama_config():
    """Create configuration file for our integration"""
    config = {
        "ollama_host": "http://localhost:11434",
        "default_model": "llama3.2:3b",
        "timeout": 30,
        "models": {
            "fast": "llama3.2:1b",
            "balanced": "llama3.2:3b", 
            "code": "codellama:7b-code"
        },
        "setup_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "setup_complete": True
    }
    
    config_path = "ollama_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved: {config_path}")
    return config_path

def main():
    """Main setup process"""
    print("🚀 Ollama Setup for Topological Cartesian DB")
    print("=" * 60)
    print("This will install and configure Ollama for local LLM integration")
    print("Finally adding REAL AI capability to our coordinate system!")
    print()
    
    # Step 1: Check if Ollama is installed
    if not check_ollama_installed():
        if os.name == 'nt':  # Windows
            print("\n📥 Installing Ollama...")
            if not install_ollama_windows():
                print("❌ Installation failed. Please install manually from https://ollama.com")
                return False
        else:
            print("❌ Please install Ollama manually:")
            print("   Linux: curl -fsSL https://ollama.com/install.sh | sh")
            print("   macOS: brew install ollama")
            return False
    
    # Step 2: Start Ollama service
    if not start_ollama_service():
        print("❌ Failed to start Ollama service")
        print("💡 Try running 'ollama serve' manually in another terminal")
        return False
    
    # Step 3: Pull recommended models
    if not pull_recommended_models():
        print("❌ Failed to pull models")
        return False
    
    # Step 4: Test functionality
    if not test_ollama_functionality():
        print("❌ Ollama functionality test failed")
        return False
    
    # Step 5: Create configuration
    config_path = create_ollama_config()
    
    print("\n🎉 Ollama setup completed successfully!")
    print("=" * 60)
    print("✅ Ollama service running")
    print("✅ Models downloaded and ready")
    print("✅ Configuration created")
    print(f"✅ Config file: {config_path}")
    
    print("\n🚀 Next steps:")
    print("1. Run the hybrid system test:")
    print("   python src/topological_cartesian/ollama_integration.py")
    print("2. Run GSM8K benchmark with REAL AI:")
    print("   python benchmarks/gsm8k_ollama_benchmark.py")
    
    print("\n🎯 You now have REAL AI capability integrated with your coordinate system!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)