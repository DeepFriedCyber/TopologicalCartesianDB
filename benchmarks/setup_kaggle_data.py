#!/usr/bin/env python3
"""
Kaggle Dataset Setup Script
===========================

This script helps set up the Kaggle Open LLM-Perf Leaderboard dataset
for benchmarking our Topological Cartesian Cube system.

It provides multiple ways to obtain the dataset:
1. Direct download via Kaggle API (requires credentials)
2. Manual download instructions
3. Synthetic data generation for testing
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import requests
from datetime import datetime

class KaggleDatasetSetup:
    """Setup and prepare Kaggle LLM performance dataset"""
    
    def __init__(self, data_dir: str = "kaggle_data"):
        self.data_dir = data_dir
        self.dataset_file = os.path.join(data_dir, "Open_LLM_Perf_Leaderboard.csv")
        os.makedirs(data_dir, exist_ok=True)
    
    def setup_kaggle_api(self) -> bool:
        """Setup Kaggle API credentials"""
        print("🔧 Setting up Kaggle API...")
        
        try:
            import kaggle
            
            # Test API access
            kaggle.api.authenticate()
            print("✅ Kaggle API authenticated successfully")
            return True
            
        except ImportError:
            print("❌ Kaggle package not installed")
            print("💡 Install with: pip install kaggle")
            return False
            
        except Exception as e:
            print(f"❌ Kaggle API authentication failed: {e}")
            print("\n💡 To set up Kaggle API:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\{username}\\.kaggle\\ (Windows)")
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def download_via_kaggle_api(self) -> bool:
        """Download dataset using Kaggle API"""
        if not self.setup_kaggle_api():
            return False
        
        try:
            import kaggle
            
            dataset_id = "warcoder/open-llm-perf-leaderboard-dataset"
            print(f"📥 Downloading dataset: {dataset_id}")
            
            kaggle.api.dataset_download_files(
                dataset_id, 
                path=self.data_dir, 
                unzip=True
            )
            
            # Find the downloaded CSV file
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if csv_files:
                original_file = os.path.join(self.data_dir, csv_files[0])
                os.rename(original_file, self.dataset_file)
                print(f"✅ Dataset downloaded and saved to: {self.dataset_file}")
                return True
            else:
                print("❌ No CSV file found in downloaded data")
                return False
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def download_via_direct_url(self) -> bool:
        """Attempt to download dataset via direct URL (may not work due to Kaggle's authentication)"""
        print("📥 Attempting direct download...")
        
        # Note: This typically won't work for Kaggle datasets due to authentication requirements
        url = "https://www.kaggle.com/datasets/warcoder/open-llm-perf-leaderboard-dataset/download"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(self.dataset_file, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Dataset downloaded to: {self.dataset_file}")
                return True
            else:
                print(f"❌ Download failed with status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Direct download failed: {e}")
            return False
    
    def create_synthetic_dataset(self) -> bool:
        """Create a synthetic dataset based on the Kaggle LLM leaderboard structure"""
        print("🔧 Creating synthetic LLM performance dataset...")
        
        # Model configurations based on real LLM architectures
        model_configs = [
            # GPT family
            ("gpt2", "GPT-2", "124M", "pytorch", "None", 35.2, 45.3, 2048),
            ("gpt2-medium", "GPT-2", "355M", "pytorch", "None", 38.7, 52.1, 3072),
            ("gpt2-large", "GPT-2", "774M", "pytorch", "None", 41.2, 68.4, 4096),
            ("gpt2-xl", "GPT-2", "1.5B", "pytorch", "None", 43.8, 89.2, 6144),
            
            # LLaMA family
            ("llama-7b", "LLaMA", "7B", "pytorch", "None", 62.5, 156.3, 13312),
            ("llama-7b", "LLaMA", "7B", "pytorch", "float16", 62.5, 78.2, 6656),
            ("llama-7b", "LLaMA", "7B", "pytorch", "LLM.int8", 61.8, 89.4, 7984),
            ("llama-7b", "LLaMA", "7B", "pytorch", "LLM.fp4", 60.2, 45.7, 3328),
            ("llama-13b", "LLaMA", "13B", "pytorch", "None", 66.1, 287.5, 25600),
            ("llama-13b", "LLaMA", "13B", "pytorch", "float16", 66.1, 143.8, 12800),
            ("llama-30b", "LLaMA", "30B", "pytorch", "float16", 69.3, 334.2, 61440),
            ("llama-65b", "LLaMA", "65B", "pytorch", "float16", 71.6, 723.8, 131072),
            
            # Mistral family
            ("mistral-7b", "Mistral", "7B", "pytorch", "None", 64.2, 142.7, 12288),
            ("mistral-7b", "Mistral", "7B", "pytorch", "float16", 64.2, 71.4, 6144),
            ("mistral-7b-instruct", "Mistral", "7B", "pytorch", "float16", 65.8, 73.2, 6144),
            
            # MPT family
            ("mpt-7b", "MPT", "7B", "pytorch", "None", 58.3, 134.5, 11264),
            ("mpt-7b", "MPT", "7B", "pytorch", "float16", 58.3, 67.3, 5632),
            ("mpt-7b-chat", "MPT", "7B", "pytorch", "float16", 59.1, 69.8, 5632),
            ("mpt-30b", "MPT", "30B", "pytorch", "float16", 63.7, 298.4, 57344),
            
            # Falcon family
            ("falcon-7b", "Falcon", "7B", "pytorch", "float16", 61.4, 145.2, 13824),
            ("falcon-40b", "Falcon", "40B", "pytorch", "float16", 67.9, 412.6, 81920),
            
            # CodeLlama family
            ("codellama-7b", "CodeLlama", "7B", "pytorch", "float16", 53.2, 78.9, 6656),
            ("codellama-13b", "CodeLlama", "13B", "pytorch", "float16", 56.8, 148.3, 12800),
            ("codellama-34b", "CodeLlama", "34B", "pytorch", "float16", 61.2, 367.4, 65536),
            
            # Vicuna family
            ("vicuna-7b", "Vicuna", "7B", "pytorch", "float16", 63.9, 76.5, 6656),
            ("vicuna-13b", "Vicuna", "13B", "pytorch", "float16", 67.2, 145.8, 12800),
            
            # ChatGLM family
            ("chatglm2-6b", "ChatGLM", "6B", "pytorch", "float16", 55.8, 124.7, 11776),
            ("chatglm2-6b", "ChatGLM", "6B", "pytorch", "LLM.int8", 55.1, 142.3, 13440),
            
            # Baichuan family
            ("baichuan-7b", "Baichuan", "7B", "pytorch", "float16", 54.3, 138.9, 12288),
            ("baichuan-13b", "Baichuan", "13B", "pytorch", "float16", 59.7, 267.4, 24576),
            
            # OPT family
            ("opt-1.3b", "OPT", "1.3B", "pytorch", "float16", 42.6, 28.4, 2560),
            ("opt-2.7b", "OPT", "2.7B", "pytorch", "float16", 46.8, 52.3, 5120),
            ("opt-6.7b", "OPT", "6.7B", "pytorch", "float16", 51.2, 123.6, 12288),
            ("opt-13b", "OPT", "13B", "pytorch", "float16", 54.7, 241.8, 24576),
            ("opt-30b", "OPT", "30B", "pytorch", "float16", 58.9, 567.2, 57344),
            
            # BLOOM family
            ("bloom-560m", "BLOOM", "560M", "pytorch", "float16", 38.9, 18.7, 2048),
            ("bloom-1b1", "BLOOM", "1.1B", "pytorch", "float16", 41.3, 34.2, 4096),
            ("bloom-3b", "BLOOM", "3B", "pytorch", "float16", 45.7, 89.4, 8192),
            ("bloom-7b1", "BLOOM", "7.1B", "pytorch", "float16", 49.2, 187.3, 16384),
            
            # T5 family
            ("t5-small", "T5", "60M", "pytorch", "float16", 32.1, 12.3, 1024),
            ("t5-base", "T5", "220M", "pytorch", "float16", 36.4, 23.7, 2048),
            ("t5-large", "T5", "770M", "pytorch", "float16", 40.8, 67.2, 4096),
            ("t5-3b", "T5", "3B", "pytorch", "float16", 44.9, 134.5, 8192),
            ("t5-11b", "T5", "11B", "pytorch", "float16", 48.3, 423.7, 24576),
        ]
        
        # Generate additional variations with different optimizations
        extended_configs = []
        for config in model_configs:
            name, model_type, params, backend, opt, score, latency, memory = config
            
            # Base configuration
            extended_configs.append(config)
            
            # Add optimized versions if base is "None"
            if opt == "None":
                # float16 version (if not already float16)
                if "float16" not in name:
                    extended_configs.append((
                        name, model_type, params, backend, "float16",
                        score * 0.98,  # Slight accuracy drop
                        latency * 0.52,  # ~2x faster
                        memory * 0.51   # ~2x less memory
                    ))
                
                # int8 quantization
                extended_configs.append((
                    name, model_type, params, backend, "LLM.int8",
                    score * 0.95,  # Accuracy drop
                    latency * 0.65,  # Faster
                    memory * 0.6    # Less memory
                ))
                
                # fp4 quantization
                extended_configs.append((
                    name, model_type, params, backend, "LLM.fp4",
                    score * 0.88,  # Larger accuracy drop
                    latency * 0.35,  # Much faster
                    memory * 0.28   # Much less memory
                ))
        
        # Create DataFrame
        data = []
        for i, config in enumerate(extended_configs):
            name, model_type, params, backend, opt, score, latency, memory = config
            
            # Add some realistic noise
            score_noise = np.random.normal(0, 1.5)
            latency_noise = np.random.normal(1, 0.1)
            memory_noise = np.random.normal(1, 0.05)
            
            data.append({
                "Model": name,
                "Type": model_type,
                "Params": params,
                "Backend": backend,
                "Optimization": opt,
                "Average": max(0, score + score_noise),
                "Latency (ms)": max(1, latency * latency_noise),
                "Peak Memory (MB)": max(100, memory * memory_noise),
                "Throughput (tokens/s)": 1000.0 / max(1, latency * latency_noise),
                "Cost per 1k tokens": self._estimate_cost(params, opt)
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_file, index=False)
        
        print(f"✅ Synthetic dataset created with {len(df)} entries")
        print(f"📄 Saved to: {self.dataset_file}")
        
        # Print sample statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"  • Models: {df['Model'].nunique()} unique models")
        print(f"  • Model Types: {', '.join(df['Type'].unique())}")
        print(f"  • Parameter Ranges: {df['Params'].unique()}")
        print(f"  • Backends: {', '.join(df['Backend'].unique())}")
        print(f"  • Optimizations: {', '.join(df['Optimization'].unique())}")
        print(f"  • Average Score Range: {df['Average'].min():.1f} - {df['Average'].max():.1f}")
        print(f"  • Latency Range: {df['Latency (ms)'].min():.1f} - {df['Latency (ms)'].max():.1f} ms")
        print(f"  • Memory Range: {df['Peak Memory (MB)'].min():.0f} - {df['Peak Memory (MB)'].max():.0f} MB")
        
        return True
    
    def _estimate_cost(self, params: str, optimization: str) -> float:
        """Estimate cost per 1k tokens based on model size and optimization"""
        # Base cost mapping
        param_costs = {
            "60M": 0.0001, "124M": 0.0002, "220M": 0.0003, "355M": 0.0005,
            "560M": 0.0008, "774M": 0.001, "1.1B": 0.0015, "1.3B": 0.002,
            "1.5B": 0.0025, "2.7B": 0.004, "3B": 0.005, "6B": 0.008,
            "6.7B": 0.009, "7B": 0.01, "7.1B": 0.011, "11B": 0.015,
            "13B": 0.018, "30B": 0.035, "34B": 0.04, "40B": 0.045,
            "65B": 0.08
        }
        
        base_cost = param_costs.get(params, 0.01)
        
        # Optimization multipliers
        opt_multipliers = {
            "None": 1.0,
            "float16": 0.75,
            "LLM.int8": 0.5,
            "LLM.fp4": 0.25,
            "BetterTransformer": 0.8
        }
        
        multiplier = opt_multipliers.get(optimization, 1.0)
        return base_cost * multiplier * np.random.uniform(0.8, 1.2)  # Add some noise
    
    def validate_dataset(self) -> bool:
        """Validate the downloaded/created dataset"""
        if not os.path.exists(self.dataset_file):
            print(f"❌ Dataset file not found: {self.dataset_file}")
            return False
        
        try:
            df = pd.read_csv(self.dataset_file)
            
            required_columns = ["Model", "Type", "Average", "Latency (ms)", "Peak Memory (MB)"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                return False
            
            print(f"✅ Dataset validation passed")
            print(f"  • Rows: {len(df)}")
            print(f"  • Columns: {len(df.columns)}")
            print(f"  • Required columns present: {required_columns}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            return False
    
    def print_manual_instructions(self):
        """Print manual download instructions"""
        print("\n" + "="*60)
        print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("If automatic download fails, you can manually download the dataset:")
        print()
        print("1. Go to: https://www.kaggle.com/datasets/warcoder/open-llm-perf-leaderboard-dataset")
        print("2. Click 'Download' button (requires Kaggle account)")
        print("3. Extract the CSV file")
        print(f"4. Save it as: {self.dataset_file}")
        print()
        print("Alternative: The benchmark can run with synthetic data if no dataset is available.")
        print("="*60)
    
    def setup_dataset(self) -> bool:
        """Main setup method - tries multiple approaches"""
        print("🚀 Setting up Kaggle LLM Performance Dataset")
        print("="*50)
        
        # Check if dataset already exists
        if os.path.exists(self.dataset_file):
            print(f"📄 Dataset already exists: {self.dataset_file}")
            if self.validate_dataset():
                return True
            else:
                print("⚠️  Existing dataset is invalid, recreating...")
        
        # Try Kaggle API download
        print("\n1️⃣ Attempting Kaggle API download...")
        if self.download_via_kaggle_api():
            if self.validate_dataset():
                return True
        
        # Try direct download (usually won't work for Kaggle)
        print("\n2️⃣ Attempting direct download...")
        if self.download_via_direct_url():
            if self.validate_dataset():
                return True
        
        # Create synthetic dataset
        print("\n3️⃣ Creating synthetic dataset...")
        if self.create_synthetic_dataset():
            if self.validate_dataset():
                print("\n✅ Synthetic dataset ready for benchmarking!")
                return True
        
        # If all fails, print manual instructions
        print("\n❌ All automatic methods failed")
        self.print_manual_instructions()
        return False

def main():
    """Main setup function"""
    print("🔧 Kaggle Dataset Setup for Topological Cartesian Cube Benchmarking")
    print("="*70)
    
    setup = KaggleDatasetSetup()
    
    if setup.setup_dataset():
        print("\n🎉 Dataset setup complete!")
        print("You can now run the benchmark with:")
        print("  python kaggle_llm_benchmark.py")
    else:
        print("\n⚠️  Dataset setup incomplete")
        print("The benchmark will use synthetic baseline data")

if __name__ == "__main__":
    main()