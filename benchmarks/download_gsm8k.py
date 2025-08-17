#!/usr/bin/env python3
"""
GSM8K Dataset Downloader
========================

Downloads the official GSM8K dataset from OpenAI's repository.
This ensures we're using the exact official data for verification.
"""

import os
import requests
import json
import hashlib
from typing import List, Dict

def download_gsm8k_dataset():
    """Download the official GSM8K test dataset"""
    
    print("📥 Downloading Official GSM8K Dataset")
    print("=" * 50)
    
    # Official GSM8K test set URL
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    
    print(f"🔗 Source: {url}")
    print("📊 Dataset: GSM8K Test Set (1,319 problems)")
    
    try:
        # Download the file
        print("⬇️  Downloading...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to file
        filename = "gsm8k_test.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✅ Downloaded: {filename}")
        
        # Verify the download
        verify_gsm8k_dataset(filename)
        
        return filename
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def verify_gsm8k_dataset(filename: str):
    """Verify the GSM8K dataset integrity and format"""
    
    print(f"\n🔍 Verifying Dataset: {filename}")
    print("-" * 30)
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    # Calculate file hash
    with open(filename, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    print(f"🔐 SHA256 Hash: {file_hash[:16]}...")
    
    # Verify format and count problems
    try:
        problems = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Verify required fields
                    if 'question' not in data or 'answer' not in data:
                        print(f"❌ Invalid format at line {line_num}")
                        return False
                    
                    problems.append(data)
                    
                except json.JSONDecodeError:
                    print(f"❌ JSON error at line {line_num}")
                    return False
        
        print(f"📊 Problems loaded: {len(problems)}")
        print(f"📋 Expected count: 1,319 (official GSM8K test set)")
        
        if len(problems) == 1319:
            print("✅ Problem count matches official GSM8K test set")
        else:
            print(f"⚠️  Problem count mismatch: {len(problems)} vs 1,319")
        
        # Show sample problem
        if problems:
            sample = problems[0]
            print(f"\n📝 Sample Problem:")
            print(f"   Question: {sample['question'][:100]}...")
            print(f"   Answer: {sample['answer'][:50]}...")
        
        print(f"\n✅ Dataset verification complete")
        print(f"🔐 File hash: {file_hash}")
        
        # Save verification info
        verification_info = {
            "filename": filename,
            "sha256_hash": file_hash,
            "problem_count": len(problems),
            "verified_date": "2025-08-07",
            "source_url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
            "official_count": 1319,
            "format_valid": True
        }
        
        with open("gsm8k_verification.json", 'w') as f:
            json.dump(verification_info, f, indent=2)
        
        print(f"📄 Verification info saved: gsm8k_verification.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Download and verify GSM8K dataset"""
    
    print("🧮 GSM8K Dataset Setup")
    print("=" * 30)
    
    # Check if already exists
    if os.path.exists("gsm8k_test.jsonl"):
        print("📁 Dataset file already exists: gsm8k_test.jsonl")
        print("🔍 Verifying existing file...")
        verify_gsm8k_dataset("gsm8k_test.jsonl")
    else:
        # Download fresh copy
        filename = download_gsm8k_dataset()
        if filename:
            print(f"\n🎉 GSM8K dataset ready: {filename}")
        else:
            print("\n❌ Failed to download GSM8K dataset")
            return False
    
    print("\n✅ GSM8K setup complete!")
    print("🚀 Ready to run verified benchmark")
    
    return True

if __name__ == "__main__":
    main()