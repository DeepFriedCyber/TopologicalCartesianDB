"""
Example demonstrating the use of Hodge Laplacian-based anomaly detection 
for cybersecurity applications
"""

import sys
import os
import json
import random
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions.hodge_anomaly import CyberHodgeDetector

def generate_sample_network_data(num_records=1000, num_anomalies=5):
    """Generate synthetic network connection data with some anomalies"""
    normal_ips = [
        f"192.168.1.{i}" for i in range(1, 50)
    ]
    
    common_ports = [80, 443, 22, 25, 53, 3389, 8080]
    
    # Generate baseline traffic
    data = []
    start_time = datetime.now() - timedelta(hours=24)
    
    for i in range(num_records):
        # Normal connections between internal IPs
        src_ip = random.choice(normal_ips)
        dest_ip = random.choice([ip for ip in normal_ips if ip != src_ip])
        port = random.choice(common_ports)
        
        # Random timestamp within last 24 hours
        timestamp = start_time + timedelta(seconds=random.randint(0, 24*3600))
        
        # Bytes transferred follow a log-normal distribution
        bytes_transferred = int(10 ** random.normalvariate(3, 1))  # mostly 1KB to 10MB
        
        data.append({
            'source_ip': src_ip,
            'dest_ip': dest_ip,
            'port': port,
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'bytes_transferred': bytes_transferred
        })
    
    # Add anomalies
    for i in range(num_anomalies):
        # Unusual connections - external IPs, weird ports, odd timings, large data transfers
        src_ip = f"10.0.0.{random.randint(1, 10)}"
        dest_ip = f"203.0.113.{random.randint(1, 255)}"  # External IP from TEST-NET-3
        port = random.randint(10000, 65000)  # Unusual high port
        
        # Late night timestamp
        hour = random.randint(1, 5)  # 1am to 5am
        timestamp = start_time.replace(hour=hour, minute=random.randint(0, 59))
        
        # Unusually large data transfer
        bytes_transferred = 10 ** random.normalvariate(6, 1)  # ~1GB-ish
        
        anomaly = {
            'source_ip': src_ip,
            'dest_ip': dest_ip,
            'port': port,
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'bytes_transferred': int(bytes_transferred)
        }
        
        # Insert anomaly at random position
        data.insert(random.randint(0, len(data)), anomaly)
    
    return data

def main():
    # Create a cyber detector instance
    detector = CyberHodgeDetector()
    
    print("Generating synthetic network data...")
    network_data = generate_sample_network_data(1000, 7)
    
    print("Establishing baseline...")
    baseline = detector.establish_network_baseline(network_data[:800])
    
    print("Detecting network anomalies...")
    results = detector.detect_network_anomalies(network_data)
    
    print(f"\nFound {len(results['anomalous_connections'])} anomalous connections out of {results['total_connections']} total")
    
    # Display the first 3 anomalies with explanations
    print("\nTop anomalies:")
    for i, idx in enumerate(results['anomaly_indices'][:3]):
        conn = network_data[idx]
        print(f"\nAnomaly #{i+1}:")
        print(f"  Source IP: {conn['source_ip']}")
        print(f"  Dest IP:   {conn['dest_ip']}")
        print(f"  Port:      {conn['port']}")
        print(f"  Time:      {conn['timestamp']}")
        print(f"  Bytes:     {conn['bytes_transferred']}")
        print("\nExplanation:")
        print(detector.explain_anomaly(idx, network_data=network_data))
    
    print("\nFile system scanning example:")
    print("Scanning current directory for anomalies...")
    
    # Scan this script's directory
    fs_results = detector.scan_filesystem(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Found {len(fs_results['anomalous_files'])} anomalous files out of {fs_results['total_files']} total")
    
    # Display first 3 anomalous files with explanations
    if fs_results['anomalous_files']:
        print("\nTop file anomalies:")
        for i, file_path in enumerate(fs_results['anomalous_files'][:3]):
            idx = fs_results['anomaly_indices'][i]
            print(f"\nAnomaly #{i+1}: {os.path.basename(file_path)}")
            print(detector.explain_anomaly(idx, file_paths=fs_results['anomalous_files']))

if __name__ == "__main__":
    main()