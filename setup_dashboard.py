#!/usr/bin/env python3
"""
TOPCART Dashboard Setup
======================

Sets up the performance dashboard for transparent comparison testing.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dashboard_requirements():
    """Install required packages for dashboard"""
    requirements = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0"
    ]
    
    print("📦 Installing dashboard requirements...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    
    return True

def create_dashboard_structure():
    """Create dashboard directory structure"""
    dashboard_dir = Path("dashboard")
    dashboard_dir.mkdir(exist_ok=True)
    
    # Create additional directories
    (dashboard_dir / "data").mkdir(exist_ok=True)
    (dashboard_dir / "results").mkdir(exist_ok=True)
    (dashboard_dir / "static").mkdir(exist_ok=True)
    
    print("📁 Created dashboard directory structure")

def create_launch_script():
    """Create script to launch dashboard"""
    launch_script = """#!/usr/bin/env python3
'''
Launch TOPCART Performance Dashboard
===================================
'''

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 Launching TOPCART Performance Dashboard...")
    print("=" * 60)
    print("📊 This dashboard provides transparent comparison between:")
    print("   🔹 Baseline LLM performance")
    print("   🚀 TOPCART-enhanced performance")
    print("=" * 60)
    
    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent / "dashboard"
    os.chdir(dashboard_dir)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "performance_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("launch_dashboard.py", "w") as f:
        f.write(launch_script)
    
    print("🚀 Created dashboard launch script")

def main():
    """Setup TOPCART dashboard"""
    print("🎯 TOPCART Dashboard Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_dashboard_requirements():
        print("❌ Failed to install requirements")
        return
    
    # Create structure
    create_dashboard_structure()
    
    # Create launch script
    create_launch_script()
    
    print("\n✅ Dashboard setup completed!")
    print("=" * 50)
    print("🚀 To launch the dashboard:")
    print("   python launch_dashboard.py")
    print("\n📊 Dashboard features:")
    print("   ✅ Side-by-side performance comparison")
    print("   ✅ Multiple test types (math, Q&A, code)")
    print("   ✅ Real-time results with visualizations")
    print("   ✅ Transparent methodology")
    print("   ✅ Reproducible results")
    print("\n🎯 This provides verifiable evidence of TOPCART's value!")

if __name__ == "__main__":
    main()