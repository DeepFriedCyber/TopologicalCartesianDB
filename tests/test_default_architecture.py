#!/usr/bin/env python3
"""
Test Default TOPCART Architecture

Verifies that the multi-cube orchestrator is used by default
and provides instructions for ensuring consistent usage.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_default_import():
    """Test that importing TOPCART automatically configures multi-cube mode"""
    
    print("Testing default TOPCART import behavior...")
    
    # Import TOPCART - this should automatically configure multi-cube mode
    import topological_cartesian as topcart
    
    # Verify configuration
    config = topcart.get_topcart_config()
    validation = topcart.validate_topcart_architecture()
    
    print(f"✅ TOPCART imported successfully")
    print(f"   Mode: {config.mode.value}")
    print(f"   Orchestrator Forced: {config.force_orchestrator}")
    print(f"   DNN Enabled: {config.enable_dnn_optimization}")
    
    # Check if multi-cube mode is active
    if config.mode.value == 'multi_cube' and config.force_orchestrator:
        print(f"✅ Multi-cube orchestrator is the default architecture")
        return True
    else:
        print(f"❌ Multi-cube orchestrator is NOT the default")
        return False

def test_default_system_creation():
    """Test creating default TOPCART system"""
    
    print(f"\nTesting default TOPCART system creation...")
    
    import topological_cartesian as topcart
    
    # Create default system
    system = topcart.create_default_topcart()
    
    # Verify it's a multi-cube orchestrator
    if hasattr(system, 'cubes') and len(system.cubes) > 1:
        print(f"✅ Default system is multi-cube orchestrator with {len(system.cubes)} cubes")
        
        # List cubes
        for cube_name, cube in system.cubes.items():
            print(f"   • {cube_name}: {cube.specialization}")
        
        return True
    else:
        print(f"❌ Default system is NOT multi-cube orchestrator")
        return False

def test_convenience_aliases():
    """Test convenience aliases work correctly"""
    
    print(f"\nTesting convenience aliases...")
    
    import topological_cartesian as topcart
    
    # Test TOPCART alias
    system1 = topcart.TOPCART()
    
    # Test MultiCubeTOPCART alias  
    system2 = topcart.MultiCubeTOPCART()
    
    # Both should be multi-cube systems
    is_system1_multicube = hasattr(system1, 'cubes') and len(system1.cubes) > 1
    is_system2_multicube = hasattr(system2, 'cubes') and len(system2.cubes) > 1
    
    if is_system1_multicube and is_system2_multicube:
        print(f"✅ Both TOPCART() and MultiCubeTOPCART() create multi-cube systems")
        return True
    else:
        print(f"❌ Convenience aliases don't create multi-cube systems")
        return False

def show_usage_examples():
    """Show examples of how to use TOPCART consistently"""
    
    print(f"\n" + "="*60)
    print("TOPCART USAGE EXAMPLES")
    print("="*60)
    
    print(f"""
🚀 RECOMMENDED USAGE (Always Multi-Cube):

1. Simple Import and Use:
   ```python
   import topological_cartesian as topcart
   
   # This automatically creates multi-cube orchestrator
   system = topcart.create_default_topcart()
   
   # Or use convenience alias
   system = topcart.TOPCART()
   ```

2. Explicit Multi-Cube Creation:
   ```python
   from topological_cartesian import MultiCubeOrchestrator
   
   system = MultiCubeOrchestrator(enable_dnn_optimization=True)
   ```

3. Force Multi-Cube Mode (if needed):
   ```python
   import topological_cartesian as topcart
   
   topcart.force_multi_cube_architecture()
   system = topcart.create_default_topcart()
   ```

4. Benchmark Mode:
   ```python
   import topological_cartesian as topcart
   
   topcart.enable_benchmark_mode()
   system = topcart.create_default_topcart()
   ```

5. Validate Architecture:
   ```python
   import topological_cartesian as topcart
   
   topcart.print_topcart_status()
   validation = topcart.validate_topcart_architecture()
   ```

🎯 ENVIRONMENT VARIABLES (Optional):
   export TOPCART_MODE=multi_cube
   export TOPCART_FORCE_ORCHESTRATOR=true
   export TOPCART_ENABLE_DNN=true
   export TOPCART_BENCHMARK=true

✅ This ensures multi-cube orchestrator is ALWAYS used!
""")

def run_all_tests():
    """Run all architecture tests"""
    
    print("TOPCART Default Architecture Tests")
    print("="*50)
    
    tests = [
        ("Default Import", test_default_import),
        ("Default System Creation", test_default_system_creation),
        ("Convenience Aliases", test_convenience_aliases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print(f"🎉 ALL TESTS PASSED - Multi-cube orchestrator is the default!")
    else:
        print(f"⚠️ Some tests failed - Check configuration")
    
    # Show usage examples
    show_usage_examples()
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print(f"\n🚀 CONCLUSION: TOPCART multi-cube orchestrator architecture is properly configured as default!")
    else:
        print(f"\n❌ CONCLUSION: Configuration needs adjustment")
    
    sys.exit(0 if success else 1)