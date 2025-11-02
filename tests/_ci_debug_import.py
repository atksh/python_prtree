#!/usr/bin/env python
"""Debug script to diagnose import issues in CI environments."""
import sys
import pathlib
import importlib.util
import traceback

print("=" * 60)
print("CI Debug Import Check")
print("=" * 60)
print(f"sys.version: {sys.version}")
print(f"sys.platform: {sys.platform}")
print(f"sys.executable: {sys.executable}")
print()

try:
    import python_prtree
    print(f"✓ python_prtree imported successfully")
    print(f"  Location: {python_prtree.__file__}")
    
    pkg_dir = pathlib.Path(python_prtree.__file__).parent
    print(f"  Package directory: {pkg_dir}")
    print(f"  Contents: {sorted(x.name for x in pkg_dir.iterdir())}")
    print()
    
    spec = importlib.util.find_spec("python_prtree.PRTree")
    print(f"  find_spec('python_prtree.PRTree'): {spec}")
    print()
    
    from python_prtree import PRTree3D
    print(f"✓ PRTree3D imported successfully")
    print(f"  PRTree3D: {PRTree3D}")
    print()
    
    print("=" * 60)
    print("All imports successful!")
    print("=" * 60)
    sys.exit(0)
    
except Exception as e:
    print(f"✗ IMPORT FAILED: {repr(e)}")
    print()
    traceback.print_exc()
    print()
    print("=" * 60)
    print("Import check failed - see traceback above")
    print("=" * 60)
    sys.exit(1)
