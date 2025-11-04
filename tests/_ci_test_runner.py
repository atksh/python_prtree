#!/usr/bin/env python
"""
CI test runner that adapts test execution based on platform.

For emulated platforms (aarch64, musllinux), skip heavy concurrency/stress tests
that can hang or take excessive time under QEMU emulation.

For native platforms (x86_64, win_amd64, macosx), run full test suite.
"""
import os
import sys
import subprocess
import platform

def get_platform_info():
    """Determine if we're running on an emulated platform."""
    platform_id = os.environ.get('CIBW_PLATFORM_ID', '')
    
    is_emulated = (
        'aarch64' in platform_id or 
        'musllinux' in platform_id or
        platform.machine() == 'aarch64'
    )
    
    return platform_id, is_emulated

def main():
    """Run tests appropriate for the current platform."""
    platform_id, is_emulated = get_platform_info()
    
    print(f"Platform ID: {platform_id}")
    print(f"Machine: {platform.machine()}")
    print(f"Is emulated/slow platform: {is_emulated}")
    
    import_test = os.path.join(os.path.dirname(__file__), '_ci_debug_import.py')
    print(f"\n=== Running import test: {import_test} ===")
    result = subprocess.run([sys.executable, import_test])
    if result.returncode != 0:
        print("Import test failed!")
        return result.returncode
    
    test_dir = os.path.dirname(__file__)
    
    if is_emulated:
        print("\n=== Running lightweight test suite (emulated platform) ===")
        ignore_args = [
            '--ignore=tests/unit/test_concurrency.py',
            '--ignore=tests/unit/test_memory_safety.py', 
            '--ignore=tests/unit/test_comprehensive_safety.py',
            '--ignore=tests/unit/test_segfault_safety.py',
        ]
        cmd = [sys.executable, '-m', 'pytest', test_dir, '-vv'] + ignore_args
    else:
        print("\n=== Running full test suite (native platform) ===")
        cmd = [sys.executable, '-m', 'pytest', test_dir, '-vv']
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
