# Segmentation Fault Safety Testing

This document describes the comprehensive segmentation fault (segfault) safety testing strategy for python_prtree.

## Overview

As python_prtree is implemented in C++/Cython, it's critical to ensure memory safety and prevent segmentation faults. Our test suite includes extensive testing for potential crash scenarios.

## Test Categories

### 1. Null Pointer Safety (`test_segfault_safety.py`)
Tests protection against null pointer dereferences:
- Query on uninitialized tree
- Erase on empty tree
- Get object on empty tree
- Access to deleted elements

### 2. Use-After-Free Protection
Tests scenarios that could cause use-after-free errors:
- Query after erase
- Access after rebuild
- Query after save
- Double-free attempts (erase same index twice)

### 3. Buffer Overflow Protection
Tests protection against buffer overflows:
- Very large indices (2^31 - 1)
- Very negative indices (-2^31)
- Extremely large coordinates (1e100+)

### 4. Array Bounds Safety
Tests protection against array bounds violations:
- Empty array input
- Wrong-shaped boxes
- 1D boxes (should be 2D array)
- 3D boxes (invalid shape)
- Mismatched array lengths

### 5. Memory Leak Detection
Tests for potential memory leaks:
- Repeated insert/erase cycles
- Repeated save/load cycles
- Tree deletion and recreation

### 6. Corrupted Data Handling
Tests handling of corrupted or invalid data:
- Loading corrupted binary files
- Loading empty files
- Loading partially truncated files
- Random bytes as input

### 7. Concurrent Access Safety
Tests thread safety and concurrent access:
- Query during modification
- Multiple threads querying
- Insert during iteration
- Save/load during queries

### 8. Object Lifecycle Management
Tests proper object lifecycle:
- Tree deletion and recreation
- Circular reference safety
- Garbage collection cycles
- Numpy array lifecycle

### 9. Extreme Inputs
Tests extreme and unusual inputs:
- All NaN boxes
- Mixed NaN and valid values
- Zero-size boxes
- Subnormal numbers
- Very large datasets (100k+ elements)

### 10. Type Safety
Tests type conversion and validation:
- Wrong dtype indices (float instead of int)
- String indices
- None inputs
- Unsigned integer indices
- Float16 boxes

## Crash Isolation Tests (`test_crash_isolation.py`)

These tests run potentially dangerous operations in isolated subprocesses to prevent crashes from affecting the test suite. Each test:
1. Runs code in a subprocess
2. Checks exit code (0 = success, -11 = segfault on Unix)
3. Verifies no segmentation fault occurred

Test categories:
- Double-free protection
- Invalid memory access
- File corruption handling
- Stress conditions
- Boundary conditions
- Object pickling safety
- Multiple tree interaction
- Race conditions

## Memory Safety Tests (`test_memory_safety.py`)

Comprehensive memory bounds checking and validation:
- Input validation (negative box dimensions, misaligned arrays)
- Memory bounds (out-of-bounds index access)
- Garbage collection interaction
- Edge case arrays (subnormal numbers, mixed special values)
- Concurrent modification protection
- Resource exhaustion handling
- Various numpy dtypes

## Concurrency Tests (`test_concurrency.py`)

Tests for Python-level concurrency:

### Threading Tests
- Concurrent queries from multiple threads
- Concurrent batch queries
- Read-only concurrent access
- Thread pool executor compatibility
- Simultaneous read-write with protection

### Multiprocessing Tests
- Concurrent queries from multiple processes
- Process pool executor compatibility
- Independent tree instances per process

### Async/Await Tests
- Async query operations
- Async batch query operations
- Event loop compatibility

### Data Race Protection
- Reader/writer thread coordination
- Lock-based protection verification

## Parallel Configuration Tests (`test_parallel_configuration.py`)

Tests for C++ std::thread parallelization in batch_query:

### Scaling Tests
- Different query counts (10, 100, 1000)
- Different tree sizes (100, 1000, 10000)
- Performance scaling verification

### Correctness Tests
- Batch vs single query consistency
- Deterministic results
- No data races in parallel execution
- Duplicate query handling

### Edge Cases
- Single query batch
- Empty tree batch query
- Single element tree

### query_intersections Parallel Tests
- Scaling with tree size
- Deterministic results
- Correctness verification

## Running Segfault Tests

### Run all safety tests
```bash
pytest tests/unit/test_segfault_safety.py -v
pytest tests/unit/test_crash_isolation.py -v
pytest tests/unit/test_memory_safety.py -v
```

### Run concurrency tests
```bash
pytest tests/unit/test_concurrency.py -v
pytest tests/unit/test_parallel_configuration.py -v
```

### Run with different thread counts
```bash
pytest tests/unit/test_concurrency.py -v -k "num_threads"
pytest tests/unit/test_parallel_configuration.py -v -k "batch_size"
```

### Run crash isolation tests (slower)
```bash
# These tests run in subprocesses and may be slower
pytest tests/unit/test_crash_isolation.py -v --timeout=60
```

## Expected Behavior

### Safe Failure
Tests verify that invalid operations fail gracefully with Python exceptions rather than crashing:
- `ValueError`: Invalid input (NaN, Inf, min > max)
- `RuntimeError`: C++ runtime error
- `KeyError`/`IndexError`: Invalid index access
- `OSError`: File I/O errors

### No Segfaults
All tests verify that operations never cause segmentation faults, even with:
- Invalid inputs
- Corrupted data
- Extreme values
- Concurrent access
- Memory exhaustion

## Coverage Goals

- **Crash Safety**: 100% of crash scenarios handled safely
- **Memory Safety**: All memory operations validated
- **Thread Safety**: All concurrent access patterns tested
- **Input Validation**: All invalid inputs rejected gracefully

## Implementation Notes

### C++ Safety Features
The library should implement:
- Null pointer checks
- Bounds checking
- Input validation
- Thread-safe data structures (or GIL protection)
- Exception handling at C++/Python boundary

### Python Safety Features
The Python wrapper should:
- Validate inputs before passing to C++
- Handle exceptions from C++ layer
- Manage object lifecycle properly
- Provide thread-safe operations (via GIL or locks)

## Debugging Segfaults

If a segfault occurs:

1. **Run under debugger**:
   ```bash
   gdb python
   (gdb) run -m pytest tests/unit/test_segfault_safety.py::test_name
   (gdb) backtrace
   ```

2. **Enable core dumps**:
   ```bash
   ulimit -c unlimited
   pytest tests/unit/test_segfault_safety.py
   # If crash occurs, analyze core dump
   gdb python core
   ```

3. **Use AddressSanitizer** (if available):
   ```bash
   # Rebuild with ASAN
   CFLAGS="-fsanitize=address" pip install -e .
   pytest tests/unit/test_segfault_safety.py
   ```

4. **Use Valgrind**:
   ```bash
   valgrind --leak-check=full python -m pytest tests/unit/test_segfault_safety.py
   ```

## Contributing

When adding new features:
1. Add corresponding safety tests
2. Test with invalid inputs
3. Test with extreme values
4. Test concurrent access if applicable
5. Run all segfault safety tests before committing

## Known Safe Operations

Based on testing, the following operations are known to be safe:
- ✅ Query on empty tree (returns empty list)
- ✅ Invalid inputs (raise ValueError/RuntimeError)
- ✅ Concurrent read-only queries
- ✅ Save/load cycles
- ✅ Large datasets (up to memory limits)
- ✅ Garbage collection
- ✅ Parallel batch queries
- ✅ Async/await contexts

## Known Limitations

Document any known limitations:
- Maximum index value (if limited)
- Maximum tree size (memory dependent)
- Thread safety guarantees (GIL-dependent vs. thread-safe)
- Concurrent modification behavior

## References

- [Python C API Memory Management](https://docs.python.org/3/c-api/memory.html)
- [Cython Best Practices](https://cython.readthedocs.io/en/latest/src/userguide/best_practices.html)
- [C++ Thread Safety](https://en.cppreference.com/w/cpp/thread)
