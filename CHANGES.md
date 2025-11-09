# PRTree Improvements

## v0.7.0 - Native Precision Support (2025-01-XX)

### Major Architectural Changes

#### 1. Native Float32/Float64 Precision
- **Previous**: Float32 tree + idx2exact map + double precision refinement
- **New**: Native float32 and float64 tree implementations
- **Benefit**: Simpler code, better performance, true precision throughout
- **Impact**: ~72 lines of code removed, no conversion overhead

**Implementation Details:**
- Templated `PRTree<T, B, D, Real>` with `Real` type parameter (float or double)
- Propagated `Real` parameter through entire class hierarchy:
  - `BB<D, Real>`: Bounding boxes
  - `DataType<T, D, Real>`: Data storage
  - `PRTreeNode<T, B, D, Real>`: Tree nodes
  - `PRTreeLeaf<T, B, D, Real>`: Leaf nodes
  - `PseudoPRTree<T, B, D, Real>`: Builder helper
- Exposed 6 C++ classes via pybind11: `_PRTree{2D,3D,4D}_{float32,float64}`
- Python wrapper auto-selects precision based on numpy dtype

**Breaking Change:**
- Previous files saved with float64 input must be loaded with the correct precision
- Solution: Auto-detection when loading from files (tries float32, then float64)

#### 2. Advanced Precision Control
- **Adaptive epsilon**: Automatically scales epsilon based on bounding box sizes
- **Configurable epsilon**: Set relative and absolute epsilon for edge cases
- **Subnormal detection**: Correctly handles denormalized floating-point numbers
- **Methods added**:
  ```python
  tree.set_adaptive_epsilon(bool)
  tree.set_relative_epsilon(float)
  tree.set_absolute_epsilon(float)
  tree.set_subnormal_detection(bool)
  tree.get_adaptive_epsilon() -> bool
  tree.get_relative_epsilon() -> float
  tree.get_absolute_epsilon() -> float
  tree.get_subnormal_detection() -> bool
  ```

#### 3. Query Precision Fixes
- **Issue**: Query methods (`find_one`, `find_all`) used hardcoded `float` type
- **Fix**: Templated with `Real` to match tree precision
- **Impact**: Float64 trees now maintain full precision in queries

#### 4. Python Wrapper Enhancements
- **Auto-detection on load**: Automatically tries both precisions when loading from file
- **Preserve settings on insert**: First insert on empty tree now preserves precision settings
- **Subnormal workaround**: Handles edge case of inserting with subnormal detection disabled

### Testing

✅ **991/991 tests pass** (including 14 new adaptive epsilon tests)

New test coverage:
- `test_adaptive_epsilon.py`: 14 tests covering edge cases
- `test_save_load_float32_no_regression`: Precision preservation across save/load
- Float32 vs float64 precision validation tests

### Performance

- **No regression**: Construction and query performance unchanged
- **Memory reduction**: Eliminated idx2exact map overhead
- **Code simplification**: ~72 lines removed, improved maintainability

### Bug Fixes

1. **Float64 precision loss in queries** (critical)
   - Query methods forced float32, losing precision
   - Fixed: Template query methods with Real parameter

2. **Precision settings lost on first insert**
   - Python wrapper recreated tree without preserving settings
   - Fixed: Preserve all precision settings when recreating

3. **File load precision mismatch**
   - Loading float32 file with float64 class caused std::bad_alloc
   - Fixed: Auto-detect precision by trying both classes

## Previous Releases

### Critical Fixes

#### 1. Windows Crash Fixed
- **Issue**: Fatal crash with `std::mutex` (not copyable, caused deadlocks)
- **Fix**: Use `std::unique_ptr<std::recursive_mutex>`
- **Result**: Thread-safe, no crashes, pybind11 compatible

#### 2. Error Messages
- Improved with context while maintaining backward compatibility
- Example: `"Given index is not found. (Index: 999, tree size: 2)"`

### Improvements Applied

- **C++20**: Migrated standard, added concepts for type safety
- **Exception Safety**: noexcept + RAII (no memory leaks)
- **Thread Safety**: Recursive mutex protects all mutable operations

### Performance Baseline

- Construction: 9-11M ops/sec (single-threaded)
- Memory: 23 bytes/element
- Parallel scaling: Limited by algorithm (Amdahl's law), not implementation

## Future Work

- Parallel partitioning algorithm for better thread scaling (2-3x expected)
- Split large prtree.h into modular components
- Additional precision validation modes
