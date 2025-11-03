# Test Strategy for python_prtree

## Overview
This document defines the comprehensive test strategy for python_prtree, including test classification, feature-perspective matrix, and test organization.

## Test Classification

### 1. Unit Tests (`tests/unit/`)
Tests for individual functions and methods in isolation.

### 2. Integration Tests (`tests/integration/`)
Tests for interactions between multiple components.

### 3. End-to-End Tests (`tests/e2e/`)
Tests for complete user workflows and scenarios.

## Feature-Perspective Matrix

| Feature | Normal | Error | Boundary | Precision | Edge Case | Consistency | Performance |
|---------|--------|-------|----------|-----------|-----------|-------------|-------------|
| **Construction** | ✓ | ✓ | ✓ | ✓ | ✓ | - | - |
| **Query (single)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| **Batch Query** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| **Point Query** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| **Insert** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| **Erase** | ✓ | ✓ | ✓ | - | ✓ | ✓ | - |
| **Save** | ✓ | ✓ | ✓ | ✓ | - | ✓ | - |
| **Load** | ✓ | ✓ | ✓ | ✓ | - | ✓ | - |
| **Rebuild** | ✓ | - | ✓ | - | - | ✓ | - |
| **Query Intersections** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| **Object Handling** | ✓ | ✓ | ✓ | - | ✓ | ✓ | - |
| **Properties (size, len)** | ✓ | - | ✓ | - | - | - | - |

## Test Perspectives

### 1. Normal Cases (正常系)
- Valid inputs with expected behavior
- Common use cases from README

### 2. Error Cases (異常系)
- Invalid inputs (NaN, Inf, negative ranges)
- Non-existent indices
- Invalid file paths
- Type errors
- Empty operations

### 3. Boundary Values (境界値)
- Empty tree (0 elements)
- Single element
- Very large datasets (10k+ elements)
- Very small/large coordinate values
- Zero-volume boxes

### 4. Precision (精度)
- float32 vs float64
- Small gaps (< 1e-5)
- Large magnitude coordinates (> 1e6)
- Precision loss scenarios

### 5. Edge Cases (エッジケース)
- Degenerate boxes (min == max)
- Overlapping boxes
- Touching boxes (closed interval semantics)
- Identical positions
- All boxes intersecting
- No boxes intersecting

### 6. Consistency (一貫性)
- query vs batch_query results
- Results after save/load
- Results after insert/erase
- Results after rebuild

### 7. Performance (パフォーマンス)
- Not covered in unit tests
- Covered in benchmarks/profiling

## Test Organization

### Unit Tests Structure
```
tests/unit/
├── test_construction.py      # Tree initialization
├── test_query.py             # Single query operations
├── test_batch_query.py       # Batch query operations
├── test_insert.py            # Insert operations
├── test_erase.py             # Erase operations
├── test_persistence.py       # Save/load operations
├── test_rebuild.py           # Rebuild operations
├── test_intersections.py     # Query intersections
├── test_object_handling.py   # Object storage/retrieval
├── test_properties.py        # Size, len, n properties
└── test_precision.py         # Float32/64 precision
```

### Integration Tests Structure
```
tests/integration/
├── test_insert_query.py           # Insert → Query workflow
├── test_erase_query.py            # Erase → Query workflow
├── test_rebuild_query.py          # Rebuild → Query workflow
├── test_persistence_query.py      # Save → Load → Query workflow
└── test_mixed_operations.py       # Complex operation sequences
```

### E2E Tests Structure
```
tests/e2e/
├── test_readme_examples.py        # All README examples
├── test_user_workflows.py         # Common user scenarios
└── test_regression.py             # Known bug fixes
```

## Coverage Goals

- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **Feature Coverage**: 100% (all public APIs)

## Test Naming Convention

```python
def test_<feature>_<scenario>_<expected_result>():
    """Test description in Japanese and English."""
    pass
```

Examples:
- `test_query_empty_tree_returns_empty_list()`
- `test_insert_nan_coordinates_raises_error()`
- `test_batch_query_float64_precision_matches_query()`

## Missing Test Cases (Identified Gaps)

### High Priority
1. ✗ Invalid input validation (NaN, Inf, min > max)
2. ✗ Error messages verification
3. ✗ Empty tree operations
4. ✗ Non-existent index operations
5. ✗ Invalid file path handling
6. ✗ Duplicate index handling
7. ✗ Property accessors (__len__, n)

### Medium Priority
1. ✗ Same position boxes
2. ✗ All identical boxes
3. ✗ Type conversion edge cases
4. ✗ Object pickling failures
5. ✗ Concurrent save/load (if supported)

### Low Priority
1. ✗ Memory leak detection
2. ✗ Performance regression tests
3. ✗ Stress tests (millions of boxes)

## Implementation Plan

1. **Phase 1**: Create test directory structure
2. **Phase 2**: Implement unit tests (high priority gaps first)
3. **Phase 3**: Implement integration tests
4. **Phase 4**: Implement E2E tests
5. **Phase 5**: Run coverage analysis and fill gaps
6. **Phase 6**: Documentation and maintenance guide

## Test Execution

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=python_prtree --cov-report=html tests/

# Run specific dimension
pytest -k "PRTree2D"
pytest -k "PRTree3D"
pytest -k "PRTree4D"
```

## Maintenance Guidelines

1. **New Features**: Add tests in all three categories (unit, integration, e2e)
2. **Bug Fixes**: Add regression test in e2e before fixing
3. **Refactoring**: Ensure all tests pass before and after
4. **Dependencies**: Update test fixtures when dependencies change
5. **Documentation**: Update this document when test strategy changes
