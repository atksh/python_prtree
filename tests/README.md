# Test Suite for python_prtree

This directory contains a comprehensive test suite for python_prtree, organized by test type and functionality.

## Directory Structure

```
tests/
├── unit/                    # Unit tests (individual features)
│   ├── test_construction.py
│   ├── test_query.py
│   ├── test_batch_query.py
│   ├── test_insert.py
│   ├── test_erase.py
│   ├── test_persistence.py
│   ├── test_rebuild.py
│   ├── test_intersections.py
│   ├── test_object_handling.py
│   ├── test_properties.py
│   └── test_precision.py
│
├── integration/             # Integration tests (feature combinations)
│   ├── test_insert_query_workflow.py
│   ├── test_erase_query_workflow.py
│   ├── test_persistence_query_workflow.py
│   ├── test_rebuild_query_workflow.py
│   └── test_mixed_operations.py
│
├── e2e/                     # End-to-end tests (user scenarios)
│   ├── test_readme_examples.py
│   ├── test_regression.py
│   └── test_user_workflows.py
│
├── legacy/                  # Original test file (kept for reference)
│   └── test_PRTree.py
│
├── conftest.py             # Shared fixtures and configuration
└── README.md               # This file

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# E2E tests only
pytest tests/e2e/
```

### Run specific test file
```bash
pytest tests/unit/test_construction.py
```

### Run tests for specific dimension
```bash
# Run all PRTree2D tests
pytest tests/ -k "PRTree2D"

# Run all PRTree3D tests
pytest tests/ -k "PRTree3D"

# Run all PRTree4D tests
pytest tests/ -k "PRTree4D"
```

### Run with coverage
```bash
pytest --cov=python_prtree --cov-report=html tests/
```

### Run with verbose output
```bash
pytest -v tests/
```

### Run specific test by name
```bash
pytest tests/unit/test_construction.py::TestNormalConstruction::test_construction_with_valid_inputs
```

## Test Organization

### Unit Tests (`tests/unit/`)
Test individual functions and methods in isolation:
- **test_construction.py**: Tree initialization and construction
- **test_query.py**: Single query operations
- **test_batch_query.py**: Batch query operations
- **test_insert.py**: Insert operations
- **test_erase.py**: Erase operations
- **test_persistence.py**: Save/load operations
- **test_rebuild.py**: Rebuild operations
- **test_intersections.py**: Query intersections operations
- **test_object_handling.py**: Object storage and retrieval
- **test_properties.py**: Properties (size, len, n)
- **test_precision.py**: Float32/64 precision handling
- **test_segfault_safety.py**: Segmentation fault safety tests
- **test_crash_isolation.py**: Crash isolation tests (subprocess)
- **test_memory_safety.py**: Memory safety and bounds checking
- **test_concurrency.py**: Python threading/multiprocessing/async tests
- **test_parallel_configuration.py**: Parallel execution configuration tests

### Integration Tests (`tests/integration/`)
Test interactions between multiple components:
- **test_insert_query_workflow.py**: Insert → Query workflows
- **test_erase_query_workflow.py**: Erase → Query workflows
- **test_persistence_query_workflow.py**: Save → Load → Query workflows
- **test_rebuild_query_workflow.py**: Rebuild → Query workflows
- **test_mixed_operations.py**: Complex operation sequences

### End-to-End Tests (`tests/e2e/`)
Test complete user workflows and scenarios:
- **test_readme_examples.py**: All examples from README
- **test_regression.py**: Known bug fixes and edge cases
- **test_user_workflows.py**: Common user scenarios

## Test Coverage

The test suite covers:
- ✅ All public APIs (PRTree2D, PRTree3D, PRTree4D)
- ✅ Normal cases (happy path)
- ✅ Error cases (invalid inputs)
- ✅ Boundary values (empty, single, large datasets)
- ✅ Precision cases (float32 vs float64)
- ✅ Edge cases (degenerate boxes, touching boxes, etc.)
- ✅ Consistency (query vs batch_query, save/load, etc.)
- ✅ Known regressions (bugs from issues)
- ✅ Memory safety (segfault prevention, bounds checking)
- ✅ Concurrency (threading, multiprocessing, async)
- ✅ Parallel execution (batch_query parallelization)

## Test Matrix

See [docs/TEST_STRATEGY.md](../docs/TEST_STRATEGY.md) for the complete feature-perspective test matrix.

## Adding New Tests

When adding new tests:

1. **Choose the right category**:
   - Unit tests: Testing a single feature in isolation
   - Integration tests: Testing multiple features together
   - E2E tests: Testing complete user workflows

2. **Follow naming conventions**:
   ```python
   def test_<feature>_<scenario>_<expected>():
       """Test description in Japanese and English."""
       pass
   ```

3. **Use parametrization** for dimension testing:
   ```python
   @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
   def test_my_feature(PRTree, dim):
       pass
   ```

4. **Use shared fixtures** from `conftest.py` when appropriate

5. **Update TEST_STRATEGY.md** if adding new test perspectives

## Continuous Integration

These tests are run automatically on:
- Every pull request
- Every push to main branch
- Scheduled daily builds

See `.github/workflows/` for CI configuration.

## Known Issues

- Some tests may take longer on slower systems due to large dataset sizes
- Float precision tests are sensitive to numpy/system math libraries
- File I/O tests require write permissions in tmp_path

## Contributing

When contributing tests:
1. Ensure all tests pass locally before submitting PR
2. Add tests for any new features or bug fixes
3. Update this README if adding new test categories
4. Aim for >90% line coverage and >85% branch coverage
