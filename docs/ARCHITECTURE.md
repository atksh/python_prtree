# Project Architecture

This document describes the architecture and directory structure of python_prtree.

## Overview

python_prtree is a Python package that provides fast spatial indexing using the Priority R-Tree data structure. It consists of:

1. **C++ Core**: High-performance implementation of the Priority R-Tree algorithm
2. **Python Bindings**: pybind11-based bindings exposing C++ functionality to Python
3. **Python Wrapper**: User-friendly Python API with additional features

## Directory Structure

```
python_prtree/
├── include/                    # C++ Public Headers (API)
│   └── prtree/
│       ├── core/               # Core algorithm headers
│       │   └── prtree.h        # Main PRTree class template
│       └── utils/              # Utility headers
│           ├── parallel.h      # Parallel processing utilities
│           └── small_vector.h  # Optimized vector implementation
│
├── src/                        # Source Code
│   ├── cpp/                    # C++ Implementation
│   │   ├── core/               # Core implementation (future)
│   │   └── bindings/           # Python bindings
│   │       └── python_bindings.cc  # pybind11 bindings
│   │
│   └── python_prtree/          # Python Package
│       ├── __init__.py         # Package entry point
│       ├── core.py             # PRTree2D/3D/4D classes
│       └── py.typed            # Type hints marker (PEP 561)
│
├── tests/                      # Test Suite
│   ├── unit/                   # Unit tests (individual features)
│   │   ├── test_construction.py
│   │   ├── test_query.py
│   │   ├── test_insert.py
│   │   ├── test_erase.py
│   │   └── ...
│   ├── integration/            # Integration tests (workflows)
│   │   ├── test_insert_query_workflow.py
│   │   ├── test_persistence_query_workflow.py
│   │   └── ...
│   ├── e2e/                    # End-to-end tests
│   │   ├── test_readme_examples.py
│   │   └── test_user_workflows.py
│   └── conftest.py             # Shared test fixtures
│
├── benchmarks/                 # Performance Benchmarks
│   ├── cpp/                    # C++ benchmarks
│   │   ├── benchmark_construction.cpp
│   │   ├── benchmark_query.cpp
│   │   ├── benchmark_parallel.cpp
│   │   └── stress_test_concurrent.cpp
│   └── python/                 # Python benchmarks (future)
│       └── README.md
│
├── docs/                       # Documentation
│   ├── examples/               # Example notebooks and scripts
│   │   └── experiment.ipynb
│   ├── images/                 # Documentation images
│   └── baseline/               # Benchmark baseline data
│
├── tools/                      # Development Tools
│   ├── analyze_baseline.py    # Benchmark analysis
│   ├── profile.py              # Profiling script
│   ├── profile.sh              # Profiling shell script
│   └── profile_all_workloads.sh
│
└── third/                      # Third-party Dependencies (git submodules)
    ├── pybind11/               # Python bindings framework
    ├── cereal/                 # Serialization library
    └── snappy/                 # Compression library
```

## Architectural Layers

### 1. Core C++ Layer (`include/prtree/core/`)

**Purpose**: Implements the Priority R-Tree algorithm

**Key Components**:
- `prtree.h`: Main template class `PRTree<T, B, D>`
  - `T`: Index type (typically `int64_t`)
  - `B`: Branching factor (default: 8)
  - `D`: Dimensions (2, 3, or 4)

**Design Principles**:
- Header-only template library for performance
- No Python dependencies at this layer
- Pure C++ with C++20 features

### 2. Utilities Layer (`include/prtree/utils/`)

**Purpose**: Supporting data structures and algorithms

**Components**:
- `parallel.h`: Thread-safe parallel processing utilities
- `small_vector.h`: Cache-friendly vector with small size optimization

**Design Principles**:
- Reusable utilities independent of PRTree
- Optimized for performance (SSE, cache-locality)

### 3. Python Bindings Layer (`src/cpp/bindings/`)

**Purpose**: Expose C++ functionality to Python using pybind11

**Key File**: `python_bindings.cc`

**Responsibilities**:
- Create Python classes from C++ templates
- Handle numpy array conversions
- Expose methods with Python-friendly signatures
- Provide module-level documentation

**Design Principles**:
- Thin binding layer (minimal logic)
- Direct mapping to C++ API
- Efficient numpy integration

### 4. Python Wrapper Layer (`src/python_prtree/`)

**Purpose**: User-friendly Python API with safety features

**Key Files**:
- `__init__.py`: Package entry point and version info
- `core.py`: Main user-facing classes (`PRTree2D`, `PRTree3D`, `PRTree4D`)

**Added Features**:
- Empty tree safety (prevent segfaults)
- Python object storage (pickle serialization)
- Convenient APIs (auto-indexing, return_obj parameter)
- Type hints and documentation

**Design Principles**:
- Safety over raw performance
- Pythonic API design
- Backwards compatibility considerations

## Data Flow

### Construction
```
User Code
  ↓ (numpy arrays)
PRTree2D/3D/4D (Python)
  ↓ (arrays + validation)
_PRTree2D/3D/4D (pybind11)
  ↓ (type conversion)
PRTree<int64_t, 8, D> (C++)
  ↓ (algorithm)
Optimized R-Tree Structure
```

### Query
```
User Code
  ↓ (query box)
PRTree2D.query() (Python)
  ↓ (empty tree check)
_PRTree2D.query() (pybind11)
  ↓ (type conversion)
PRTree::find_one() (C++)
  ↓ (tree traversal)
Result Indices
  ↓ (optional: object retrieval)
User Code
```

## Separation of Concerns

### By Functionality

1. **Core Algorithm** (`include/prtree/core/`)
   - Spatial indexing logic
   - Tree construction and traversal
   - No I/O, no Python

2. **Utilities** (`include/prtree/utils/`)
   - Generic helpers
   - Reusable across projects

3. **Bindings** (`src/cpp/bindings/`)
   - Python/C++ bridge
   - Type conversions only

4. **Python API** (`src/python_prtree/`)
   - User interface
   - Safety and convenience

### By Testing

1. **Unit Tests** (`tests/unit/`)
   - Test individual features in isolation
   - Fast, focused tests
   - Examples: `test_insert.py`, `test_query.py`

2. **Integration Tests** (`tests/integration/`)
   - Test feature interactions
   - Workflow-based tests
   - Examples: `test_insert_query_workflow.py`

3. **E2E Tests** (`tests/e2e/`)
   - Test complete user scenarios
   - Documentation examples
   - Examples: `test_readme_examples.py`

## Build System

### CMake Configuration

**Key Variables**:
- `PRTREE_SOURCES`: Source files to compile
- `PRTREE_INCLUDE_DIRS`: Header search paths

**Targets**:
- `PRTree`: Main Python extension module
- `benchmark_*`: C++ benchmark executables (optional)

**Options**:
- `BUILD_BENCHMARKS`: Enable benchmark compilation
- `ENABLE_PROFILING`: Build with profiling symbols
- `ENABLE_ASAN/TSAN/UBSAN`: Enable sanitizers

### Build Process

```
User runs: pip install -e .
  ↓
setup.py invoked
  ↓
CMakeBuild.build_extension()
  ↓
CMake configuration
  - Find dependencies (pybind11, cereal, snappy)
  - Set compiler flags
  - Configure include paths
  ↓
CMake build
  - Compile C++ to shared library (.so/.pyd)
  - Link dependencies
  ↓
Extension installed in src/python_prtree/
```

## Design Decisions

### Header-Only Core

**Decision**: Keep core PRTree as header-only template library

**Rationale**:
- Enables full compiler optimization
- Simplifies distribution
- No need for .cc files at core layer

**Trade-offs**:
- Longer compilation times
- Larger binary size

### Separate Bindings File

**Decision**: Single `python_bindings.cc` file separate from core

**Rationale**:
- Clear separation: core C++ vs. Python interface
- Core can be reused in C++-only projects
- Easier to maintain Python API changes

### Python Wrapper Layer

**Decision**: Add Python wrapper on top of pybind11 bindings

**Rationale**:
- Safety: prevent segfaults on empty trees
- Convenience: Pythonic APIs, object storage
- Evolution: can change API without C++ recompilation

**Trade-offs**:
- Extra layer adds slight overhead
- More code to maintain

### Test Organization

**Decision**: Three-tier test structure (unit/integration/e2e)

**Rationale**:
- Fast feedback loop with unit tests
- Comprehensive coverage with integration tests
- Real-world validation with e2e tests
- Easy to run subsets: `pytest tests/unit -v`

## Future Improvements

1. **Split prtree.h**: Large monolithic header could be split into:
   - `prtree_fwd.h`: Forward declarations
   - `prtree_node.h`: Node implementation
   - `prtree_query.h`: Query algorithms
   - `prtree_insert.h`: Insert/erase logic

2. **C++ Core Library**: Extract core into `src/cpp/core/` for:
   - Faster compilation
   - Better code organization
   - Easier testing of C++ layer independently

3. **Python Benchmarks**: Add `benchmarks/python/` for:
   - Performance regression testing
   - Comparison with other Python libraries
   - Memory profiling

4. **Documentation**: Add `docs/api/` with:
   - Sphinx-generated API docs
   - Architecture diagrams
   - Performance tuning guide

## Contributing

When adding new features, follow the separation of concerns:

1. **Core algorithm changes**: Modify `include/prtree/core/prtree.h`
2. **Expose to Python**: Update `src/cpp/bindings/python_bindings.cc`
3. **Python API enhancements**: Update `src/python_prtree/core.py`
4. **Add tests**: Unit tests for features, integration tests for workflows

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed contribution guidelines.

## References

- **Priority R-Tree Paper**: Arge et al., SIGMOD 2004
- **pybind11**: https://pybind11.readthedocs.io/
- **Python Packaging**: PEP 517, PEP 518, PEP 621
