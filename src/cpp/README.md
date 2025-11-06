# C++ Source Code

This directory contains C++ implementation files.

## Structure

```
src/cpp/
├── bindings/          # Python bindings (pybind11)
│   └── python_bindings.cc
└── core/             # Core implementation (future)
```

## Current Organization

### bindings/

Python bindings using pybind11. This layer:
- Exposes C++ PRTree to Python
- Handles numpy array conversions
- Provides Python-friendly method signatures
- Documents the Python API

**Key File**: `python_bindings.cc`
- Defines Python module `PRTree`
- Exposes `_PRTree2D`, `_PRTree3D`, `_PRTree4D` classes
- Handles type conversions between Python and C++

## Design Principles

1. **Thin Bindings**: Keep binding layer minimal
2. **Direct Mapping**: Map C++ methods to Python 1:1
3. **Type Safety**: Use pybind11 type checking
4. **Documentation**: Provide docstrings at binding level

## Future Organization

As the codebase grows, implementation files may be added:

```
src/cpp/
├── core/             # Core implementation files (.cc)
│   ├── prtree.cc    # PRTree implementation (if split from header)
│   └── ...
└── bindings/        # Python bindings
    └── python_bindings.cc
```

## For Contributors

### Adding New Methods

1. Implement in C++ header (`include/prtree/core/prtree.h`)
2. Expose in bindings (`bindings/python_bindings.cc`)
3. Add Python wrapper if needed (`src/python_prtree/core.py`)
4. Add tests (`tests/`)

### Building

```bash
# Build C++ extension
make build

# Or directly with setup.py
python setup.py build_ext --inplace
```

See [DEVELOPMENT.md](../../DEVELOPMENT.md) for complete build instructions.
