# Python Package

This directory contains the Python package for python_prtree.

## Structure

```
python_prtree/
├── __init__.py       # Package entry point
├── core.py           # PRTree2D/3D/4D classes
└── py.typed          # PEP 561 type hints marker
```

## Module Responsibilities

### `__init__.py`
- Package initialization
- Version information
- Public API exports (`PRTree2D`, `PRTree3D`, `PRTree4D`)
- Top-level documentation

### `core.py`
- Main user-facing classes
- Python wrapper around C++ bindings
- Safety features (empty tree handling)
- Convenience features (object storage, auto-indexing)
- Type hints and comprehensive docstrings

### `py.typed`
- Marker file for PEP 561
- Indicates package supports type checking
- Enables IDE autocompletion with types

## Architecture

```
User Code
    ↓
PRTree2D/3D/4D (core.py)
    ↓ (Python wrapper with safety)
_PRTree2D/3D/4D (C++ binding)
    ↓ (pybind11 bridge)
PRTree<T,B,D> (C++ core)
```

## Design Principles

1. **Pythonic API**: Natural Python interface
2. **Safety First**: Prevent segfaults, validate inputs
3. **Type Hints**: Full typing support
4. **Documentation**: Comprehensive docstrings
5. **Backwards Compatibility**: Maintain API stability

## For Contributors

### Adding New Features

1. **C++ Side**: Implement in `include/prtree/core/prtree.h`
2. **Binding**: Expose in `src/cpp/bindings/python_bindings.cc`
3. **Python Wrapper**: Add to `core.py` with safety checks
4. **Export**: Add to `__all__` in `__init__.py`
5. **Document**: Add docstrings and type hints
6. **Test**: Add tests in `tests/`

### Example: Adding a new method

```python
# In core.py
class PRTreeBase:
    def new_method(self, param: int) -> List[int]:
        """
        Description of new method.

        Args:
            param: Parameter description

        Returns:
            List of results
        """
        # Safety checks
        if self.n == 0:
            return []

        # Call C++ implementation
        return self._tree.new_method(param)
```

### Code Style

- Follow PEP 8
- Use type hints everywhere
- Write comprehensive docstrings (Google style)
- Run `make format` and `make lint` before committing

See [DEVELOPMENT.md](../../DEVELOPMENT.md) for complete development guidelines.
