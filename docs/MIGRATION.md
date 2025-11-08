# Migration Guide

This document helps users migrate between major versions and structural changes.

## v0.7.0 Project Restructuring

### Overview

Version 0.7.0 introduces a major project restructuring with clear separation of concerns. **The Python API remains 100% backwards compatible** - no code changes are needed.

### What Changed

#### For End Users (Python API)

**No action required!** All existing code continues to work:

```python
from python_prtree import PRTree2D

# All existing code works exactly the same
tree = PRTree2D([1, 2], [[0, 0, 1, 1], [2, 2, 3, 3]])
results = tree.query([0.5, 0.5, 2.5, 2.5])
```

#### For Contributors (Project Structure)

If you've been developing on the codebase, note these changes:

**Directory Structure Changes:**

```
Old Structure              →  New Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cpp/                      →  include/prtree/core/
  ├── prtree.h            →    └── prtree.h
  ├── parallel.h          →  include/prtree/utils/parallel.h
  ├── small_vector.h      →  include/prtree/utils/small_vector.h
  └── main.cc             →  src/cpp/bindings/python_bindings.cc

src/python_prtree/        →  src/python_prtree/
  └── __init__.py         →    ├── __init__.py (simplified)
                          →    ├── core.py (new, main classes)
                          →    └── py.typed (new, type hints)

benchmarks/               →  benchmarks/
  └── *.cpp               →    ├── cpp/ (C++ benchmarks)
                          →    └── python/ (future)

docs/                     →  docs/
  ├── experiment.ipynb    →    ├── examples/experiment.ipynb
  ├── images/             →    ├── images/
  └── baseline/           →    └── baseline/

scripts/                  →  tools/ (consolidated)
run_*.sh                  →  tools/*.sh
```

**Build System:**

- `requirements.txt` → removed (use `pyproject.toml`)
- `requirements-dev.txt` → removed (use `pip install -e ".[dev]"`)
- CMake paths updated to use `include/` and `src/cpp/`

**Development Workflow:**

```bash
# Old way
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# New way (single command)
pip install -e ".[dev]"
```

### Migration Steps for Contributors

#### 1. Update Your Development Environment

```bash
# Clean old build artifacts
make clean

# Update dependencies
pip install -e ".[dev]"

# Rebuild
make build
```

#### 2. Update Include Paths (if you have C++ code)

```cpp
// Old includes
#include "prtree.h"
#include "parallel.h"

// New includes
#include "prtree/core/prtree.h"
#include "prtree/utils/parallel.h"
```

#### 3. Update Git Submodules

```bash
git submodule update --init --recursive
```

#### 4. Update Your Fork

```bash
git pull upstream main
git push origin main
```

### Benefits of New Structure

1. **Clear Separation**: C++ core, bindings, and Python API are clearly separated
2. **Better Documentation**: Each layer has its own README
3. **Modern Tooling**: Uses pyproject.toml, type hints, modern linters
4. **Easier Contribution**: Clear where to add code for different types of changes
5. **Future-Ready**: Structure supports future modularization and improvements

### Troubleshooting

#### Build Errors

**Error**: `prtree.h: No such file or directory`

**Solution**: Clean and rebuild:
```bash
make clean
git submodule update --init --recursive
make build
```

#### Import Errors

**Error**: `ImportError: cannot import name 'PRTree2D'`

**Solution**: Reinstall the package:
```bash
pip uninstall python-prtree
pip install -e ".[dev]"
```

#### Test Failures

**Error**: Tests fail after upgrading

**Solution**: Ensure you're on the latest version:
```bash
git pull
pip install -e ".[dev]"
make test
```

### Getting Help

If you encounter issues during migration:

1. Check existing [GitHub Issues](https://github.com/atksh/python_prtree/issues)
2. See [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions
3. See [ARCHITECTURE.md](ARCHITECTURE.md) for structure details
4. Open a new issue with:
   - Your Python version
   - Your OS
   - Error messages
   - Steps you've tried

## Future Migrations

### v0.8.0 (Planned): C++ Modularization

The large `prtree.h` file (1617 lines) will be split into modules:

```
prtree.h → {
  prtree/core/detail/types.h
  prtree/core/detail/bounding_box.h
  prtree/core/detail/nodes.h
  prtree/core/detail/pseudo_tree.h
  prtree/core/prtree.h (main interface)
}
```

**Impact**: None for Python users. C++ users will need to include the main header only.

### v1.0.0 (Future): Stable API

Version 1.0 will mark API stability:
- Semantic versioning strictly followed
- No breaking changes without major version bump
- Long-term support for stable API

Stay tuned for updates!
