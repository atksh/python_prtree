# C++ Public Headers

This directory contains the public C++ API for python_prtree.

## Structure

```
include/prtree/
├── core/              # Core algorithm implementation
│   ├── prtree.h      # Main PRTree class template
│   └── detail/       # Implementation details (future modularization)
└── utils/            # Utility headers
    ├── parallel.h    # Parallel processing utilities
    └── small_vector.h # Optimized small vector
```

## Usage

### From C++ (if using as library)

```cpp
#include "prtree/core/prtree.h"

// Use the PRTree
PRTree<int64_t, 8, 2> tree;
```

### Include Paths

When building, add this to your include path:
```cmake
target_include_directories(your_target PRIVATE ${PROJECT_SOURCE_DIR}/include)
```

## Design Principles

1. **Header-Only**: Core algorithm is template-based, header-only
2. **Modular**: Separate concerns (core, utils, bindings)
3. **No Python Dependencies**: Core can be used independently of Python
4. **C++20**: Uses modern C++ features (concepts, ranges, etc.)

## Modularization

The current `prtree.h` is a large file (1617 lines). See `core/detail/README.md` for the planned modularization strategy.

## For Contributors

- Core algorithm changes: modify `core/prtree.h`
- Utility additions: add to `utils/`
- Keep headers self-contained (include all dependencies)
- Document public APIs with doxygen-style comments
- Follow C++ Core Guidelines

For more details, see [ARCHITECTURE.md](../ARCHITECTURE.md).
