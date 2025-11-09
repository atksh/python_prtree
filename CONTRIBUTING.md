# Contributing Guide

Thank you for your interest in contributing to python_prtree!

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CMake 3.12 or higher
- C++20-compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- Git

### Quick Start

```bash
# Clone the repository
git clone --recursive https://github.com/atksh/python_prtree
cd python_prtree

# Setup development environment (first time only)
make dev

# Build and test
make build
make test
```

## Makefile Commands

We provide a Makefile to streamline the development workflow.

### Show Help

```bash
make help
```

### Initial Setup

```bash
make init              # Initialize submodules + check dependencies
make install-deps      # Install development dependencies
make dev              # Run init + install-deps + build at once
```

### Building

```bash
make build            # Build in debug mode
make build-release    # Build in release mode
make rebuild          # Clean build (clean + build)
make debug-build      # Build with debug symbols
```

### Testing

```bash
make test                    # Run all tests
make test-verbose            # Run tests in verbose mode
make test-fast               # Run tests in parallel (faster)
make test-coverage           # Run tests with coverage
make test-one TEST=<name>    # Run specific test(s)
```

### Cleanup

```bash
make clean      # Remove build artifacts
make clean-all  # Remove everything including submodules
```

### Packaging

```bash
make wheel     # Build wheel package
make sdist     # Build source distribution
make release   # Build release packages (wheel + sdist)
```

### Other Commands

```bash
make format      # Format C++ code (requires clang-format)
make lint        # Lint code
make info        # Show project information
make check       # Run build and tests (for CI)
make quick       # Quick test (clean + build + test)
```

## Development Workflow

### Adding New Features

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Setup development environment (first time only)**
   ```bash
   make dev
   ```

3. **Make changes**
   - C++ core: `include/prtree/core/prtree.h`
   - Python bindings: `src/cpp/bindings/python_bindings.cc`
   - Python wrapper: `src/python_prtree/core.py`
   - Tests: `tests/unit/`, `tests/integration/`, `tests/e2e/`

4. **Build and test**
   ```bash
   make rebuild
   make test
   ```

5. **Check code quality**
   ```bash
   make format  # Format code
   make lint    # Lint code
   ```

6. **Commit**
   ```bash
   git add -A
   git commit -m "Add: description of new feature"
   ```

7. **Create pull request**

### Test-Driven Development (TDD)

1. **Write tests first**
   ```python
   # tests/test_PRTree.py
   def test_new_feature():
       # Write test code
       pass
   ```

2. **Verify test fails**
   ```bash
   make test-one TEST=test_new_feature
   ```

3. **Implement feature**
   ```cpp
   // include/prtree/core/prtree.h
   // Add implementation
   ```

4. **Build and test**
   ```bash
   make build
   make test-one TEST=test_new_feature
   ```

5. **Run all tests**
   ```bash
   make test
   ```

### Debugging

Build with debug symbols for debugging:

```bash
make debug-build
gdb python3
(gdb) run -c "from python_prtree import PRTree2D; ..."
```

### Checking Coverage

```bash
make test-coverage
# Open htmlcov/index.html in browser
```

## Coding Standards

### C++

- **Style**: Follow Google C++ Style Guide
- **Formatting**: Use clang-format (`make format`)
- **Naming conventions**:
  - Classes: `PascalCase` (e.g., `PRTree`)
  - Functions/methods: `snake_case` (e.g., `batch_query`)
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`

### Python

- **Style**: Follow PEP 8
- **Line length**: Maximum 100 characters
- **Documentation**: Use docstrings

### Tests

- Add tests for all new features
- Test case naming: `test_<feature>_<case>`
- Cover edge cases
- Use parameterized tests (`@pytest.mark.parametrize`)

## Project Structure

```
python_prtree/
├── include/               # C++ public headers
│   └── prtree/
│       ├── core/          # Core algorithm headers
│       │   └── prtree.h   # PRTree core implementation
│       └── utils/         # Utility headers
│           ├── parallel.h # Parallel processing utilities
│           └── small_vector.h # Optimized vector
├── src/
│   ├── cpp/               # C++ implementation
│   │   └── bindings/      # Python bindings
│   │       └── python_bindings.cc
│   └── python_prtree/     # Python wrapper
│       ├── __init__.py    # Package entry point
│       └── core.py        # Main user-facing classes
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── third/                 # Third-party libraries (submodules)
│   ├── pybind11/
│   └── snappy/
├── CMakeLists.txt         # CMake configuration
├── pyproject.toml         # Project metadata and dependencies
├── setup.py               # Build configuration
├── Makefile              # Development workflow
└── README.md             # User documentation
```

## Troubleshooting

### Submodules Not Found

```bash
make clean-all
make init
```

### Build Errors

```bash
make clean
make build
```

### Test Failures

1. Verify build succeeded
   ```bash
   make build
   ```

2. Check environment variables
   ```bash
   echo $PYTHONPATH  # Should include src directory
   ```

3. Run in verbose mode
   ```bash
   make test-verbose
   ```

### CMake Errors

Clear CMake cache:
```bash
rm -rf build
make build
```

## Continuous Integration (CI)

When you create a pull request, the following checks run automatically:

- Build verification
- All tests
- Code coverage

Run the same checks locally:
```bash
make check
```

## Release Process

1. **Update version**
   - Update version number in `setup.py`

2. **Update changelog**
   - Update "New Features and Changes" section in `README.md`

3. **Run tests**
   ```bash
   make clean
   make check
   ```

4. **Build release packages**
   ```bash
   make release
   ```

5. **Create tag**
   ```bash
   git tag -a v0.x.x -m "Release v0.x.x"
   git push origin v0.x.x
   ```

## Questions and Support

- **Issues**: https://github.com/atksh/python_prtree/issues
- **Discussions**: https://github.com/atksh/python_prtree/discussions

## License

Contributions to this project will be released under the same license as the project (MIT).
