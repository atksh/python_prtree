# Development Guide

Welcome to the python_prtree development guide! This document will help you get started with contributing to the project.

## Project Structure

```
python_prtree/
├── src/                    # Python source code
│   └── python_prtree/     # Main package
├── cpp/                    # C++ implementation
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── tools/                  # Development tools and scripts
├── benchmarks/             # Performance benchmarks
├── docs/                   # Documentation
├── .github/workflows/      # CI/CD configuration
└── third/                  # Third-party dependencies (git submodules)
```

## Prerequisites

- Python 3.8 or higher
- CMake 3.22 or higher
- C++17 compatible compiler
- Git (for submodules)

### Platform-Specific Requirements

**macOS:**
```bash
brew install cmake
```

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake build-essential
```

**Windows:**
- Visual Studio 2019 or later with C++ development tools
- CMake (can be installed via Visual Studio installer or from cmake.org)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/atksh/python_prtree.git
cd python_prtree
```

### 2. Initialize Submodules

The project uses git submodules for third-party dependencies:

```bash
git submodule update --init --recursive
```

Or use the Makefile:

```bash
make init
```

### 3. Set Up Development Environment

#### Using pip (recommended)

```bash
# Install in development mode with all dependencies
pip install -e ".[dev,docs,benchmark]"
```

#### Using make

```bash
# Initialize submodules and install dependencies
make dev
```

This will:
- Initialize git submodules
- Install the package in editable mode
- Install all development dependencies

### 4. Build the C++ Extension

```bash
# Build in debug mode (default)
make build

# Or build in release mode
make build-release
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run tests in parallel (faster)
make test-fast

# Run tests with coverage report
make test-coverage

# Run specific test
make test-one TEST=test_insert
```

Or use pytest directly:

```bash
pytest tests -v
pytest tests/unit/test_insert.py -v
pytest tests -k "test_insert" -v
```

### Code Quality

#### Format Code

```bash
# Format both Python and C++ code
make format

# Format only Python (uses black)
python -m black src/ tests/

# Format only C++ (uses clang-format)
clang-format -i cpp/*.cc cpp/*.h
```

#### Lint Code

```bash
# Lint all code
make lint

# Lint only Python (uses ruff)
make lint-python

# Lint only C++ (uses clang-tidy)
make lint-cpp

# Type check Python code (uses mypy)
make type-check
```

### Building Documentation

```bash
make docs
```

### Cleaning Build Artifacts

```bash
# Remove build artifacts
make clean

# Clean everything including submodules
make clean-all
```

## Project Configuration

All project metadata and dependencies are defined in `pyproject.toml`:

- **Project metadata**: name, version, description, authors
- **Dependencies**: runtime and development dependencies
- **Build system**: setuptools with CMake integration
- **Tool configurations**: pytest, black, ruff, mypy, coverage

## Testing Guidelines

### Test Organization

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Tests for component interactions
- `tests/e2e/`: End-to-end workflow tests
- `tests/legacy/`: Legacy test suite

### Writing Tests

```python
import pytest
from python_prtree import PRTree

def test_basic_insertion():
    """Test basic rectangle insertion."""
    tree = PRTree()
    tree.insert([0, 0, 10, 10], "rect1")
    assert tree.size() == 1

def test_query():
    """Test rectangle query."""
    tree = PRTree()
    tree.insert([0, 0, 10, 10], "rect1")
    results = tree.query([5, 5, 15, 15])
    assert len(results) > 0
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit -v

# Run only integration tests
pytest tests/integration -v

# Run only e2e tests
pytest tests/e2e -v
```

## C++ Development

### Building with Debug Symbols

```bash
make debug-build
```

### Profiling

```bash
# Run profiling scripts
./tools/profile.sh
python tools/profile.py
```

### Benchmarks

```bash
# Run benchmarks (if available)
make benchmark
```

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **Pull Requests**: Runs unit tests on multiple platforms (Linux, macOS, Windows) and Python versions (3.8-3.14)
- **Main Branch**: Builds wheels for all platforms and Python versions
- **Version Tags**: Publishes packages to PyPI

## Making Changes

### Workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes and write tests

3. Run tests and linting:
   ```bash
   make test
   make lint
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. Push and create a pull request:
   ```bash
   git push origin feature/my-feature
   ```

### Code Style

- **Python**: Follow PEP 8, use black for formatting (100 char line length)
- **C++**: Follow Google C++ Style Guide, use clang-format
- **Commits**: Use conventional commit messages
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `test:` for test changes
  - `refactor:` for refactoring
  - `chore:` for maintenance tasks

## Troubleshooting

### Submodules Not Initialized

```bash
git submodule update --init --recursive
```

### Build Fails

1. Ensure CMake is installed and up to date
2. Check that all submodules are initialized
3. Try cleaning and rebuilding:
   ```bash
   make clean
   make build
   ```

### Tests Fail

1. Ensure the extension is built:
   ```bash
   make build
   ```

2. Check that all dependencies are installed:
   ```bash
   pip install -e ".[dev]"
   ```

### Import Errors

Ensure you've installed the package in development mode:
```bash
pip install -e .
```

## Additional Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [README.md](README.md) - Project overview
- [CHANGES.md](CHANGES.md) - Version history
- [GitHub Issues](https://github.com/atksh/python_prtree/issues) - Bug reports and feature requests

## Questions?

If you have questions or need help, please:

1. Check existing [GitHub Issues](https://github.com/atksh/python_prtree/issues)
2. Open a new issue with your question
3. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details

Happy coding! 🎉
