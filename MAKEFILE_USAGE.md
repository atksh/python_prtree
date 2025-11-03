# Makefile Usage Guide

This document provides a quick reference for all available Make commands in the python_prtree project.

## Quick Start

```bash
# First time setup
make dev

# Build and test
make build
make test
```

## Command Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make dev` | Complete development setup (init + install-deps + build) |
| `make build` | Build C++ extension |
| `make test` | Run all tests |
| `make clean` | Remove build artifacts |

### Initialization

| Command | Description |
|---------|-------------|
| `make init` | Initialize submodules and check dependencies |
| `make check-deps` | Verify required tools are installed |
| `make init-submodules` | Initialize git submodules |
| `make install-deps` | Install Python development dependencies |

### Building

| Command | Description |
|---------|-------------|
| `make build` | Build in debug mode (default) |
| `make build-release` | Build optimized release version |
| `make rebuild` | Clean and rebuild |
| `make debug-build` | Build with debug symbols |

### Testing

| Command | Description | Example |
|---------|-------------|---------|
| `make test` | Run all tests | |
| `make test-verbose` | Run tests with detailed output | |
| `make test-fast` | Run tests in parallel | |
| `make test-coverage` | Generate coverage report | |
| `make test-one` | Run specific test(s) | `make test-one TEST=test_result` |

### Code Quality

| Command | Description | Requirements |
|---------|-------------|--------------|
| `make format` | Format C++ code | clang-format |
| `make lint-cpp` | Lint C++ code | clang-tidy |
| `make lint-python` | Lint Python code | flake8 |
| `make lint` | Lint all code | clang-tidy, flake8 |

### Packaging

| Command | Description |
|---------|-------------|
| `make wheel` | Build wheel package |
| `make sdist` | Build source distribution |
| `make release` | Build both wheel and sdist |

### Maintenance

| Command | Description |
|---------|-------------|
| `make clean` | Remove build artifacts |
| `make clean-all` | Remove everything including submodules |
| `make info` | Show project and environment info |
| `make check` | Run build + test (for CI) |

### Other

| Command | Description | Requirements |
|---------|-------------|--------------|
| `make docs` | Generate documentation | Doxygen |
| `make benchmark` | Run benchmarks | benchmark.py |
| `make watch-test` | Auto-run tests on file changes | pytest-watch |

## Common Workflows

### First Time Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/atksh/python_prtree
cd python_prtree

# Setup development environment
make dev
```

### Daily Development

```bash
# Make changes to code...

# Build and test
make rebuild
make test

# Or use quick command
make quick  # clean + build + test
```

### Before Committing

```bash
# Format and lint
make format
make lint

# Run full test suite
make test

# Check everything
make check
```

### Testing Specific Features

```bash
# Run tests matching a pattern
make test-one TEST=test_query

# This will run all tests with "test_query" in the name
```

### Release Preparation

```bash
# Clean everything
make clean

# Run all checks
make check

# Build release packages
make release
```

## Troubleshooting

### "Submodules not initialized"

```bash
make init
```

### Build failures

```bash
make clean
make build
```

### Test failures

```bash
# Run in verbose mode to see details
make test-verbose

# Check environment
make info
```

### CMake cache issues

```bash
rm -rf build
make build
```

## Environment Variables

The Makefile automatically sets:

- `PYTHONPATH`: Includes `src/` directory for imports

You can customize:

- `PYTHON`: Python executable (default: `python3`)
- `CMAKE_BUILD_TYPE`: Build type for CMake

Example:
```bash
PYTHON=python3.11 make build
```

## Tips

1. **Parallel Testing**: Use `make test-fast` to run tests in parallel
2. **Coverage Reports**: Use `make test-coverage` and open `htmlcov/index.html`
3. **Watch Mode**: Install pytest-watch (`pip install pytest-watch`) and use `make watch-test`
4. **Incremental Builds**: `make build` only rebuilds changed files
5. **Clean Slate**: Use `make rebuild` or `make quick` for a fresh build

## Integration with IDEs

### VS Code

Add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build",
      "type": "shell",
      "command": "make build",
      "group": "build"
    },
    {
      "label": "Test",
      "type": "shell",
      "command": "make test",
      "group": "test"
    }
  ]
}
```

### PyCharm

Configure External Tools:
- Settings → Tools → External Tools → Add
- Program: `make`
- Arguments: `build` (or any other command)
- Working directory: `$ProjectFileDir$`

## See Also

- `CONTRIBUTING.md`: Full development guide
- `README.md`: User documentation
- `make help`: List all commands
