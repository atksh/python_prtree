# Documentation Directory

This directory contains comprehensive documentation for python_prtree developers and contributors.

## Contents

### Core Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Project architecture and design decisions
  - Directory structure and separation of concerns
  - Data flow diagrams
  - Build system overview
  - Native precision support architecture

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development environment setup
  - Prerequisites and installation
  - Build instructions
  - Testing and code quality tools
  - Troubleshooting guide

- **[MIGRATION.md](MIGRATION.md)** - Migration guides between versions
  - v0.7.0 project restructuring guide
  - Breaking changes and migration steps
  - Planned future migrations

### Supplementary Resources

- **baseline/** - Performance baseline measurements
  - System information
  - Benchmark results and analysis
  - Used for regression testing and performance comparison

- **examples/** - Example notebooks and scripts
  - Experimental notebooks for exploring the API
  - Usage demonstrations
  - Prototyping and development examples

- **images/** - Documentation images
  - Benchmark graphs used in README
  - Performance comparison charts
  - Referenced by main documentation

## For Users

If you're a user looking for usage documentation, see:
- [README.md](../README.md) - Main user documentation with examples
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute to the project
- [CHANGES.md](../CHANGES.md) - Version history and changelog

## For Developers

Start with these files in order:
1. [README.md](../README.md) - Understand what the library does
2. [DEVELOPMENT.md](DEVELOPMENT.md) - Set up your development environment
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the codebase structure
4. [CONTRIBUTING.md](../CONTRIBUTING.md) - Learn the contribution workflow

## Keeping Documentation Updated

When making changes:
- Update ARCHITECTURE.md if you change the project structure
- Update DEVELOPMENT.md if you change build/test processes
- Update MIGRATION.md when introducing breaking changes
- Regenerate benchmarks if performance characteristics change
