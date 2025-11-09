.PHONY: help init build build-release test test-verbose test-coverage clean clean-all install dev-install format lint docs check-submodules

# Default target
.DEFAULT_GOAL := help

# Colors for output
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# Python environment
PYTHON := python3
PYTEST := $(PYTHON) -m pytest
PIP := $(PYTHON) -m pip

# Project directories
SRC_DIR := src/python_prtree
CPP_DIR := src/cpp
TEST_DIR := tests
BUILD_DIR := build
DIST_DIR := dist

# Set PYTHONPATH
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

help: ## Show help message
	@echo "$(BOLD)$(BLUE)python_prtree Development Makefile$(RESET)"
	@echo ""
	@echo "$(BOLD)Available commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Development workflow:$(RESET)"
	@echo "  1. $(YELLOW)make init$(RESET)        - Initial setup"
	@echo "  2. $(YELLOW)make build$(RESET)       - Build C++ extension"
	@echo "  3. $(YELLOW)make test$(RESET)        - Run tests"
	@echo "  4. $(YELLOW)make clean$(RESET)       - Clean up"
	@echo ""

init: check-deps init-submodules ## Initial setup (initialize submodules + install dependencies)
	@echo "$(BOLD)$(GREEN)✓ Initialization complete$(RESET)"

check-deps: ## Check for required dependencies
	@echo "$(BOLD)Checking dependencies...$(RESET)"
	@command -v git >/dev/null 2>&1 || { echo "$(BOLD)Error: git is not installed$(RESET)" >&2; exit 1; }
	@command -v cmake >/dev/null 2>&1 || { echo "$(BOLD)Error: cmake is not installed$(RESET)" >&2; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(BOLD)Error: python3 is not installed$(RESET)" >&2; exit 1; }
	@echo "$(GREEN)✓ All required dependencies found$(RESET)"

init-submodules: ## Initialize git submodules
	@echo "$(BOLD)Initializing submodules...$(RESET)"
	@if [ ! -f third/pybind11/CMakeLists.txt ]; then \
		git submodule update --init --recursive; \
		echo "$(GREEN)✓ Submodules initialized$(RESET)"; \
	else \
		echo "$(YELLOW)Submodules already initialized$(RESET)"; \
	fi

check-submodules: ## Check submodule status
	@if [ ! -f third/pybind11/CMakeLists.txt ]; then \
		echo "$(BOLD)$(YELLOW)Warning: Submodules not initialized$(RESET)"; \
		echo "$(YELLOW)Please run: make init$(RESET)"; \
		exit 1; \
	fi

build: check-submodules ## Build C++ extension (in-place, debug mode)
	@echo "$(BOLD)Building C++ extension (debug mode)...$(RESET)"
	$(PYTHON) setup.py build_ext --inplace
	@echo "$(GREEN)✓ Build complete$(RESET)"

build-release: check-submodules ## Build C++ extension (release mode)
	@echo "$(BOLD)Building C++ extension (release mode)...$(RESET)"
	CMAKE_BUILD_TYPE=Release $(PYTHON) setup.py build_ext --inplace
	@echo "$(GREEN)✓ Release build complete$(RESET)"

rebuild: clean build ## Clean build (clean + build)

test: build ## Run tests
	@echo "$(BOLD)Running tests...$(RESET)"
	$(PYTEST) $(TEST_DIR) -v
	@echo "$(GREEN)✓ All tests passed$(RESET)"

test-verbose: build ## Run tests in verbose mode
	@echo "$(BOLD)Running tests (verbose mode)...$(RESET)"
	$(PYTEST) $(TEST_DIR) -vv --tb=long

test-fast: build ## Run tests in parallel (fast)
	@echo "$(BOLD)Running tests in parallel...$(RESET)"
	$(PYTEST) $(TEST_DIR) -v -n auto
	@echo "$(GREEN)✓ All tests passed$(RESET)"

test-coverage: build ## Run tests with coverage
	@echo "$(BOLD)Running tests with coverage...$(RESET)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(RESET)"

test-one: build ## Run specific tests (e.g., make test-one TEST=test_result)
	@if [ -z "$(TEST)" ]; then \
		echo "$(BOLD)Error: TEST variable not specified$(RESET)"; \
		echo "Example: make test-one TEST=test_result"; \
		exit 1; \
	fi
	@echo "$(BOLD)Running test $(TEST)...$(RESET)"
	$(PYTEST) $(TEST_DIR) -k "$(TEST)" -v

clean: ## Remove build artifacts
	@echo "$(BOLD)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.so' -delete
	find . -type f -name '*.a' -delete
	find $(SRC_DIR) -name '*.so' -delete 2>/dev/null || true
	find $(SRC_DIR) -name '*.a' -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"

clean-all: clean ## Clean everything (including submodules)
	@echo "$(BOLD)Cleaning everything...$(RESET)"
	git submodule deinit -f --all 2>/dev/null || true
	rm -rf third/pybind11/*
	rm -rf third/snappy/*
	@echo "$(GREEN)✓ Complete cleanup done$(RESET)"
	@echo "$(YELLOW)Note: Please run 'make init' again$(RESET)"

install: ## Install package
	@echo "$(BOLD)Installing package...$(RESET)"
	$(PIP) install .
	@echo "$(GREEN)✓ Installation complete$(RESET)"

dev-install: ## Install in development mode with all dependencies
	@echo "$(BOLD)Installing in development mode...$(RESET)"
	$(PIP) install -e ".[dev,docs,benchmark]"
	@echo "$(GREEN)✓ Development installation complete$(RESET)"

install-deps: ## Install development dependencies
	@echo "$(BOLD)Installing development dependencies...$(RESET)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

format: ## Format code (Python with black, C++ with clang-format)
	@echo "$(BOLD)Formatting Python code...$(RESET)"
	@if command -v black >/dev/null 2>&1 || $(PYTHON) -m black --version >/dev/null 2>&1; then \
		$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR); \
		echo "$(GREEN)✓ Python formatting complete$(RESET)"; \
	else \
		echo "$(YELLOW)Warning: black not installed (pip install black)$(RESET)"; \
	fi
	@echo "$(BOLD)Formatting C++ code...$(RESET)"
	@if command -v clang-format >/dev/null 2>&1; then \
		find $(CPP_DIR) -name '*.h' -o -name '*.cc' | xargs clang-format -i; \
		echo "$(GREEN)✓ C++ formatting complete$(RESET)"; \
	else \
		echo "$(YELLOW)Warning: clang-format not installed$(RESET)"; \
	fi

lint-cpp: ## Lint C++ code (requires clang-tidy)
	@if command -v clang-tidy >/dev/null 2>&1; then \
		echo "$(BOLD)Linting C++ code...$(RESET)"; \
		find $(CPP_DIR) -name '*.cc' | xargs clang-tidy; \
	else \
		echo "$(YELLOW)Warning: clang-tidy not installed$(RESET)"; \
	fi

lint-python: ## Lint Python code (requires ruff)
	@echo "$(BOLD)Linting Python code with ruff...$(RESET)"
	@if command -v ruff >/dev/null 2>&1 || $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
		$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR); \
		echo "$(GREEN)✓ Linting complete$(RESET)"; \
	else \
		echo "$(YELLOW)Warning: ruff not installed (pip install ruff)$(RESET)"; \
	fi

type-check: ## Type check Python code (requires mypy)
	@echo "$(BOLD)Type checking Python code...$(RESET)"
	@if command -v mypy >/dev/null 2>&1 || $(PYTHON) -m mypy --version >/dev/null 2>&1; then \
		$(PYTHON) -m mypy $(SRC_DIR); \
		echo "$(GREEN)✓ Type checking complete$(RESET)"; \
	else \
		echo "$(YELLOW)Warning: mypy not installed (pip install mypy)$(RESET)"; \
	fi

lint: lint-cpp lint-python type-check ## Lint all code

docs: ## Generate documentation (requires Doxygen)
	@if command -v doxygen >/dev/null 2>&1; then \
		echo "$(BOLD)Generating documentation...$(RESET)"; \
		doxygen Doxyfile 2>/dev/null || echo "$(YELLOW)Doxyfile not found$(RESET)"; \
	else \
		echo "$(YELLOW)Warning: doxygen not installed$(RESET)"; \
	fi

benchmark: build ## Run benchmarks (if benchmark script exists)
	@if [ -f benchmark.py ]; then \
		echo "$(BOLD)Running benchmarks...$(RESET)"; \
		$(PYTHON) benchmark.py; \
	else \
		echo "$(YELLOW)benchmark.py not found$(RESET)"; \
	fi

wheel: check-submodules ## Build wheel package
	@echo "$(BOLD)Building wheel package...$(RESET)"
	$(PYTHON) setup.py bdist_wheel
	@echo "$(GREEN)✓ Wheel package created: $(DIST_DIR)/$(RESET)"
	@ls -lh $(DIST_DIR)/*.whl 2>/dev/null || true

sdist: ## Build source distribution
	@echo "$(BOLD)Building source distribution...$(RESET)"
	$(PYTHON) setup.py sdist
	@echo "$(GREEN)✓ Source distribution created: $(DIST_DIR)/$(RESET)"

release: clean check-submodules wheel sdist ## Build release packages (wheel + sdist)
	@echo "$(BOLD)$(GREEN)✓ Release packages ready$(RESET)"
	@echo "Distribution files:"
	@ls -lh $(DIST_DIR)/

check: build test ## Run build and tests (for CI)
	@echo "$(BOLD)$(GREEN)✓ All checks passed$(RESET)"

watch-test: ## Run tests in watch mode (requires pytest-watch)
	@if command -v ptw >/dev/null 2>&1; then \
		echo "$(BOLD)Starting test watch mode...$(RESET)"; \
		ptw -- $(TEST_DIR) -v; \
	else \
		echo "$(YELLOW)pytest-watch not installed$(RESET)"; \
		echo "Install: pip install pytest-watch"; \
	fi

debug-build: ## Build with debug information
	@echo "$(BOLD)Building with debug info...$(RESET)"
	CMAKE_BUILD_TYPE=Debug $(PYTHON) setup.py build_ext --inplace
	@echo "$(GREEN)✓ Debug build complete$(RESET)"

info: ## Show project information
	@echo "$(BOLD)$(BLUE)Project Information$(RESET)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "CMake: $$(cmake --version | head -n1)"
	@echo "Git: $$(git --version)"
	@echo ""
	@echo "$(BOLD)Project structure:$(RESET)"
	@echo "Source directory: $(SRC_DIR)"
	@echo "C++ directory: $(CPP_DIR)"
	@echo "Test directory: $(TEST_DIR)"
	@echo ""
	@echo "$(BOLD)Submodule status:$(RESET)"
	@git submodule status

# Quick development targets
quick: clean build test ## Quick test (clean + build + test)

dev: init install-deps build ## Setup development environment
	@echo "$(BOLD)$(GREEN)✓ Development environment setup complete$(RESET)"
	@echo ""
	@echo "Next steps:"
	@echo "  - $(YELLOW)make test$(RESET) to run tests"
	@echo "  - $(YELLOW)make watch-test$(RESET) to start auto-testing (requires pytest-watch)"
