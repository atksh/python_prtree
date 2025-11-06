#!/bin/bash
# Phase 0: Automated Profiling Script
# Runs all benchmarks with hardware counters, flamegraphs, and cache analysis

set -e  # Exit on error

# Configuration
BUILD_DIR="build_profile"
BASELINE_DIR="docs/baseline"
REPORTS_DIR="${BASELINE_DIR}/reports"
FLAMEGRAPH_DIR="${BASELINE_DIR}/flamegraphs"
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,branch-instructions,branch-misses"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Phase 0: Microarchitectural Profiling"
echo "========================================"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found: $BUILD_DIR${NC}"
    echo "Please run:"
    echo "  mkdir -p $BUILD_DIR"
    echo "  cd $BUILD_DIR"
    echo "  cmake -DBUILD_BENCHMARKS=ON -DENABLE_PROFILING=ON .."
    echo "  make -j\$(nproc)"
    exit 1
fi

# Check if benchmarks exist
if [ ! -f "$BUILD_DIR/benchmark_construction" ]; then
    echo -e "${RED}Error: Benchmarks not built${NC}"
    echo "Please build with: cd $BUILD_DIR && make -j\$(nproc)"
    exit 1
fi

# Create output directories
mkdir -p "$REPORTS_DIR"
mkdir -p "$FLAMEGRAPH_DIR"

# Workloads to profile
WORKLOADS=("small_uniform" "large_uniform" "clustered" "skewed" "sequential")

echo -e "${GREEN}Step 1: Hardware Performance Counters${NC}"
echo "========================================"
echo ""

# Function to run perf stat for a benchmark
run_perf_stat() {
    local benchmark=$1
    local workload=$2
    local output_file=$3

    echo "Profiling: $benchmark - $workload"

    if command -v perf &> /dev/null; then
        perf stat -e $PERF_EVENTS \
            -o "$output_file" \
            "$BUILD_DIR/$benchmark" "$workload" 2>&1 | tee -a "$output_file"
        echo -e "${GREEN}✓${NC} Saved to $output_file"
    else
        echo -e "${YELLOW}⚠ perf not available, skipping${NC}"
        echo "Install with: sudo apt-get install linux-tools-generic" >> "$output_file"
    fi
    echo ""
}

# Profile construction benchmarks
echo "Construction Benchmarks:"
for workload in "${WORKLOADS[@]}"; do
    run_perf_stat "benchmark_construction" "$workload" "$REPORTS_DIR/perf_construction_${workload}.txt"
done

# Profile query benchmarks
echo "Query Benchmarks:"
for workload in "${WORKLOADS[@]}"; do
    run_perf_stat "benchmark_query" "$workload" "$REPORTS_DIR/perf_query_${workload}.txt"
done

echo -e "${GREEN}Step 2: Flamegraph Generation${NC}"
echo "========================================"
echo ""

# Function to generate flamegraph
generate_flamegraph() {
    local benchmark=$1
    local workload=$2
    local output_svg=$3

    echo "Generating flamegraph: $benchmark - $workload"

    if command -v perf &> /dev/null; then
        # Record
        perf record --call-graph dwarf -F 99 -o "$REPORTS_DIR/perf_${benchmark}_${workload}.data" \
            "$BUILD_DIR/$benchmark" "$workload" > /dev/null 2>&1

        # Generate call graph report
        perf report --stdio -i "$REPORTS_DIR/perf_${benchmark}_${workload}.data" \
            > "$REPORTS_DIR/callgraph_${benchmark}_${workload}.txt" 2>&1

        # Generate flamegraph if flamegraph tool available
        if [ -d "FlameGraph" ]; then
            perf script -i "$REPORTS_DIR/perf_${benchmark}_${workload}.data" | \
                FlameGraph/stackcollapse-perf.pl | \
                FlameGraph/flamegraph.pl > "$output_svg"
            echo -e "${GREEN}✓${NC} Flamegraph saved to $output_svg"
        else
            echo -e "${YELLOW}⚠ FlameGraph tool not found${NC}"
            echo "Clone with: git clone https://github.com/brendangregg/FlameGraph.git"
        fi
    else
        echo -e "${YELLOW}⚠ perf not available, skipping${NC}"
    fi
    echo ""
}

# Generate flamegraphs for key workloads
generate_flamegraph "benchmark_construction" "small_uniform" "$FLAMEGRAPH_DIR/construction_small.svg"
generate_flamegraph "benchmark_construction" "large_uniform" "$FLAMEGRAPH_DIR/construction_large.svg"
generate_flamegraph "benchmark_construction" "clustered" "$FLAMEGRAPH_DIR/construction_clustered.svg"
generate_flamegraph "benchmark_query" "small_uniform" "$FLAMEGRAPH_DIR/query_small.svg"
generate_flamegraph "benchmark_query" "large_uniform" "$FLAMEGRAPH_DIR/query_large.svg"

# Parallel benchmark flamegraph
if [ -f "$BUILD_DIR/benchmark_parallel" ]; then
    echo "Generating flamegraph: benchmark_parallel"
    if command -v perf &> /dev/null; then
        perf record --call-graph dwarf -F 99 -o "$REPORTS_DIR/perf_parallel.data" \
            "$BUILD_DIR/benchmark_parallel" > /dev/null 2>&1
        perf report --stdio -i "$REPORTS_DIR/perf_parallel.data" \
            > "$REPORTS_DIR/callgraph_parallel.txt" 2>&1

        if [ -d "FlameGraph" ]; then
            perf script -i "$REPORTS_DIR/perf_parallel.data" | \
                FlameGraph/stackcollapse-perf.pl | \
                FlameGraph/flamegraph.pl > "$FLAMEGRAPH_DIR/batch_query_parallel.svg"
            echo -e "${GREEN}✓${NC} Flamegraph saved"
        fi
    fi
    echo ""
fi

echo -e "${GREEN}Step 3: Cache Analysis (Cachegrind)${NC}"
echo "========================================"
echo ""

# Function to run cachegrind
run_cachegrind() {
    local benchmark=$1
    local workload=$2
    local output_file=$3

    echo "Cache profiling: $benchmark - $workload"

    if command -v valgrind &> /dev/null; then
        valgrind --tool=cachegrind \
            --cachegrind-out-file="$REPORTS_DIR/cachegrind_${benchmark}_${workload}.out" \
            "$BUILD_DIR/$benchmark" "$workload" > /dev/null 2>&1

        if command -v cg_annotate &> /dev/null; then
            cg_annotate "$REPORTS_DIR/cachegrind_${benchmark}_${workload}.out" \
                > "$output_file" 2>&1
            echo -e "${GREEN}✓${NC} Cache report saved to $output_file"
        fi
    else
        echo -e "${YELLOW}⚠ Valgrind not available, skipping${NC}"
        echo "Install with: sudo apt-get install valgrind"
    fi
    echo ""
}

# Run cachegrind on key workloads (skip large ones as they're slow)
run_cachegrind "benchmark_construction" "small_uniform" "$REPORTS_DIR/cache_construction_small.txt"
run_cachegrind "benchmark_query" "small_uniform" "$REPORTS_DIR/cache_query_small.txt"

echo -e "${GREEN}Step 4: False Sharing Detection (perf c2c)${NC}"
echo "========================================"
echo ""

if [ -f "$BUILD_DIR/benchmark_parallel" ]; then
    echo "Running cache-to-cache transfer analysis..."
    if command -v perf &> /dev/null && perf c2c --help &> /dev/null; then
        perf c2c record -o "$REPORTS_DIR/perf_c2c.data" \
            "$BUILD_DIR/benchmark_parallel" > /dev/null 2>&1

        perf c2c report -i "$REPORTS_DIR/perf_c2c.data" --stdio \
            > "$REPORTS_DIR/c2c_parallel.txt" 2>&1

        echo -e "${GREEN}✓${NC} C2C report saved to $REPORTS_DIR/c2c_parallel.txt"
    else
        echo -e "${YELLOW}⚠ perf c2c not available${NC}"
        echo "Requires recent Linux kernel with c2c support"
    fi
else
    echo -e "${YELLOW}⚠ Parallel benchmark not built${NC}"
fi
echo ""

echo -e "${GREEN}Step 5: System Information${NC}"
echo "========================================"
echo ""

# Collect system info
SYSINFO_FILE="$BASELINE_DIR/system_info.txt"
{
    echo "System Information"
    echo "=================="
    echo ""
    echo "CPU:"
    lscpu | grep -E "Model name|Thread|Core|Socket|Cache"
    echo ""
    echo "Memory:"
    free -h
    echo ""
    echo "Kernel:"
    uname -a
    echo ""
    echo "Compiler:"
    g++ --version | head -1
    clang++ --version | head -1 2>/dev/null || echo "clang++ not found"
} > "$SYSINFO_FILE"

cat "$SYSINFO_FILE"
echo ""

echo "========================================"
echo -e "${GREEN}Profiling Complete!${NC}"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Performance counters: $REPORTS_DIR/perf_*.txt"
echo "  - Flamegraphs: $FLAMEGRAPH_DIR/*.svg"
echo "  - Cache analysis: $REPORTS_DIR/cache_*.txt"
echo "  - Call graphs: $REPORTS_DIR/callgraph_*.txt"
echo "  - System info: $SYSINFO_FILE"
echo ""
echo "Next steps:"
echo "  1. Review flamegraphs to identify hotspots"
echo "  2. Analyze cache miss rates in perf reports"
echo "  3. Run: python3 scripts/analyze_baseline.py"
echo "  4. Fill out: docs/baseline/BASELINE_SUMMARY.md"
echo ""
