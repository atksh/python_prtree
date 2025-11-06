# Phase 0: Microarchitectural Baseline Profiling

This directory contains the baseline performance characteristics of PRTree before any optimizations are applied. All measurements must be completed and documented before proceeding with Phase 1.

## 🔴 CRITICAL: Go/No-Go Gate

**Phase 0 is complete ONLY when:**
- ✅ All artifacts generated for all workloads
- ✅ Baseline summary memo reviewed and approved
- ✅ Raw data committed to repository (for regression detection)
- ✅ Automated benchmark suite integrated into CI
- ✅ Performance regression detection scripts validated

**If metrics cannot be collected: STOP. Fix tooling before proceeding.**

## Directory Structure

```
baseline/
├── README.md                      # This file
├── BASELINE_SUMMARY.md           # Executive summary (REQUIRED)
├── perf_counters.md              # Hardware counter baselines
├── hotspots.md                   # Top performance bottlenecks
├── layout_analysis.md            # Data structure memory layout
├── numa_analysis.md              # NUMA behavior (if applicable)
├── flamegraphs/                  # Flamegraph visualizations
│   ├── construction_small.svg
│   ├── construction_large.svg
│   ├── construction_clustered.svg
│   ├── query_small.svg
│   ├── query_large.svg
│   └── batch_query_parallel.svg
└── reports/                      # Raw profiling data
    ├── construction_*.txt        # Call-graph reports
    ├── cache_*.txt               # Cachegrind reports
    └── c2c_*.txt                 # Cache-to-cache transfer reports
```

## Required Tooling

### Linux Tools (Mandatory)
```bash
# Hardware performance counters
sudo apt-get install linux-tools-generic linux-tools-$(uname -r)

# Cache topology
sudo apt-get install hwloc lstopo

# Valgrind with Cachegrind
sudo apt-get install valgrind

# FlameGraph generator
git clone https://github.com/brendangregg/FlameGraph.git
```

### macOS Tools
```bash
# Instruments (part of Xcode)
xcode-select --install

# Homebrew tools
brew install hwloc valgrind
```

## Standard Workloads

All benchmarks must be run with these representative workloads:

1. **small_uniform**: 10,000 elements, uniform distribution, 1,000 small queries
2. **large_uniform**: 1,000,000 elements, uniform distribution, 10,000 medium queries
3. **clustered**: 500,000 elements, clustered distribution (10 clusters), 5,000 mixed queries
4. **skewed**: 1,000,000 elements, Zipfian distribution, 10,000 large queries
5. **sequential**: 100,000 elements, sequential data, 1,000 small queries

## Metrics to Collect

### Construction Phase
For each workload, collect:
- **Performance Counters**: cycles, instructions, IPC, cache misses (L1/L2/L3), TLB misses, branch misses
- **Call Graph**: Hotspot functions with CPU time percentages
- **Cache Behavior**: Cachegrind annotations showing cache line utilization
- **Memory Usage**: Peak RSS, allocations

### Query Phase
Same metrics as construction phase, plus:
- **Query throughput**: Queries per second
- **Latency distribution**: P50, P95, P99

### Multithreaded Construction
For parallel construction, collect:
- **Thread scaling**: 1, 2, 4, 8, 16 threads
- **NUMA effects**: Local vs remote memory access
- **Cache-to-cache transfers**: False sharing detection
- **Parallel speedup**: Actual vs theoretical

## How to Run Profiling

### Step 1: Build with Profiling Symbols
```bash
mkdir -p build_profile
cd build_profile
cmake -DBUILD_BENCHMARKS=ON -DENABLE_PROFILING=ON ..
make -j$(nproc)
```

### Step 2: Run Benchmarks and Collect Metrics
```bash
# From repository root
./scripts/profile_all_workloads.sh
```

This will:
1. Run each benchmark with `perf stat` for hardware counters
2. Run with `perf record` for flamegraphs
3. Run with `valgrind --tool=cachegrind` for cache analysis
4. Generate reports in `docs/baseline/reports/`
5. Generate flamegraphs in `docs/baseline/flamegraphs/`

### Step 3: Analyze and Document
```bash
# Generate summary analysis
./scripts/analyze_baseline.py
```

This creates:
- `perf_counters.md` - Tabulated counter results
- `hotspots.md` - Top 10 functions by various metrics
- `BASELINE_SUMMARY.md` - Executive summary with recommendations

## Validation Checklist

Before considering Phase 0 complete, verify:

- [ ] All 5 workloads profiled successfully
- [ ] Hardware counters collected for all workloads
- [ ] Flamegraphs generated and readable
- [ ] Cachegrind reports show detailed cache line info
- [ ] Hotspot analysis identifies top bottlenecks
- [ ] Data structure layout documented with `pahole`
- [ ] Thread scaling measured (if applicable)
- [ ] NUMA analysis complete (if multi-socket system)
- [ ] Baseline summary memo written and reviewed
- [ ] All raw data committed to git
- [ ] CI integration tested and passing

## Expected Timeline

- **Tooling setup**: 2 hours
- **Benchmark implementation**: 4 hours
- **Data collection**: 2 hours (automated)
- **Analysis and documentation**: 4 hours
- **Review and approval**: 2 hours

**Total: 2-3 days**

## Troubleshooting

### "perf_event_open failed: Permission denied"
```bash
# Temporary (until reboot)
sudo sysctl -w kernel.perf_event_paranoid=-1

# Permanent
echo 'kernel.perf_event_paranoid = -1' | sudo tee -a /etc/sysctl.conf
```

### "Cannot find debug symbols"
Ensure you built with `-DENABLE_PROFILING=ON` which adds `-g` and `-fno-omit-frame-pointer`.

### "Cachegrind too slow"
For large workloads, you can sample:
```bash
valgrind --tool=cachegrind --cachegrind-out-file=cache.out \
  --I1=32768,8,64 --D1=32768,8,64 --LL=8388608,16,64 \
  ./benchmark_construction large_uniform
```

## References

- [perf documentation](https://perf.wiki.kernel.org/index.php/Tutorial)
- [Cachegrind manual](https://valgrind.org/docs/manual/cg-manual.html)
- [FlameGraph guide](https://www.brendangregg.com/flamegraphs.html)
- [Intel VTune tutorial](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html)
