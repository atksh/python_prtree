// Phase 0: Construction Phase Benchmark
// Measures tree construction performance across different workloads

#include "workloads.h"
#include "benchmark_utils.h"

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <unistd.h>

// Simple BB class for benchmarking (extracted from prtree.h)
template <int D = 2>
class BB {
private:
  float values[2 * D];

public:
  BB() {
    for (int i = 0; i < 2 * D; i++) values[i] = 0.0f;
  }

  BB(const float *minima, const float *maxima) {
    for (int i = 0; i < D; i++) {
      values[i] = minima[i];
      values[i + D] = maxima[i];
    }
  }

  inline float min(int i) const { return values[i]; }
  inline float max(int i) const { return values[i + D]; }

  inline float operator[](const int i) const { return values[i]; }
};

// Simple DataType class for benchmarking
template <class T, int D = 2>
class DataType {
public:
  T first;
  BB<D> second;

  DataType() = default;
  DataType(const T &f, const BB<D> &s) : first(f), second(s) {}
};

// Simplified tree construction simulation
// This mimics the core construction logic without full PRTree implementation
template<typename T, int D>
class SimplePRTreeBenchmark {
public:
    using BBox = std::array<float, D * 2>;
    using Data = DataType<T, D>;

    void construct(const std::vector<BBox>& data) {
        elements_.clear();
        elements_.reserve(data.size());

        // Convert input data to DataType format
        for (size_t i = 0; i < data.size(); ++i) {
            float minima[D], maxima[D];
            for (int d = 0; d < D; ++d) {
                minima[d] = data[i][d];
                maxima[d] = data[i][d + D];
            }
            BB<D> bb(minima, maxima);
            elements_.emplace_back(static_cast<T>(i), bb);
        }

        // Simulate partitioning/sorting work
        // This represents the dominant cost in PRTree construction
        std::sort(elements_.begin(), elements_.end(),
                  [](const Data& a, const Data& b) {
                      return a.second[0] < b.second[0];
                  });

        // Simulate tree building overhead
        build_tree_structure();
    }

    size_t size() const { return elements_.size(); }

private:
    std::vector<Data> elements_;

    void build_tree_structure() {
        // Simulate recursive tree building
        // In real PRTree, this involves complex partitioning
        if (elements_.size() <= 6) return;

        // Simulate some memory access patterns
        float total = 0.0f;
        for (const auto& elem : elements_) {
            for (int d = 0; d < D; ++d) {
                total += elem.second.min(d) + elem.second.max(d);
            }
        }
        // Prevent optimization
        if (total < 0) std::cout << total;
    }
};

// Get current memory usage (RSS) in bytes
size_t get_memory_usage() {
    long rss = 0L;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp) {
        if (fscanf(fp, "%*s%ld", &rss) == 1) {
            fclose(fp);
            return rss * sysconf(_SC_PAGESIZE);
        }
        fclose(fp);
    }
    return 0;
}

void run_construction_benchmark(const benchmark::WorkloadConfig& config,
                                benchmark::BenchmarkReporter& reporter) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Running construction benchmark: " << config.name << "\n";
    std::cout << "Elements: " << config.n_elements << "\n";
    std::cout << std::string(60, '=') << "\n";

    // Generate data
    benchmark::DataGenerator<2> generator;
    auto data = generator.generate(config);

    size_t mem_before = get_memory_usage();

    // Benchmark construction
    SimplePRTreeBenchmark<int64_t, 2> tree;
    benchmark::Timer timer;
    tree.construct(data);
    double elapsed_ms = timer.elapsed_ms();

    size_t mem_after = get_memory_usage();
    size_t mem_delta = (mem_after > mem_before) ? (mem_after - mem_before) : 0;

    // Calculate throughput
    double throughput = (config.n_elements / elapsed_ms) * 1000.0;

    // Record results
    benchmark::BenchmarkResult result;
    result.workload_name = config.name;
    result.operation = "construction";
    result.n_elements = config.n_elements;
    result.n_queries = 0;
    result.time_ms = elapsed_ms;
    result.throughput = throughput;
    result.memory_bytes = mem_delta;

    result.print();
    reporter.add_result(result);
}

int main(int argc, char** argv) {
    std::cout << "PRTree Phase 0: Construction Benchmark\n";
    std::cout << "========================================\n\n";

    benchmark::BenchmarkReporter reporter;

    // Get workloads to run
    auto workloads = benchmark::get_standard_workloads();

    // If specific workload requested via command line
    if (argc > 1) {
        std::string requested = argv[1];
        auto it = std::find_if(workloads.begin(), workloads.end(),
                              [&requested](const auto& w) {
                                  return w.name == requested;
                              });
        if (it != workloads.end()) {
            run_construction_benchmark(*it, reporter);
        } else {
            std::cerr << "Unknown workload: " << requested << "\n";
            std::cerr << "Available workloads:\n";
            for (const auto& w : workloads) {
                std::cerr << "  - " << w.name << "\n";
            }
            return 1;
        }
    } else {
        // Run all workloads
        for (const auto& workload : workloads) {
            run_construction_benchmark(workload, reporter);
        }
    }

    // Print summary and save results
    reporter.print_summary();
    reporter.save_csv("construction_benchmark_results.csv");

    return 0;
}
