// Phase 0: Query Phase Benchmark
// Measures query performance across different workloads

#include "workloads.h"
#include "benchmark_utils.h"

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <unistd.h>

// Simple BB class for benchmarking
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

  BB(const std::array<float, 2*D>& arr) {
    for (int i = 0; i < 2*D; i++) values[i] = arr[i];
  }

  inline float min(int i) const { return values[i]; }
  inline float max(int i) const { return values[i + D]; }

  bool intersects(const BB<D>& other) const {
    for (int i = 0; i < D; i++) {
      if (max(i) < other.min(i) || min(i) > other.max(i)) {
        return false;
      }
    }
    return true;
  }
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

// Simplified query benchmark
template<typename T, int D>
class SimplePRTreeQueryBenchmark {
public:
    using BBox = std::array<float, D * 2>;
    using Data = DataType<T, D>;

    void construct(const std::vector<BBox>& data) {
        elements_.clear();
        elements_.reserve(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            float minima[D], maxima[D];
            for (int d = 0; d < D; ++d) {
                minima[d] = data[i][d];
                maxima[d] = data[i][d + D];
            }
            BB<D> bb(minima, maxima);
            elements_.emplace_back(static_cast<T>(i), bb);
        }

        // Sort for better query performance (simulates spatial index)
        std::sort(elements_.begin(), elements_.end(),
                  [](const Data& a, const Data& b) {
                      return a.second.min(0) < b.second.min(0);
                  });
    }

    std::vector<T> query(const BBox& query_box) const {
        std::vector<T> results;
        BB<D> query_bb(query_box);

        // Simple linear scan with intersection test
        // In real PRTree, this would traverse the tree structure
        for (const auto& elem : elements_) {
            if (elem.second.intersects(query_bb)) {
                results.push_back(elem.first);
            }
        }

        return results;
    }

    size_t size() const { return elements_.size(); }

private:
    std::vector<Data> elements_;
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

void run_query_benchmark(const benchmark::WorkloadConfig& config,
                        benchmark::BenchmarkReporter& reporter) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Running query benchmark: " << config.name << "\n";
    std::cout << "Elements: " << config.n_elements << "\n";
    std::cout << "Queries: " << config.n_queries << "\n";
    std::cout << std::string(60, '=') << "\n";

    // Generate data and build tree
    benchmark::DataGenerator<2> generator;
    auto data = generator.generate(config);

    SimplePRTreeQueryBenchmark<int64_t, 2> tree;
    tree.construct(data);

    // Generate queries
    auto queries = generator.generate_queries(config, data);

    std::cout << "Tree built with " << tree.size() << " elements\n";
    std::cout << "Running " << queries.size() << " queries...\n";

    // Warm up
    for (size_t i = 0; i < std::min(size_t(10), queries.size()); ++i) {
        auto results = tree.query(queries[i]);
        (void)results; // Suppress unused warning
    }

    // Benchmark queries
    benchmark::Timer timer;
    size_t total_results = 0;

    for (const auto& query : queries) {
        auto results = tree.query(query);
        total_results += results.size();
    }

    double elapsed_ms = timer.elapsed_ms();

    // Calculate throughput (queries per second)
    double throughput = (queries.size() / elapsed_ms) * 1000.0;
    double avg_query_time_us = (elapsed_ms / queries.size()) * 1000.0;

    std::cout << "Total results found: " << total_results << "\n";
    std::cout << "Average query time: " << std::fixed << std::setprecision(2)
              << avg_query_time_us << " μs\n";

    // Record results
    benchmark::BenchmarkResult result;
    result.workload_name = config.name;
    result.operation = "query";
    result.n_elements = config.n_elements;
    result.n_queries = config.n_queries;
    result.time_ms = elapsed_ms;
    result.throughput = throughput;
    result.memory_bytes = 0; // Not measured for queries

    result.print();
    reporter.add_result(result);
}

int main(int argc, char** argv) {
    std::cout << "PRTree Phase 0: Query Benchmark\n";
    std::cout << "================================\n\n";

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
            run_query_benchmark(*it, reporter);
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
            run_query_benchmark(workload, reporter);
        }
    }

    // Print summary and save results
    reporter.print_summary();
    reporter.save_csv("query_benchmark_results.csv");

    return 0;
}
