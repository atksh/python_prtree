// Phase 0: Parallel/Multithreaded Benchmark
// Measures thread scaling for parallel construction

#include "workloads.h"
#include "benchmark_utils.h"

#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <algorithm>
#include <unistd.h>

// Simple BB class
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
};

// Simple DataType class
// Phase 7: Thread-local buffers eliminate need for alignas(64)
template <class T, int D = 2>
class DataType {
public:
  T first;
  BB<D> second;

  DataType() = default;
  DataType(const T &f, const BB<D> &s) : first(f), second(s) {}
};

// Parallel construction benchmark
template<typename T, int D>
class ParallelPRTreeBenchmark {
public:
    using BBox = std::array<float, D * 2>;
    using Data = DataType<T, D>;

    void construct_parallel(const std::vector<BBox>& data, size_t n_threads) {
        elements_.clear();

        // Phase 7: Use thread-local buffers to eliminate contention
        const size_t chunk_size = (data.size() + n_threads - 1) / n_threads;
        std::vector<std::vector<Data>> thread_local_buffers(n_threads);
        std::vector<std::thread> threads;

        // Phase 1: Parallel data conversion (thread-local)
        for (size_t t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, data.size());

                // Reserve space in thread-local buffer
                auto& local_buffer = thread_local_buffers[t];
                local_buffer.reserve(end - start);

                for (size_t i = start; i < end; ++i) {
                    float minima[D], maxima[D];
                    for (int d = 0; d < D; ++d) {
                        minima[d] = data[i][d];
                        maxima[d] = data[i][d + D];
                    }
                    BB<D> bb(minima, maxima);
                    // Write to thread-local buffer (no contention!)
                    local_buffer.emplace_back(static_cast<T>(i), bb);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Phase 2: Merge thread-local buffers
        size_t total_size = 0;
        for (const auto& buffer : thread_local_buffers) {
            total_size += buffer.size();
        }
        elements_.reserve(total_size);

        for (auto& buffer : thread_local_buffers) {
            elements_.insert(elements_.end(),
                           std::make_move_iterator(buffer.begin()),
                           std::make_move_iterator(buffer.end()));
        }

        // Sort phase (single-threaded for simplicity)
        std::sort(elements_.begin(), elements_.end(),
                  [](const Data& a, const Data& b) {
                      return a.second.min(0) < b.second.min(0);
                  });

        // Simulate tree building
        build_tree_structure();
    }

    size_t size() const { return elements_.size(); }

private:
    std::vector<Data> elements_;

    void build_tree_structure() {
        if (elements_.size() <= 6) return;
        float total = 0.0f;
        for (const auto& elem : elements_) {
            for (int d = 0; d < D; ++d) {
                total += elem.second.min(d) + elem.second.max(d);
            }
        }
        if (total < 0) std::cout << total;
    }
};

void run_parallel_benchmark(const benchmark::WorkloadConfig& config,
                           size_t n_threads,
                           benchmark::BenchmarkReporter& reporter) {
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "Threads: " << n_threads << "\n";

    // Generate data
    benchmark::DataGenerator<2> generator;
    auto data = generator.generate(config);

    // Benchmark parallel construction
    ParallelPRTreeBenchmark<int64_t, 2> tree;
    benchmark::Timer timer;
    tree.construct_parallel(data, n_threads);
    double elapsed_ms = timer.elapsed_ms();

    // Calculate throughput
    double throughput = (config.n_elements / elapsed_ms) * 1000.0;

    std::cout << "Time: " << elapsed_ms << " ms\n";
    std::cout << "Throughput: " << throughput << " elements/sec\n";

    // Record results
    benchmark::BenchmarkResult result;
    result.workload_name = config.name + "_threads_" + std::to_string(n_threads);
    result.operation = "parallel_construction";
    result.n_elements = config.n_elements;
    result.n_queries = 0;
    result.time_ms = elapsed_ms;
    result.throughput = throughput;
    result.memory_bytes = 0;

    reporter.add_result(result);
}

int main(int argc, char** argv) {
    std::cout << "PRTree Phase 0: Parallel/Thread Scaling Benchmark\n";
    std::cout << "==================================================\n\n";

    benchmark::BenchmarkReporter reporter;

    // Use large_uniform workload for thread scaling
    auto workloads = benchmark::get_standard_workloads();
    auto it = std::find_if(workloads.begin(), workloads.end(),
                          [](const auto& w) { return w.name == "large_uniform"; });

    if (it == workloads.end()) {
        std::cerr << "large_uniform workload not found\n";
        return 1;
    }

    const auto& config = *it;

    // Thread counts to test
    std::vector<size_t> thread_counts = {1, 2, 4, 8};
    size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads > 8) {
        thread_counts.push_back(16);
    }

    std::cout << "Workload: " << config.name << "\n";
    std::cout << "Elements: " << config.n_elements << "\n";
    std::cout << "Hardware threads: " << hw_threads << "\n\n";

    // Baseline (single-threaded)
    double baseline_time = 0.0;

    for (size_t n_threads : thread_counts) {
        run_parallel_benchmark(config, n_threads, reporter);

        if (n_threads == 1) {
            baseline_time = reporter.get_results().back().time_ms;
        }
    }

    // Print scaling analysis
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "THREAD SCALING ANALYSIS\n";
    std::cout << std::string(60, '=') << "\n\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Threads | Time (ms) | Speedup | Efficiency\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& result : reporter.get_results()) {
        // Extract thread count from workload name
        size_t pos = result.workload_name.find("_threads_");
        if (pos != std::string::npos) {
            size_t threads = std::stoul(result.workload_name.substr(pos + 9));
            double speedup = baseline_time / result.time_ms;
            double efficiency = (speedup / threads) * 100.0;

            std::cout << std::setw(7) << threads << " | "
                     << std::setw(9) << result.time_ms << " | "
                     << std::setw(7) << speedup << "x | "
                     << std::setw(6) << efficiency << "%\n";
        }
    }

    std::cout << "\n";

    // Save results
    reporter.save_csv("parallel_benchmark_results.csv");

    return 0;
}
