// Phase 0: Benchmark Utilities
// Helper functions for timing and reporting

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace benchmark {

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

struct BenchmarkResult {
    std::string workload_name;
    std::string operation;
    size_t n_elements;
    size_t n_queries;
    double time_ms;
    double throughput;  // operations per second
    size_t memory_bytes;

    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Workload: " << workload_name << "\n";
        std::cout << "Operation: " << operation << "\n";
        std::cout << "Elements: " << n_elements << "\n";
        if (n_queries > 0) {
            std::cout << "Queries: " << n_queries << "\n";
        }
        std::cout << "Time: " << time_ms << " ms\n";
        std::cout << "Throughput: " << throughput << " ops/sec\n";
        if (memory_bytes > 0) {
            std::cout << "Memory: " << (memory_bytes / 1024.0 / 1024.0) << " MB\n";
        }
        std::cout << std::string(60, '-') << "\n";
    }

    std::string to_csv_header() const {
        return "workload,operation,n_elements,n_queries,time_ms,throughput_ops_sec,memory_mb";
    }

    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << workload_name << ","
            << operation << ","
            << n_elements << ","
            << n_queries << ","
            << time_ms << ","
            << throughput << ","
            << (memory_bytes / 1024.0 / 1024.0);
        return oss.str();
    }
};

class BenchmarkReporter {
public:
    void add_result(const BenchmarkResult& result) {
        results_.push_back(result);
    }

    void print_summary() const {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "BENCHMARK SUMMARY\n";
        std::cout << std::string(60, '=') << "\n\n";

        for (const auto& result : results_) {
            result.print();
        }

        std::cout << "\nTotal benchmarks run: " << results_.size() << "\n";
    }

    void save_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return;
        }

        if (!results_.empty()) {
            file << results_[0].to_csv_header() << "\n";
            for (const auto& result : results_) {
                file << result.to_csv() << "\n";
            }
        }

        file.close();
        std::cout << "Results saved to: " << filename << "\n";
    }

    const std::vector<BenchmarkResult>& get_results() const {
        return results_;
    }

private:
    std::vector<BenchmarkResult> results_;
};

// Statistics helper
struct Stats {
    double mean;
    double median;
    double std_dev;
    double min;
    double max;

    static Stats compute(std::vector<double> values) {
        if (values.empty()) {
            return {0, 0, 0, 0, 0};
        }

        std::sort(values.begin(), values.end());

        Stats s;
        s.min = values.front();
        s.max = values.back();

        // Mean
        s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

        // Median
        size_t mid = values.size() / 2;
        if (values.size() % 2 == 0) {
            s.median = (values[mid - 1] + values[mid]) / 2.0;
        } else {
            s.median = values[mid];
        }

        // Standard deviation
        double sq_sum = 0.0;
        for (double v : values) {
            sq_sum += (v - s.mean) * (v - s.mean);
        }
        s.std_dev = std::sqrt(sq_sum / values.size());

        return s;
    }

    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Mean: " << mean << " ms\n";
        std::cout << "Median: " << median << " ms\n";
        std::cout << "Std Dev: " << std_dev << " ms\n";
        std::cout << "Min: " << min << " ms\n";
        std::cout << "Max: " << max << " ms\n";
    }
};

// Memory estimation helper
inline size_t estimate_memory_usage() {
    // This is a simple estimation - actual measurement would require platform-specific code
    // On Linux, you could parse /proc/self/status
    return 0;
}

} // namespace benchmark

#endif // BENCHMARK_UTILS_H
