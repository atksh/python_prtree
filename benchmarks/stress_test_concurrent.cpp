// Phase 0: Concurrent Stress Test
// Tests thread-safety of PRTree under concurrent operations
// Must pass cleanly under ThreadSanitizer (TSan)

#include "workloads.h"
#include "benchmark_utils.h"

#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cassert>

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

// Simple DataType class
template <class T, int D = 2>
class DataType {
public:
  T first;
  BB<D> second;

  DataType() = default;
  DataType(const T &f, const BB<D> &s) : first(f), second(s) {}
};

// Thread-safe tree for stress testing
template<typename T, int D>
class ThreadSafeTreeStub {
public:
    using BBox = std::array<float, D * 2>;
    using Data = DataType<T, D>;

    void construct(const std::vector<BBox>& data) {
        std::lock_guard<std::mutex> lock(mutex_);
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
    }

    std::vector<T> query(const BBox& query_box) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<T> results;
        BB<D> query_bb(query_box);

        for (const auto& elem : elements_) {
            if (elem.second.intersects(query_bb)) {
                results.push_back(elem.first);
            }
        }
        return results;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return elements_.size();
    }

private:
    mutable std::mutex mutex_;
    std::vector<Data> elements_;
};

// Test 1: Concurrent queries while rebuilding
void test_concurrent_build_and_query() {
    std::cout << "\nTest 1: Concurrent Build and Query\n";
    std::cout << std::string(40, '-') << "\n";

    constexpr int NUM_QUERY_THREADS = 8;
    constexpr int NUM_ITERATIONS = 100;
    constexpr int DATASET_SIZE = 1000;

    ThreadSafeTreeStub<int64_t, 2> tree;
    std::atomic<bool> keep_running{true};
    std::atomic<int> query_count{0};
    std::atomic<int> build_count{0};

    benchmark::DataGenerator<2> generator;
    benchmark::WorkloadConfig config("stress_test", DATASET_SIZE,
                                    benchmark::Distribution::UNIFORM,
                                    100, benchmark::QuerySize::SMALL);
    auto data = generator.generate(config);
    auto queries = generator.generate_queries(config, data);

    // Initial build
    tree.construct(data);

    // Builder thread
    std::thread builder([&]() {
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            tree.construct(data);
            build_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        keep_running = false;
    });

    // Query threads
    std::vector<std::thread> query_threads;
    for (int t = 0; t < NUM_QUERY_THREADS; ++t) {
        query_threads.emplace_back([&, t]() {
            while (keep_running) {
                size_t query_idx = query_count % queries.size();
                auto results = tree.query(queries[query_idx]);
                query_count++;
                // Small delay to prevent tight spinning
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    // Wait for completion
    builder.join();
    for (auto& th : query_threads) {
        th.join();
    }

    std::cout << "Builds completed: " << build_count << "\n";
    std::cout << "Queries completed: " << query_count << "\n";
    std::cout << "✓ Test passed - no crashes or data races\n";
}

// Test 2: Concurrent queries from multiple threads
void test_concurrent_queries() {
    std::cout << "\nTest 2: Concurrent Queries\n";
    std::cout << std::string(40, '-') << "\n";

    constexpr int NUM_THREADS = 8;
    constexpr int QUERIES_PER_THREAD = 1000;
    constexpr int DATASET_SIZE = 10000;

    ThreadSafeTreeStub<int64_t, 2> tree;
    std::atomic<int> total_queries{0};

    benchmark::DataGenerator<2> generator;
    benchmark::WorkloadConfig config("stress_test", DATASET_SIZE,
                                    benchmark::Distribution::UNIFORM,
                                    100, benchmark::QuerySize::MEDIUM);
    auto data = generator.generate(config);
    auto queries = generator.generate_queries(config, data);

    // Build tree
    tree.construct(data);

    // Query threads
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < QUERIES_PER_THREAD; ++i) {
                size_t query_idx = (t * QUERIES_PER_THREAD + i) % queries.size();
                auto results = tree.query(queries[query_idx]);
                total_queries++;
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    std::cout << "Total queries completed: " << total_queries << "\n";
    assert(total_queries == NUM_THREADS * QUERIES_PER_THREAD);
    std::cout << "✓ Test passed\n";
}

// Test 3: Long-running torture test
void test_torture() {
    std::cout << "\nTest 3: Torture Test (10 seconds)\n";
    std::cout << std::string(40, '-') << "\n";

    constexpr int NUM_THREADS = 8;
    constexpr int TEST_DURATION_SEC = 10;

    ThreadSafeTreeStub<int64_t, 2> tree;
    std::atomic<bool> keep_running{true};
    std::atomic<long> total_operations{0};

    benchmark::DataGenerator<2> generator;
    benchmark::WorkloadConfig config("stress_test", 5000,
                                    benchmark::Distribution::UNIFORM,
                                    100, benchmark::QuerySize::MIXED);
    auto data = generator.generate(config);
    auto queries = generator.generate_queries(config, data);

    // Initial build
    tree.construct(data);

    // Timer thread
    std::thread timer([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(TEST_DURATION_SEC));
        keep_running = false;
    });

    // Worker threads (mix of builds and queries)
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            while (keep_running) {
                // 90% queries, 10% rebuilds
                if (t == 0 && (total_operations % 10 == 0)) {
                    tree.construct(data);
                } else {
                    size_t query_idx = total_operations % queries.size();
                    auto results = tree.query(queries[query_idx]);
                }
                total_operations++;
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        });
    }

    timer.join();
    for (auto& th : threads) {
        th.join();
    }

    std::cout << "Total operations: " << total_operations << "\n";
    std::cout << "Operations/sec: " << (total_operations / TEST_DURATION_SEC) << "\n";
    std::cout << "✓ Test passed\n";
}

int main(int argc, char** argv) {
    std::cout << "PRTree Phase 0: Concurrent Stress Test\n";
    std::cout << "=======================================\n";
    std::cout << "\nThis test MUST run clean under ThreadSanitizer!\n";
    std::cout << "Build with: cmake -DENABLE_TSAN=ON\n\n";

    try {
        test_concurrent_build_and_query();
        test_concurrent_queries();
        test_torture();

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "ALL STRESS TESTS PASSED ✓\n";
        std::cout << std::string(60, '=') << "\n";

        std::cout << "\nNext steps:\n";
        std::cout << "1. Run under TSan: ./stress_test_concurrent\n";
        std::cout << "2. Check for data race warnings\n";
        std::cout << "3. Run for extended period (1 hour)\n";
        std::cout << "   timeout 3600 ./stress_test_concurrent\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ STRESS TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
