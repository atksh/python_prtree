// Phase 0: Benchmark Workload Definitions
// Defines representative workloads for microarchitectural profiling

#ifndef BENCHMARK_WORKLOADS_H
#define BENCHMARK_WORKLOADS_H

#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace benchmark {

enum class Distribution {
    UNIFORM,      // Uniform random distribution
    CLUSTERED,    // 10 clusters with normal distribution
    ZIPF,         // Heavy-tailed distribution (Zipfian)
    SEQUENTIAL    // Sequential/sorted data
};

enum class QuerySize {
    SMALL,        // 1% of space
    MEDIUM,       // 10% of space
    LARGE,        // 50% of space
    MIXED         // Mix of all sizes
};

struct WorkloadConfig {
    std::string name;
    size_t n_elements;
    Distribution distribution;
    size_t n_queries;
    QuerySize query_size;
    int dimensions;  // Default 2D

    WorkloadConfig(const std::string& n, size_t ne, Distribution d,
                   size_t nq, QuerySize qs, int dim = 2)
        : name(n), n_elements(ne), distribution(d),
          n_queries(nq), query_size(qs), dimensions(dim) {}
};

// Standard workloads covering real-world usage
inline std::vector<WorkloadConfig> get_standard_workloads() {
    return {
        WorkloadConfig("small_uniform", 10000, Distribution::UNIFORM, 1000, QuerySize::SMALL),
        WorkloadConfig("large_uniform", 1000000, Distribution::UNIFORM, 10000, QuerySize::MEDIUM),
        WorkloadConfig("clustered", 500000, Distribution::CLUSTERED, 5000, QuerySize::MIXED),
        WorkloadConfig("skewed", 1000000, Distribution::ZIPF, 10000, QuerySize::LARGE),
        WorkloadConfig("sequential", 100000, Distribution::SEQUENTIAL, 1000, QuerySize::SMALL)
    };
}

// Helper class to generate test data based on workload configuration
template<int D>
class DataGenerator {
public:
    using BBox = std::array<float, D * 2>;

    DataGenerator(uint64_t seed = 42) : rng_(seed), dist_01_(0.0, 1.0) {}

    // Generate bounding boxes based on distribution type
    std::vector<BBox> generate(const WorkloadConfig& config) {
        std::vector<BBox> data;
        data.reserve(config.n_elements);

        switch (config.distribution) {
            case Distribution::UNIFORM:
                return generate_uniform(config.n_elements);
            case Distribution::CLUSTERED:
                return generate_clustered(config.n_elements, 10);
            case Distribution::ZIPF:
                return generate_zipf(config.n_elements, 1.5);
            case Distribution::SEQUENTIAL:
                return generate_sequential(config.n_elements);
        }

        return data;
    }

    // Generate query rectangles
    std::vector<BBox> generate_queries(const WorkloadConfig& config,
                                       const std::vector<BBox>& data) {
        std::vector<BBox> queries;
        queries.reserve(config.n_queries);

        float size = get_query_size_fraction(config.query_size);

        for (size_t i = 0; i < config.n_queries; ++i) {
            if (config.query_size == QuerySize::MIXED) {
                // Randomly choose size for mixed queries
                float r = dist_01_(rng_);
                if (r < 0.33) size = 0.01f;
                else if (r < 0.66) size = 0.10f;
                else size = 0.50f;
            }

            queries.push_back(generate_query_box(size));
        }

        return queries;
    }

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> dist_01_;

    float get_query_size_fraction(QuerySize qs) {
        switch (qs) {
            case QuerySize::SMALL: return 0.01f;   // 1%
            case QuerySize::MEDIUM: return 0.10f;  // 10%
            case QuerySize::LARGE: return 0.50f;   // 50%
            case QuerySize::MIXED: return 0.10f;   // Default for mixed
        }
        return 0.01f;
    }

    BBox generate_query_box(float size) {
        BBox box;
        for (int d = 0; d < D; ++d) {
            float center = dist_01_(rng_);
            float half_size = size * 0.5f;
            box[d * 2] = std::max(0.0f, center - half_size);
            box[d * 2 + 1] = std::min(1.0f, center + half_size);
        }
        return box;
    }

    std::vector<BBox> generate_uniform(size_t n) {
        std::vector<BBox> data;
        data.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            BBox box;
            for (int d = 0; d < D; ++d) {
                float min_val = dist_01_(rng_);
                float max_val = min_val + dist_01_(rng_) * 0.01f; // Small boxes
                box[d * 2] = min_val;
                box[d * 2 + 1] = std::min(1.0f, max_val);
            }
            data.push_back(box);
        }

        return data;
    }

    std::vector<BBox> generate_clustered(size_t n, int n_clusters) {
        std::vector<BBox> data;
        data.reserve(n);

        // Generate cluster centers
        std::vector<std::array<float, D>> centers;
        for (int c = 0; c < n_clusters; ++c) {
            std::array<float, D> center;
            for (int d = 0; d < D; ++d) {
                center[d] = dist_01_(rng_);
            }
            centers.push_back(center);
        }

        std::normal_distribution<float> cluster_dist(0.0, 0.05);

        for (size_t i = 0; i < n; ++i) {
            int cluster_id = i % n_clusters;
            const auto& center = centers[cluster_id];

            BBox box;
            for (int d = 0; d < D; ++d) {
                float offset = cluster_dist(rng_);
                float min_val = std::clamp(center[d] + offset, 0.0f, 1.0f);
                float max_val = std::clamp(min_val + dist_01_(rng_) * 0.01f, 0.0f, 1.0f);
                box[d * 2] = min_val;
                box[d * 2 + 1] = max_val;
            }
            data.push_back(box);
        }

        return data;
    }

    std::vector<BBox> generate_zipf(size_t n, double s) {
        std::vector<BBox> data;
        data.reserve(n);

        // Generate Zipfian distribution - heavy concentration in certain regions
        double c = 0.0;
        for (size_t i = 1; i <= n; ++i) {
            c += 1.0 / std::pow(i, s);
        }
        c = 1.0 / c;

        for (size_t i = 0; i < n; ++i) {
            double sum_prob = 0.0;
            double z = dist_01_(rng_);
            size_t rank = 1;

            for (size_t k = 1; k <= n; ++k) {
                sum_prob += c / std::pow(k, s);
                if (sum_prob >= z) {
                    rank = k;
                    break;
                }
            }

            // Map rank to spatial location (lower ranks = concentrated area)
            float spatial_factor = static_cast<float>(rank) / n;

            BBox box;
            for (int d = 0; d < D; ++d) {
                float min_val = spatial_factor + dist_01_(rng_) * 0.1f;
                min_val = std::clamp(min_val, 0.0f, 1.0f);
                float max_val = std::clamp(min_val + dist_01_(rng_) * 0.01f, 0.0f, 1.0f);
                box[d * 2] = min_val;
                box[d * 2 + 1] = max_val;
            }
            data.push_back(box);
        }

        return data;
    }

    std::vector<BBox> generate_sequential(size_t n) {
        std::vector<BBox> data;
        data.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            float base = static_cast<float>(i) / n;

            BBox box;
            for (int d = 0; d < D; ++d) {
                float min_val = base;
                float max_val = std::min(1.0f, base + 0.01f);
                box[d * 2] = min_val;
                box[d * 2 + 1] = max_val;
            }
            data.push_back(box);
        }

        return data;
    }
};

} // namespace benchmark

#endif // BENCHMARK_WORKLOADS_H
