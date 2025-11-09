#pragma once

// Standard Library Includes
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// C++20 features
#include <concepts>

// External Dependencies
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/atomic.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <snappy.h>

// PRTree Modular Components
#include "prtree/core/detail/types.h"
#include "prtree/core/detail/bounding_box.h"
#include "prtree/core/detail/data_type.h"
#include "prtree/core/detail/pseudo_tree.h"
#include "prtree/core/detail/nodes.h"

#include "prtree/utils/parallel.h"
#include "prtree/utils/small_vector.h"

#ifdef MY_DEBUG
#include <gperftools/profiler.h>
#endif

namespace py = pybind11;

template <IndexType T, int B = 6, int D = 2, typename Real = float> class PRTree {
private:
  vec<PRTreeElement<T, B, D, Real>> flat_tree;
  std::unordered_map<T, BB<D, Real>> idx2bb;
  std::unordered_map<T, std::string> idx2data;
  int64_t n_at_build = 0;
  std::atomic<T> global_idx = 0;

  mutable std::unique_ptr<std::recursive_mutex> tree_mutex_;

  // Precision control parameters
  Real relative_epsilon_ = 1e-6;   // Relative epsilon for adaptive precision
  Real absolute_epsilon_ = 1e-8;   // Absolute epsilon (backward compatible default)
  bool use_adaptive_epsilon_ = true;  // Use adaptive epsilon based on coordinate magnitude
  bool detect_subnormal_ = true;   // Detect and handle subnormal numbers

  // Helper: Calculate adaptive epsilon for insert operations
  Real calculate_adaptive_epsilon(const BB<D, Real> &bb) const {
    if (!use_adaptive_epsilon_) {
      return absolute_epsilon_;
    }

    // Calculate the maximum extent of the bounding box
    Real max_extent = 0.0;
    for (int i = 0; i < D; ++i) {
      Real extent = bb.max(i) - bb.min(i);
      if (extent > max_extent) {
        max_extent = extent;
      }
    }

    // For degenerate boxes (points), use the magnitude of coordinates
    if (max_extent < std::numeric_limits<Real>::epsilon()) {
      Real max_magnitude = 0.0;
      for (int i = 0; i < D; ++i) {
        Real mag = std::max(std::abs(bb.min(i)), std::abs(bb.max(i)));
        if (mag > max_magnitude) {
          max_magnitude = mag;
        }
      }
      max_extent = max_magnitude;
    }

    // Adaptive epsilon: relative to the scale + absolute minimum
    // This ensures reasonable behavior across different coordinate scales
    Real adaptive_eps = max_extent * relative_epsilon_ + absolute_epsilon_;

    // Clamp to reasonable bounds to avoid numerical issues
    Real min_epsilon = std::numeric_limits<Real>::epsilon() * 10.0;
    Real max_epsilon = max_extent * 0.01;  // At most 1% of the extent

    return std::max(min_epsilon, std::min(adaptive_eps, max_epsilon));
  }

public:
  template <class Archive> void serialize(Archive &archive) {
    archive(flat_tree, idx2bb, idx2data, global_idx, n_at_build);
  }

  void save(const std::string& fname) const {
    std::lock_guard<std::recursive_mutex> lock(*tree_mutex_);
    std::ofstream ofs(fname, std::ios::binary);
    cereal::PortableBinaryOutputArchive o_archive(ofs);
    o_archive(cereal::make_nvp("flat_tree", flat_tree),
              cereal::make_nvp("idx2bb", idx2bb),
              cereal::make_nvp("idx2data", idx2data),
              cereal::make_nvp("global_idx", global_idx),
              cereal::make_nvp("n_at_build", n_at_build));
  }

  void load(const std::string& fname) {
    std::lock_guard<std::recursive_mutex> lock(*tree_mutex_);
    std::ifstream ifs(fname, std::ios::binary);
    cereal::PortableBinaryInputArchive i_archive(ifs);
    i_archive(cereal::make_nvp("flat_tree", flat_tree),
              cereal::make_nvp("idx2bb", idx2bb),
              cereal::make_nvp("idx2data", idx2data),
              cereal::make_nvp("global_idx", global_idx),
              cereal::make_nvp("n_at_build", n_at_build));
  }

  PRTree() : tree_mutex_(std::make_unique<std::recursive_mutex>()) {}

  PRTree(const std::string& fname) : tree_mutex_(std::make_unique<std::recursive_mutex>()) {
    load(fname);
  }

  // Helper: Validate bounding box coordinates (reject NaN/Inf, enforce min <=
  // max)
  template <typename CoordType>
  void validate_box(const CoordType *coords, int dim_count) const {
    for (int i = 0; i < dim_count; ++i) {
      CoordType min_val = coords[i];
      CoordType max_val = coords[i + dim_count];

      // Check for NaN or Inf
      if (!std::isfinite(min_val) || !std::isfinite(max_val)) {
        throw std::runtime_error(
            "Bounding box coordinates must be finite (no NaN or Inf)");
      }

      // Check for subnormal numbers if detection is enabled
      if (detect_subnormal_) {
        // A number is subnormal if it's non-zero but not normal
        bool min_subnormal = (min_val != 0.0) && !std::isnormal(min_val);
        bool max_subnormal = (max_val != 0.0) && !std::isnormal(max_val);

        if (min_subnormal || max_subnormal) {
          throw std::runtime_error(
              "Bounding box contains subnormal numbers which may cause "
              "precision issues. Consider rescaling coordinates or using "
              "larger values. Subnormal detection can be disabled if needed.");
        }
      }

      // Enforce min <= max
      if (min_val > max_val) {
        throw std::runtime_error(
            "Bounding box minimum must be <= maximum in each dimension");
      }
    }
  }

  // Unified constructor for any Real type (float32 or float64)
  PRTree(const py::array_t<T> &idx, const py::array_t<Real> &x)
      : tree_mutex_(std::make_unique<std::recursive_mutex>()) {
    const auto &buff_info_idx = idx.request();
    const auto &shape_idx = buff_info_idx.shape;
    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    if (unlikely(shape_idx[0] != shape_x[0])) {
      throw std::runtime_error(
          "Both index and bounding box must have the same length");
    }
    if (unlikely(shape_x[1] != 2 * D)) {
      throw std::runtime_error(
          "Bounding box must have the shape (length, 2 * dim)");
    }

    auto ri = idx.template unchecked<1>();
    auto rx = x.template unchecked<2>();
    T length = shape_idx[0];
    idx2bb.reserve(length);

    DataType<T, D, Real> *b, *e;
    // Phase 1: RAII memory management to prevent leaks on exception
    struct MallocDeleter {
      void operator()(void* ptr) const {
        if (ptr) std::free(ptr);
      }
    };
    std::unique_ptr<void, MallocDeleter> placement(
        std::malloc(sizeof(DataType<T, D, Real>) * length)
    );
    if (!placement) {
      throw std::bad_alloc();
    }
    b = reinterpret_cast<DataType<T, D, Real> *>(placement.get());
    e = b + length;

    for (T i = 0; i < length; i++) {
      Real minima[D];
      Real maxima[D];

      for (int j = 0; j < D; ++j) {
        minima[j] = rx(i, j);  // Direct assignment with native Real type
        maxima[j] = rx(i, j + D);
      }

      // Validate bounding box (reject NaN/Inf, enforce min <= max)
      Real coords[2 * D];
      for (int j = 0; j < D; ++j) {
        coords[j] = minima[j];
        coords[j + D] = maxima[j];
      }
      validate_box(coords, D);

      auto bb = BB<D, Real>(minima, maxima);
      auto ri_i = ri(i);
      new (b + i) DataType<T, D, Real>{std::move(ri_i), std::move(bb)};
    }

    for (T i = 0; i < length; i++) {
      Real minima[D];
      Real maxima[D];
      for (int j = 0; j < D; ++j) {
        minima[j] = rx(i, j);
        maxima[j] = rx(i, j + D);
      }
      auto bb = BB<D, Real>(minima, maxima);
      auto ri_i = ri(i);
      idx2bb.emplace_hint(idx2bb.end(), std::move(ri_i), std::move(bb));
    }
    build(b, e, placement.get());
    // Phase 1: No need to free - unique_ptr handles cleanup automatically
  }

  void set_obj(const T &idx,
               std::optional<std::string> objdumps = std::nullopt) {
    if (objdumps) {
      auto val = objdumps.value();
      idx2data.emplace(idx, compress(val));
    }
  }

  py::object get_obj(const T &idx) {
    py::object obj = py::none();
    auto search = idx2data.find(idx);
    if (likely(search != idx2data.end())) {
      auto val = idx2data.at(idx);
      obj = py::cast<py::object>(py::bytes(decompress(val)));
    }
    return obj;
  }

  // Unified insert for any Real type (float32 or float64)
  void insert(const T &idx, const py::array_t<Real> &x,
              const std::optional<std::string> objdumps = std::nullopt) {
    // Phase 1: Thread-safety - protect entire insert operation
    std::lock_guard<std::recursive_mutex> lock(*tree_mutex_);

#ifdef MY_DEBUG
    ProfilerStart("insert.prof");
    std::cout << "profiler start of insert" << std::endl;
#endif
    vec<size_t> cands;
    BB<D, Real> bb;

    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    const auto &ndim = buff_info_x.ndim;
    // Phase 4: Improved error messages with context
    if (unlikely((shape_x[0] != 2 * D || ndim != 1))) {
      throw std::runtime_error(
          "Invalid shape for bounding box array. Expected shape (" +
          std::to_string(2 * D) + ",) but got shape (" +
          std::to_string(shape_x[0]) + ",) with ndim=" + std::to_string(ndim));
    }
    auto it = idx2bb.find(idx);
    if (unlikely(it != idx2bb.end())) {
      throw std::runtime_error(
          "Index already exists in tree: " + std::to_string(idx));
    }
    {
      Real minima[D];
      Real maxima[D];
      for (int i = 0; i < D; ++i) {
        minima[i] = *x.data(i);
        maxima[i] = *x.data(i + D);
      }

      // Validate bounding box (reject NaN/Inf, enforce min <= max)
      Real coords[2 * D];
      for (int j = 0; j < D; ++j) {
        coords[j] = minima[j];
        coords[j + D] = maxima[j];
      }
      validate_box(coords, D);

      bb = BB<D, Real>(minima, maxima);
    }
    idx2bb.emplace(idx, bb);
    set_obj(idx, objdumps);

    // Use adaptive epsilon based on bounding box scale
    Real adaptive_eps = calculate_adaptive_epsilon(bb);
    Real delta[D];
    for (int i = 0; i < D; ++i) {
      delta[i] = bb.max(i) - bb.min(i) + adaptive_eps;
    }

    // find the leaf node to insert
    Real c = 0.0;
    size_t count = flat_tree.size();
    while (cands.empty()) {
      Real d[D];
      for (int i = 0; i < D; ++i) {
        d[i] = delta[i] * c;
      }
      bb.expand(d);
      c = (c + 1) * 2;

      queue<size_t> que;
      auto qpush_if_intersect = [&](const size_t &i) {
        if (flat_tree[i](bb)) {
          que.emplace(i);
        }
      };

      qpush_if_intersect(0);
      while (!que.empty()) {
        size_t i = que.front();
        que.pop();
        PRTreeElement<T, B, D, Real> &elem = flat_tree[i];

        if (elem.leaf && elem.leaf->mbb(bb)) {
          cands.push_back(i);
        } else {
          for (size_t offset = 0; offset < B; offset++) {
            size_t j = i * B + offset + 1;
            if (j < count)
              qpush_if_intersect(j);
          }
        }
      }
    }

    if (unlikely(cands.empty()))
      throw std::runtime_error("cannnot determine where to insert");

    // Now cands is the list of candidate leaf nodes to insert
    bb = idx2bb.at(idx);
    size_t min_leaf = 0;
    if (cands.size() == 1) {
      min_leaf = cands[0];
    } else {
      Real min_diff_area = 1e100;
      for (const auto &i : cands) {
        PRTreeLeaf<T, B, D, Real> *leaf = flat_tree[i].leaf.get();
        PRTreeLeaf<T, B, D, Real> tmp_leaf = PRTreeLeaf<T, B, D, Real>(*leaf);
        Real diff_area = -tmp_leaf.area();
        tmp_leaf.push(idx, bb);
        diff_area += tmp_leaf.area();
        if (diff_area < min_diff_area) {
          min_diff_area = diff_area;
          min_leaf = i;
        }
      }
    }
    flat_tree[min_leaf].leaf->push(idx, bb);
    // update mbbs of all cands and their parents
    size_t i = min_leaf;
    while (true) {
      PRTreeElement<T, B, D, Real> &elem = flat_tree[i];

      if (elem.leaf)
        elem.mbb += elem.leaf->mbb;

      if (i > 0) {
        size_t j = (i - 1) / B;
        flat_tree[j].mbb += flat_tree[i].mbb;
      }
      if (i == 0)
        break;
      i = (i - 1) / B;
    }

    if (size() > REBUILD_THRE * n_at_build) {
      rebuild();
    }
#ifdef MY_DEBUG
    ProfilerStop();
    std::cout << "profiler end of insert" << std::endl;
#endif
  }

  void rebuild() {
    // Phase 1: Thread-safety - protect entire rebuild operation
    std::lock_guard<std::recursive_mutex> lock(*tree_mutex_);

    std::stack<size_t> sta;
    T length = idx2bb.size();
    DataType<T, D, Real> *b, *e;

    // Phase 1: RAII memory management to prevent leaks on exception
    struct MallocDeleter {
      void operator()(void* ptr) const {
        if (ptr) std::free(ptr);
      }
    };
    std::unique_ptr<void, MallocDeleter> placement(
        std::malloc(sizeof(DataType<T, D, Real>) * length)
    );
    if (!placement) {
      throw std::bad_alloc();
    }
    b = reinterpret_cast<DataType<T, D, Real> *>(placement.get());
    e = b + length;

    T i = 0;
    sta.push(0);
    while (!sta.empty()) {
      size_t idx = sta.top();
      sta.pop();

      PRTreeElement<T, B, D, Real> &elem = flat_tree[idx];

      if (elem.leaf) {
        for (const auto &datum : elem.leaf->data) {
          new (b + i) DataType<T, D, Real>{datum.first, datum.second};
          i++;
        }
      } else {
        for (size_t offset = 0; offset < B; offset++) {
          size_t jdx = idx * B + offset + 1;
          if (likely(flat_tree[jdx].is_used)) {
            sta.push(jdx);
          }
        }
      }
    }

    build(b, e, placement.get());
    // Phase 1: No need to free - unique_ptr handles cleanup automatically
  }

  template <class iterator>
  void build(const iterator &b, const iterator &e, void *placement) {
#ifdef MY_DEBUG
    ProfilerStart("build.prof");
    std::cout << "profiler start of build" << std::endl;
#endif
    std::unique_ptr<PRTreeNode<T, B, D, Real>> root;
    {
      n_at_build = size();
      vec<std::unique_ptr<PRTreeNode<T, B, D, Real>>> prev_nodes;
      std::unique_ptr<PRTreeNode<T, B, D, Real>> p, q, r;

      auto first_tree = PseudoPRTree<T, B, D, Real>(b, e);
      auto first_leaves = first_tree.get_all_leaves(e - b);
      for (auto &leaf : first_leaves) {
        auto pp = std::make_unique<PRTreeNode<T, B, D, Real>>(leaf);
        prev_nodes.push_back(std::move(pp));
      }
      auto [bb, ee] = first_tree.as_X(placement, e - b);
      while (prev_nodes.size() > 1) {
        auto tree = PseudoPRTree<T, B, D, Real>(bb, ee);
        auto leaves = tree.get_all_leaves(ee - bb);
        auto leaves_size = leaves.size();

        vec<std::unique_ptr<PRTreeNode<T, B, D, Real>>> tmp_nodes;
        tmp_nodes.reserve(leaves_size);

        for (auto &leaf : leaves) {
          int idx, jdx;
          int len = leaf->data.size();
          auto pp = std::make_unique<PRTreeNode<T, B, D, Real>>(leaf->mbb);
          if (likely(!leaf->data.empty())) {
            for (int i = 1; i < len; i++) {
              idx = leaf->data[len - i - 1].first; // reversed way
              jdx = leaf->data[len - i].first;
              prev_nodes[idx]->next = std::move(prev_nodes[jdx]);
            }
            idx = leaf->data[0].first;
            pp->head = std::move(prev_nodes[idx]);
            if (unlikely(!pp->head)) {
              throw std::runtime_error("ppp");
            }
            tmp_nodes.push_back(std::move(pp));
          } else {
            throw std::runtime_error("what????");
          }
        }

        prev_nodes.swap(tmp_nodes);
        if (prev_nodes.size() > 1) {
          auto tmp = tree.as_X(placement, ee - bb);
          bb = std::move(tmp.first);
          ee = std::move(tmp.second);
        }
      }
      if (unlikely(prev_nodes.size() != 1)) {
        throw std::runtime_error("#roots is not 1.");
      }
      root = std::move(prev_nodes[0]);
    }
    // flatten built tree
    {
      queue<std::pair<PRTreeNode<T, B, D, Real> *, size_t>> que;
      PRTreeNode<T, B, D, Real> *p, *q;

      int depth = 0;

      p = root.get();
      while (p->head) {
        p = p->head.get();
        depth++;
      }

      // resize
      {
        flat_tree.clear();
        flat_tree.shrink_to_fit();
        size_t count = 0;
        for (int i = 0; i <= depth; i++) {
          count += std::pow(B, depth);
        }
        flat_tree.resize(count);
      }

      // assign
      que.emplace(root.get(), 0);
      while (!que.empty()) {
        auto tmp = que.front();
        que.pop();
        p = tmp.first;
        size_t idx = tmp.second;

        flat_tree[idx] = PRTreeElement(*p);
        size_t child_idx = 0;
        if (p->head) {
          size_t jdx = idx * B + child_idx + 1;
          ++child_idx;

          q = p->head.get();
          que.emplace(q, jdx);
          while (q->next) {
            jdx = idx * B + child_idx + 1;
            ++child_idx;

            q = q->next.get();
            que.emplace(q, jdx);
          }
        }
      }
    }

#ifdef MY_DEBUG
    ProfilerStop();
    std::cout << "profiler end of build" << std::endl;
#endif
  }

  auto find_all(const py::array_t<Real> &x) {
#ifdef MY_DEBUG
    ProfilerStart("find_all.prof");
    std::cout << "profiler start of find_all" << std::endl;
#endif
    const auto &buff_info_x = x.request();
    const auto &ndim = buff_info_x.ndim;
    const auto &shape_x = buff_info_x.shape;
    bool is_point = false;
    if (unlikely(ndim == 1 && (!(shape_x[0] == 2 * D || shape_x[0] == D)))) {
      throw std::runtime_error("Invalid Bounding box size");
    }
    if (unlikely((ndim == 2 && (!(shape_x[1] == 2 * D || shape_x[1] == D))))) {
      throw std::runtime_error(
          "Bounding box must have the shape (length, 2 * dim)");
    }
    if (unlikely(ndim > 3)) {
      throw std::runtime_error("invalid shape");
    }

    if (ndim == 1) {
      if (shape_x[0] == D) {
        is_point = true;
      }
    } else {
      if (shape_x[1] == D) {
        is_point = true;
      }
    }
    vec<BB<D, Real>> X;
    X.reserve(ndim == 1 ? 1 : shape_x[0]);
    BB<D, Real> bb;
    if (ndim == 1) {
      {
        Real minima[D];
        Real maxima[D];
        for (int i = 0; i < D; ++i) {
          minima[i] = *x.data(i);
          if (is_point) {
            maxima[i] = minima[i];
          } else {
            maxima[i] = *x.data(i + D);
          }
        }
        bb = BB<D, Real>(minima, maxima);
      }
      X.push_back(std::move(bb));
    } else {
      X.reserve(shape_x[0]);
      for (long int i = 0; i < shape_x[0]; i++) {
        {
          Real minima[D];
          Real maxima[D];
          for (int j = 0; j < D; ++j) {
            minima[j] = *x.data(i, j);
            if (is_point) {
              maxima[j] = minima[j];
            } else {
              maxima[j] = *x.data(i, j + D);
            }
          }
          bb = BB<D, Real>(minima, maxima);
        }
        X.push_back(std::move(bb));
      }
    }
    // Build exact query coordinates for refinement
    vec<std::array<double, 2 * D>> queries_exact;
    queries_exact.reserve(X.size());

    if (ndim == 1) {
      std::array<double, 2 * D> qe;
      for (int i = 0; i < D; ++i) {
        qe[i] = static_cast<double>(*x.data(i));
        if (is_point) {
          qe[i + D] = qe[i];
        } else {
          qe[i + D] = static_cast<double>(*x.data(i + D));
        }
      }
      queries_exact.push_back(qe);
    } else {
      for (long int i = 0; i < shape_x[0]; i++) {
        std::array<double, 2 * D> qe;
        for (int j = 0; j < D; ++j) {
          qe[j] = static_cast<double>(*x.data(i, j));
          if (is_point) {
            qe[j + D] = qe[j];
          } else {
            qe[j + D] = static_cast<double>(*x.data(i, j + D));
          }
        }
        queries_exact.push_back(qe);
      }
    }

    vec<vec<T>> out;
    out.resize(X.size()); // Pre-size for index-based parallel access
#ifdef MY_DEBUG
    for (size_t i = 0; i < X.size(); ++i) {
      auto candidates = find(X[i]);
      out[i] = candidates;
    }
#else
    // Index-based parallel loop (safe, no pointer arithmetic)
    const size_t n_queries = X.size();

    // Early return if no queries
    if (n_queries == 0) {
      return out;
    }

    // Guard against hardware_concurrency() returning 0 (can happen on macOS)
    size_t hw = std::thread::hardware_concurrency();
    size_t n_threads = hw ? hw : 1;
    n_threads = std::min(n_threads, n_queries);

    const size_t chunk_size = (n_queries + n_threads - 1) / n_threads;

    vec<std::thread> threads;
    threads.reserve(n_threads);

    for (size_t t = 0; t < n_threads; ++t) {
      threads.emplace_back([&, t]() {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n_queries);
        for (size_t i = start; i < end; ++i) {
          auto candidates = find(X[i]);
          out[i] = candidates;
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
#endif
#ifdef MY_DEBUG
    ProfilerStop();
    std::cout << "profiler end of find_all" << std::endl;
#endif
    return out;
  }

  auto find_all_array(const py::array_t<Real> &x) {
    return list_list_to_arrays(std::move(find_all(x)));
  }

  auto find_one(const vec<Real> &x) {
    bool is_point = false;
    if (unlikely(!(x.size() == 2 * D || x.size() == D))) {
      throw std::runtime_error("invalid shape");
    }
    Real minima[D];
    Real maxima[D];

    if (x.size() == D) {
      is_point = true;
    }
    for (int i = 0; i < D; ++i) {
      minima[i] = x.at(i);

      if (is_point) {
        maxima[i] = minima[i];
      } else {
        maxima[i] = x.at(i + D);
      }
    }
    const auto bb = BB<D, Real>(minima, maxima);
    auto candidates = find(bb);

    return candidates;
  }

  // Helper method: Check intersection with double precision (closed interval
  // semantics)
  bool intersects_exact(const std::array<double, 2 * D> &box_a,
                        const std::array<double, 2 * D> &box_b) const {
    for (int i = 0; i < D; ++i) {
      double a_min = box_a[i];
      double a_max = box_a[i + D];
      double b_min = box_b[i];
      double b_max = box_b[i + D];

      // Closed interval: boxes touch if a_max == b_min or b_max == a_min
      if (a_min > b_max || b_min > a_max) {
        return false;
      }
    }
    return true;
  }

  vec<T> find(const BB<D, Real> &target) {
    vec<T> out;
    auto find_func = [&](std::unique_ptr<PRTreeLeaf<T, B, D, Real>> &leaf) {
      (*leaf)(target, out);
    };

    bfs<T, B, D, Real>(std::move(find_func), flat_tree, target);
    std::sort(out.begin(), out.end());
    return out;
  }

  void erase(const T idx) {
    // Phase 1: Thread-safety - protect entire erase operation
    std::lock_guard<std::recursive_mutex> lock(*tree_mutex_);

    auto it = idx2bb.find(idx);
    if (unlikely(it == idx2bb.end())) {
      // Phase 4: Improved error message with context (backward compatible)
      throw std::runtime_error(
          "Given index is not found. (Index: " + std::to_string(idx) +
          ", tree size: " + std::to_string(idx2bb.size()) + ")");
    }
    BB<D, Real> target = it->second;

    auto erase_func = [&](std::unique_ptr<PRTreeLeaf<T, B, D, Real>> &leaf) {
      leaf->del(idx, target);
    };

    bfs<T, B, D, Real>(std::move(erase_func), flat_tree, target);

    idx2bb.erase(idx);
    idx2data.erase(idx);
    if (unlikely(REBUILD_THRE * size() < n_at_build)) {
      rebuild();
    }
  }

  int64_t size() const noexcept {
    std::lock_guard<std::recursive_mutex> lock(*tree_mutex_);
    return static_cast<int64_t>(idx2bb.size());
  }

  bool empty() const noexcept {
    std::lock_guard<std::recursive_mutex> lock(*tree_mutex_);
    return idx2bb.empty();
  }

  /**
   * Find all pairs of intersecting AABBs in the tree.
   * Returns a numpy array of shape (n_pairs, 2) where each row contains
   * a pair of indices (i, j) with i < j representing intersecting AABBs.
   *
   * This method is optimized for performance by:
   * - Using parallel processing for queries
   * - Avoiding duplicate pairs by enforcing i < j
   * - Performing intersection checks in C++ to minimize Python overhead
   * - Using double-precision refinement when exact coordinates are available
   *
   * @return py::array_t<T> Array of shape (n_pairs, 2) containing index pairs
   */
  py::array_t<T> query_intersections() {
    // Collect all indices and bounding boxes
    vec<T> indices;
    vec<BB<D, Real>> bboxes;

    if (unlikely(idx2bb.empty())) {
      // Return empty array of shape (0, 2)
      vec<T> empty_data;
      std::unique_ptr<vec<T>> data_ptr =
          std::make_unique<vec<T>>(std::move(empty_data));
      auto capsule = py::capsule(data_ptr.get(), [](void *p) {
        std::unique_ptr<vec<T>>(reinterpret_cast<vec<T> *>(p));
      });
      data_ptr.release();
      return py::array_t<T>({0, 2}, {2 * sizeof(T), sizeof(T)}, nullptr,
                            capsule);
    }

    indices.reserve(idx2bb.size());
    bboxes.reserve(idx2bb.size());

    for (const auto &pair : idx2bb) {
      indices.push_back(pair.first);
      bboxes.push_back(pair.second);
    }

    const size_t n_items = indices.size();

    // Use thread-local storage to collect pairs
    // Guard against hardware_concurrency() returning 0 (can happen on some
    // systems)
    size_t hw = std::thread::hardware_concurrency();
    size_t n_threads = hw ? hw : 1;
    n_threads = std::min(n_threads, n_items);
    vec<vec<std::pair<T, T>>> thread_pairs(n_threads);

#ifdef MY_PARALLEL
    vec<std::thread> threads;
    threads.reserve(n_threads);

    for (size_t t = 0; t < n_threads; ++t) {
      threads.emplace_back([&, t]() {
        vec<std::pair<T, T>> local_pairs;

        for (size_t i = t; i < n_items; i += n_threads) {
          const T idx_i = indices[i];
          const BB<D, Real> &bb_i = bboxes[i];

          // Find all intersections with this bounding box
          auto candidates = find(bb_i);

          // Keep only pairs where idx_i < idx_j to avoid duplicates
          for (const T &idx_j : candidates) {
            if (idx_i < idx_j) {
              local_pairs.emplace_back(idx_i, idx_j);
            }
          }
        }

        thread_pairs[t] = std::move(local_pairs);
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
#else
    // Single-threaded version
    vec<std::pair<T, T>> local_pairs;

    for (size_t i = 0; i < n_items; ++i) {
      const T idx_i = indices[i];
      const BB<D, Real> &bb_i = bboxes[i];

      // Find all intersections with this bounding box
      auto candidates = find(bb_i);

      // Keep only pairs where idx_i < idx_j to avoid duplicates
      for (const T &idx_j : candidates) {
        if (idx_i < idx_j) {
          local_pairs.emplace_back(idx_i, idx_j);
        }
      }
    }

    thread_pairs[0] = std::move(local_pairs);
#endif

    // Merge results from all threads into a flat vector
    vec<T> flat_pairs;
    size_t total_pairs = 0;
    for (const auto &pairs : thread_pairs) {
      total_pairs += pairs.size();
    }
    flat_pairs.reserve(total_pairs * 2);

    for (const auto &pairs : thread_pairs) {
      for (const auto &pair : pairs) {
        flat_pairs.push_back(pair.first);
        flat_pairs.push_back(pair.second);
      }
    }

    // Create output numpy array using the same pattern as as_pyarray
    auto data = flat_pairs.data();
    std::unique_ptr<vec<T>> data_ptr =
        std::make_unique<vec<T>>(std::move(flat_pairs));
    auto capsule = py::capsule(data_ptr.get(), [](void *p) {
      std::unique_ptr<vec<T>>(reinterpret_cast<vec<T> *>(p));
    });
    data_ptr.release();

    // Return 2D array with shape (total_pairs, 2)
    return py::array_t<T>(
        {static_cast<py::ssize_t>(total_pairs), py::ssize_t(2)}, // shape
        {2 * sizeof(T), sizeof(T)}, // strides (row-major)
        data,                       // data pointer
        capsule                     // capsule for cleanup
    );
  }

  // Precision control methods

  /**
   * Set relative epsilon for adaptive precision calculation.
   * This epsilon is multiplied by the coordinate scale to determine
   * the precision threshold for insert operations.
   *
   * @param epsilon Relative epsilon (default: 1e-6)
   */
  void set_relative_epsilon(Real epsilon) {
    if (epsilon <= 0.0 || !std::isfinite(epsilon)) {
      throw std::runtime_error("Relative epsilon must be positive and finite");
    }
    relative_epsilon_ = epsilon;
  }

  /**
   * Set absolute epsilon for precision calculation.
   * This is the minimum epsilon used regardless of coordinate scale,
   * ensuring backward compatibility and reasonable behavior for small coordinates.
   *
   * @param epsilon Absolute epsilon (default: 1e-8)
   */
  void set_absolute_epsilon(Real epsilon) {
    if (epsilon <= 0.0 || !std::isfinite(epsilon)) {
      throw std::runtime_error("Absolute epsilon must be positive and finite");
    }
    absolute_epsilon_ = epsilon;
  }

  /**
   * Enable or disable adaptive epsilon calculation.
   * When disabled, only absolute_epsilon is used (backward compatible behavior).
   *
   * @param enabled True to enable adaptive epsilon (default: true)
   */
  void set_adaptive_epsilon(bool enabled) {
    use_adaptive_epsilon_ = enabled;
  }

  /**
   * Enable or disable subnormal number detection.
   * When enabled, validation will reject subnormal numbers to avoid precision issues.
   *
   * @param enabled True to enable detection (default: true)
   */
  void set_subnormal_detection(bool enabled) {
    detect_subnormal_ = enabled;
  }

  // Getters for precision parameters

  Real get_relative_epsilon() const { return relative_epsilon_; }
  Real get_absolute_epsilon() const { return absolute_epsilon_; }
  bool get_adaptive_epsilon() const { return use_adaptive_epsilon_; }
  bool get_subnormal_detection() const { return detect_subnormal_; }
};
