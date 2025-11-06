/**
 * @file types.h
 * @brief Common types, concepts, and utility functions for PRTree
 *
 * This file contains:
 * - Type aliases and concepts
 * - Utility functions for Python/C++ interop
 * - Common constants and macros
 */
#pragma once

#include <concepts>
#include <deque>
#include <queue>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "prtree/utils/small_vector.h"

namespace py = pybind11;

// === Versioning ===

constexpr uint16_t PRTREE_VERSION_MAJOR = 1;
constexpr uint16_t PRTREE_VERSION_MINOR = 0;

// === C++20 Concepts ===

template <typename T>
concept IndexType = std::integral<T> && !std::same_as<T, bool>;

template <typename T>
concept SignedIndexType = IndexType<T> && std::is_signed_v<T>;

// === Type Aliases ===

template <class T>
using vec = std::vector<T>;

template <class T, size_t StaticCapacity>
using svec = itlib::small_vector<T, StaticCapacity>;

template <class T>
using deque = std::deque<T>;

template <class T>
using queue = std::queue<T, deque<T>>;

// === Constants ===

static const float REBUILD_THRE = 1.25;

// === Branch Prediction Hints ===

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// === Python Interop Utilities ===

/**
 * @brief Convert a C++ sequence to a numpy array with zero-copy
 *
 * Transfers ownership of the sequence data to Python.
 */
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &seq) {
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr =
      std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
  });
  seq_ptr.release();
  return py::array(size, data, capsule);
}

/**
 * @brief Convert nested vector to tuple of numpy arrays
 *
 * Returns (sizes, flattened_data) where sizes[i] is the length of out_ll[i]
 * and flattened_data contains all elements concatenated.
 */
template <typename T>
auto list_list_to_arrays(vec<vec<T>> out_ll) {
  vec<T> out_s;
  out_s.reserve(out_ll.size());
  std::size_t sum = 0;
  for (auto &&i : out_ll) {
    out_s.push_back(i.size());
    sum += i.size();
  }
  vec<T> out;
  out.reserve(sum);
  for (const auto &v : out_ll)
    out.insert(out.end(), v.begin(), v.end());

  return make_tuple(std::move(as_pyarray(out_s)), std::move(as_pyarray(out)));
}

// === Compression Utilities ===

#include <snappy.h>
#include <string>

inline std::string compress(std::string &data) {
  std::string output;
  snappy::Compress(data.data(), data.size(), &output);
  return output;
}

inline std::string decompress(std::string &data) {
  std::string output;
  snappy::Uncompress(data.data(), data.size(), &output);
  return output;
}
