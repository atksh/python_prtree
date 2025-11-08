/**
 * @file data_type.h
 * @brief Data storage structures for PRTree
 *
 * Contains DataType class for storing index-bounding box pairs
 * and related utility functions.
 */
#pragma once

#include <utility>

#include "prtree/core/detail/bounding_box.h"
#include "prtree/core/detail/types.h"

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int D = 2> class DataType {
public:
  BB<D> second;
  T first;

  DataType() noexcept = default;

  DataType(const T &f, const BB<D> &s) {
    first = f;
    second = s;
  }

  DataType(T &&f, BB<D> &&s) noexcept {
    first = std::move(f);
    second = std::move(s);
  }

  void swap(DataType& other) noexcept {
    using std::swap;
    swap(first, other.first);
    swap(second, other.second);
  }

  template <class Archive> void serialize(Archive &ar) { ar(first, second); }
};

template <class T, int D = 2>
void clean_data(DataType<T, D> *b, DataType<T, D> *e) {
  for (DataType<T, D> *it = e - 1; it >= b; --it) {
    it->~DataType<T, D>();
  }
}
