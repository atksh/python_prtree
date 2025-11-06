/**
 * @file bounding_box.h
 * @brief Axis-Aligned Bounding Box (AABB) implementation
 *
 * Provides the BB<D> class for D-dimensional bounding boxes with
 * geometric operations like intersection, union, and containment tests.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include <cereal/cereal.hpp>

#include "prtree/core/detail/types.h"

using Real = float;

/**
 * @brief D-dimensional Axis-Aligned Bounding Box
 *
 * Stores min/max coordinates for each dimension and provides
 * geometric operations.
 *
 * @tparam D Number of dimensions (2, 3, or 4)
 */
template <int D = 2>
class BB {
public:
  std::array<Real, D> lo;  ///< Minimum coordinates
  std::array<Real, D> hi;  ///< Maximum coordinates

  /// Default constructor - creates an invalid/empty box
  BB() {
    for (int i = 0; i < D; i++) {
      lo[i] = std::numeric_limits<Real>::max();
      hi[i] = -std::numeric_limits<Real>::max();
    }
  }

  /// Constructor from coordinate arrays
  BB(const std::array<Real, D> &lo_, const std::array<Real, D> &hi_)
      : lo(lo_), hi(hi_) {}

  /// Constructor from iterators (for compatibility with span/vector)
  template <typename Iterator>
  BB(Iterator lo_begin, Iterator lo_end, Iterator hi_begin, Iterator hi_end) {
    std::copy(lo_begin, lo_end, lo.begin());
    std::copy(hi_begin, hi_end, hi.begin());
  }

  /**
   * @brief Check if this box intersects with another
   *
   * Two boxes intersect if they overlap in all dimensions.
   */
  bool intersects(const BB &other) const {
    for (int i = 0; i < D; i++) {
      if (hi[i] < other.lo[i] || lo[i] > other.hi[i])
        return false;
    }
    return true;
  }

  /**
   * @brief Check if this box contains a point
   */
  bool contains_point(const std::array<Real, D> &point) const {
    for (int i = 0; i < D; i++) {
      if (point[i] < lo[i] || point[i] > hi[i])
        return false;
    }
    return true;
  }

  /**
   * @brief Check if this box completely contains another
   */
  bool contains(const BB &other) const {
    for (int i = 0; i < D; i++) {
      if (other.lo[i] < lo[i] || other.hi[i] > hi[i])
        return false;
    }
    return true;
  }

  /**
   * @brief Compute the union of this box with another
   *
   * Returns the smallest box that contains both boxes.
   */
  BB union_with(const BB &other) const {
    BB result;
    for (int i = 0; i < D; i++) {
      result.lo[i] = std::min(lo[i], other.lo[i]);
      result.hi[i] = std::max(hi[i], other.hi[i]);
    }
    return result;
  }

  /**
   * @brief Compute the intersection of this box with another
   *
   * Returns an empty box if they don't intersect.
   */
  BB intersection_with(const BB &other) const {
    BB result;
    for (int i = 0; i < D; i++) {
      result.lo[i] = std::max(lo[i], other.lo[i]);
      result.hi[i] = std::min(hi[i], other.hi[i]);
      if (result.lo[i] > result.hi[i])
        return BB(); // Empty box
    }
    return result;
  }

  /**
   * @brief Compute the volume (area in 2D) of the box
   */
  Real volume() const {
    Real vol = 1.0;
    for (int i = 0; i < D; i++) {
      Real extent = hi[i] - lo[i];
      if (extent < 0)
        return 0; // Invalid box
      vol *= extent;
    }
    return vol;
  }

  /**
   * @brief Compute the perimeter (in 2D) or surface area (in 3D)
   */
  Real perimeter() const {
    if constexpr (D == 2) {
      return 2 * ((hi[0] - lo[0]) + (hi[1] - lo[1]));
    } else if constexpr (D == 3) {
      Real dx = hi[0] - lo[0];
      Real dy = hi[1] - lo[1];
      Real dz = hi[2] - lo[2];
      return 2 * (dx * dy + dy * dz + dz * dx);
    } else {
      // For other dimensions, return sum of extents
      Real sum = 0;
      for (int i = 0; i < D; i++)
        sum += hi[i] - lo[i];
      return sum;
    }
  }

  /**
   * @brief Compute the center point of the box
   */
  std::array<Real, D> center() const {
    std::array<Real, D> c;
    for (int i = 0; i < D; i++)
      c[i] = (lo[i] + hi[i]) / 2;
    return c;
  }

  /**
   * @brief Check if the box is valid (min <= max for all dimensions)
   */
  bool is_valid() const {
    for (int i = 0; i < D; i++) {
      if (lo[i] > hi[i])
        return false;
    }
    return true;
  }

  /**
   * @brief Check if the box is empty (zero volume)
   */
  bool is_empty() const { return volume() == 0; }

  /**
   * @brief Expand the box to include a point
   */
  void expand_to_include(const std::array<Real, D> &point) {
    for (int i = 0; i < D; i++) {
      lo[i] = std::min(lo[i], point[i]);
      hi[i] = std::max(hi[i], point[i]);
    }
  }

  /**
   * @brief Expand the box to include another box
   */
  void expand_to_include(const BB &other) {
    for (int i = 0; i < D; i++) {
      lo[i] = std::min(lo[i], other.lo[i]);
      hi[i] = std::max(hi[i], other.hi[i]);
    }
  }

  /// Serialization support
  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(lo), CEREAL_NVP(hi));
  }
};
