/**
 * @file bounding_box.h
 * @brief Axis-Aligned Bounding Box (AABB) implementation
 *
 * Provides the BB<D> class for D-dimensional bounding boxes with
 * geometric operations like intersection, union, and area calculation.
 */
#pragma once

#include <algorithm>
#include <stdexcept>

#include <cereal/cereal.hpp>

#include "prtree/core/detail/types.h"

template <int D = 2, typename Real = float> class BB {
private:
  Real values[2 * D];

public:
  BB() { clear(); }

  BB(const Real (&minima)[D], const Real (&maxima)[D]) {
    Real v[2 * D];
    for (int i = 0; i < D; ++i) {
      v[i] = -minima[i];
      v[i + D] = maxima[i];
    }
    validate(v);
    for (int i = 0; i < D; ++i) {
      values[i] = v[i];
      values[i + D] = v[i + D];
    }
  }

  BB(const Real (&v)[2 * D]) {
    validate(v);
    for (int i = 0; i < D; ++i) {
      values[i] = v[i];
      values[i + D] = v[i + D];
    }
  }

  Real min(const int dim) const {
    if (unlikely(dim < 0 || D <= dim)) {
      throw std::runtime_error("Invalid dim");
    }
    return -values[dim];
  }
  Real max(const int dim) const {
    if (unlikely(dim < 0 || D <= dim)) {
      throw std::runtime_error("Invalid dim");
    }
    return values[dim + D];
  }

  bool validate(const Real (&v)[2 * D]) const {
    bool flag = false;
    for (int i = 0; i < D; ++i) {
      if (unlikely(-v[i] > v[i + D])) {
        flag = true;
        break;
      }
    }
    if (unlikely(flag)) {
      throw std::runtime_error("Invalid Bounding Box");
    }
    return flag;
  }
  void clear() noexcept {
    for (int i = 0; i < 2 * D; ++i) {
      values[i] = -1e100;
    }
  }

  Real val_for_comp(const int &axis) const noexcept {
    const int axis2 = (axis + 1) % (2 * D);
    return values[axis] + values[axis2];
  }

  BB operator+(const BB &rhs) const {
    Real result[2 * D];
    for (int i = 0; i < 2 * D; ++i) {
      result[i] = std::max(values[i], rhs.values[i]);
    }
    return BB<D>(result);
  }

  BB operator+=(const BB &rhs) {
    for (int i = 0; i < 2 * D; ++i) {
      values[i] = std::max(values[i], rhs.values[i]);
    }
    return *this;
  }

  void expand(const Real (&delta)[D]) noexcept {
    for (int i = 0; i < D; ++i) {
      values[i] += delta[i];
      values[i + D] += delta[i];
    }
  }

  bool operator()(
      const BB &target) const { // whether this and target has any intersect

    Real minima[D];
    Real maxima[D];
    bool flags[D];
    bool flag = true;

    for (int i = 0; i < D; ++i) {
      minima[i] = std::min(values[i], target.values[i]);
      maxima[i] = std::min(values[i + D], target.values[i + D]);
    }
    for (int i = 0; i < D; ++i) {
      flags[i] = -minima[i] <= maxima[i];
    }
    for (int i = 0; i < D; ++i) {
      flag &= flags[i];
    }
    return flag;
  }

  Real area() const {
    Real result = 1;
    for (int i = 0; i < D; ++i) {
      result *= max(i) - min(i);
    }
    return result;
  }

  inline Real operator[](const int i) const { return values[i]; }

  template <class Archive> void serialize(Archive &ar) { ar(values); }
};
