/**
 * @file pseudo_tree.h
 * @brief Pseudo PRTree structures used during construction
 *
 * Contains Leaf, PseudoPRTreeNode, and PseudoPRTree classes that form
 * the intermediate data structure during PRTree construction.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <thread>
#include <utility>

#include "prtree/core/detail/bounding_box.h"
#include "prtree/core/detail/data_type.h"
#include "prtree/core/detail/types.h"

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int B = 6, int D = 2, typename Real = float> class Leaf {
public:
  BB<D, Real> mbb;
  svec<DataType<T, D, Real>, B> data; // You can swap when filtering
  int axis = 0;

  // T is type of keys(ids) which will be returned when you post a query.
  Leaf() { mbb = BB<D, Real>(); }
  Leaf(const int _axis) {
    axis = _axis;
    mbb = BB<D, Real>();
  }

  void set_axis(const int &_axis) { axis = _axis; }

  void push(const T &key, const BB<D, Real> &target) {
    data.emplace_back(key, target);
    update_mbb();
  }

  void update_mbb() {
    mbb.clear();
    for (const auto &datum : data) {
      mbb += datum.second;
    }
  }

  bool filter(DataType<T, D, Real> &value) { // false means given value is ignored
    // Phase 2: C++20 requires explicit 'this' capture
    auto comp = [this](const auto &a, const auto &b) noexcept {
      return a.second.val_for_comp(axis) < b.second.val_for_comp(axis);
    };

    if (data.size() < B) { // if there is room, just push the candidate
      auto iter = std::lower_bound(data.begin(), data.end(), value, comp);
      DataType<T, D, Real> tmp_value = DataType<T, D, Real>(value);
      data.insert(iter, std::move(tmp_value));
      mbb += value.second;
      return true;
    } else { // if there is no room, check the priority and swap if needed
      if (data[0].second.val_for_comp(axis) < value.second.val_for_comp(axis)) {
        size_t n_swap =
            std::lower_bound(data.begin(), data.end(), value, comp) -
            data.begin();
        std::swap(*data.begin(), value);
        auto iter = data.begin();
        for (size_t i = 0; i < n_swap - 1; ++i) {
          std::swap(*(iter + i), *(iter + i + 1));
        }
        update_mbb();
      }
      return false;
    }
  }
};

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int B = 6, int D = 2, typename Real = float> class PseudoPRTreeNode {
public:
  Leaf<T, B, D, Real> leaves[2 * D];
  std::unique_ptr<PseudoPRTreeNode> left, right;

  PseudoPRTreeNode() {
    for (int i = 0; i < 2 * D; i++) {
      leaves[i].set_axis(i);
    }
  }
  PseudoPRTreeNode(const int axis) {
    for (int i = 0; i < 2 * D; i++) {
      const int j = (axis + i) % (2 * D);
      leaves[i].set_axis(j);
    }
  }

  template <class Archive> void serialize(Archive &archive) {
    // archive(cereal::(left), cereal::defer(right), leaves);
    archive(left, right, leaves);
  }

  void address_of_leaves(vec<Leaf<T, B, D, Real> *> &out) {
    for (auto &leaf : leaves) {
      if (leaf.data.size() > 0) {
        out.emplace_back(&leaf);
      }
    }
  }

  template <class iterator> auto filter(const iterator &b, const iterator &e) {
    auto out = std::remove_if(b, e, [&](auto &x) {
      for (auto &l : leaves) {
        if (l.filter(x)) {
          return true;
        }
      }
      return false;
    });
    return out;
  }
};

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int B = 6, int D = 2, typename Real = float> class PseudoPRTree {
public:
  std::unique_ptr<PseudoPRTreeNode<T, B, D, Real>> root;
  vec<Leaf<T, B, D, Real> *> cache_children;
  const int nthreads = std::max(1, (int)std::thread::hardware_concurrency());

  PseudoPRTree() { root = std::make_unique<PseudoPRTreeNode<T, B, D, Real>>(); }

  template <class iterator> PseudoPRTree(const iterator &b, const iterator &e) {
    if (!root) {
      root = std::make_unique<PseudoPRTreeNode<T, B, D, Real>>();
    }
    construct(root.get(), b, e, 0);
    clean_data<T, D, Real>(b, e);
  }

  template <class Archive> void serialize(Archive &archive) {
    archive(root);
    // archive.serializeDeferments();
  }

  template <class iterator>
  void construct(PseudoPRTreeNode<T, B, D, Real> *node, const iterator &b,
                 const iterator &e, const int depth) {
    if (e - b > 0 && node != nullptr) {
      bool use_recursive_threads = std::pow(2, depth + 1) <= nthreads;
#ifdef MY_DEBUG
      use_recursive_threads = false;
#endif

      vec<std::thread> threads;
      threads.reserve(2);
      PseudoPRTreeNode<T, B, D, Real> *node_left, *node_right;

      const int axis = depth % (2 * D);
      auto ee = node->filter(b, e);
      auto m = b;
      std::advance(m, (ee - b) / 2);
      std::nth_element(b, m, ee,
                       [axis](const DataType<T, D, Real> &lhs,
                              const DataType<T, D, Real> &rhs) noexcept {
                         return lhs.second[axis] < rhs.second[axis];
                       });

      if (m - b > 0) {
        node->left = std::make_unique<PseudoPRTreeNode<T, B, D, Real>>(axis);
        node_left = node->left.get();
        if (use_recursive_threads) {
          threads.push_back(
              std::thread([&]() { construct(node_left, b, m, depth + 1); }));
        } else {
          construct(node_left, b, m, depth + 1);
        }
      }
      if (ee - m > 0) {
        node->right = std::make_unique<PseudoPRTreeNode<T, B, D, Real>>(axis);
        node_right = node->right.get();
        if (use_recursive_threads) {
          threads.push_back(
              std::thread([&]() { construct(node_right, m, ee, depth + 1); }));
        } else {
          construct(node_right, m, ee, depth + 1);
        }
      }
      std::for_each(threads.begin(), threads.end(),
                    [&](std::thread &x) { x.join(); });
    }
  }

  auto get_all_leaves(const int hint) {
    if (cache_children.empty()) {
      using U = PseudoPRTreeNode<T, B, D, Real>;
      cache_children.reserve(hint);
      auto node = root.get();
      queue<U *> que;
      que.emplace(node);

      while (!que.empty()) {
        node = que.front();
        que.pop();
        node->address_of_leaves(cache_children);
        if (node->left)
          que.emplace(node->left.get());
        if (node->right)
          que.emplace(node->right.get());
      }
    }
    return cache_children;
  }

  std::pair<DataType<T, D, Real> *, DataType<T, D, Real> *> as_X(void *placement,
                                                     const int hint) {
    DataType<T, D, Real> *b, *e;
    auto children = get_all_leaves(hint);
    T total = children.size();
    b = reinterpret_cast<DataType<T, D, Real> *>(placement);
    e = b + total;
    for (T i = 0; i < total; i++) {
      new (b + i) DataType<T, D, Real>{i, children[i]->mbb};
    }
    return {b, e};
  }
};
