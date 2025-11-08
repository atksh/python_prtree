/**
 * @file nodes.h
 * @brief PRTree node implementations
 *
 * Contains PRTreeLeaf, PRTreeNode, PRTreeElement classes and utility
 * functions for the actual PRTree structure.
 */
#pragma once

#include <algorithm>
#include <functional>
#include <memory>

#include "prtree/core/detail/bounding_box.h"
#include "prtree/core/detail/data_type.h"
#include "prtree/core/detail/pseudo_tree.h"
#include "prtree/core/detail/types.h"

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int B = 6, int D = 2, typename Real = float> class PRTreeLeaf {
public:
  BB<D, Real> mbb;
  svec<DataType<T, D, Real>, B> data;

  PRTreeLeaf() { mbb = BB<D, Real>(); }

  PRTreeLeaf(const Leaf<T, B, D, Real> &leaf) {
    mbb = leaf.mbb;
    data = leaf.data;
  }

  Real area() const { return mbb.area(); }

  void update_mbb() {
    mbb.clear();
    for (const auto &datum : data) {
      mbb += datum.second;
    }
  }

  void operator()(const BB<D, Real> &target, vec<T> &out) const {
    if (mbb(target)) {
      for (const auto &x : data) {
        if (x.second(target)) {
          out.emplace_back(x.first);
        }
      }
    }
  }

  void del(const T &key, const BB<D, Real> &target) {
    if (mbb(target)) {
      auto remove_it =
          std::remove_if(data.begin(), data.end(), [&](auto &datum) {
            return datum.second(target) && datum.first == key;
          });
      data.erase(remove_it, data.end());
    }
  }

  void push(const T &key, const BB<D, Real> &target) {
    data.emplace_back(key, target);
    update_mbb();
  }

  template <class Archive> void save(Archive &ar) const {
    vec<DataType<T, D, Real>> _data;
    for (const auto &datum : data) {
      _data.push_back(datum);
    }
    ar(mbb, _data);
  }

  template <class Archive> void load(Archive &ar) {
    vec<DataType<T, D, Real>> _data;
    ar(mbb, _data);
    for (const auto &datum : _data) {
      data.push_back(datum);
    }
  }
};

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int B = 6, int D = 2, typename Real = float> class PRTreeNode {
public:
  BB<D, Real> mbb;
  std::unique_ptr<Leaf<T, B, D, Real>> leaf;
  std::unique_ptr<PRTreeNode<T, B, D, Real>> head, next;

  PRTreeNode() {}
  PRTreeNode(const BB<D, Real> &_mbb) { mbb = _mbb; }

  PRTreeNode(BB<D, Real> &&_mbb) noexcept { mbb = std::move(_mbb); }

  PRTreeNode(Leaf<T, B, D, Real> *l) {
    leaf = std::make_unique<Leaf<T, B, D, Real>>();
    mbb = l->mbb;
    leaf->mbb = std::move(l->mbb);
    leaf->data = std::move(l->data);
  }

  bool operator()(const BB<D, Real> &target) { return mbb(target); }
};

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int B = 6, int D = 2, typename Real = float> class PRTreeElement {
public:
  BB<D, Real> mbb;
  std::unique_ptr<PRTreeLeaf<T, B, D, Real>> leaf;
  bool is_used = false;

  PRTreeElement() {
    mbb = BB<D, Real>();
    is_used = false;
  }

  PRTreeElement(const PRTreeNode<T, B, D, Real> &node) {
    mbb = BB<D, Real>(node.mbb);
    if (node.leaf) {
      Leaf<T, B, D, Real> tmp_leaf = Leaf<T, B, D, Real>(*node.leaf.get());
      leaf = std::make_unique<PRTreeLeaf<T, B, D, Real>>(tmp_leaf);
    }
    is_used = true;
  }

  bool operator()(const BB<D, Real> &target) { return is_used && mbb(target); }

  template <class Archive> void serialize(Archive &archive) {
    archive(mbb, leaf, is_used);
  }
};

// Phase 8: Apply C++20 concept constraints
template <IndexType T, int B = 6, int D = 2, typename Real = float>
void bfs(
    const std::function<void(std::unique_ptr<PRTreeLeaf<T, B, D, Real>> &)> &func,
    vec<PRTreeElement<T, B, D, Real>> &flat_tree, const BB<D, Real> target) {
  queue<size_t> que;
  auto qpush_if_intersect = [&](const size_t &i) {
    PRTreeElement<T, B, D, Real> &r = flat_tree[i];
    // std::cout << "i " << (long int) i << " : " << (bool) r.leaf << std::endl;
    if (r(target)) {
      // std::cout << " is pushed" << std::endl;
      que.emplace(i);
    }
  };

  // std::cout << "size: " << flat_tree.size() << std::endl;
  qpush_if_intersect(0);
  while (!que.empty()) {
    size_t idx = que.front();
    // std::cout << "idx: " << (long int) idx << std::endl;
    que.pop();
    PRTreeElement<T, B, D, Real> &elem = flat_tree[idx];

    if (elem.leaf) {
      // std::cout << "func called for " << (long int) idx << std::endl;
      func(elem.leaf);
    } else {
      for (size_t offset = 0; offset < B; offset++) {
        size_t jdx = idx * B + offset + 1;
        qpush_if_intersect(jdx);
      }
    }
  }
}
