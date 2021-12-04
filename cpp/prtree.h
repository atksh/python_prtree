#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stack>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include <optional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp> //for smart pointers
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/atomic.hpp>

#include <snappy.h>
#include "parallel.h"

using Real = float;

namespace py = pybind11;
using std::swap;

template <class T, class U> using pair = std::pair<T, U>;

template <class T> using vec = std::vector<T>;

template <class T> using deque = std::deque<T>;

template <class T> using queue = std::queue<T, deque<T>>;

static std::mt19937 rand_src(42);

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif


std::string compress(std::string& data){
  std::string output;
  snappy::Compress(data.data(), data.size(), &output);
  return output;
}

std::string decompress(std::string& data){
  std::string output;
  snappy::Uncompress(data.data(), data.size(), &output);
  return output;
}


template <typename Iter> Iter select_randomly(Iter start, Iter end) {
  std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
  std::advance(start, dis(rand_src));
  return start;
};

template <int D = 2> class BB {
private:
  std::array<Real, 2 * D> values;

public:
  BB() { clear(); }

  template <typename Iter> BB(const Iter &minima, const Iter &maxima) {
    std::array<Real, 2 * D> v;
    if (unlikely(minima.size() != maxima.size())) {
      throw std::runtime_error("Invalid size");
    }
    const int n = minima.size();
    if (unlikely(n != D)) {
      throw std::runtime_error("Invalid size");
    }
    for (int i = 0; i < n; ++i) {
      v[i] = -minima[i];
      v[i + D] = maxima[i];
    }
    validate(v);
    values = v;
  }

  BB(const std::array<Real, 2 * D> v) {
    validate(v);
    values = v;
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

  bool validate(const std::array<Real, 2 * D> &v) const {
    bool flag = false;
    for (int i = 0; i < 2; ++i) {
      if (-v[i] > v[i + D]) {
        flag = true;
        break;
      }
    }
    if (flag) {
      throw std::runtime_error("Invalid Bounding Box");
    }
    return flag;
  }
  void clear() { std::fill_n(values.begin(), 2 * D, -1e100); }

  BB operator+(const BB &rhs) const {
    std::array<Real, 2 * D> result;
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

  void expand(const std::array<Real, D> &delta) {
    for (int i = 0; i < D; ++i) {
      values[i] += delta[i];
      values[i + D] += delta[i];
    }
  }

  bool operator()(
      const BB &target) const { // whether this and target has any intersect
    std::array<Real, 2 * D> result;
    for (int i = 0; i < 2 * D; ++i) {
      result[i] = std::min(values[i], target.values[i]);
    }
    bool flag = true;
    for (int i = 0; i < D; ++i) {
      flag = flag && (-result[i] <= result[i + D]);
    }
    return flag;
  }

  Real operator[](const int i) const { return values[i]; }
  template <class Archive> void serialize(Archive &ar) { ar(values); }
};

template <class T, int D = 2> class DataType {
public:
  T first;
  BB<D> second;
  DataType(){};

  DataType(const T &f, const BB<D> &s) {
    first = f;
    second = s;
  }

  DataType(T &&f, BB<D> &&s) noexcept {
    first = std::move(f);
    second = std::move(s);
  }

  template <class Archive> void serialize(Archive &ar) { ar(first, second); }
};

template <class T, int B = 6, int D = 2> class Leaf {
public:
  int axis = 0;
  BB<D> mbb;
  vec<DataType<T, D>> data; // You can swap when filtering
  // T is type of keys(ids) which will be returned when you post a query.
  Leaf() {
    mbb = BB<D>();
    data.reserve(B);
  }
  Leaf(int &_axis) {
    axis = _axis;
    mbb = BB<D>();
    data.reserve(B);
  }

  ~Leaf() {
    data.clear();
    vec<DataType<T, D>>().swap(data);
  }

  template <class Archive> void serialize(Archive &ar) { ar(axis, mbb, data); }

  void set_axis(const int &_axis) { axis = _axis; }

  void push(const T &key, const BB<D> &target) {
    data.emplace_back(key, target);
    update_mbb();
  }

  void update_mbb() {
    mbb.clear();
    for (const auto &datum : data) {
      mbb += datum.second;
    }
  }

  template <typename F>
  bool filter(DataType<T, D> &value,
              const F &comp) { // false means given value is ignored
    if (unlikely(data.size() <
                 B)) { // if there is room, just push the candidate
      mbb += value.second;
      data.push_back(std::move(value));
      return true;
    } else { // if there is no room, check the priority and swap if needed
      auto iter = std::upper_bound(data.begin(), data.end(), value, comp);
      if (unlikely(iter != data.end())) {
        swap(*iter, value);
        update_mbb();
      }
      return false;
    }
  }

  void operator()(const BB<D> &target, vec<T> &out) const {
    if (unlikely(mbb(target))) {
      for (const auto &x : data) {
        if (unlikely(x.second(target))) {
          out.emplace_back(x.first);
        }
      }
    }
  }

  void del(const T &key, const BB<D> &target) {
    if (unlikely(mbb(target))) {
      auto remove_it =
          std::remove_if(data.begin(), data.end(), [&](auto &datum) {
            return datum.second(target) && datum.first == key;
          });
      data.erase(remove_it, data.end());
    }
  }
};

template <class T, int B = 6, int D = 2> class PseudoPRTreeNode {
public:
  std::array<Leaf<T, B, D>, 2 * D> leaves;
  std::unique_ptr<PseudoPRTreeNode> left, right;

  PseudoPRTreeNode() {
    for (int i = 0; i < 2 * D; i++) {
      leaves[i].set_axis(i);
    }
  }

  template <class Archive> void serialize(Archive &archive) {
    // archive(cereal::(left), cereal::defer(right), leaves);
    archive(left, right, leaves);
  }

  void address_of_leaves(vec<Leaf<T, B, D> *> &out) {
    for (auto &leaf : leaves) {
      if (likely(leaf.data.size() > 0)) {
        out.emplace_back(&leaf);
      }
    }
  }

  template <class iterator> auto filter(iterator &b, iterator &e) {
    vec<std::function<bool(const DataType<T, D> &, const DataType<T, D> &)>>
        comps;
    for (int axis = 0; axis < 2 * D; ++axis) {
      comps.emplace_back(
          [axis](const DataType<T, D> &lhs, const DataType<T, D> &rhs) {
            return lhs.second[axis] < rhs.second[axis];
          });
    }

    auto out = std::remove_if(b, e, [&](auto &x) {
      for (auto &l : leaves) {
        if (unlikely(l.filter(x, comps[l.axis]))) {
          return true;
        }
      }
      return false;
    });
    return out;
  }
};

template <class T, int B = 6, int D = 2> class PseudoPRTree {
public:
  std::unique_ptr<PseudoPRTreeNode<T, B, D>> root;
  vec<Leaf<T, B, D> *> cache_children;
  const int nthreads = std::max(1, (int)std::thread::hardware_concurrency());

  PseudoPRTree() { root = std::make_unique<PseudoPRTreeNode<T, B, D>>(); }

  ~PseudoPRTree() {
    cache_children.clear();
    vec<Leaf<T, B, D> *>().swap(cache_children);
  }

  template <class iterator> PseudoPRTree(iterator &b, iterator &e) {
    if (likely(!root)) {
      root = std::make_unique<PseudoPRTreeNode<T, B, D>>();
    }
    construct(root.get(), b, e, 0);

    for (DataType<T, D> *it = e - 1; it >= b; --it) {
      it->~DataType<T, D>();
    }
    b = nullptr;
    e = nullptr;
  }

  template <class Archive> void serialize(Archive &archive) {
    archive(root);
    // archive.serializeDeferments();
  }

  template <class iterator>
  void construct(PseudoPRTreeNode<T, B, D> *node, iterator &b, iterator &e,
                 const size_t depth) {
    if (e - b > 0 && node != nullptr) {
      bool use_recursive_threads = std::pow(2, depth + 1) <= nthreads;

      vec<std::thread> threads;
      threads.reserve(2);
      PseudoPRTreeNode<T, B, D> *node_left, *node_right;

      const int axis = depth % (2 * D);
      auto ee = node->filter(b, e);
      auto m = b;
      std::advance(m, (ee - b) / 2);
      auto mm = m;
      std::nth_element(b, m, ee,
                       [axis](const DataType<T, D> &lhs,
                              const DataType<T, D> &rhs) noexcept {
                         return lhs.second[axis] < rhs.second[axis];
                       });

      if (m - b > 0) {
        node->left = std::make_unique<PseudoPRTreeNode<T, B, D>>();
        node_left = node->left.get();
        if (use_recursive_threads) {
          threads.push_back(
              std::thread([&]() { construct(node_left, b, m, depth + 1); }));
        } else {
          construct(node_left, b, m, depth + 1);
        }
      }
      if (ee - mm > 0) {
        node->right = std::make_unique<PseudoPRTreeNode<T, B, D>>();
        node_right = node->right.get();
        if (use_recursive_threads) {
          threads.push_back(
              std::thread([&]() { construct(node_right, mm, ee, depth + 1); }));
        } else {
          construct(node_right, mm, ee, depth + 1);
        }
      }
      std::for_each(threads.begin(), threads.end(),
                    [&](std::thread &x) { x.join(); });
      threads.clear();
      vec<std::thread>().swap(threads);
    }
  }

  auto get_all_leaves(const int hint) {
    if (cache_children.empty()) {
      using U = PseudoPRTreeNode<T, B, D>;
      cache_children.reserve(hint);
      auto node = root.get();
      queue<U *> que;
      que.emplace(node);

      while (likely(!que.empty())) {
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

  std::pair<DataType<int, D> *, DataType<int, D> *> as_X(void *placement,
                                                         const int hint) {
    DataType<int, D> *b, *e;
    auto children = get_all_leaves(hint);
    int total = children.size();
    b = (DataType<int, D> *)placement;
    e = b + total;
    parallel_for_each(b, e, [&](auto &p) {
      int i = &p - b;
      new (b + i) DataType<int, D>{i, children[i]->mbb};
    });
    return {b, e};
  }
};

template <class T, int B = 6, int D = 2> class PRTreeNode {
public:
  BB<D> mbb;
  std::unique_ptr<Leaf<T, B, D>> leaf;
  std::unique_ptr<PRTreeNode<T, B, D>> head, next;

  PRTreeNode() {}
  PRTreeNode(const BB<D> &_mbb) { mbb = _mbb; }

  PRTreeNode(BB<D> &&_mbb) { mbb = std::move(_mbb); }

  PRTreeNode(Leaf<T, B, D> *l) {
    leaf = std::make_unique<Leaf<T, B, D>>();
    mbb = l->mbb;
    leaf->mbb = std::move(l->mbb);
    leaf->data = std::move(l->data);
  }

  bool operator()(const BB<D> &target) { return mbb(target); }

  template <class Archive> void serialize(Archive &archive) {
    // archive(cereal::defer(leaf), cereal::defer(head), cereal::defer(next));
    archive(mbb, leaf, head, next);
  }
};

template <class T, int B = 6, int D = 2> class PRTree {
private:
  std::unique_ptr<PRTreeNode<T, B, D>> root;
  std::unordered_map<T, BB<D>> idx2bb;
  std::unordered_map<T, std::string> idx2data;
  int64_t n_at_build = 0;
  std::atomic<T> global_idx = 0;

public:
  template <class Archive> void serialize(Archive &archive) {
    archive(root, idx2bb, idx2data, global_idx, n_at_build);
    // archive.serializeDeferments();
  }

  void save(std::string fname) {
    {
      {
        std::ofstream ofs(fname, std::ios::binary);
        cereal::PortableBinaryOutputArchive o_archive(ofs);
        o_archive(cereal::make_nvp("root", root),
                  cereal::make_nvp("idx2bb", idx2bb),
                  cereal::make_nvp("idx2data", idx2data),
                  cereal::make_nvp("global_idx", global_idx),
                  cereal::make_nvp("n_at_build", n_at_build));
      }
    }
  }

  PRTree() {
    root = std::make_unique<PRTreeNode<T, B, D>>();
  }

  ~PRTree() {
    root.reset();
    idx2bb.clear();
    idx2data.clear();
    std::unordered_map<T, BB<D>>().swap(idx2bb);
    std::unordered_map<T, std::string>().swap(idx2data);
  }

  PRTree(std::string fname) { load(fname); }

  void load(std::string fname) {
    {
      {
        std::ifstream ifs(fname, std::ios::binary);
        cereal::PortableBinaryInputArchive i_archive(ifs);
        // cereal::JSONInputArchive i_archive(ifs);
        i_archive(cereal::make_nvp("root", root),
                  cereal::make_nvp("idx2bb", idx2bb),
                  cereal::make_nvp("idx2data", idx2data),
                  cereal::make_nvp("global_idx", global_idx),
                  cereal::make_nvp("n_at_build", n_at_build));
      }
    }
  }

  PRTree(const py::array_t<T> &idx, const py::array_t<float> &x) {
    const auto &buff_info_idx = idx.request();
    const auto &shape_idx = buff_info_idx.shape;
    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    if (shape_idx[0] != shape_x[0]) {
      throw std::runtime_error(
          "Both index and boudning box must have the same length");
    } else if (shape_x[1] != 2 * D) {
      throw std::runtime_error(
          "Bounding box must have the shape (length, 2 * dim)");
    }

    auto ri = idx.template unchecked<1>();
    auto rx = x.template unchecked<2>();
    size_t length = shape_idx[0];
    idx2bb.reserve(length);

    DataType<T, D> *b, *e;
    void *placement = std::malloc(
        std::max(sizeof(DataType<T, D>), sizeof(DataType<int, D>)) * length);
    b = reinterpret_cast<DataType<T, D> *>(placement);
    e = b + length;

    parallel_for_each(b, e, [&](auto &it) {
      int i = &it - b;
      std::array<Real, D> minima;
      std::array<Real, D> maxima;
      for (int j = 0; j < D; ++j) {
        minima[j] = rx(i, j);
        maxima[j] = rx(i, j + D);
      }
      auto bb = BB<D>(minima, maxima);
      new (b + i) DataType<T, D>{ri(i), std::move(bb)};
    });

    for (size_t i = 0; i < length; i++) {
      std::array<Real, D> minima;
      std::array<Real, D> maxima;
      for (int j = 0; j < D; ++j) {
        minima[j] = rx(i, j);
        maxima[j] = rx(i, j + D);
      }
      auto bb = BB<D>(minima, maxima);
      idx2bb.emplace_hint(idx2bb.end(), ri(i), std::move(bb));
    }
    build(b, e, placement);
    std::free(placement);
  }

  void set_obj(const T &idx, std::optional<std::string> objdumps = std::nullopt){
    if (objdumps){
      auto val = objdumps.value();
      idx2data.emplace(idx, compress(val));
    }
  }

  py::object get_obj(const T &idx){
    py::object obj = py::none();
    auto search = idx2data.find(idx);
    if (likely(search != idx2data.end())){
      auto val = idx2data.at(idx);
      obj = py::cast<py::object>(py::bytes(decompress(val)));
    }
    return obj;
  }

  void insert(const T &idx, const py::array_t<float> &x, const std::optional<std::string> objdumps = std::nullopt) {
    vec<Leaf<T, B, D> *> cands;
    queue<PRTreeNode<T, B, D> *> que;
    BB<D> bb;
    PRTreeNode<T, B, D> *p, *q;

    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    const auto &ndim = buff_info_x.ndim;
    if (shape_x[0] != 2 * D || ndim != 1) {
      throw std::runtime_error("invalid shape.");
    }
    auto it = idx2bb.find(idx);
    if (unlikely(it != idx2bb.end())) {
      throw std::runtime_error("Given index is already included.");
    }
    if (size() > 1.25 * n_at_build){
      rebuild();
    }
    {
      std::array<Real, D> minima;
      std::array<Real, D> maxima;
      for (int i = 0; i < D; ++i) {
        minima[i] = *x.data(i);
        maxima[i] = *x.data(i + D);
      }
      bb = BB<D>(minima, maxima);
    }
    idx2bb.emplace(idx, bb);
    set_obj(idx, objdumps);

    std::array<Real, D> delta;
    for (int i = 0; i < D; ++i) {
      delta[i] = bb.max(i) - bb.min(i) + 0.00000001;
    }
    Real c = 0.0;
    std::stack<PRTreeNode<T, B, D> *> sta;
    while (likely(cands.size() == 0)) {
      while (likely(!sta.empty())) {
        sta.pop();
      }
      std::array<Real, D> d;
      for (int i = 0; i < D; ++i) {
        d[i] = delta[i] * c;
      }
      bb.expand(d);
      c = (c + 1) * 2;
      auto qpush = [&](PRTreeNode<T, B, D> *r) {
        if (unlikely((*r)(bb))) {
          que.emplace(r);
          sta.push(r);
        }
      };

      p = root.get();
      qpush(p);
      while (likely(!que.empty())) {
        p = que.front();
        que.pop();

        if (unlikely(p->leaf)) {
          cands.push_back(p->leaf.get());
        } else {
          if (likely(p->head)) {
            q = p->head.get();
            qpush(q);
            while (likely(q->next)) {
              q = q->next.get();
              qpush(q);
            }
          }
        }
      }
    }

    bb = idx2bb.at(idx);
    auto tg = *select_randomly(cands.begin(), cands.end());
    tg->push(idx, bb);
    while (likely(!sta.empty())) {
      p = sta.top();
      sta.pop();
      if (unlikely(p->leaf)) {
        p->mbb = p->leaf->mbb;
      } else if (likely(p->head)) {
        BB<D> mbb;
        q = p->head.get();
        mbb += q->mbb;
        while (likely(q->next)) {
          q = q->next.get();
          mbb += q->mbb;
        }
        p->mbb = mbb;
      }
    }
  }

  void rebuild(){
    size_t length = idx2bb.size();
    DataType<T, D> *b, *e;
    void *placement = std::malloc(
        std::max(sizeof(DataType<T, D>), sizeof(DataType<int, D>)) * length);
    b = reinterpret_cast<DataType<T, D> *>(placement);
    e = b + length;
    int i = 0;
    for (const auto& [key, value] : idx2bb ) {
      new (b + i) DataType<T, D>{key, value};
      i++;
    }
    build(b, e, placement);
    std::free(placement);
  }

  template <class iterator>
  void build(iterator &b, iterator &e, void *placement) {
    n_at_build = size();
    vec<std::unique_ptr<PRTreeNode<T, B, D>>> prev_nodes;
    std::unique_ptr<PRTreeNode<T, B, D>> p, q, r;

    auto first_tree = PseudoPRTree<T, B, D>(b, e);
    auto first_leaves = first_tree.get_all_leaves(e - b);
    parallel_for_each(first_leaves.begin(), first_leaves.end(), prev_nodes,
                      [&](auto &leaf, auto &o) {
                        auto pp = std::make_unique<PRTreeNode<T, B, D>>(leaf);
                        o.push_back(std::move(pp));
                      });
    auto [bb, ee] = first_tree.as_X(placement, e - b);
    while (likely(prev_nodes.size() > 1)) {
      auto tree = PseudoPRTree<int, B, D>(bb, ee);
      vec<Leaf<int, B, D> *> leaves = tree.get_all_leaves(ee - bb);
      auto leaves_size = leaves.size();

      vec<std::unique_ptr<PRTreeNode<T, B, D>>> tmp_nodes;
      tmp_nodes.reserve(leaves_size);

      parallel_for_each(
          leaves.begin(), leaves.end(), tmp_nodes, [&](auto &leaf, auto &o) {
            int idx, jdx;
            int len = leaf->data.size();
            auto pp = std::make_unique<PRTreeNode<T, B, D>>(leaf->mbb);
            if (likely(!leaf->data.empty())) {
              for (int i = 1; i < len; i++) {
                idx = leaf->data[len - i - 1].first; // reversed way
                jdx = leaf->data[len - i].first;
                prev_nodes[idx]->next = std::move(prev_nodes[jdx]);
              }
              idx = leaf->data[0].first;
              pp->head = std::move(prev_nodes[idx]);
              if (!pp->head) {
                throw std::runtime_error("ppp");
              }
              o.push_back(std::move(pp));
            } else {
              throw std::runtime_error("what????");
            }
          });

      leaves.clear();
      vec<Leaf<int, B, D> *>().swap(leaves);
      prev_nodes.swap(tmp_nodes);
      tmp_nodes.clear();
      vec<std::unique_ptr<PRTreeNode<T, B, D>>>().swap(tmp_nodes);

      {
        auto tmp = tree.as_X(placement, ee - bb);
        bb = std::move(tmp.first);
        ee = std::move(tmp.second);
      }
    }
    if (unlikely(prev_nodes.size() != 1)) {
      throw std::runtime_error("#roots is not 1.");
    }
    root = std::move(prev_nodes[0]);
    prev_nodes.clear();
    vec<std::unique_ptr<PRTreeNode<T, B, D>>>().swap(prev_nodes);
  }

  auto find_all(const py::array_t<float> &x) {
    const auto &buff_info_x = x.request();
    const auto &ndim = buff_info_x.ndim;
    const auto &shape_x = buff_info_x.shape;
    bool is_point = false;
    if (ndim == 1 && (!(shape_x[0] == 2 * D || shape_x[0] == D))) {
      throw std::runtime_error("Invalid Bounding box size");
    } else if (ndim == 2 && (!(shape_x[1] == 2 * D || shape_x[1] == D))) {
      throw std::runtime_error(
          "Bounding box must have the shape (length, 2 * dim)");
    } else if (ndim > 3) {
      throw std::runtime_error("invalid shape");
    }

    if (ndim == 1){
      if (shape_x[0] == D){
        is_point = true; 
      }
    } else {
      if (shape_x[1] == D){
        is_point = true; 
      }
    }
    vec<BB<D>> X;
    X.reserve(ndim == 1 ? 1 : shape_x[0]);
    BB<D> bb;
    if (ndim == 1) {
      {
        std::array<Real, D> minima;
        std::array<Real, D> maxima;
        for (int i = 0; i < D; ++i) {
          minima[i] = *x.data(i);
          if (is_point){
            maxima[i] = minima[i];
          } else {
            maxima[i] = *x.data(i + D);
          }
        }
        bb = BB<D>(minima, maxima);
      }
      X.push_back(std::move(bb));
    } else {
      X.reserve(shape_x[0]);
      for (long int i = 0; i < shape_x[0]; i++) {
        {
          std::array<Real, D> minima;
          std::array<Real, D> maxima;
          for (int j = 0; j < D; ++j) {
            minima[j] = *x.data(i, j);
            if (is_point){
              maxima[j] = minima[j];
            } else {
              maxima[j] = *x.data(i, j + D);
            }
          }
          bb = BB<D>(minima, maxima);
        }
        X.push_back(std::move(bb));
      }
    }
    size_t length = X.size();
    vec<vec<T>> out;
    out.reserve(length);
    parallel_for_each(X.begin(), X.end(), out,
                      [&](const BB<D> &x, auto &o) { o.push_back(find(x)); });
    return out;
  }

  vec<T> find(const BB<D> &target) {
    vec<T> out;
    queue<PRTreeNode<T, B, D> *> que;
    PRTreeNode<T, B, D> *p, *q;
    auto qpush = [&](PRTreeNode<T, B, D> *r) {
      if (unlikely((*r)(target))) {
        que.emplace(r);
      }
    };

    p = root.get();
    qpush(p);
    while (likely(!que.empty())) {
      p = que.front();
      que.pop();

      if (unlikely(p->leaf)) {
        (*p->leaf)(target, out);
      } else {
        if (likely(p->head)) {
          q = p->head.get();
          qpush(q);
          while (likely(q->next)) {
            q = q->next.get();
            qpush(q);
          }
        }
      }
    }
    return out;
  }

  auto find_one(const vec<float> &x) {
    bool is_point = false;
    if (unlikely(!(x.size() == 2 * D || x.size() == D))) {
      throw std::runtime_error("invalid shape");
    }
    std::array<Real, D> minima;
    std::array<Real, D> maxima;
    if (x.size() == D){
      is_point = true;
    }
    for (int i = 0; i < D; ++i) {
      minima[i] = x.at(i);
      if (is_point){
        maxima[i] = minima[i];
      } else {
        maxima[i] = x.at(i + D);
      }
    }
    const auto bb = BB<D>(minima, maxima);
    auto out = find(bb);
    return out;
  }

  void erase(const T idx) {
    auto it = idx2bb.find(idx);
    if (unlikely(it == idx2bb.end())) {
      throw std::runtime_error("Given index is not found.");
    }
    BB<D> target = it->second;
    queue<PRTreeNode<T, B, D> *> que;
    PRTreeNode<T, B, D> *p, *q;
    auto qpush = [&](PRTreeNode<T, B, D> *r) {
      if (unlikely((*r)(target))) {
        que.emplace(r);
      }
    };

    p = root.get();
    qpush(p);
    while (likely(!que.empty())) {
      p = que.front();
      que.pop();

      if (unlikely(p->leaf)) {
        p->leaf->del(idx, target);
      } else {
        if (likely(p->head)) {
          q = p->head.get();
          qpush(q);
          while (likely(q->next)) {
            q = q->next.get();
            qpush(q);
          }
        }
      }
    }
    idx2bb.erase(idx);
    idx2data.erase(idx);
  }
  int64_t size(){
    return static_cast<int64_t>(idx2bb.size());
  }
};
