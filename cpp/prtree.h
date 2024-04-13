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
#include <stack>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include <optional>
#include <functional>

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
#include "small_vector.h"

#ifdef MY_DEBUG
#include <gperftools/profiler.h>
#endif

using Real = float;

namespace py = pybind11;

template <class T>
using vec = std::vector<T>;

template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &seq)
{

  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr = std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p)
                             { std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p)); });
  seq_ptr.release();
  return py::array(size, data, capsule);
}

template <typename T>
auto list_list_to_arrays(vec<vec<T>> out_ll)
{
  vec<T> out_s;
  out_s.reserve(out_ll.size());
  std::size_t sum = 0;
  for (auto &&i : out_ll)
  {
    out_s.push_back(i.size());
    sum += i.size();
  }
  vec<T> out;
  out.reserve(sum);
  for (const auto &v : out_ll)
    out.insert(out.end(), v.begin(), v.end());

  return make_tuple(
      std::move(as_pyarray(out_s)),
      std::move(as_pyarray(out)));
}

template <class T, size_t StaticCapacity>
using svec = itlib::small_vector<T, StaticCapacity>;

template <class T>
using deque = std::deque<T>;

template <class T>
using queue = std::queue<T, deque<T>>;

static const float REBUILD_THRE = 1.25;

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

std::string compress(std::string &data)
{
  std::string output;
  snappy::Compress(data.data(), data.size(), &output);
  return output;
}

std::string decompress(std::string &data)
{
  std::string output;
  snappy::Uncompress(data.data(), data.size(), &output);
  return output;
}

template <int D = 2>
class BB
{
private:
  Real values[2 * D];

public:
  BB() { clear(); }

  BB(const Real (&minima)[D], const Real (&maxima)[D])
  {
    Real v[2 * D];
    for (int i = 0; i < D; ++i)
    {
      v[i] = -minima[i];
      v[i + D] = maxima[i];
    }
    validate(v);
    for (int i = 0; i < D; ++i)
    {
      values[i] = v[i];
      values[i + D] = v[i + D];
    }
  }

  BB(const Real (&v)[2 * D])
  {
    validate(v);
    for (int i = 0; i < D; ++i)
    {
      values[i] = v[i];
      values[i + D] = v[i + D];
    }
  }

  Real min(const int dim) const
  {
    if (unlikely(dim < 0 || D <= dim))
    {
      throw std::runtime_error("Invalid dim");
    }
    return -values[dim];
  }
  Real max(const int dim) const
  {
    if (unlikely(dim < 0 || D <= dim))
    {
      throw std::runtime_error("Invalid dim");
    }
    return values[dim + D];
  }

  bool validate(const Real (&v)[2 * D]) const
  {
    bool flag = false;
    for (int i = 0; i < D; ++i)
    {
      if (unlikely(-v[i] > v[i + D]))
      {
        flag = true;
        break;
      }
    }
    if (unlikely(flag))
    {
      throw std::runtime_error("Invalid Bounding Box");
    }
    return flag;
  }
  void clear()
  {
    for (int i = 0; i < 2 * D; ++i)
    {
      values[i] = -1e100;
    }
  }

  Real val_for_comp(const int &axis) const
  {
    const int axis2 = (axis + 1) % (2 * D);
    return values[axis] + values[axis2];
  }

  BB operator+(const BB &rhs) const
  {
    Real result[2 * D];
    for (int i = 0; i < 2 * D; ++i)
    {
      result[i] = std::max(values[i], rhs.values[i]);
    }
    return BB<D>(result);
  }

  BB operator+=(const BB &rhs)
  {
    for (int i = 0; i < 2 * D; ++i)
    {
      values[i] = std::max(values[i], rhs.values[i]);
    }
    return *this;
  }

  void expand(const Real (&delta)[D])
  {
    for (int i = 0; i < D; ++i)
    {
      values[i] += delta[i];
      values[i + D] += delta[i];
    }
  }

  bool operator()(const BB &target) const
  { // whether this and target has any intersect

    Real minima[D];
    Real maxima[D];
    bool flags[D];
    bool flag = true;

    for (int i = 0; i < D; ++i)
    {
      minima[i] = std::min(values[i], target.values[i]);
      maxima[i] = std::min(values[i + D], target.values[i + D]);
    }
    for (int i = 0; i < D; ++i)
    {
      flags[i] = -minima[i] <= maxima[i];
    }
    for (int i = 0; i < D; ++i)
    {
      flag &= flags[i];
    }
    return flag;
  }

  Real area() const
  {
    Real result = 1;
    for (int i = 0; i < D; ++i)
    {
      result *= max(i) - min(i);
    }
    return result;
  }

  inline Real operator[](const int i) const { return values[i]; }

  template <class Archive>
  void serialize(Archive &ar) { ar(values); }
};

template <class T, int D = 2>
class DataType
{
public:
  BB<D> second;
  T first;

  DataType(){};

  DataType(const T &f, const BB<D> &s)
  {
    first = f;
    second = s;
  }

  DataType(T &&f, BB<D> &&s) noexcept
  {
    first = std::move(f);
    second = std::move(s);
  }

  template <class Archive>
  void serialize(Archive &ar) { ar(first, second); }
};

template <class T, int D = 2>
void clean_data(DataType<T, D> *b, DataType<T, D> *e)
{
  for (DataType<T, D> *it = e - 1; it >= b; --it)
  {
    it->~DataType<T, D>();
  }
}

template <class T, int B = 6, int D = 2>
class Leaf
{
public:
  BB<D> mbb;
  svec<DataType<T, D>, B> data; // You can swap when filtering
  int axis = 0;

  // T is type of keys(ids) which will be returned when you post a query.
  Leaf()
  {
    mbb = BB<D>();
  }
  Leaf(const int _axis)
  {
    axis = _axis;
    mbb = BB<D>();
  }

  void set_axis(const int &_axis) { axis = _axis; }

  void push(const T &key, const BB<D> &target)
  {
    data.emplace_back(key, target);
    update_mbb();
  }

  void update_mbb()
  {
    mbb.clear();
    for (const auto &datum : data)
    {
      mbb += datum.second;
    }
  }

  bool filter(DataType<T, D> &value)
  { // false means given value is ignored
    auto comp = [=](const auto &a, const auto &b) noexcept
    { return a.second.val_for_comp(axis) < b.second.val_for_comp(axis); };

    if (data.size() < B)
    { // if there is room, just push the candidate
      auto iter = std::lower_bound(data.begin(), data.end(), value, comp);
      DataType<T, D> tmp_value = DataType<T, D>(value);
      data.insert(iter, std::move(tmp_value));
      mbb += value.second;
      return true;
    }
    else
    { // if there is no room, check the priority and swap if needed
      if (data[0].second.val_for_comp(axis) < value.second.val_for_comp(axis))
      {
        size_t n_swap = std::lower_bound(data.begin(), data.end(), value, comp) - data.begin();
        std::swap(*data.begin(), value);
        auto iter = data.begin();
        for (size_t i = 0; i < n_swap - 1; ++i)
        {
          std::swap(*(iter + i), *(iter + i + 1));
        }
        update_mbb();
      }
      return false;
    }
  }
};

template <class T, int B = 6, int D = 2>
class PseudoPRTreeNode
{
public:
  Leaf<T, B, D> leaves[2 * D];
  std::unique_ptr<PseudoPRTreeNode> left, right;

  PseudoPRTreeNode()
  {
    for (int i = 0; i < 2 * D; i++)
    {
      leaves[i].set_axis(i);
    }
  }
  PseudoPRTreeNode(const int axis)
  {
    for (int i = 0; i < 2 * D; i++)
    {
      const int j = (axis + i) % (2 * D);
      leaves[i].set_axis(j);
    }
  }

  template <class Archive>
  void serialize(Archive &archive)
  {
    // archive(cereal::(left), cereal::defer(right), leaves);
    archive(left, right, leaves);
  }

  void address_of_leaves(vec<Leaf<T, B, D> *> &out)
  {
    for (auto &leaf : leaves)
    {
      if (leaf.data.size() > 0)
      {
        out.emplace_back(&leaf);
      }
    }
  }

  template <class iterator>
  auto filter(const iterator &b, const iterator &e)
  {
    auto out = std::remove_if(b, e, [&](auto &x)
                              {
      for (auto &l : leaves) {
        if (l.filter(x)) {
          return true;
        }
      }
      return false; });
    return out;
  }
};

template <class T, int B = 6, int D = 2>
class PseudoPRTree
{
public:
  std::unique_ptr<PseudoPRTreeNode<T, B, D>> root;
  vec<Leaf<T, B, D> *> cache_children;
  const int nthreads = std::max(1, (int)std::thread::hardware_concurrency());

  PseudoPRTree() { root = std::make_unique<PseudoPRTreeNode<T, B, D>>(); }

  template <class iterator>
  PseudoPRTree(const iterator &b, const iterator &e)
  {
    if (!root)
    {
      root = std::make_unique<PseudoPRTreeNode<T, B, D>>();
    }
    construct(root.get(), b, e, 0);
    clean_data<T, D>(b, e);
  }

  template <class Archive>
  void serialize(Archive &archive)
  {
    archive(root);
    // archive.serializeDeferments();
  }

  template <class iterator>
  void construct(PseudoPRTreeNode<T, B, D> *node, const iterator &b, const iterator &e,
                 const int depth)
  {
    if (e - b > 0 && node != nullptr)
    {
      bool use_recursive_threads = std::pow(2, depth + 1) <= nthreads;
#ifdef MY_DEBUG
      use_recursive_threads = false;
#endif

      vec<std::thread> threads;
      threads.reserve(2);
      PseudoPRTreeNode<T, B, D> *node_left, *node_right;

      const int axis = depth % (2 * D);
      auto ee = node->filter(b, e);
      auto m = b;
      std::advance(m, (ee - b) / 2);
      std::nth_element(b, m, ee,
                       [axis](const DataType<T, D> &lhs,
                              const DataType<T, D> &rhs) noexcept
                       {
                         return lhs.second[axis] < rhs.second[axis];
                       });

      if (m - b > 0)
      {
        node->left = std::make_unique<PseudoPRTreeNode<T, B, D>>(axis);
        node_left = node->left.get();
        if (use_recursive_threads)
        {
          threads.push_back(
              std::thread([&]()
                          { construct(node_left, b, m, depth + 1); }));
        }
        else
        {
          construct(node_left, b, m, depth + 1);
        }
      }
      if (ee - m > 0)
      {
        node->right = std::make_unique<PseudoPRTreeNode<T, B, D>>(axis);
        node_right = node->right.get();
        if (use_recursive_threads)
        {
          threads.push_back(
              std::thread([&]()
                          { construct(node_right, m, ee, depth + 1); }));
        }
        else
        {
          construct(node_right, m, ee, depth + 1);
        }
      }
      std::for_each(threads.begin(), threads.end(),
                    [&](std::thread &x)
                    { x.join(); });
    }
  }

  auto get_all_leaves(const int hint)
  {
    if (cache_children.empty())
    {
      using U = PseudoPRTreeNode<T, B, D>;
      cache_children.reserve(hint);
      auto node = root.get();
      queue<U *> que;
      que.emplace(node);

      while (!que.empty())
      {
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

  std::pair<DataType<T, D> *, DataType<T, D> *> as_X(void *placement, const int hint)
  {
    DataType<T, D> *b, *e;
    auto children = get_all_leaves(hint);
    T total = children.size();
    b = reinterpret_cast<DataType<T, D> *>(placement);
    e = b + total;
    for (T i = 0; i < total; i++)
    {
      new (b + i) DataType<T, D>{i, children[i]->mbb};
    }
    return {b, e};
  }
};

template <class T, int B = 6, int D = 2>
class PRTreeLeaf
{
public:
  BB<D> mbb;
  svec<DataType<T, D>, B> data;

  PRTreeLeaf()
  {
    mbb = BB<D>();
  }

  PRTreeLeaf(const Leaf<T, B, D> &leaf)
  {
    mbb = leaf.mbb;
    data = leaf.data;
  }

  Real area() const
  {
    return mbb.area();
  }

  void update_mbb()
  {
    mbb.clear();
    for (const auto &datum : data)
    {
      mbb += datum.second;
    }
  }

  void operator()(const BB<D> &target, vec<T> &out) const
  {
    if (mbb(target))
    {
      for (const auto &x : data)
      {
        if (x.second(target))
        {
          out.emplace_back(x.first);
        }
      }
    }
  }

  void del(const T &key, const BB<D> &target)
  {
    if (mbb(target))
    {
      auto remove_it =
          std::remove_if(data.begin(), data.end(), [&](auto &datum)
                         { return datum.second(target) && datum.first == key; });
      data.erase(remove_it, data.end());
    }
  }

  void push(const T &key, const BB<D> &target)
  {
    data.emplace_back(key, target);
    update_mbb();
  }

  template <class Archive>
  void save(Archive &ar) const
  {
    vec<DataType<T, D>> _data;
    for (const auto &datum : data)
    {
      _data.push_back(datum);
    }
    ar(mbb, _data);
  }

  template <class Archive>
  void load(Archive &ar)
  {
    vec<DataType<T, D>> _data;
    ar(mbb, _data);
    for (const auto &datum : _data)
    {
      data.push_back(datum);
    }
  }
};

template <class T, int B = 6, int D = 2>
class PRTreeNode
{
public:
  BB<D> mbb;
  std::unique_ptr<Leaf<T, B, D>> leaf;
  std::unique_ptr<PRTreeNode<T, B, D>> head, next;

  PRTreeNode() {}
  PRTreeNode(const BB<D> &_mbb) { mbb = _mbb; }

  PRTreeNode(BB<D> &&_mbb) noexcept { mbb = std::move(_mbb); }

  PRTreeNode(Leaf<T, B, D> *l)
  {
    leaf = std::make_unique<Leaf<T, B, D>>();
    mbb = l->mbb;
    leaf->mbb = std::move(l->mbb);
    leaf->data = std::move(l->data);
  }

  bool operator()(const BB<D> &target) { return mbb(target); }
};

template <class T, int B = 6, int D = 2>
class PRTreeElement
{
public:
  BB<D> mbb;
  std::unique_ptr<PRTreeLeaf<T, B, D>> leaf;
  bool is_used = false;

  PRTreeElement()
  {
    mbb = BB<D>();
    is_used = false;
  }

  PRTreeElement(const PRTreeNode<T, B, D> &node)
  {
    mbb = BB<D>(node.mbb);
    if (node.leaf)
    {
      Leaf<T, B, D> tmp_leaf = Leaf<T, B, D>(*node.leaf.get());
      leaf = std::make_unique<PRTreeLeaf<T, B, D>>(tmp_leaf);
    }
    is_used = true;
  }

  bool operator()(const BB<D> &target) { return is_used && mbb(target); }

  template <class Archive>
  void serialize(Archive &archive)
  {
    archive(mbb, leaf, is_used);
  }
};

template <class T, int B = 6, int D = 2>
void bfs(const std::function<void(std::unique_ptr<PRTreeLeaf<T, B, D>> &)> &func, vec<PRTreeElement<T, B, D>> &flat_tree, const BB<D> target)
{
  queue<size_t> que;
  auto qpush_if_intersect = [&](const size_t &i)
  {
    PRTreeElement<T, B, D> &r = flat_tree[i];
    // std::cout << "i " << (long int) i << " : " << (bool) r.leaf << std::endl;
    if (r(target))
    {
      // std::cout << " is pushed" << std::endl;
      que.emplace(i);
    }
  };

  // std::cout << "size: " << flat_tree.size() << std::endl;
  qpush_if_intersect(0);
  while (!que.empty())
  {
    size_t idx = que.front();
    // std::cout << "idx: " << (long int) idx << std::endl;
    que.pop();
    PRTreeElement<T, B, D> &elem = flat_tree[idx];

    if (elem.leaf)
    {
      // std::cout << "func called for " << (long int) idx << std::endl;
      func(elem.leaf);
    }
    else
    {
      for (size_t offset = 0; offset < B; offset++)
      {
        size_t jdx = idx * B + offset + 1;
        qpush_if_intersect(jdx);
      }
    }
  }
}

template <class T, int B = 6, int D = 2>
class PRTree
{
private:
  vec<PRTreeElement<T, B, D>> flat_tree;
  std::unordered_map<T, BB<D>> idx2bb;
  std::unordered_map<T, std::string> idx2data;
  int64_t n_at_build = 0;
  std::atomic<T> global_idx = 0;

public:
  template <class Archive>
  void serialize(Archive &archive)
  {
    archive(flat_tree, idx2bb, idx2data, global_idx, n_at_build);
  }

  void save(std::string fname)
  {
    {
      {
        std::ofstream ofs(fname, std::ios::binary);
        cereal::PortableBinaryOutputArchive o_archive(ofs);
        o_archive(cereal::make_nvp("flat_tree", flat_tree),
                  cereal::make_nvp("idx2bb", idx2bb),
                  cereal::make_nvp("idx2data", idx2data),
                  cereal::make_nvp("global_idx", global_idx),
                  cereal::make_nvp("n_at_build", n_at_build));
      }
    }
  }

  void load(std::string fname)
  {
    {
      {
        std::ifstream ifs(fname, std::ios::binary);
        cereal::PortableBinaryInputArchive i_archive(ifs);
        i_archive(cereal::make_nvp("flat_tree", flat_tree),
                  cereal::make_nvp("idx2bb", idx2bb),
                  cereal::make_nvp("idx2data", idx2data),
                  cereal::make_nvp("global_idx", global_idx),
                  cereal::make_nvp("n_at_build", n_at_build));
      }
    }
  }

  PRTree() {}

  PRTree(std::string fname) { load(fname); }

  PRTree(const py::array_t<T> &idx, const py::array_t<float> &x)
  {
    const auto &buff_info_idx = idx.request();
    const auto &shape_idx = buff_info_idx.shape;
    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    if (unlikely(shape_idx[0] != shape_x[0]))
    {
      throw std::runtime_error(
          "Both index and boudning box must have the same length");
    }
    if (unlikely(shape_x[1] != 2 * D))
    {
      throw std::runtime_error(
          "Bounding box must have the shape (length, 2 * dim)");
    }

    auto ri = idx.template unchecked<1>();
    auto rx = x.template unchecked<2>();
    T length = shape_idx[0];
    idx2bb.reserve(length);

    DataType<T, D> *b, *e;
    void *placement = std::malloc(sizeof(DataType<T, D>) * length);
    b = reinterpret_cast<DataType<T, D> *>(placement);
    e = b + length;

    for (T i = 0; i < length; i++)
    {
      Real minima[D];
      Real maxima[D];
      for (int j = 0; j < D; ++j)
      {
        minima[j] = rx(i, j);
        maxima[j] = rx(i, j + D);
      }
      auto bb = BB<D>(minima, maxima);
      auto ri_i = ri(i);
      new (b + i) DataType<T, D>{std::move(ri_i), std::move(bb)};
    }

    for (T i = 0; i < length; i++)
    {
      Real minima[D];
      Real maxima[D];
      for (int j = 0; j < D; ++j)
      {
        minima[j] = rx(i, j);
        maxima[j] = rx(i, j + D);
      }
      auto bb = BB<D>(minima, maxima);
      auto ri_i = ri(i);
      idx2bb.emplace_hint(idx2bb.end(), std::move(ri_i), std::move(bb));
    }
    build(b, e, placement);
    std::free(placement);
  }

  void set_obj(const T &idx, std::optional<std::string> objdumps = std::nullopt)
  {
    if (objdumps)
    {
      auto val = objdumps.value();
      idx2data.emplace(idx, compress(val));
    }
  }

  py::object get_obj(const T &idx)
  {
    py::object obj = py::none();
    auto search = idx2data.find(idx);
    if (likely(search != idx2data.end()))
    {
      auto val = idx2data.at(idx);
      obj = py::cast<py::object>(py::bytes(decompress(val)));
    }
    return obj;
  }

  void insert(const T &idx, const py::array_t<float> &x, const std::optional<std::string> objdumps = std::nullopt)
  {
#ifdef MY_DEBUG
    ProfilerStart("insert.prof");
    std::cout << "profiler start of insert" << std::endl;
#endif
    vec<size_t> cands;
    BB<D> bb;

    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    const auto &ndim = buff_info_x.ndim;
    if (unlikely((shape_x[0] != 2 * D || ndim != 1)))
    {
      throw std::runtime_error("invalid shape.");
    }
    auto it = idx2bb.find(idx);
    if (unlikely(it != idx2bb.end()))
    {
      throw std::runtime_error("Given index is already included.");
    }
    {
      Real minima[D];
      Real maxima[D];
      for (int i = 0; i < D; ++i)
      {
        minima[i] = *x.data(i);
        maxima[i] = *x.data(i + D);
      }
      bb = BB<D>(minima, maxima);
    }
    idx2bb.emplace(idx, bb);
    set_obj(idx, objdumps);

    Real delta[D];
    for (int i = 0; i < D; ++i)
    {
      delta[i] = bb.max(i) - bb.min(i) + 0.00000001;
    }

    // find the leaf node to insert
    Real c = 0.0;
    size_t count = flat_tree.size();
    while (cands.empty())
    {
      Real d[D];
      for (int i = 0; i < D; ++i)
      {
        d[i] = delta[i] * c;
      }
      bb.expand(d);
      c = (c + 1) * 2;

      queue<size_t> que;
      auto qpush_if_intersect = [&](const size_t &i)
      {
        if (flat_tree[i](bb))
        {
          que.emplace(i);
        }
      };

      qpush_if_intersect(0);
      while (!que.empty())
      {
        size_t i = que.front();
        que.pop();
        PRTreeElement<T, B, D> &elem = flat_tree[i];

        if (elem.leaf && elem.leaf->mbb(bb))
        {
          cands.push_back(i);
        }
        else
        {
          for (size_t offset = 0; offset < B; offset++)
          {
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
    if (cands.size() == 1)
    {
      min_leaf = cands[0];
    }
    else
    {
      Real min_diff_area = 1e100;
      for (const auto &i : cands)
      {
        PRTreeLeaf<T, B, D> *leaf = flat_tree[i].leaf.get();
        PRTreeLeaf<T, B, D> tmp_leaf = PRTreeLeaf<T, B, D>(*leaf);
        Real diff_area = -tmp_leaf.area();
        tmp_leaf.push(idx, bb);
        diff_area += tmp_leaf.area();
        if (diff_area < min_diff_area)
        {
          min_diff_area = diff_area;
          min_leaf = i;
        }
      }
    }
    flat_tree[min_leaf].leaf->push(idx, bb);
    // update mbbs of all cands and their parents
    size_t i = min_leaf;
    while (true)
    {
      PRTreeElement<T, B, D> &elem = flat_tree[i];

      if (elem.leaf)
        elem.mbb += elem.leaf->mbb;

      if (i > 0)
      {
        size_t j = (i - 1) / B;
        flat_tree[j].mbb += flat_tree[i].mbb;
      }
      if (i == 0)
        break;
      i = (i - 1) / B;
    }

    if (size() > REBUILD_THRE * n_at_build)
    {
      rebuild();
    }
#ifdef MY_DEBUG
    ProfilerStop();
    std::cout << "profiler end of insert" << std::endl;
#endif
  }

  void rebuild()
  {
    std::stack<size_t> sta;
    T length = idx2bb.size();
    DataType<T, D> *b, *e;

    void *placement = std::malloc(sizeof(DataType<T, D>) * length);
    b = reinterpret_cast<DataType<T, D> *>(placement);
    e = b + length;

    T i = 0;
    sta.push(0);
    while (!sta.empty())
    {
      size_t idx = sta.top();
      sta.pop();

      PRTreeElement<T, B, D> &elem = flat_tree[idx];

      if (elem.leaf)
      {
        for (const auto &datum : elem.leaf->data)
        {
          new (b + i) DataType<T, D>{datum.first, datum.second};
          i++;
        }
      }
      else
      {
        for (size_t offset = 0; offset < B; offset++)
        {
          size_t jdx = idx * B + offset + 1;
          if (likely(flat_tree[jdx].is_used))
          {
            sta.push(jdx);
          }
        }
      }
    }

    build(b, e, placement);
    std::free(placement);
  }

  template <class iterator>
  void build(const iterator &b, const iterator &e, void *placement)
  {
#ifdef MY_DEBUG
    ProfilerStart("build.prof");
    std::cout << "profiler start of build" << std::endl;
#endif
    std::unique_ptr<PRTreeNode<T, B, D>> root;
    {
      n_at_build = size();
      vec<std::unique_ptr<PRTreeNode<T, B, D>>> prev_nodes;
      std::unique_ptr<PRTreeNode<T, B, D>> p, q, r;

      auto first_tree = PseudoPRTree<T, B, D>(b, e);
      auto first_leaves = first_tree.get_all_leaves(e - b);
      for (auto &leaf : first_leaves)
      {
        auto pp = std::make_unique<PRTreeNode<T, B, D>>(leaf);
        prev_nodes.push_back(std::move(pp));
      }
      auto [bb, ee] = first_tree.as_X(placement, e - b);
      while (prev_nodes.size() > 1)
      {
        auto tree = PseudoPRTree<T, B, D>(bb, ee);
        auto leaves = tree.get_all_leaves(ee - bb);
        auto leaves_size = leaves.size();

        vec<std::unique_ptr<PRTreeNode<T, B, D>>> tmp_nodes;
        tmp_nodes.reserve(leaves_size);

        for (auto &leaf : leaves)
        {
          int idx, jdx;
          int len = leaf->data.size();
          auto pp = std::make_unique<PRTreeNode<T, B, D>>(leaf->mbb);
          if (likely(!leaf->data.empty()))
          {
            for (int i = 1; i < len; i++)
            {
              idx = leaf->data[len - i - 1].first; // reversed way
              jdx = leaf->data[len - i].first;
              prev_nodes[idx]->next = std::move(prev_nodes[jdx]);
            }
            idx = leaf->data[0].first;
            pp->head = std::move(prev_nodes[idx]);
            if (unlikely(!pp->head))
            {
              throw std::runtime_error("ppp");
            }
            tmp_nodes.push_back(std::move(pp));
          }
          else
          {
            throw std::runtime_error("what????");
          }
        }

        prev_nodes.swap(tmp_nodes);
        if (prev_nodes.size() > 1)
        {
          auto tmp = tree.as_X(placement, ee - bb);
          bb = std::move(tmp.first);
          ee = std::move(tmp.second);
        }
      }
      if (unlikely(prev_nodes.size() != 1))
      {
        throw std::runtime_error("#roots is not 1.");
      }
      root = std::move(prev_nodes[0]);
    }
    // flatten built tree
    {
      queue<std::pair<PRTreeNode<T, B, D> *, size_t>> que;
      PRTreeNode<T, B, D> *p, *q;

      int depth = 0;

      p = root.get();
      while (p->head)
      {
        p = p->head.get();
        depth++;
      }

      // resize
      {
        flat_tree.clear();
        flat_tree.shrink_to_fit();
        size_t count = 0;
        for (int i = 0; i <= depth; i++)
        {
          count += std::pow(B, depth);
        }
        flat_tree.resize(count);
      }

      // assign
      que.emplace(root.get(), 0);
      while (!que.empty())
      {
        auto tmp = que.front();
        que.pop();
        p = tmp.first;
        size_t idx = tmp.second;

        flat_tree[idx] = PRTreeElement(*p);
        size_t child_idx = 0;
        if (p->head)
        {
          size_t jdx = idx * B + child_idx + 1;
          ++child_idx;

          q = p->head.get();
          que.emplace(q, jdx);
          while (q->next)
          {
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

  auto find_all(const py::array_t<float> &x)
  {
#ifdef MY_DEBUG
    ProfilerStart("find_all.prof");
    std::cout << "profiler start of find_all" << std::endl;
#endif
    const auto &buff_info_x = x.request();
    const auto &ndim = buff_info_x.ndim;
    const auto &shape_x = buff_info_x.shape;
    bool is_point = false;
    if (unlikely(ndim == 1 && (!(shape_x[0] == 2 * D || shape_x[0] == D))))
    {
      throw std::runtime_error("Invalid Bounding box size");
    }
    if (unlikely((ndim == 2 && (!(shape_x[1] == 2 * D || shape_x[1] == D)))))
    {
      throw std::runtime_error(
          "Bounding box must have the shape (length, 2 * dim)");
    }
    if (unlikely(ndim > 3))
    {
      throw std::runtime_error("invalid shape");
    }

    if (ndim == 1)
    {
      if (shape_x[0] == D)
      {
        is_point = true;
      }
    }
    else
    {
      if (shape_x[1] == D)
      {
        is_point = true;
      }
    }
    vec<BB<D>> X;
    X.reserve(ndim == 1 ? 1 : shape_x[0]);
    BB<D> bb;
    if (ndim == 1)
    {
      {
        Real minima[D];
        Real maxima[D];
        for (int i = 0; i < D; ++i)
        {
          minima[i] = *x.data(i);
          if (is_point)
          {
            maxima[i] = minima[i];
          }
          else
          {
            maxima[i] = *x.data(i + D);
          }
        }
        bb = BB<D>(minima, maxima);
      }
      X.push_back(std::move(bb));
    }
    else
    {
      X.reserve(shape_x[0]);
      for (long int i = 0; i < shape_x[0]; i++)
      {
        {
          Real minima[D];
          Real maxima[D];
          for (int j = 0; j < D; ++j)
          {
            minima[j] = *x.data(i, j);
            if (is_point)
            {
              maxima[j] = minima[j];
            }
            else
            {
              maxima[j] = *x.data(i, j + D);
            }
          }
          bb = BB<D>(minima, maxima);
        }
        X.push_back(std::move(bb));
      }
    }
    vec<vec<T>> out;
    out.reserve(X.size());
#ifdef MY_DEBUG
    std::for_each(X.begin(), X.end(),
                  [&](const BB<D> &x)
                  { out.push_back(find(x)); });
#else
    parallel_for_each(X.begin(), X.end(), out,
                      [&](const BB<D> &x, auto &o)
                      { o.push_back(find(x)); });
#endif
#ifdef MY_DEBUG
    ProfilerStop();
    std::cout << "profiler end of find_all" << std::endl;
#endif
    return out;
  }

  auto find_all_array(const py::array_t<float> &x)
  {
    return list_list_to_arrays(std::move(find_all(x)));
  }

  auto find_one(const vec<float> &x)
  {
    bool is_point = false;
    if (unlikely(!(x.size() == 2 * D || x.size() == D)))
    {
      throw std::runtime_error("invalid shape");
    }
    Real minima[D];
    Real maxima[D];
    if (x.size() == D)
    {
      is_point = true;
    }
    for (int i = 0; i < D; ++i)
    {
      minima[i] = x.at(i);
      if (is_point)
      {
        maxima[i] = minima[i];
      }
      else
      {
        maxima[i] = x.at(i + D);
      }
    }
    const auto bb = BB<D>(minima, maxima);
    auto out = find(bb);
    return out;
  }

  vec<T> find(const BB<D> &target)
  {
    vec<T> out;
    auto find_func = [&](std::unique_ptr<PRTreeLeaf<T, B, D>> &leaf)
    {
      (*leaf)(target, out);
    };

    bfs<T, B, D>(std::move(find_func), flat_tree, target);
    std::sort(out.begin(), out.end());
    return out;
  }

  void erase(const T idx)
  {
    auto it = idx2bb.find(idx);
    if (unlikely(it == idx2bb.end()))
    {
      throw std::runtime_error("Given index is not found.");
    }
    BB<D> target = it->second;

    auto erase_func = [&](std::unique_ptr<PRTreeLeaf<T, B, D>> &leaf)
    {
      leaf->del(idx, target);
    };

    bfs<T, B, D>(std::move(erase_func), flat_tree, target);

    idx2bb.erase(idx);
    idx2data.erase(idx);
    if (unlikely(REBUILD_THRE * size() < n_at_build))
    {
      rebuild();
    }
  }

  int64_t size()
  {
    return static_cast<int64_t>(idx2bb.size());
  }
};
