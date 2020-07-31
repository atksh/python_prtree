#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <unordered_map>
#include <queue>
#include <stack>
#include <algorithm>
#include <numeric>
#include <memory>
#include <iterator>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <future>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp> //for smart pointers
#include <cereal/types/string.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>

#include "xsimd/xsimd.hpp"
#include "parallel.h"
#include "function_ref.h"

namespace xs = xsimd;

using Real = float;

namespace py = pybind11;
using std::swap;

template <class T, class U>
using pair = std::pair<T, U>;

template <class T>
using vec = std::vector<T>;

template <class T>
using deque = std::deque<T>;

template <class T>
using queue = std::queue<T, deque<T>>;

static std::mt19937 rand_src(42);

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

template <typename Iter>
inline Iter select_randomly(Iter start, Iter end)
{
  std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
  std::advance(start, dis(rand_src));
  return start;
};

static const auto zero_values = xs::batch<Real, 4>(-1e100, -1e100, -1e100, -1e100);
class BB
{
private:
  xs::batch<Real, 4> values;

public:
  BB()
  {
    clear();
  }

  BB(const Real &_xmin, const Real &_xmax, const Real &_ymin, const Real &_ymax)
  {
    if (unlikely(_xmin > _xmax || _ymin > _ymax))
    {
      throw std::runtime_error("Impossible rectange was given. xmin < xmax and ymin < ymax must be satisfied.");
    }
    values = xs::batch<Real, 4>(-_xmin, _xmax, -_ymin, _ymax);
  }

  BB(const xs::batch<Real, 4> &v)
  {
    if (unlikely(-v[0] > v[1] || -v[2] > v[3]))
    {
      throw std::runtime_error("Impossible rectange was given. xmin < xmax and ymin < ymax must be satisfied.");
    }
    values = v;
  }

  inline Real xmin() const { return -values[0]; }
  inline Real xmax() const { return values[1]; }
  inline Real ymin() const { return -values[2]; }
  inline Real ymax() const { return values[3]; }

  inline void clear()
  {
    values += zero_values;
  }

  inline BB operator+(const BB &rhs) const
  {
    return BB(xs::fmax(values, rhs.values));
  }

  inline BB operator+=(const BB &rhs)
  {
    values = xs::fmax(values, rhs.values);
    return *this;
  }

  inline void expand(const Real &dx, const Real &dy)
  {
    auto d = xs::batch<Real, 4>(dx, dx, dy, dy);
    values += d;
  }

  inline bool operator()(const BB &target) const
  { // whether this and target has any intersect
    const auto m = xs::fmin(values, target.values);
    //const bool c1 = unlikely(std::max(xmin(), target.xmin()) <= std::min(xmax(), target.xmax()));
    //const bool c2 = unlikely(std::max(ymin(), target.ymin()) <= std::min(ymax(), target.ymax()));
    //return unlikely(c1 && c2);
    return (-m[0] <= m[1]) && (-m[2] <= m[3]);
  }

  inline Real operator[](const int &i) const
  {
    return values[i];
  }
  template <class Archive>
  void serialize(Archive &ar)
  {
    ar(values[0], values[1], values[2], values[3]);
  }
};

template <class T>
class DataType
{
public:
  T first;
  BB second;
  DataType(){};

  DataType(const T &f, const BB &s)
  {
    first = f;
    second = s;
  }

  DataType(T &&f, BB &&s)
  {
    first = std::move(f);
    second = std::move(s);
  }

  template <class Archive>
  void serialize(Archive &ar)
  {
    ar(first, second);
  }
};

template <class T, int B = 6>
class Leaf
{
public:
  int axis = 0;
  BB mbb;
  vec<DataType<T>> data; // You can swap when filtering
  // T is type of keys(ids) which will be returned when you post a query.
  Leaf()
  {
    mbb = BB();
    data.reserve(B);
  }
  Leaf(int &_axis)
  {
    axis = _axis;
    mbb = BB();
    data.reserve(B);
  }

  ~Leaf()
  {
    data.clear();
    vec<DataType<T>>().swap(data);
  }

  template <class Archive>
  void serialize(Archive &ar)
  {
    ar(axis, mbb, data);
  }

  inline void set_axis(const int &_axis)
  {
    axis = _axis;
  }

  inline void push(const T &key, const BB &target)
  {
    data.emplace_back(key, target);
    update_mbb();
  }

  inline void update_mbb()
  {
    mbb.clear();
    for (const auto &datum : data)
    {
      mbb += datum.second;
    }
  }

  template <typename F>
  inline bool filter(DataType<T> &value, const F &comp)
  { // false means given value is ignored
    if (unlikely(data.size() < B))
    { // if there is room, just push the candidate
      mbb += value.second;
      data.push_back(std::move(value));
      return true;
    }
    else
    { // if there is no room, check the priority and swap if needed
      auto iter = upper_bound(data.begin(), data.end(), value, comp);
      if (unlikely(iter != data.end()))
      {
        swap(*iter, value);
        update_mbb();
      }
      return false;
    }
  }

  void operator()(const BB &target, vec<T> &out) const
  {
    if (unlikely(mbb(target)))
    {
      for (const auto &x : data)
      {
        if (unlikely(x.second(target)))
        {
          out.emplace_back(x.first);
        }
      }
    }
  }

  void del(const T &key, const BB &target)
  {
    if (unlikely(mbb(target)))
    {
      auto remove_it = std::remove_if(data.begin(), data.end(), [&](auto &datum) {
        return datum.second(target) && datum.first == key;
      });
      data.erase(remove_it, data.end());
    }
  }
};

template <class T, int B = 6>
class PseudoPRTreeNode
{
public:
  std::array<Leaf<T, B>, 4> leaves;
  std::unique_ptr<PseudoPRTreeNode> left, right;

  PseudoPRTreeNode()
  {
    for (int i = 0; i < 4; i++)
    {
      leaves[i].set_axis(i);
    }
  }

  template <class Archive>
  void serialize(Archive &archive)
  {
    //archive(cereal::(left), cereal::defer(right), leaves);
    archive(left, right, leaves);
  }

  inline void address_of_leaves(vec<Leaf<T, B> *> &out)
  {
    for (auto &leaf : leaves)
    {
      if (likely(leaf.data.size() > 0))
      {
        out.emplace_back(&leaf);
      }
    }
  }

  template <class iterator>
  inline auto filter(iterator &b, iterator &e)
  {
    vec<tl::function_ref<bool(const DataType<T> &, const DataType<T> &)>> comps;
    for (int axis = 0; axis < 4; axis++)
    {
      comps.emplace_back(
          [axis](const DataType<T> &lhs, const DataType<T> &rhs) noexcept {
            return lhs.second[axis] < rhs.second[axis];
          });
    }

    auto out = std::remove_if(b, e, [&](auto &x) {
      for (auto &l : leaves)
      {
        if (unlikely(l.filter(x, comps[l.axis])))
        {
          return true;
        }
      }
      return false;
    });
    return out;
  }
};

template <class T, int B = 6>
class PseudoPRTree
{
public:
  std::unique_ptr<PseudoPRTreeNode<T, B>> root;
  vec<Leaf<T, B> *> cache_children;
  const int nthreads = std::max(1, (int)std::thread::hardware_concurrency());

  PseudoPRTree()
  {
    root = std::make_unique<PseudoPRTreeNode<T, B>>();
  }

  ~PseudoPRTree()
  {
    cache_children.clear();
    vec<Leaf<T, B> *>().swap(cache_children);
  }

  template <class iterator>
  PseudoPRTree(iterator &b, iterator &e)
  {
    if (likely(!root))
    {
      root = std::make_unique<PseudoPRTreeNode<T, B>>();
    }
    construct(root.get(), b, e, 0);

    for (DataType<T> *it = e - 1; it >= b; --it)
    {
      it->~DataType<T>();
    }
    b = nullptr;
    e = nullptr;
  }

  template <class Archive>
  void serialize(Archive &archive)
  {
    archive(root);
    //archive.serializeDeferments();
  }

  template <class iterator>
  inline void construct(PseudoPRTreeNode<T, B> *node, iterator &b, iterator &e, const size_t depth)
  {
    if (e - b > 0 && node != nullptr)
    {
      bool use_recursive_threads = std::pow(2, depth + 1) <= nthreads;

      vec<std::thread> threads;
      threads.reserve(2);
      PseudoPRTreeNode<T, B> *node_left, *node_right;

      const int axis = depth % 4;
      auto ee = node->filter(b, e);
      auto m = b;
      std::advance(m, (ee - b) / 2);
      auto mm = m;
      std::nth_element(b, m, ee, [axis](const DataType<T> &lhs, const DataType<T> &rhs) noexcept {
        return lhs.second[axis] < rhs.second[axis];
      });

      if (m - b > 0)
      {
        node->left = std::make_unique<PseudoPRTreeNode<T, B>>();
        node_left = node->left.get();
        if (use_recursive_threads)
        {
          threads.push_back(std::thread([&]() { construct(node_left, b, m, depth + 1); }));
        }
        else
        {
          construct(node_left, b, m, depth + 1);
        }
      }
      if (ee - mm > 0)
      {
        node->right = std::make_unique<PseudoPRTreeNode<T, B>>();
        node_right = node->right.get();
        if (use_recursive_threads)
        {
          threads.push_back(std::thread([&]() { construct(node_right, mm, ee, depth + 1); }));
        }
        else
        {
          construct(node_right, mm, ee, depth + 1);
        }
      }
      std::for_each(threads.begin(), threads.end(), [&](std::thread &x) { x.join(); });
      threads.clear();
      vec<std::thread>().swap(threads);
    }
  }

  inline auto get_all_leaves(const int hint)
  {
    if (cache_children.empty())
    {
      using U = PseudoPRTreeNode<T, B>;
      cache_children.reserve(hint);
      auto node = root.get();
      queue<U *> que;
      que.emplace(node);

      while (likely(!que.empty()))
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

  std::pair<DataType<int> *, DataType<int> *> as_X(void *placement, const int hint)
  {
    DataType<int> *b, *e;
    auto children = get_all_leaves(hint);
    int total = children.size();
    b = (DataType<int> *)placement;
    e = b + total;
    parallel_for_each(b, e, [&](auto &p) {
      int i = &p - b;
      new (b + i) DataType<int>{i, children[i]->mbb};
    });
    return {b, e};
  }
};

template <class T, int B = 6>
class PRTreeNode
{
public:
  BB mbb;
  std::unique_ptr<Leaf<T, B>> leaf;
  std::unique_ptr<PRTreeNode<T, B>> head, next;

  PRTreeNode() {}
  PRTreeNode(const BB &_mbb)
  {
    mbb = _mbb;
  }

  PRTreeNode(BB &&_mbb)
  {
    mbb = std::move(_mbb);
  }

  PRTreeNode(Leaf<T, B> *l)
  {
    leaf = std::make_unique<Leaf<T, B>>();
    mbb = l->mbb;
    leaf->mbb = std::move(l->mbb);
    leaf->data = std::move(l->data);
  }

  inline bool operator()(const BB &target)
  {
    return mbb(target);
  }

  template <class Archive>
  void serialize(Archive &archive)
  {
    //archive(cereal::defer(leaf), cereal::defer(head), cereal::defer(next));
    archive(mbb, leaf, head, next);
  }
};

template <class T, int B = 6>
class PRTree
{
private:
  std::unique_ptr<PRTreeNode<T, B>> root;
  std::unordered_map<T, BB> umap;

public:
  template <class Archive>
  void serialize(Archive &archive)
  {
    archive(root, umap);
    //archive.serializeDeferments();
  }

  void save(std::string fname)
  {
    {
      {
        std::ofstream ofs(fname, std::ios::binary);
        cereal::PortableBinaryOutputArchive o_archive(ofs);
        //cereal::JSONOutputArchive o_archive(ofs);
        o_archive(cereal::make_nvp("root", root), cereal::make_nvp("umap", umap));
      }
    }
  }

  PRTree() {}

  ~PRTree()
  {
    umap.clear();
    std::unordered_map<T, BB>().swap(umap);
  }

  PRTree(std::string fname)
  {
    load(fname);
  }

  void load(std::string fname)
  {
    {
      {
        std::ifstream ifs(fname, std::ios::binary);
        cereal::PortableBinaryInputArchive i_archive(ifs);
        //cereal::JSONInputArchive i_archive(ifs);
        i_archive(cereal::make_nvp("root", root), cereal::make_nvp("umap", umap));
      }
    }
  }

  PRTree(const py::array_t<T> &idx, const py::array_t<float> &x)
  {
    const auto &buff_info_idx = idx.request();
    const auto &shape_idx = buff_info_idx.shape;
    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    if (shape_idx[0] != shape_x[0])
    {
      throw std::runtime_error("Both index and boudning box must have the same length");
    }
    else if (shape_x[1] != 4)
    {
      throw std::runtime_error("Bounding box must have the shape (length, 4)");
    }

    auto ri = idx.template unchecked<1>();
    auto rx = x.template unchecked<2>();
    size_t length = shape_idx[0];
    umap.reserve(length);

    DataType<T> *b, *e;
    void *placement = std::malloc(std::max(sizeof(DataType<T>), sizeof(DataType<int>)) * length);
    b = reinterpret_cast<DataType<T> *>(placement);
    e = b + length;

    parallel_for_each(b, e,
                      [&](auto &it) {
                        int i = &it - b;
                        auto bb = BB(rx(i, 0), rx(i, 1), rx(i, 2), rx(i, 3));
                        new (b + i) DataType<T>{ri(i), std::move(bb)};
                      });

    auto build_umap = [&] {for (size_t i = 0; i < length; i++){
        auto bb = BB(rx(i, 0), rx(i, 1), rx(i, 2), rx(i, 3));
        umap.emplace_hint(umap.end(), ri(i), std::move(bb));
      } };

    //ProfilerStart("construct.prof");
    auto t2 = std::thread([&] { build(b, e, placement); });
    auto t1 = std::thread(std::move(build_umap));
    t1.join();
    t2.join();
    //ProfilerStop();
    std::free(placement);
  }

  void insert(const T &idx, const py::array_t<float> &x)
  {
    vec<Leaf<T, B> *> cands;
    queue<PRTreeNode<T, B> *> que;
    BB bb;
    PRTreeNode<T, B> *p, *q;

    const auto &buff_info_x = x.request();
    const auto &shape_x = buff_info_x.shape;
    const auto &ndim = buff_info_x.ndim;
    if (shape_x[0] != 4 || ndim != 1)
    {
      throw std::runtime_error("invalid shape.");
    }
    auto it = umap.find(idx);
    if (unlikely(it != umap.end()))
    {
      throw std::runtime_error("Given index is already included.");
    }

    bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
    Real dx = bb.xmax() - bb.xmin() + 0.000000001;
    Real dy = bb.ymax() - bb.ymin() + 0.000000001;
    Real c = 0.0;
    std::stack<PRTreeNode<T, B> *> sta;
    while (likely(cands.size() == 0))
    {
      while (likely(!sta.empty()))
      {
        sta.pop();
      }
      bb.expand(dx * c, dy * c);
      c = (c + 1) * 2;
      auto qpush = [&](PRTreeNode<T, B> *r) {
        if (unlikely((*r)(bb)))
        {
          que.emplace(r);
          sta.push(r);
        }
      };

      p = root.get();
      qpush(p);
      while (likely(!que.empty()))
      {
        p = que.front();
        que.pop();

        if (unlikely(p->leaf))
        {
          cands.push_back(p->leaf.get());
        }
        else
        {
          if (likely(p->head))
          {
            q = p->head.get();
            qpush(q);
            while (likely(q->next))
            {
              q = q->next.get();
              qpush(q);
            }
          }
        }
      }
    }
    auto tg = *select_randomly(cands.begin(), cands.end());
    bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
    tg->push(idx, bb);
    umap[idx] = bb;
    while (likely(!sta.empty()))
    {
      p = sta.top();
      sta.pop();
      if (unlikely(p->leaf))
      {
        p->mbb = p->leaf->mbb;
      }
      else if (likely(p->head))
      {
        BB mbb;
        q = p->head.get();
        mbb += q->mbb;
        while (likely(q->next))
        {
          q = q->next.get();
          mbb += q->mbb;
        }
        p->mbb = mbb;
      }
    }
  }

  template <class iterator>
  void build(iterator &b, iterator &e, void *placement)
  {
    vec<std::unique_ptr<PRTreeNode<T, B>>> prev_nodes;
    std::unique_ptr<PRTreeNode<T, B>> p, q, r;

    auto first_tree = PseudoPRTree<T, B>(b, e);
    auto first_leaves = first_tree.get_all_leaves(e - b);
    parallel_for_each(first_leaves.begin(), first_leaves.end(), prev_nodes,
                      [&](auto &leaf, auto &o) {
                        auto pp = std::make_unique<PRTreeNode<T, B>>(leaf);
                        o.push_back(std::move(pp));
                      });

    auto [bb, ee] = first_tree.as_X(placement, e - b);
    while (likely(prev_nodes.size() > 1))
    {
      auto tree = PseudoPRTree<int, B>(bb, ee);
      vec<Leaf<int, B> *> leaves = tree.get_all_leaves(ee - bb);
      auto leaves_size = leaves.size();

      vec<std::unique_ptr<PRTreeNode<T, B>>> tmp_nodes;
      tmp_nodes.reserve(leaves_size);

      parallel_for_each(leaves.begin(), leaves.end(), tmp_nodes,
                        [&](auto &leaf, auto &o) {
                          int idx, jdx;
                          int len = leaf->data.size();
                          auto pp = std::make_unique<PRTreeNode<T, B>>(leaf->mbb);
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
                            if (!pp->head)
                            {
                              throw std::runtime_error("ppp");
                            }
                            o.push_back(std::move(pp));
                          }
                          else
                          {
                            throw std::runtime_error("what????");
                          }
                        });

      /*size_t c = 0;
        for (const auto& n : prev_nodes){
          if (unlikely(n)){
            c++;
          }
        }
        if (unlikely(c > 0)){
          throw std::runtime_error("eee");
        }*/
      leaves.clear();
      vec<Leaf<int, B> *>().swap(leaves);
      prev_nodes.swap(tmp_nodes);
      tmp_nodes.clear();
      vec<std::unique_ptr<PRTreeNode<T, B>>>().swap(tmp_nodes);

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
    prev_nodes.clear();
    vec<std::unique_ptr<PRTreeNode<T, B>>>().swap(prev_nodes);
  }

  auto find_all(const py::array_t<float> &x)
  {
    const auto &buff_info_x = x.request();
    const auto &ndim = buff_info_x.ndim;
    const auto &shape_x = buff_info_x.shape;
    if (ndim == 1 && shape_x[0] != 4)
    {
      throw std::runtime_error("Bounding box must have the shape 4 with (xmin, xmax, ymin, ymax)");
    }
    else if (ndim == 2 && shape_x[1] != 4)
    {
      throw std::runtime_error("Bounding box must have the shape (length, 4)");
    }
    else if (ndim > 3)
    {
      throw std::runtime_error("invalid shape");
    }
    vec<BB> X;
    X.reserve(ndim == 1 ? 1 : shape_x[0]);
    BB bb;
    if (ndim == 1)
    {
      bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
      X.push_back(std::move(bb));
    }
    else
    {
      X.reserve(shape_x[0]);
      for (long int i = 0; i < shape_x[0]; i++)
      {
        bb = BB(*x.data(i, 0), *x.data(i, 1), *x.data(i, 2), *x.data(i, 3));
        X.push_back(std::move(bb));
      }
    }
    size_t length = X.size();
    vec<vec<T>> out;
    out.reserve(length);
    parallel_for_each(X.begin(), X.end(), out, [&](const BB &x, auto &o) { o.push_back(find(x)); });
    return out;
  }

  vec<T> find_one(const py::array_t<float> &x)
  {
    const auto &buff_info_x = x.request();
    const auto &ndim = buff_info_x.ndim;
    const auto &shape_x = buff_info_x.shape;
    if (unlikely(ndim != 1 || shape_x[0] != 4))
    {
      throw std::runtime_error("invalid shape");
    }
    const BB bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
    return find(bb);
  }

  vec<T> find(const BB &target)
  {
    vec<T> out;
    queue<PRTreeNode<T, B> *> que;
    PRTreeNode<T, B> *p, *q;
    auto qpush = [&](PRTreeNode<T, B> *r) {
      if (unlikely((*r)(target)))
      {
        que.emplace(r);
      }
    };

    p = root.get();
    qpush(p);
    while (likely(!que.empty()))
    {
      p = que.front();
      que.pop();

      if (unlikely(p->leaf))
      {
        (*p->leaf)(target, out);
      }
      else
      {
        if (likely(p->head))
        {
          q = p->head.get();
          qpush(q);
          while (likely(q->next))
          {
            q = q->next.get();
            qpush(q);
          }
        }
      }
    }
    return out;
  }

  void erase(const T idx)
  {
    auto it = umap.find(idx);
    if (unlikely(it == umap.end()))
    {
      throw std::runtime_error("Given index is not found.");
    }
    BB target = it->second;
    queue<PRTreeNode<T, B> *> que;
    PRTreeNode<T, B> *p, *q;
    auto qpush = [&](PRTreeNode<T, B> *r) {
      if (unlikely((*r)(target)))
      {
        que.emplace(r);
      }
    };

    p = root.get();
    qpush(p);
    while (likely(!que.empty()))
    {
      p = que.front();
      que.pop();

      if (unlikely(p->leaf))
      {
        p->leaf->del(idx, target);
      }
      else
      {
        if (likely(p->head))
        {
          q = p->head.get();
          qpush(q);
          while (likely(q->next))
          {
            q = q->next.get();
            qpush(q);
          }
        }
      }
    }
    umap.erase(idx);
  }
};
