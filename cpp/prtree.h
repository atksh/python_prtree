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
#include <cereal/types/memory.hpp>  //for smart pointers
#include <cereal/types/string.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>



namespace py = pybind11;
using std::swap;

template<class T, class U>
using pair = std::pair<T, U>;

template<class T>
using vec = std::vector<T>;

static std::mt19937 rand_src(42);


#if defined(__GNUC__) || defined(__clang__)
#define likely(x)       __builtin_expect(!!(x),1)
#define unlikely(x)     __builtin_expect(!!(x),0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif


template<typename Iter>
inline Iter select_randomly(Iter start, Iter end) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(rand_src));
    return start;
};

class Uncopyable {
  protected:
    Uncopyable() = default;
    ~Uncopyable() = default;
    Uncopyable(const Uncopyable&) = delete;
    Uncopyable& operator=(const Uncopyable&) = delete;
};

class BB{
  public:
    double xmin, xmax, ymin, ymax;
    BB() {
      clear();
    }

    BB(const double& _xmin, const double& _xmax, const double& _ymin, const double& _ymax){
      if (unlikely(_xmin > _xmax || _ymin > _ymax)){
        throw std::runtime_error("Impossible rectange was given. xmin < xmax and ymin < ymax must be satisfied.");
      }
      xmin = _xmin;
      xmax = _xmax;
      ymin = _ymin;
      ymax = _ymax;
    }

    inline void clear(){
      xmin = 1e100;
      xmax = -1e100;
      ymin = 1e100;
      ymax= -1e100;
    }

    inline BB operator +(const BB& rhs) const{
      return BB(std::min(xmin, rhs.xmin), std::max(xmax, rhs.xmax), std::min(ymin, rhs.ymin), std::max(ymax, rhs.ymax));
    }

    inline BB operator +=(const BB& rhs){
      xmin = std::min(xmin, rhs.xmin);
      xmax = std::max(xmax, rhs.xmax);
      ymin = std::min(ymin, rhs.ymin);
      ymax = std::max(ymax, rhs.ymax);
      return *this;
    }

    inline void expand(const double& dx, const double& dy){
      xmin -= dx;
      xmax += dx;
      ymin -= dy;
      ymax += dy;
    }

    inline bool operator ()(const BB& target) const{ // whether this and target has any intersect
      const bool c1 = unlikely(std::max(xmin, target.xmin) <= std::min(xmax, target.xmax));
      const bool c2 = unlikely(std::max(ymin, target.ymin) <= std::min(ymax, target.ymax));
      return unlikely(c1 && c2);
    }

    inline double operator [](const int& i)const{
      if (i == 0){
        return xmin;
      } else if (i == 1){
        return -xmax;
      } else if (i == 2){
        return ymin;
      } else{
        return -ymax;
      }
    }
    template <class Archive>
    void serialize( Archive & ar ){
      ar(xmin, xmax, ymin, ymax);
    }
};


template<class T>
class DataType{
  public:
    T first;
    BB second;
    DataType() = default;

    DataType(const T& f, const BB& s){
      first = f;
      second = s;
    }

    DataType(const DataType& obj) noexcept{// copy constructor for vector
      first = obj.first;
      second = obj.second;
    }

    DataType(DataType&& obj) noexcept{// move constructor for vector
      first = std::move(obj.first);
      second = std::move(obj.second);
    }

    DataType& operator=(DataType&& obj) noexcept{
      if (likely(this != &obj)){
        first = std::move(obj.first);
        second = std::move(obj.second);
      }
      return *this;
    }
    template <class Archive>
    void serialize( Archive & ar ){
      ar(first, second);
    }
};


template <class T, int B=6>
class Leaf : Uncopyable{
  public:
    int axis = 0;
    BB mbb;
    vec<DataType<T>> data; // You can swap when filtering
    // T is type of keys(ids) which will be returned when you post a query.
    Leaf(){
      mbb = BB();
      data.reserve(B);
    }
    Leaf(int& _axis){
      axis = _axis;
      mbb = BB();
      data.reserve(B);
    }

    template <class Archive>
    void serialize( Archive & ar ){
      ar(axis, mbb, data);
    }

    inline void set_axis(const int& _axis){
      axis = _axis;
    }

    inline void push(const T& key, const BB& target){
      data.emplace_back(key, target);
      update_mbb();
    }

    inline void update_mbb(){
      mbb.clear();
      for (const auto& datum : data){
        mbb += datum.second;
      }
    }

    template <typename F>
    inline bool filter(DataType<T>& value, const F comp){// false means given value is ignored
      if (unlikely(data.size() < B)){ // if there is room, just push the candidate
        data.emplace_back(std::move(value));
        update_mbb();
        return true;
      } else { // if there is no room, check the priority and swap if needed
        auto iter = upper_bound(data.begin(), data.end(), value, comp);
        if (unlikely(iter != data.end())){
          //iter->swap(value);
          swap(iter->first, value.first);
          swap(iter->second, value.second);
          update_mbb();
        }
        return false;
      }
    }

    void operator ()(const BB& target, vec<T>& out)const{
      if (unlikely(mbb(target))){
        for (const auto& x : data){
          if (unlikely(x.second(target))){
            out.emplace_back(x.first);
          }
        }
      }
    }

    void del(const T& key, const BB& target){
      if (unlikely(mbb(target))){
        auto remove_it = std::remove_if(data.begin(), data.end(), [&](auto& datum){
              return datum.second(target) && datum.first == key;
            });
        data.erase(remove_it, data.end());
      }
    }
};


template<class T, int B=6>
class PseudoPRTreeNode : Uncopyable{
  public:
    std::array<Leaf<T, B>, 4> leaves;
    std::unique_ptr<PseudoPRTreeNode> left, right;

    PseudoPRTreeNode(){
      for (int i = 0; i < 4; i++){
        leaves[i].set_axis(i);
      }
    }
    template<class Archive>
    void serialize(Archive & archive){
      //archive(cereal::(left), cereal::defer(right), leaves);
      archive(left, right, leaves);
    }

    inline auto address_of_leaves() {
      vec<Leaf<T, B>*> out;
      for (auto& leaf : leaves){
        if (likely(leaf.data.size() > 0)){
          out.emplace_back(&leaf);
        }
      }
      return out;
    }

    inline void address_of_leaves(vec<Leaf<T, B>*>& out) {
      for (auto& leaf : leaves){
        if (likely(leaf.data.size() > 0)){
          out.emplace_back(&leaf);
        }
      }
    }

    template<class iterator, class Function>
    inline auto filter (iterator b, iterator e, const int axis, const Function comp){
      auto out = std::remove_if(b, e, [&](auto& x){
          for (auto& l : leaves){
            if (unlikely(l.filter(x, comp))){
              return true;
            }
          }
          return false;
      });
      return out;
    }
    
};


template<class T, int B=6>
class PseudoPRTree : Uncopyable{
  public:
    std::unique_ptr<PseudoPRTreeNode<T, B>> root;
    vec<Leaf<T, B>*> cache_children;
    const int nthreads = std::max(1, (int) std::thread::hardware_concurrency());

    PseudoPRTree(){
      root = std::make_unique<PseudoPRTreeNode<T, B>>();
    }

    template<class iterator>
    PseudoPRTree(iterator& b, iterator& e){
      if (likely(!root)){
        root = std::make_unique<PseudoPRTreeNode<T, B>>();
      }
      construct(root.get(), b, e, 0);
      b->~DataType<T>();
      for (DataType<T>* it = b; it < e; ++it){
        it->~DataType<T>();
      }
      b = nullptr;
      e = nullptr;
    }

    template<class Archive>
    void serialize(Archive & archive){
      archive(root);
      //archive.serializeDeferments();
    }

    template<class iterator>
    inline void construct(PseudoPRTreeNode<T, B>* node, iterator& b, iterator& e, const size_t depth){
      if (e - b > 0 && node != nullptr) {
        bool use_recursive_threads = std::pow(2, depth + 1) <= nthreads;

        vec<std::thread> threads;
        PseudoPRTreeNode<T, B> *node_left, *node_right;

        const int axis = depth % 4;
        auto comp = [&](const DataType<T>& lhs, const DataType<T>& rhs){
          return lhs.second[axis] < rhs.second[axis];
        };
        auto ee = node->filter(b, e, axis, std::ref(comp));
        auto m = b;
        std::advance(m, (ee - b) / 2);
        auto mm = m;
        std::nth_element(b, m, ee, comp); // about 25~30 % of construction time
        if (m - b > 0){
          node->left = std::make_unique<PseudoPRTreeNode<T, B>>();
          node_left = node->left.get();
          if (use_recursive_threads){
            threads.emplace_back(std::thread([&](){construct(node_left, b, m, depth + 1);}));
          } else {
            construct(node_left, b, m, depth + 1);
          }
        }
        if (ee - mm > 0){
          node->right = std::make_unique<PseudoPRTreeNode<T, B>>();
          node_right = node->right.get();
          if (use_recursive_threads){
            threads.emplace_back(std::thread([&](){construct(node_right, mm, ee, depth + 1);}));
          } else {
            construct(node_right, mm, ee, depth + 1);
          }
        }
        std::for_each(threads.begin(), threads.end(), [&](std::thread& x){x.join();});
      }
    }

    inline auto get_all_leaves(const int hint){
      if (cache_children.empty()){
        using U = PseudoPRTreeNode<T, B>;
        cache_children.reserve(hint);
        auto node = root.get();
        std::queue<U*> que;
        que.emplace(node);

        while (likely(!que.empty())){
          node = que.front();
          que.pop();
          node->address_of_leaves(cache_children);
          if (node->left) que.emplace(node->left.get());
          if (node->right) que.emplace(node->right.get());
        }
      }
      return cache_children;
    }

    std::pair<DataType<int>*, DataType<int>*> as_X(void* placement, const int hint){
      const size_t nthreads = std::max(1, (int) std::thread::hardware_concurrency());
      DataType<int> *b, *e;
      auto children = get_all_leaves(hint);
      int total = children.size();
      b = (DataType<int>*)placement;
      e = b + total;
      {
        vec<std::thread> threads(nthreads);
        for (int t = 0; t < nthreads; t++){
          threads[t] = std::thread(std::bind(
            [&](const int bi, const int ei, const int t){
              for (int i = bi; i < ei; i++){
                new(b + i) DataType<int>{i, children[i]->mbb};
              }
            }, t * total/nthreads, unlikely((t+1)==nthreads)?total:(t+1)*total/nthreads, t)
          );
        }
        std::for_each(threads.begin(), threads.end(), [&](std::thread& x){x.join();});
      }
      return {b, e};
    }
};


template<class T, int B=6>
class PRTreeNode : Uncopyable{
  public:
    BB mbb;
    std::unique_ptr<Leaf<T, B>> leaf;
    std::unique_ptr<PRTreeNode<T, B>> head, next;

    PRTreeNode(){}
    PRTreeNode(const BB& _mbb){
      mbb = _mbb;
    }
    PRTreeNode(BB&& _mbb){
      mbb = std::move(_mbb);
    }
    PRTreeNode(Leaf<T, B>* l){
      leaf = std::make_unique<Leaf<T, B>>();
      leaf->mbb = l->mbb;
      mbb = l->mbb;
      for (const auto& d : l->data){
        leaf->push(d.first, d.second);
      }
    }

    inline bool operator ()(const BB& target){
      return mbb(target);
    }

    template<class Archive>
    void serialize(Archive & archive){
      //archive(cereal::defer(leaf), cereal::defer(head), cereal::defer(next));
      archive(mbb, leaf, head, next);
    }
};


template<class T, int B=6>
class PRTree : Uncopyable{
  private:
    std::unique_ptr<PRTreeNode<T, B>> root;
    std::unordered_map<T, BB> umap;

  public:
    template<class Archive>
    void serialize(Archive & archive){
      archive(root, umap);
      //archive.serializeDeferments();
    }

    void save(std::string fname){
      {
        {
          std::ofstream ofs(fname, std::ios::binary);
          cereal::PortableBinaryOutputArchive o_archive(ofs);
          //cereal::JSONOutputArchive o_archive(ofs);
          o_archive(cereal::make_nvp("root", root), cereal::make_nvp("umap", umap));
        }
      }
    }

    PRTree(){}

    PRTree(std::string fname){
      load(fname);
    }

    void load(std::string fname){
      {
        {
          std::ifstream ifs(fname, std::ios::binary);
          cereal::PortableBinaryInputArchive i_archive(ifs);
          //cereal::JSONInputArchive i_archive(ifs);
          i_archive(cereal::make_nvp("root", root), cereal::make_nvp("umap", umap));
        }
      }
    }


    PRTree(const py::array_t<T>& idx, const py::array_t<double>& x){
      const auto &buff_info_idx = idx.request();
      const auto &shape_idx = buff_info_idx.shape;
      const auto &buff_info_x = x.request();
      const auto &shape_x = buff_info_x.shape;
      if (shape_idx[0] != shape_x[0]){
        throw std::runtime_error("Both index and boudning box must have the same length");
      } else if (shape_x[1] != 4){
        throw std::runtime_error("Bounding box must have the shape (length, 4)");
      }
      size_t length = shape_idx[0];
      umap.reserve(length);

      DataType<T> *b, *e;
      void *placement = std::malloc(std::max(sizeof(DataType<T>), sizeof(DataType<int>)) * length);
      b = (DataType<T>*)placement;
      e = b;
      for (size_t i = 0; i < length; i++){
        auto bb = BB(*x.data(i, 0), *x.data(i, 1), *x.data(i, 2), *x.data(i, 3));

        new(e) DataType<T>{*idx.data(i), bb};
        e++;

        umap.emplace_hint(umap.end(), *idx.data(i), std::move(bb));
      }
      //ProfilerStart("construct.prof");
      build(b, e, placement);
      //ProfilerStop();
      std::free(placement);
    }

    void insert(const T& idx, const py::array_t<double>& x){
      vec<Leaf<T, B>*> cands;
      std::queue<PRTreeNode<T, B>*, std::deque<PRTreeNode<T, B>*>> que;
      BB bb;
      PRTreeNode<T, B>* p, * q;

      const auto &buff_info_x = x.request();
      const auto &shape_x = buff_info_x.shape;
      const auto &ndim = buff_info_x.ndim;
      if (shape_x[0] != 4 || ndim != 1){
        throw std::runtime_error("invalid shape.");
      }
      auto it = umap.find(idx);
      if (unlikely(it != umap.end())){
        throw std::runtime_error("Given index is already included.");
      }

      bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
      double dx = bb.xmax - bb.xmin + 0.000000001;
      double dy = bb.ymax - bb.ymin + 0.000000001;
      double c = 0.0;
      std::stack<PRTreeNode<T, B>*> sta;
      while (likely(cands.size() == 0)){
        while (likely(!sta.empty())){
          sta.pop();
        }
        bb.expand(dx * c, dy * c);
        c = (c + 1) * 2;
        auto qpush = [&](PRTreeNode<T, B>* r){
          if (unlikely((*r)(bb))){
            que.emplace(r);
            sta.push(r);
          }
        };

        p = root.get();
        qpush(p);
        while (likely(!que.empty())){
          p = que.front();
          que.pop();

          if (unlikely(p->leaf)){
            cands.emplace_back(p->leaf.get());
          } else {
            if (likely(p->head)){
              q = p->head.get();
              qpush(q);
              while (likely(q->next)){
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
      while (likely(!sta.empty())){
        p = sta.top();
        sta.pop();
        if (unlikely(p->leaf)){
          p->mbb = p->leaf->mbb;
        } else if (likely(p->head)) {
          BB mbb;
          q = p->head.get();
          mbb += q->mbb;
          while (likely(q->next)){
            q = q->next.get();
            mbb += q->mbb;
          }
          p->mbb = mbb;
        }
      }
    }


    template<class iterator>
    void build(iterator b, iterator e, void *placement){
      const int nthreads = std::max(1, (int) std::thread::hardware_concurrency());
      vec<Leaf<int, B>*> leaves;
      vec<std::unique_ptr<PRTreeNode<T, B>>> tmp_nodes, prev_nodes;
      std::unique_ptr<PRTreeNode<T, B>> p, q, r;

      auto first_tree = PseudoPRTree<T, B>(b, e);
      for (auto& leaf : first_tree.get_all_leaves(e - b)){
        p = std::make_unique<PRTreeNode<T, B>>(leaf);
        prev_nodes.emplace_back(std::move(p));
      }

      auto [bb, ee] = first_tree.as_X(placement, e - b);
      while (likely(prev_nodes.size() > 1)){
        auto tree = PseudoPRTree<int, B>(bb, ee);
        leaves = tree.get_all_leaves(ee - bb);
        auto leaves_size = leaves.size();
        tmp_nodes.clear();
        tmp_nodes.reserve(leaves_size);

        vec<decltype(tmp_nodes)> out_privates(nthreads);
        {
          vec<std::thread> threads(nthreads);
          for (auto& o : out_privates){
            o.reserve(leaves_size/nthreads*2);
          }
          for (int t = 0; t < nthreads; t++){
            threads[t] = std::thread(std::bind(
                  [&](const int bi, const int ei, const int t){
                  for (size_t k = bi; k < ei; k++){
                    int idx, jdx;
                    auto& leaf = leaves[k];
                    int len = leaf->data.size();
                    auto pp = std::make_unique<PRTreeNode<T, B>>(leaf->mbb);
                    if (likely(!leaf->data.empty())){
                      for (int i = 1; i < len; i++){
                        idx = leaf->data[len - i - 1].first; // reversed way
                        jdx = leaf->data[len - i].first;
                        prev_nodes[idx]->next = std::move(prev_nodes[jdx]);
                      }
                      idx = leaf->data[0].first;
                      pp->head = std::move(prev_nodes[idx]);
                      if (!pp->head){throw std::runtime_error("ppp");}
                      out_privates[t].emplace_back(std::move(pp));
                    } else {
                      throw std::runtime_error("what????");
                    }
                  }
                }, t*leaves_size/nthreads, unlikely((t+1)==nthreads)?leaves_size:(t+1)*leaves_size/nthreads, t));
          }
          std::for_each(threads.begin(), threads.end(), [&](std::thread& x){x.join();});

          for (int t = 0; t < nthreads; t++){
            tmp_nodes.insert(tmp_nodes.end(),
                std::make_move_iterator(out_privates[t].begin()),
                std::make_move_iterator(out_privates[t].end()));
          }
        }

        {
          auto tmp = tree.as_X(placement, ee - bb);
          bb = std::move(tmp.first);
          ee = std::move(tmp.second);
        }

        /*size_t c = 0;
        for (const auto& n : prev_nodes){
          if (unlikely(n)){
            c++;
          }
        }
        if (unlikely(c > 0)){
          throw std::runtime_error("eee");
        }*/

        prev_nodes.swap(tmp_nodes);
      }
      if (unlikely(prev_nodes.size() != 1)){
        throw std::runtime_error("#roots is not 1.");
      }
      root = std::move(prev_nodes[0]);

      vec<Leaf<int, B>*>().swap(leaves);
      vec<std::unique_ptr<PRTreeNode<T, B>>>().swap(tmp_nodes);
      vec<std::unique_ptr<PRTreeNode<T, B>>>().swap(prev_nodes);
    }
    

    auto find_all(const py::array_t<double>& x){
      const auto &buff_info_x = x.request();
      const auto &ndim= buff_info_x.ndim;
      const auto &shape_x = buff_info_x.shape;
      if (ndim == 1 && shape_x[0] != 4){
        throw std::runtime_error("Bounding box must have the shape 4 with (xmin, xmax, ymin, ymax)");
      } else if (ndim == 2 && shape_x[1] != 4){
        throw std::runtime_error("Bounding box must have the shape (length, 4)");
      } else if (ndim > 3){
        throw std::runtime_error("invalid shape");
      }
      vec<BB> X;
      BB bb;
      if (ndim == 1){
        bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
        X.emplace_back(std::move(bb));
      } else {
        X.reserve(shape_x[0]);
        for (long int i = 0; i < shape_x[0]; i++){
          bb = BB(*x.data(i, 0), *x.data(i, 1), *x.data(i, 2), *x.data(i, 3));
          X.emplace_back(std::move(bb));
        }
      }
      size_t length = X.size();
      const int nthreads = std::max(1, (int) std::thread::hardware_concurrency());
      vec<vec<vec<T>>> out_privates(nthreads);
      {
        vec<std::thread> threads(nthreads);
        for (auto& o : out_privates){
          o.reserve(length/nthreads+1);
        }
        for (int t = 0; t < nthreads; t++){
          threads[t] = std::thread(std::bind(
                [&](const int bi, const int ei, const int t){
                for (int i = bi; i < ei; i++){
                  out_privates[t].emplace_back(std::move(find(X[i])));
                }
              }, t*length/nthreads, unlikely((t+1)==nthreads)?length:(t+1)*length/nthreads, t));
        }
        std::for_each(threads.begin(), threads.end(), [&](std::thread& x){x.join();});
      }
      vec<vec<T>> out;
      out.reserve(length);
      for (int t = 0; t < nthreads; t++){
        out.insert(out.end(),
            std::make_move_iterator(out_privates[t].begin()),
            std::make_move_iterator(out_privates[t].end()));
        vec<vec<T>>().swap(out_privates[t]);
      }
      return out;
    }

    vec<T> find_one(const py::array_t<double>& x){
      const auto &buff_info_x = x.request();
      const auto &ndim= buff_info_x.ndim;
      const auto &shape_x = buff_info_x.shape;
      if (unlikely(ndim != 1 || shape_x[0] != 4)){
        throw std::runtime_error("invalid shape");
      } 
      const BB bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
      return find(bb);
    } 

    vec<T> find(const BB& target){
      vec<T> out;
      std::queue<PRTreeNode<T, B>*, std::deque<PRTreeNode<T, B>*>> que;
      PRTreeNode<T, B>* p, * q;
      auto qpush = [&](PRTreeNode<T, B>* r){
        if (unlikely((*r)(target))){
          que.emplace(r);
        }
      };

      p = root.get();
      qpush(p);
      while (likely(!que.empty())){
        p = que.front();
        que.pop();

        if (unlikely(p->leaf)){
          (*p->leaf)(target, out); 
        } else {
          if (likely(p->head)){
            q = p->head.get();
            qpush(q);
            while (likely(q->next)){
              q = q->next.get();
              qpush(q);
            }
          }
        }
      }
      return out;
    }

    void erase(const T idx){
      auto it = umap.find(idx);
      if (unlikely(it== umap.end())){
        throw std::runtime_error("Given index is not found.");
      }
      BB target = it->second;
      std::queue<PRTreeNode<T, B>*> que;
      PRTreeNode<T, B>* p, * q;
      auto qpush = [&](PRTreeNode<T, B>* r){
        if (unlikely((*r)(target))){
          que.emplace(r);
        }
      };

      p = root.get();
      qpush(p);
      while (likely(!que.empty())){
        p = que.front();
        que.pop();

        if (unlikely(p->leaf)){
          p->leaf->del(idx, target); 
        } else {
          if (likely(p->head)){
            q = p->head.get();
            qpush(q);
            while (likely(q->next)){
              q = q->next.get();
              qpush(q);
            }
          }
        }
      }
      umap.erase(idx);
    }

};

