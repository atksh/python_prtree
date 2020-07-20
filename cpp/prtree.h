#pragma once
#include <mimalloc.h>
#include <array>
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
#include <mutex>
#include <cstdlib>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


//#include <gperftools/profiler.h>


#define MAX_PARALLEL_RECURSIVE_LEVEL 5

namespace py = pybind11;
using std::swap;
template<class T, class U>
using pair = std::pair<T, U>;
template<class T>
using vec = std::vector<T, mi_stl_allocator<T>>;
static std::mt19937 rand_src(42);


#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

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
    float xmin, xmax, ymin, ymax;
    BB() {
      clear();
    }

    BB(const float& _xmin, const float& _xmax, const float& _ymin, const float& _ymax){
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

    inline void expand(const float& dx, const float& dy){
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

    inline float operator [](const int& i)const{
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

    DataType(DataType&& obj) noexcept{// move constructor for vector
      first = obj.first;
      second.xmin = obj.second.xmin;
      second.xmax = obj.second.xmax;
      second.ymin = obj.second.ymin;
      second.ymax = obj.second.ymax;
      obj.second.clear();
    }

    DataType& operator=(DataType&& obj) noexcept{
      if (likely(this != &obj)){
        first = obj.first;
        second.xmin = obj.second.xmin;
        second.xmax = obj.second.xmax;
        second.ymin = obj.second.ymin;
        second.ymax = obj.second.ymax;
        obj.second.clear();
      }
      return *this;
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
      //auto init = std::make_DataType<T>(std::forward<T>(data[0].first), BB());
      //mbb = std::reduce(data.begin(), data.end(), init, [&](DataType<T>& lhs, DataType<T>& rhs) {
      //      return std::make_DataType<T>(std::forward<T>(lhs.first), std::forward<BB>(lhs.second + rhs.second));
      //    }).second;
    }

    template <typename F>
    bool filter(DataType<T>& value, const F &comp){// false means given value is ignored
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
        for (auto it = data.begin(); it != data.end();++it){
          if (unlikely(it->second(target) && it->first == key)){
            it = data.erase(it);
            break;
          }
        }
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

    auto filter (vec<DataType<T>>& X, int axis){
      auto _comp = [&](const DataType<T>& lhs, const DataType<T>& rhs){
        return lhs.second[axis] < rhs.second[axis];
      };
      auto comp = std::ref(_comp);
      auto out = std::remove_if(X.begin(), X.end(), [&](auto& x) {
          for (auto& l : leaves){
            if (unlikely(l.filter(x, comp))){
              return true;
            }
          }
          return false;
      });
      X.erase(out, X.end());
      return out;
    }
    
};


template<class T, int B=6>
class PseudoPRTree : Uncopyable{
  public:
    std::unique_ptr<PseudoPRTreeNode<T, B>> root;

    PseudoPRTree(){
      root = std::make_unique<PseudoPRTreeNode<T, B>>();
    }
    PseudoPRTree(vec<DataType<T>>& X){
      if (likely(!root)){
        root = std::make_unique<PseudoPRTreeNode<T, B>>();
      }
      #pragma omp parallel
      {
        #pragma omp single nowait
        {
          construct(root.get(), X, 0);
        }
      }
    }

    void construct(PseudoPRTreeNode<T, B>* node, vec<DataType<T>>& X, const unsigned int& depth){
      if (likely(X.size() > 0 && node != nullptr)) {
        int axis = depth % 4;
        auto comp = [&](const DataType<T>& lhs, const DataType<T>& rhs){
          return lhs.second[axis] < rhs.second[axis];
        };
        auto e = node->filter(X, axis);
        auto b = X.begin();
        auto m = b;
        std::advance(m, (e - b) / 2);
        std::nth_element(b, m, e, comp);
        if (depth < MAX_PARALLEL_RECURSIVE_LEVEL) {
          if (likely(m - b > 0)){
            node->left = std::make_unique<PseudoPRTreeNode<T, B>>();
            auto node_left = node->left.get();
            #pragma omp task default(none) firstprivate(node_left, depth) shared(b, m) 
            {
              vec<DataType<T>> X_left(std::make_move_iterator(b), std::make_move_iterator(m));
              construct(node_left, X_left, depth + 1);
            }
          }
          if (likely(e - m > 0)){
            node->right = std::make_unique<PseudoPRTreeNode<T, B>>();
            auto node_right = node->right.get();
            #pragma omp task default(none) firstprivate(node_right, depth) shared(m, e)
            {
              vec<DataType<T>> X_right(std::make_move_iterator(m), std::make_move_iterator(e));
              construct(node_right, X_right, depth + 1);
            }
          }
          #pragma omp taskwait
        } else {
          if (likely(m - b > 0)){
            node->left = std::make_unique<PseudoPRTreeNode<T, B>>();
            auto node_left = node->left.get();
            vec<DataType<T>> X_left(std::make_move_iterator(b), std::make_move_iterator(m));
            construct(node_left, X_left, depth + 1);
            vec<DataType<T>>().swap(X_left);
          }
          if (likely(e - m > 0)){
            node->right = std::make_unique<PseudoPRTreeNode<T, B>>();
            auto node_right = node->right.get();
            vec<DataType<T>> X_right(std::make_move_iterator(m), std::make_move_iterator(e));
            construct(node_right, X_right, depth + 1);
            vec<DataType<T>>().swap(X_right);
          }
        }
      }
    }

    inline auto get_all_leaves(){
      using U = PseudoPRTreeNode<T, B>;
      vec<Leaf<T, B>*> out;
      auto node = root.get();
      std::queue<U*> que;
      vec<U*> cands;
      que.emplace(node);

      while (likely(!que.empty())){
        node = que.front();
        que.pop();
        //node->address_of_leaves(out);
        if (node->left) que.emplace(node->left.get());
        if (node->right) que.emplace(node->right.get());
        cands.emplace_back(std::move(node));
      }

      int total = cands.size();
      #pragma omp parallel
      {
        vec<Leaf<T, B>*> out_private;
        #pragma omp for nowait schedule(static)
        for (int i = 0; i<total; i++){
          auto tmp = cands[i]->address_of_leaves();
          out_private.insert(out_private.end(),
                        std::make_move_iterator(tmp.begin()),
                        std::make_move_iterator(tmp.end()));
        }
        #pragma omp for schedule(static) ordered
        for (int i = 0; i < omp_get_num_threads(); i++){
          #pragma omp ordered
          out.insert(out.end(),
                        std::make_move_iterator(out_private.begin()),
                        std::make_move_iterator(out_private.end()));
        }
      }
      vec<U*>().swap(cands);
      return out;
    }

    inline vec<DataType<int>> as_X(){
      vec<DataType<int>> out;
      auto children = get_all_leaves();
      int total = children.size();
      out.reserve(total);

      #pragma omp parallel
      {
        vec<DataType<int>> out_private;
        #pragma omp for nowait schedule(static)
        for (int i = 0; i<total; i++){
          out_private.emplace_back(i, children[i]->mbb);
        }
        #pragma omp for schedule(static) ordered
        for (int i = 0; i < omp_get_num_threads(); i++){
          #pragma omp ordered
          out.insert(out.end(),
                        std::make_move_iterator(out_private.begin()),
                        std::make_move_iterator(out_private.end()));
        }
      }
      return out;
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

};


template<class T, int B=6>
class PRTree : Uncopyable{
  private:
    std::unique_ptr<PRTreeNode<T, B>> root;
    std::unordered_map<T, BB> umap;

  public:
    PRTree(const py::array_t<T>& idx, const py::array_t<float>& x){
      const auto &buff_info_idx = idx.request();
      const auto &shape_idx = buff_info_idx.shape;
      const auto &buff_info_x = x.request();
      const auto &shape_x = buff_info_x.shape;
      if (shape_idx[0] != shape_x[0]){
        throw std::runtime_error("Both index and boudning box must have the same length");
      } else if (shape_x[1] != 4){
        throw std::runtime_error("Bounding box must have the shape (length, 4)");
      }
      unsigned int length = shape_idx[0];
      vec<DataType<T>> X;
      X.reserve(length);
      #pragma omp parallel
      {
        vec<DataType<T>> X_private;
        #pragma omp single
        for (unsigned int i = 0; i < length; i++){
          auto bb = BB(*x.data(i, 0), *x.data(i, 1), *x.data(i, 2), *x.data(i, 3));
          umap[*idx.data(i)] = bb;
        }
        #pragma omp for nowait schedule(static)
        for (unsigned int i = 0; i < length; i++){
          auto bb = BB(*x.data(i, 0), *x.data(i, 1), *x.data(i, 2), *x.data(i, 3));
          X_private.emplace_back(*idx.data(i), bb);
        }

        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++) {
          #pragma omp ordered
          X.insert(X.end(),
                        std::make_move_iterator(X_private.begin()),
                        std::make_move_iterator(X_private.end()));
        }
        vec<DataType<T>>().swap(X_private);
      }
      //ProfilerStart("construct.prof");
      build(X);
      //ProfilerStop();
      vec<DataType<T>>().swap(X);
    }

    void insert(const T& idx, const py::array_t<float>& x){
      vec<Leaf<T, B>*> cands;
      std::queue<PRTreeNode<T, B>*, std::deque<PRTreeNode<T, B>*>> que;
      BB bb;
      PRTreeNode<T, B>* p, * q;

      const auto &buff_info_x = x.request();
      const auto &shape_x = buff_info_x.shape;
      const auto &ndim = buff_info_x.ndim;
      if (shape_x[0] != 4 or ndim != 1){
        throw std::runtime_error("invalid shape.");
      }
      auto it = umap.find(idx);
      if (unlikely(it != umap.end())){
        throw std::runtime_error("Given index is already included.");
      }

      bb = BB(*x.data(0), *x.data(1), *x.data(2), *x.data(3));
      float dx = bb.xmax - bb.xmin + 0.000000001;
      float dy = bb.ymax - bb.ymin + 0.000000001;
      float c = 0.0;
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


    void build(vec<DataType<T>>& X){
      vec<Leaf<int, B>*> leaves;
      vec<std::unique_ptr<PRTreeNode<T, B>>> tmp_nodes, prev_nodes;
      std::unique_ptr<PRTreeNode<T, B>> p, q, r;

      auto first_tree = PseudoPRTree<T, B>(X);
      for (auto& leaf : first_tree.get_all_leaves()){
        p = std::make_unique<PRTreeNode<T, B>>(leaf);
        prev_nodes.emplace_back(std::move(p));
      }
      vec<DataType<int>> as_X = first_tree.as_X();
      while (likely(prev_nodes.size() > 1)){
        auto tree = PseudoPRTree<int, B>(as_X);
        leaves = tree.get_all_leaves();
        auto leaves_size = leaves.size();
        tmp_nodes.clear();
        tmp_nodes.reserve(leaves_size);
        #pragma omp parallel
        {
          vec<std::unique_ptr<PRTreeNode<T, B>>> tmp_nodes_private;
          #pragma omp for nowait schedule(static)
          for (unsigned int k=0; k<leaves_size; k++){
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
              tmp_nodes_private.emplace_back(std::move(pp));
            } else {
              throw std::runtime_error("what????");
            }
          }
          #pragma omp for schedule(static) ordered
          for(int i=0; i<omp_get_num_threads(); i++) {
              #pragma omp ordered
              tmp_nodes.insert(tmp_nodes.end(),
                            std::make_move_iterator(tmp_nodes_private.begin()),
                            std::make_move_iterator(tmp_nodes_private.end()));
          }
        }
        as_X = tree.as_X();
        unsigned int c = 0;
        for (const auto& n : prev_nodes){
          if (unlikely(n)){
            c++;
          }
        }
        if (unlikely(c > 0)){
          throw std::runtime_error("eee");
        }
        prev_nodes.swap(tmp_nodes);
      }
      if (unlikely(prev_nodes.size() != 1)){
        throw std::runtime_error("#roots is not 1.");
      }
      root = std::move(prev_nodes[0]);

      vec<Leaf<int, B>*>().swap(leaves);
      vec<std::unique_ptr<PRTreeNode<T, B>>>().swap(tmp_nodes);
      vec<std::unique_ptr<PRTreeNode<T, B>>>().swap(prev_nodes);
      vec<DataType<int>>().swap(as_X);
    }
    

    auto find_all(const py::array_t<float> x){
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
      py::gil_scoped_release release;
      vec<vec<T>> out;
      unsigned int length = X.size();
      out.reserve(length);
      #pragma omp parallel
      {
        vec<vec<T>> out_private;
        #pragma omp for nowait schedule(static)
        for(unsigned int i=0; i<length; i++){
          out_private.emplace_back(std::move(find(X[i])));
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++) {
          #pragma omp ordered
          out.insert(out.end(),
                        std::make_move_iterator(out_private.begin()),
                        std::make_move_iterator(out_private.end()));
        }
      }
      vec<BB>().swap(X);
      return out;
    }

    vec<T> find_one(const py::array_t<float> x){
      const auto &buff_info_x = x.request();
      const auto &ndim= buff_info_x.ndim;
      const auto &shape_x = buff_info_x.shape;
      if (ndim != 1 or shape_x[0] != 4){
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
