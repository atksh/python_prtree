#pragma once
#include <array>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <numeric>
#include <memory>
#include <mutex>
#include <iterator>
#include <iostream>
#include <omp.h>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)


class BoundingBox{
  public:
    double xmin, xmax, ymin, ymax;
    BoundingBox() {}
    BoundingBox(double _xmin, double _xmax, double _ymin, double _ymax){
      xmin = _xmin;
      xmax = _xmax;
      ymin = _ymin;
      ymax = _ymax;
    }

    BoundingBox operator +(const BoundingBox& rhs) const{
      return BoundingBox(std::min(xmin, rhs.xmin), std::max(xmax, rhs.xmax), std::min(ymin, rhs.ymin), std::max(ymax, rhs.ymax));
    }

    BoundingBox operator +=(const BoundingBox& rhs){
      xmin = std::min(xmin, rhs.xmin);
      xmax = std::max(xmax, rhs.xmax);
      ymin = std::min(ymin, rhs.ymin);
      ymax = std::max(ymax, rhs.ymax);
      return *this;
    }

    inline bool operator ()(const BoundingBox& target) const{ // whether this and target has any intersect
      const bool c1 = std::max(xmin, target.xmin) <= std::min(xmax, target.xmax);
      const bool c2 = std::max(ymin, target.ymin) <= std::min(ymax, target.ymax);
      return c1 && c2;
    }

    inline double operator [](int i)const{
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

template <class T, int B=6>
class Leaf{
  public:
    int axis = 0;
    BoundingBox mbb;
    std::vector<std::pair<T, BoundingBox>> data; // You can swap when filtering
    // T is type of keys(ids) which will be returned when you post a query.
    Leaf(){}
    Leaf(int _axis){
      mbb = BoundingBox(1e100, -1e100, 1e100, -1e100);
      axis = _axis;
      data.reserve(B);
    }

    inline void push(T key, BoundingBox target){
      data.emplace_back(key, target);
      update_mbb();
    }

    void update_mbb(){
      BoundingBox bb(1e100, -1e100, 1e100, -1e100);
      auto init = std::make_pair<T, BoundingBox>(std::forward<T>(data[0].first), std::forward<BoundingBox>(bb));
      mbb = std::reduce(data.begin(), data.end(), init, [&](std::pair<T, BoundingBox>& lhs, std::pair<T, BoundingBox>& rhs) {
            return std::make_pair<T, BoundingBox>(std::forward<T>(lhs.first), std::forward<BoundingBox>(lhs.second + rhs.second));
          }).second;
    }

    bool filter(std::pair<T, BoundingBox>& value){// nullopt or lower priority one
      if (unlikely(data.size() < B)){ // if there is room, just push the candidate
        data.push_back(value);
        update_mbb();
        return true;
      } else { // if there is no room, check the priority and swap if needed
        auto comp = [&](const std::pair<T, BoundingBox>& lhs, const std::pair<T, BoundingBox>& rhs){
          return lhs.second[axis] < rhs.second[axis];
        };
        auto iter = upper_bound(data.begin(), data.end(), value, comp);
        if (unlikely(iter != data.end())){
          iter->swap(value);
          update_mbb();
        }
        return false;
      }
    }

    inline void operator ()(const BoundingBox& target, std::vector<T>& out)const{
      if (unlikely(mbb(target))){
        for (const auto& x : data){
          if (unlikely(x.second(target))){
            out.emplace_back(x.first);
          }
        }
      }
    }

};


template<class T, int B=6>
class PseudoPRTreeNode{
  public:

    std::array<Leaf<T, B>, 4> leaves;
    std::shared_ptr<PseudoPRTreeNode> left, right;

    PseudoPRTreeNode(){
      for (int i = 0; i < 4; i++){
        leaves[i] = Leaf<T, B>(i);
      }
    }

    template<class iterator>
    auto filter (const iterator& b, const iterator& e){
      using U = std::pair<T, BoundingBox>;
      std::vector<U> out;
      const int length = e - b;
      out.reserve(length);
      std::for_each(b, e, [&](U x) mutable{
          out.emplace_back(x);
          for (auto& l : leaves){
            if (unlikely(l.filter(out.back()))){
              out.pop_back();
              break;
            }
          }
      });
      return out;
    }
    
};


template<class T, int B=6>
class PseudoPRTree{
  public:
    std::shared_ptr<PseudoPRTreeNode<T, B>> root;

    PseudoPRTree(){
      root = std::make_shared<PseudoPRTreeNode<T, B>>();
    }
    PseudoPRTree(std::vector<std::pair<T, BoundingBox>> X){
      if (likely(!root)){
        root = std::make_shared<PseudoPRTreeNode<T, B>>();
      }
      construct(root, X.begin(), X.end(), 0);
      auto tmp = get_all_leaves();
    }

    template<class iterator>
    void construct(std::shared_ptr<PseudoPRTreeNode<T, B>>& node, iterator b, iterator e, int depth){
      if (likely(e - b > 0 && node)) {
        int axis = depth % 4;
        auto X_r = node->filter(b, e);
        auto comp = [&](const std::pair<T, BoundingBox>& lhs, const std::pair<T, BoundingBox>& rhs){
          return lhs.second[axis] < rhs.second[axis];
        };
        auto b = X_r.begin();
        auto e = X_r.end();
        auto m = b;
        std::advance(m, X_r.size() / 2);
        std::nth_element(b, m, e, comp);
        if (likely(m - b > 0)){
          node->left = std::make_shared<PseudoPRTreeNode<T, B>>();
          construct(node->left, b, m, depth + 1);
        }
        if (likely(e - m > 0)){
          node->right = std::make_shared<PseudoPRTreeNode<T, B>>();
          construct(node->right, m, e, depth + 1);
        }
      }
      return;
    }

    auto get_all_leaves(){
      using U = PseudoPRTreeNode<T, B>;
      std::vector<Leaf<T, B>> out;
      auto node = std::shared_ptr<U>(root);
      std::queue<std::shared_ptr<U>> que;
      que.emplace(node);
      while (likely(!que.empty())){
        node = que.front();
        que.pop();
        for (auto& l : node->leaves){
          if (likely(l.data.size() > 0)){
            out.emplace_back(l);
          }
        }
        std::shared_ptr<U> left = node->left;
        std::shared_ptr<U> right = node->right;
        if (likely(left)) que.emplace(left);
        if (likely(right)) que.emplace(right);
      }
      return out;
    }

    auto as_X(){
      std::vector<std::pair<int, BoundingBox>> out;
      auto children = get_all_leaves();
      int i = 0;
      for (auto c : children){
        out.emplace_back(i, c.mbb);
        i++;
      }
      return out;
    }
};


template<class T, int B=6>
class PRTreeNode{
  public:
    BoundingBox mbb;
    std::unique_ptr<Leaf<T, B>> leaf;
    std::unique_ptr<PRTreeNode<T, B>> head, next;

    PRTreeNode(){}
    PRTreeNode(const BoundingBox& _mbb){
      mbb = _mbb;
    }
    PRTreeNode(Leaf<T, B>& l){
      leaf = std::make_unique<Leaf<T, B>>();
      leaf->mbb = l.mbb;
      mbb = l.mbb;
      for (auto& d : l.data){
        leaf->push(d.first, d.second);
      }
    }

    inline bool operator ()(BoundingBox& target){
      return mbb(target);
    }

};


template<class T, int B=6>
class PRTree{
  public:
    std::unique_ptr<PRTreeNode<T, B>> root;

    PRTree(std::vector<std::pair<T, BoundingBox>>& X){
      PseudoPRTree<int, B> tree;
      std::vector<Leaf<int, B>> leaves;
      std::vector<std::unique_ptr<PRTreeNode<T, B>>> tmp_nodes, prev_nodes;
      std::unique_ptr<PRTreeNode<T, B>> p, q, r;
      int i, len, idx, jdx;

      auto first_tree = PseudoPRTree<T, B>(X);
      for (auto& leaf : first_tree.get_all_leaves()){
        p = std::make_unique<PRTreeNode<T, B>>(leaf);
        prev_nodes.push_back(std::move(p));
      }
      auto as_X = first_tree.as_X();
      while (likely(prev_nodes.size() > 1)){
        tree = PseudoPRTree<int, B>(as_X);
        leaves = tree.get_all_leaves();
        tmp_nodes.clear();
        tmp_nodes.reserve(leaves.size());
        for (auto& leaf : leaves){
          len = leaf.data.size();
          p = std::make_unique<PRTreeNode<T, B>>(leaf.mbb);
          if (likely(!leaf.data.empty())){
            for (i = 1; i < len; i++){
              idx = leaf.data[len - i - 1].first; // reversed way
              jdx = leaf.data[len - i].first;
              prev_nodes[idx]->next = std::move(prev_nodes[jdx]);
            }
            idx = leaf.data[0].first;
            p->head = std::move(prev_nodes[idx]);
            if (!p->head){throw std::runtime_error("ppp");}
            tmp_nodes.push_back(std::move(p));
          } else {
            throw std::runtime_error("what????");
          }
        }
        as_X = tree.as_X();
        int c = 0;
        int cc = 0;
        for (auto& n : prev_nodes){
          if (unlikely(n)){
            std::cout << cc << std::endl;
            c+=1;
          }
          cc++;
        }
        if (unlikely(c > 0)){
          std::cout << c << std::endl;
          throw std::runtime_error("eee");
        }
        prev_nodes.swap(tmp_nodes);
      }
      if (unlikely(prev_nodes.size() != 1)){
        throw std::runtime_error("#roots is not 1.");
      }
      root = std::move(prev_nodes[0]);
    }

    std::vector<T> operator ()(BoundingBox& target){
      std::vector<T> out;
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
};

