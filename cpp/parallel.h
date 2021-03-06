#pragma once
#include <vector>
#include <thread>
#include <algorithm>

template <typename F, typename Iter, typename T>
void parallel_for_each(const Iter first, const Iter last, T &result, const F &func)
{
  auto f = std::ref(func);
  const int nthreads = std::max(1, (int)std::thread::hardware_concurrency());
  const size_t total = std::distance(first, last);
  std::vector<T> rr(nthreads);
  {
    std::vector<std::thread> threads;
    std::vector<Iter> iters;
    auto step = total / nthreads;
    auto remaining = total % nthreads;
    auto n = first;
    iters.emplace_back(first);
    for (int i = 0; i < nthreads - 1; ++i)
    {
      std::advance(n, i < remaining ? step + 1 : step);
      iters.emplace_back(n);
    }
    iters.emplace_back(last);

    result.reserve(total);
    for (auto &&r : rr)
    {
      r.reserve(total / nthreads + 1);
    }
    for (int t = 0; t < nthreads; t++)
    {
      threads.emplace_back(std::thread([&, t] { std::for_each(iters[t], iters[t + 1], [&](auto &x) { f(x, rr[t]); }); }));
    }
    std::for_each(threads.begin(), threads.end(), [&](std::thread &x) { x.join(); });
    threads.clear();
    std::vector<std::thread>().swap(threads);
    iters.clear();
    std::vector<Iter>().swap(iters);
  }
  for (int t = 0; t < nthreads; t++)
  {
    result.insert(result.end(),
                  std::make_move_iterator(rr[t].begin()),
                  std::make_move_iterator(rr[t].end()));
    rr[t].clear();
    T().swap(rr[t]);
  }
  std::vector<T>().swap(rr);
}

template <typename F, typename Iter>
void parallel_for_each(const Iter first, const Iter last, const F &func)
{
  auto f = std::ref(func);
  const int nthreads = std::max(1, (int)std::thread::hardware_concurrency());
  const size_t total = std::distance(first, last);
  {
    std::vector<std::thread> threads;
    std::vector<Iter> iters;
    auto step = total / nthreads;
    auto remaining = total % nthreads;
    auto n = first;
    iters.emplace_back(first);
    for (int i = 0; i < nthreads - 1; ++i)
    {
      std::advance(n, i < remaining ? step + 1 : step);
      iters.emplace_back(n);
    }
    iters.emplace_back(last);
    for (int t = 0; t < nthreads; t++)
    {
      threads.emplace_back(std::thread([&, t] { std::for_each(iters[t], iters[t + 1], [&](auto &x) { f(x); }); }));
    }
    std::for_each(threads.begin(), threads.end(), [&](std::thread &x) { x.join(); });
    threads.clear();
    std::vector<std::thread>().swap(threads);
    iters.clear();
    std::vector<Iter>().swap(iters);
  }
}
