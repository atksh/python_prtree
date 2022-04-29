#pragma once
#include <vector>
#include <thread>
#include <algorithm>

template <typename F, typename Iter, typename T>
void parallel_for_each(const Iter first, const Iter last, T &result, const F &func)
{
  auto f = std::ref(func);
  const size_t nthreads = (size_t)std::max(1, (int)std::thread::hardware_concurrency());
  const size_t total = std::distance(first, last);
  std::vector<T> rr(nthreads);
  {
    std::vector<std::thread> threads;
    std::vector<Iter> iters;
    size_t step = total / nthreads;
    size_t remaining = total % nthreads;
    Iter n = first;
    iters.emplace_back(first);
    for (size_t i = 0; i < nthreads - 1; ++i)
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
    for (size_t t = 0; t < nthreads; t++)
    {
      threads.emplace_back(std::thread([&, t]
                                       { std::for_each(iters[t], iters[t + 1], [&](auto &x)
                                                       { f(x, rr[t]); }); }));
    }
    std::for_each(threads.begin(), threads.end(), [&](std::thread &x)
                  { x.join(); });
  }
  for (size_t t = 0; t < nthreads; t++)
  {
    result.insert(result.end(),
                  std::make_move_iterator(rr[t].begin()),
                  std::make_move_iterator(rr[t].end()));
  }
}

template <typename F, typename Iter>
void parallel_for_each(const Iter first, const Iter last, const F &func)
{
  auto f = std::ref(func);
  const size_t nthreads = (size_t)std::max(1, (int)std::thread::hardware_concurrency());
  const size_t total = std::distance(first, last);
  {
    std::vector<std::thread> threads;
    std::vector<Iter> iters;
    size_t step = total / nthreads;
    size_t remaining = total % nthreads;
    Iter n = first;
    iters.emplace_back(first);
    for (size_t i = 0; i < nthreads - 1; ++i)
    {
      std::advance(n, i < remaining ? step + 1 : step);
      iters.emplace_back(n);
    }
    iters.emplace_back(last);
    for (size_t t = 0; t < nthreads; t++)
    {
      threads.emplace_back(std::thread([&, t]
                                       { std::for_each(iters[t], iters[t + 1], [&](auto &x)
                                                       { f(x); }); }));
    }
    std::for_each(threads.begin(), threads.end(), [&](std::thread &x)
                  { x.join(); });
  }
}
