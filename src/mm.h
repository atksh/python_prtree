#pragma once
#include <boost/heap/pairing_heap.hpp>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <memory>
#include <iterator>
#include <iostream>
#include <omp.h>
#include "prtree.h"

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)



template<class T>
class RoadGraphNode{
};


