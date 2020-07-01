#include <vector>
#include <random>
#include <time.h>
#include <omp.h>
#include "prtree.h"

int add(int x, int y){
  std::mt19937 engine(0);
  std::vector<std::pair<int, BoundingBox>> v;
  std::uniform_real_distribution<> normal(0.0, 1.0);
  BoundingBox bb(0.0, 0.0, 0.0, 0.0);
  for (int i=0; i < 100000; i++){
    double r0 = normal(engine);
    double r1 = normal(engine);
    double r2 = normal(engine);
    double r3 = normal(engine);
    double r4 = normal(engine);
    bb.xmin = r1;
    bb.xmax = r1 + r2/100;
    bb.ymin = r3;
    bb.ymax = r3 + r4/100;
    v.emplace_back(i, bb);
  }
  PRTree<int, 4> tree(v);
  double avg=0;
  auto start = std::chrono::system_clock::now();
  for (int i=0; i<10000;i++){
    auto out = tree(v[i].second);
    avg+=out.size();
  }
  auto end = std::chrono::system_clock::now();
  auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "duration = " <<  msec << "msec.\n";
  std::cout << avg/10000 << std::endl;

  start = std::chrono::system_clock::now();
  avg = 0.0;
  for (int i=0; i<10000;i++){
    for (auto& vv : v){
      if (vv.second(v[i].second)){
        avg += 1.0;
      }
    }
  }
  end = std::chrono::system_clock::now();
  msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "duration = " <<  msec << "msec.\n";
  std::cout << avg/10000 << std::endl;
  return x + y;
}
