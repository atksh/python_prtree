#include <vector>
#include <random>
#include <time.h>
#include <omp.h>
#include "prtree.h"

int add(int x, int y){
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::vector<std::pair<double, BoundingBox>> v;
  std::uniform_real_distribution<> normal(0.0, 1.0);
  BoundingBox bb(0.0, 0.0, 0.0, 0.0);
  for (int i=0; i < 1000000; i++){
    double r0 = normal(engine);
    double r1 = normal(engine);
    double r2 = normal(engine);
    double r3 = normal(engine);
    double r4 = normal(engine);
    bb.xmin = r1;
    bb.xmax = r1 + r2/100;
    bb.ymin = r3;
    bb.ymax = r3 + r4/100;
    v.emplace_back(r0, bb);
  }
  PRTree<double, 6> tree(v);
  double avg=0;
  auto start = std::chrono::system_clock::now();
  //#pragma omp parallel for
  for (int i=0; i<1000;i++){
    auto out = tree(v[i].second);
    avg+=out.size();
  }
  auto end = std::chrono::system_clock::now();
  auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "duration = " <<  msec << "msec.\n";
  std::cout << avg/1000 << std::endl;
  return x + y;
}
