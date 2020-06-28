set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_CPP_STANDARD -std=c++1z)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)


project(mapmatching)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


add_subdirectory(pybind11)
set(CMAKE_CXX_FLAGS "-O3 -Ofast -ffast-math -Wall -fopenmp -lpthread -fPIC -shared -march=native -flto -static-libgcc -static-libstdc++")
pybind11_add_module(_math src/main.cpp)
