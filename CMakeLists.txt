set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_CPP_STANDARD -std=c++1z)
cmake_minimum_required(VERSION 3.8)

project(PRTree)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


add_subdirectory(pybind11)
pybind11_add_module(PRTree src/main.cc)
set(CMAKE_CXX_FLAGS "-O3 -Ofast -ffast-math -Wall -fopenmp -lpthread -fPIC -shared -mtune=native -march=native -flto -static-libgcc -static-libstdc++")

