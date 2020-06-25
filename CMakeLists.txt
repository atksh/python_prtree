cmake_minimum_required(VERSION 3.10)
project(mapmatching)

add_subdirectory(pybind11)
pybind11_add_module(_math src/main.cpp)
