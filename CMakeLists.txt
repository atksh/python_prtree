cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_CPP_STANDARD -std=c++1z)

if(MSVC)
  set(CMAKE_CXX_FLAGS "/O2 /pthread")
else()
  set(CMAKE_CXX_FLAGS "-O2 -pthread")
endif()

project(PRTree)

add_subdirectory(./pybind11/)
pybind11_add_module(PRTree cpp/main.cc cpp/prtree.h)

