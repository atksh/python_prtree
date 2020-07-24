cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_CPP_STANDARD -std=c++1z)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS "-O3 -Ofast -ffast-math -ftemplate-backtrace-limit=0 -flto")

project(PRTree)
find_package(Threads REQUIRED)

add_subdirectory(./pybind11/)
pybind11_add_module(PRTree cpp/main.cc cpp/prtree.h)
target_link_libraries(PRTree PRIVATE Threads::Threads)

