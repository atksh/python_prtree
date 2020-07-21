cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_CPP_STANDARD -std=c++1z)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-O3 -Ofast -ffast-math -fPIC -Wall -mtune=native -march=native -flto")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -ffast-math -fPIC -Wall -mtune=native -march=native -flto")

project(PRTree)

add_subdirectory(./pybind11/)
add_subdirectory(./mimalloc/)
pybind11_add_module(PRTree cpp/main.cc cpp/prtree.h)
target_link_libraries(PRTree PRIVATE OpenMP::OpenMP_CXX mimalloc)