cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_CPP_STANDARD -std=c++1z)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-O3 -Ofast -ffast-math -fPIC -Wall -mtune=native -march=native -flto -ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -ffast-math -fPIC -Wall -mtune=native -march=native -flto -ggdb3")

project(PRTree)
find_package(OpenMP REQUIRED)

add_subdirectory(pybind11)
add_subdirectory(mimalloc)
pybind11_add_module(PRTree src/main.cc)
target_link_libraries(PRTree PRIVATE profiler OpenMP::OpenMP_CXX mimalloc-static)

target_link_libraries(PRTree PUBLIC)

