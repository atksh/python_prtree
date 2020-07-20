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

set(CMAKE_CXX_FLAGS "-O3 -Ofast -ffast-math -fPIC -shared -Wall -fopenmp -mtune=native -march=native -flto -static-libgcc -static-libstdc++")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -ffast-math -fPIC -shared -Wall -fopenmp -mtune=native -march=native -flto -static-libgcc -static-libstdc++")

project(PRTree)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


add_subdirectory(pybind11)
pybind11_add_module(PRTree src/main.cc)
target_link_libraries(PRTree PRIVATE OpenMP::OpenMP_CXX)

