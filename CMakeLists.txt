cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_CPP_STANDARD -std=c++1z)

if(MSVC)
  set(CMAKE_C_FLAGS "/O2 /pthread /GL /LTCG")
  set(CMAKE_CXX_FLAGS "/O2 /pthread /GL /LTCG")
else()
  set(CMAKE_C_FLAGS "-O2 -pthread -flto")
  set(CMAKE_CXX_FLAGS "-O2 -pthread -flto")
endif()

project(PRTree)

add_subdirectory(./pybind11/)
add_subdirectory(./mimalloc/)

pybind11_add_module(PRTree cpp/main.cc cpp/prtree.h)
target_include_directories(PRTree PUBLIC mimalloc/include mimalloc/src)
target_link_libraries(PRTree PRIVATE mimalloc-static)

