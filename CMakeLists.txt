cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_CPP_STANDARD -std=c++1z)

if(MSVC)
  set(CMAKE_C_FLAGS "/O2 /pthread /GL /LTCG")
  set(CMAKE_CXX_FLAGS "/O2 /pthread /GL /LTCG")
else()
  set(CMAKE_C_FLAGS "-O2 -pthread -flto")
  set(CMAKE_CXX_FLAGS "-O2 -pthread -flto -I./mimalloc/include/")
endif()

project(PRTree)
file(GLOB MYCPP "cpp/*")

set(mi_sources
    mimalloc/src/stats.c
    mimalloc/src/random.c
    mimalloc/src/os.c
    mimalloc/src/arena.c
    mimalloc/src/region.c
    mimalloc/src/segment.c
    mimalloc/src/page.c
    mimalloc/src/alloc.c
    mimalloc/src/alloc-aligned.c
    mimalloc/src/alloc-posix.c
    mimalloc/src/heap.c
    mimalloc/src/options.c
    mimalloc/src/init.c)


add_subdirectory(./pybind11/)
add_library(mimalloc-static STATIC ${mi_sources})
set_target_properties(mimalloc-static PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(mimalloc-static PUBLIC ./mimalloc/include/)

pybind11_add_module(PRTree ${MYCPP})
target_link_libraries(PRTree PRIVATE mimalloc-static)

