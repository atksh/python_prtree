cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(Boost_USE_STATIC_LIBS ON)

if(WIN32)
  set(CMAKE_CXX_FLAGS "/O3 /pthread /GL /LTCG")
elseif(APPLE)
  set(CMAKE_CXX_FLAGS "-O3 -pthread -flto")
else()
  set(CMAKE_CXX_FLAGS "-O3 -pthread -Wall -flto")
endif()

project(PRTree)
file(GLOB MYCPP ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*)

option(SNAPPY_BUILD_TESTS "" OFF)
option(SNAPPY_BUILD_BENCHMARKS "" OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/pybind11/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/cereal/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/snappy/)

pybind11_add_module(PRTree ${MYCPP})
target_link_libraries(PRTree PRIVATE 
  pybind11::module pybind11::lto pybind11::windows_extras
  cereal snappy
)

pybind11_extension(PRTree)
pybind11_strip(PRTree)
