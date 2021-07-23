cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_CPP_STANDARD -std=c++1z)

if(MSVC)
  set(CMAKE_C_FLAGS "/O3 /pthread /GL /LTCG")
  set(CMAKE_CXX_FLAGS "/O3 /pthread /GL /LTCG")
elseif(APPLE)
  set(CMAKE_C_FLAGS "-O2 -pthread -flto")
  set(CMAKE_CXX_FLAGS "-O2 -pthread -flto")
else()
  set(CMAKE_C_FLAGS "-O3 -pthread -flto")
  set(CMAKE_CXX_FLAGS "-O3 -pthread -flto")
endif()

project(PRTree)
file(GLOB MYCPP "./cpp/*")

add_subdirectory(./third/pybind11/)
add_subdirectory(./third/cereal/)

pybind11_add_module(PRTree ${MYCPP})
target_link_libraries(PRTree PRIVATE cereal)

