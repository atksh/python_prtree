cmake_minimum_required(VERSION 3.0)

if(WIN32)
  set(CMAKE_CXX_FLAGS "/O3 /pthread")
elseif(APPLE)
  set(CMAKE_CXX_FLAGS "-O3 -pthread")
else()
  set(CMAKE_CXX_FLAGS "-O3 -pthread")
endif()

project(PRTree)
file(GLOB MYCPP ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*)

option(SNAPPY_BUILD_TESTS "" OFF)
option(SNAPPY_BUILD_BENCHMARKS "" OFF)
option(SNAPPY_INSTALL "" OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/pybind11/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/cereal/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/snappy/)

pybind11_add_module(PRTree ${MYCPP})
set_target_properties(PRTree PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED TRUE
  CXX_EXTENSIONS FALSE
  POSITION_INDEPENDENT_CODE ON
	C_VISIBILITY_PRESET hidden
	CXX_VISIBILITY_PRESET hidden
)
set_target_properties(snappy PROPERTIES
  POSITION_INDEPENDENT_CODE ON
	C_VISIBILITY_PRESET hidden
	CXX_VISIBILITY_PRESET hidden
)

target_link_libraries(PRTree PRIVATE
  pybind11::module pybind11::lto pybind11::windows_extras
  cereal snappy
)

pybind11_extension(PRTree)
pybind11_strip(PRTree)
