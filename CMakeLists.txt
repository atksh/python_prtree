cmake_minimum_required(VERSION 3.5)

# Phase 0: Profiling and Sanitizer Infrastructure
option(ENABLE_PROFILING "Build with profiling symbols and frame pointers" OFF)
option(CI_MODE "CI environment - enables mandatory sanitizers" OFF)
option(ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)

if(WIN32)
  set(CMAKE_CXX_FLAGS "/O2 /EHsc")
elseif(APPLE)
  set(CMAKE_CXX_FLAGS "-O3 -pthread")
else()
  set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g -lprofiler -ltcmalloc -DMY_DEBUG")
  set(CMAKE_CXX_FLAGS "-O3 -pthread")
endif()

# Profiling support
if(ENABLE_PROFILING)
    message(STATUS "Building with profiling support")
    add_compile_options(-g -fno-omit-frame-pointer)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fno-inline-functions)
    endif()
endif()

# Sanitizer support (mandatory in CI mode)
if(CI_MODE OR ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
elseif(ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
elseif(ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
endif()

project(PRTree)

# Source files
set(PRTREE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/bindings/python_bindings.cc
)

# Include directories
set(PRTREE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp  # Backward compatibility during migration
)

option(SNAPPY_BUILD_TESTS "" OFF)
option(SNAPPY_BUILD_BENCHMARKS "" OFF)
option(SNAPPY_INSTALL "" OFF)

option(SKIP_PERFORMANCE_COMPARISON "" ON)
option(BUILD_TESTS "" OFF)
option(BUILD_SANDBOX "" OFF)
option(BUILD_DOC "" OFF)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/pybind11/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/cereal/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third/snappy/)

pybind11_add_module(PRTree ${PRTREE_SOURCES})

# Include directories
target_include_directories(PRTree PRIVATE
    ${PRTREE_INCLUDE_DIRS}
)

set_target_properties(snappy PROPERTIES
  POSITION_INDEPENDENT_CODE ON
	C_VISIBILITY_PRESET hidden
	CXX_VISIBILITY_PRESET hidden
)

target_link_libraries(PRTree PRIVATE
  cereal
  snappy
)

set_target_properties(PRTree PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED TRUE
  CXX_EXTENSIONS FALSE
  POSITION_INDEPENDENT_CODE ON
	C_VISIBILITY_PRESET hidden
	CXX_VISIBILITY_PRESET hidden
  INTERPROCEDURAL_OPTIMIZATION TRUE
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}"
  ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}"
  ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}"
)

# Phase 0: Benchmark targets
if(BUILD_BENCHMARKS)
    message(STATUS "Building performance benchmarks")

    # Construction benchmark
    add_executable(benchmark_construction benchmarks/cpp/benchmark_construction.cpp)
    target_include_directories(benchmark_construction PRIVATE ${PRTREE_INCLUDE_DIRS})
    target_link_libraries(benchmark_construction PRIVATE cereal snappy)
    set_target_properties(benchmark_construction PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
    )

    # Query benchmark
    add_executable(benchmark_query benchmarks/cpp/benchmark_query.cpp)
    target_include_directories(benchmark_query PRIVATE ${PRTREE_INCLUDE_DIRS})
    target_link_libraries(benchmark_query PRIVATE cereal snappy)
    set_target_properties(benchmark_query PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
    )

    # Multithreaded benchmark
    add_executable(benchmark_parallel benchmarks/cpp/benchmark_parallel.cpp)
    target_include_directories(benchmark_parallel PRIVATE ${PRTREE_INCLUDE_DIRS})
    target_link_libraries(benchmark_parallel PRIVATE cereal snappy)
    set_target_properties(benchmark_parallel PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
    )

    # Stress test
    add_executable(stress_test_concurrent benchmarks/cpp/stress_test_concurrent.cpp)
    target_include_directories(stress_test_concurrent PRIVATE ${PRTREE_INCLUDE_DIRS})
    target_link_libraries(stress_test_concurrent PRIVATE cereal snappy pthread)
    set_target_properties(stress_test_concurrent PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
    )
endif()
