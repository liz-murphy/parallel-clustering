cmake_minimum_required(VERSION 2.8)
project(kmedoids)

set(CUDA_HOST_COMPILER /opt/local/bin/gcc)
set(CMAKE_BUILD_TYPE "Debug")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

find_package(CUDA REQUIRED)
find_package(OPENCV2 REQUIRED)
find_package(EIGEN3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV2_INCLUDE_DIRS})

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

set(CUDA_NVCC_FLAGS "-arch=sm_30 --pre-include ${CMAKE_SOURCE_DIR}/compatibility.h -DTHRUST_DEBUG" CACHE STRING "nvcc flags" FORCE)

set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE static)
CUDA_ADD_LIBRARY(kmedoids_lib src/kmedoids.cu)
ADD_EXECUTABLE(main src/main.cpp)
target_link_libraries(main kmedoids_lib ${OpenCV2_LIBRARIES})
