cmake_minimum_required(VERSION 3.10)
project(moana LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OptiX REQUIRED)

add_executable(moana src/main.cpp)
target_include_directories(moana PRIVATE "${OptiX_INCLUDE}")
target_include_directories(moana PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
