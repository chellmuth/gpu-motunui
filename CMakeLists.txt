cmake_minimum_required(VERSION 3.10)
project(moana LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OptiX REQUIRED)

add_library(libmoana src/driver.cpp)
target_include_directories(libmoana PUBLIC include)
target_include_directories(libmoana PUBLIC "${OptiX_INCLUDE}")
target_include_directories(libmoana PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

add_subdirectory(app)
