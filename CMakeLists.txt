cmake_minimum_required(VERSION 3.18)
project(moana LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)

add_library(moana src/driver.cpp)
target_include_directories(moana PUBLIC include)
target_include_directories(moana PUBLIC "${OptiX_INCLUDE}")
target_link_libraries(moana CUDA::cudart dl)

add_subdirectory(optix)
add_dependencies(moana ptx_headers)

add_subdirectory(app)
