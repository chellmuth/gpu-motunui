cmake_minimum_required(VERSION 3.18)
project(moana LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)

add_library(moana
  src/driver.cpp
  src/scene.cpp

  src/core/camera.cpp
  src/core/transform.cpp

  src/io/image.cpp

  src/parsers/obj_parser.cpp
  src/parsers/string_util.cpp

  src/scene/archive.cpp
  src/scene/as_arena.cpp
  src/scene/bay_cedar_a1_element.cpp
  src/scene/coastline_element.cpp
  src/scene/container.cpp
  src/scene/coral_element.cpp
  src/scene/dunes_a_element.cpp
  src/scene/dunes_b_element.cpp
  src/scene/element.cpp
  src/scene/gas.cpp
  src/scene/hibiscus_element.cpp
  src/scene/ias.cpp
  src/scene/instances_bin.cpp
  src/scene/ironwood_a1_element.cpp
  src/scene/mountain_a_element.cpp
  src/scene/mountain_b_element.cpp
)
target_include_directories(moana PUBLIC include)
target_include_directories(moana PUBLIC "${OptiX_INCLUDE}")
target_include_directories(moana PRIVATE src)
target_include_directories(moana PRIVATE ext/tinyexr)
target_compile_definitions(moana PRIVATE MOANA_ROOT="${MOANA_ROOT}")
target_link_libraries(moana CUDA::cudart dl)

add_subdirectory(optix)
add_dependencies(moana ptx_headers)

add_subdirectory(app)
