add_library(ptx_files OBJECT kernel.cu)
target_include_directories(ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:ptx_files>"
  "-DVARIABLE_NAME=mainRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS ptx_files
  COMMENT "Convert ptx files to header file"
)

# fixme: macro
add_library(shadow_ptx_files OBJECT shadow_ray_kernel.cu)
target_include_directories(shadow_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(shadow_ptx_files PRIVATE "${PROJECT_CURRENT_SOURCE_DIR}")
target_include_directories(shadow_ptx_files PRIVATE "${OptiX_INCLUDE}")
target_include_directories(shadow_ptx_files PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(shadow_ptx_files PRIVATE "${PROJECT_SOURCE_DIR}/ext/optix_sdk")
set_property(TARGET shadow_ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET shadow_ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(shadow_ptx_files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(
  shadow_ray_ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/shadow_ray_kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:shadow_ptx_files>"
  "-DVARIABLE_NAME=shadowRaySource"
  "-DNAMESPACE=PTX"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS shadow_ptx_files
  COMMENT "Convert ptx files to header file"
)

