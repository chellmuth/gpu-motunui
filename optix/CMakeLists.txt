add_library(ptx_files OBJECT kernel.cu)
target_include_directories(ptx_files PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(ptx_files PUBLIC "${OptiX_INCLUDE}")
target_include_directories(ptx_files PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
set_property(TARGET ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)

add_custom_target(
  ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${PROJECT_SOURCE_DIR}/src/kernel.hpp"
  "-DPTX_FILE=$<TARGET_OBJECTS:ptx_files>"
  "-DVARIABLE_NAME=ptxSource"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${PROJECT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS ptx_files
  COMMENT "Convert ptx files to header file"
)

