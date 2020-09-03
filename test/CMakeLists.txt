add_subdirectory(${PROJECT_SOURCE_DIR}/ext/Catch2 Catch2)
add_executable(test
  test.cpp

  test_obj_parser.cpp
)
target_link_libraries(test Catch2::Catch2 moana)
