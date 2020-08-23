add_executable(moana-app main.cpp)
target_link_libraries(moana-app moana)

set_target_properties(moana-app PROPERTIES OUTPUT_NAME moana)
