add_executable(example Example.cpp)

if(${CMAKE_CXX_COMPILER_ID} MATCHES GNU AND ${LSLOpt_WITH_COVERAGE})
  message(STATUS "Enabled coverage build")
  set_target_properties(example PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -coverage"
                        LINK_FLAGS "-coverage")
endif()

set_property(TARGET example PROPERTY CXX_STANDARD 14)

if(MSVC)
  set_target_properties(example PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif(MSVC)

target_link_libraries (example LSLOpt)

add_test(NAME example_test COMMAND example)
add_dependencies(build_and_test example)