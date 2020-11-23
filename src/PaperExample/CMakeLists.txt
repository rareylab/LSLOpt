add_executable(paper_example PaperExample.cpp LJLike_0.cpp)

set_property(TARGET paper_example PROPERTY CXX_STANDARD 14)

if(MSVC)
  set_target_properties(paper_example PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif(MSVC)

target_link_libraries (paper_example LSLOpt)

add_test(NAME paper_example_test COMMAND paper_example)
