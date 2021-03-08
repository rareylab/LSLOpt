function(add_paper_example executable file)
  add_executable("${executable}" ModelSystem.cpp LJLike_0.cpp "${file}")

  set_property(TARGET "${executable}" PROPERTY CXX_STANDARD 14)

  if(MSVC)
    set_target_properties("${executable}" PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  endif(MSVC)

  target_link_libraries ("${executable}" LSLOpt)

  add_test(NAME "${executable}_test" COMMAND "${executable}")
endfunction()

add_paper_example(paper_example PaperExample.cpp)
add_paper_example(paper_example_plot1d Plot1d.cpp)
add_paper_example(paper_example_plot2d Plot2d.cpp)
add_paper_example(paper_example_plot2d_opt Plot2dOpt.cpp)
