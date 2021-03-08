function(build_test TEST_NAME TEST_FILE FLAGS)
  set(TEST_EXECUTABLE "${TEST_NAME}_tester")
  add_executable(${TEST_EXECUTABLE} main.cpp ${TEST_FILE})

  set_property(TARGET ${TEST_EXECUTABLE} PROPERTY CXX_STANDARD 14)
  if(${CMAKE_CXX_COMPILER_ID} MATCHES GNU AND ${LSLOpt_WITH_COVERAGE})
    message(STATUS "Enabled coverage build")
    set_target_properties("${TEST_EXECUTABLE}" PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -coverage"
                          LINK_FLAGS "-coverage")
  endif()

  if(MSVC)
    set_target_properties("${TEST_EXECUTABLE}" PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  endif(MSVC)

  target_link_libraries (${TEST_EXECUTABLE} LSLOpt)
  target_link_libraries (${TEST_EXECUTABLE} gtest_main)

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_EXECUTABLE} ${FLAGS})
  add_dependencies(build_and_test ${TEST_EXECUTABLE})
endfunction()

build_test(MinusInfinity MinusInfinityTest.cpp "")
build_test(Unbounded UnboundedTest.cpp "")
build_test(WrongDerivative WrongDerivativeTest.cpp "")
build_test(WrongBounds WrongBoundsTest.cpp "")
build_test(Reparameterize ReparameterizeTest.cpp "")
build_test(Rosenbrock RosenbrockTest.cpp "")
build_test(RosenbrockDebug RosenbrockDebugTest.cpp "")
build_test(Termination TerminationTests.cpp "")
build_test(RosenbrockMultiprecision RosenbrockMultiprecisionTest.cpp "--gtest_filter=RosenbrockMultiprecisionTester*Default/*")
