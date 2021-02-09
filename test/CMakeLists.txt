add_subdirectory(gtest)

include_directories(./gtest/googletest/include)

set(test_src
  run_all.cc
  matrix_mult_test.cc
)

add_executable(test_matmul ${test_src})

target_link_libraries(test_matmul
  gtest
  ${LINK_LIBRARY} 
  ${PROJECT_LIB_NAME}
  # ${DG_LIB_NAME}
  # ${CMAKE_THREAD_LIBS_INIT}
  # ${MATH_LIBS}
)

add_test(
  NAME tests
  COMMAND test_matmul
)
