add_subdirectory(gtest)

include_directories(./gtest/googletest/include)

set(test_src
  run_all.cc
  matrix_mult_test.cc
  reduction_test.cc
)

add_executable(test_all ${test_src})

target_link_libraries(test_all
  gtest
  ${LINK_LIBRARY} 
  ${MATMUL_LIB_NAME}
  ${REDUCTION_LIB_NAME}
)


add_test(
  NAME tests
  COMMAND test_all
)
