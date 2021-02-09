#include <iostream>
#include "gtest/gtest.h"
#include "../matmul/parser.h"


TEST(MatrixMult, CudaMMSameAsNaive) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0,  \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6,  \
                      4.0, 6.0, 3.5, 0.1, \
                      3.0, 7.0, 2.4, 9.5, \
                      1.0, 0.5, 1.0, 7.4};
  
  float ele_c[12] = { 53, 82.5, 11.5, 60.5, \
                      51, 78.5, 36.1, 118.3,\
                     102,168.5, 58.0, 238.0};

  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,4);
  test_parser.naive(C);

  for (auto i = 0u; i < 12; ++i)
    EXPECT_FLOAT_EQ(ele_c[i], C.elements[i]);
}
