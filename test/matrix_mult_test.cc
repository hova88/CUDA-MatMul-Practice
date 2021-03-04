#include <iostream>
#include "gtest/gtest.h"
#include "../src/parser.h"


TEST(MatrixMult, __Naive__) {
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
  test_parser.matmul_naive(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
  EXPECT_FLOAT_EQ(ele_c[3], C.elements[3]);
  EXPECT_FLOAT_EQ(ele_c[4], C.elements[4]);
  EXPECT_FLOAT_EQ(ele_c[5], C.elements[5]);
  EXPECT_FLOAT_EQ(ele_c[6], C.elements[6]);
  EXPECT_FLOAT_EQ(ele_c[7], C.elements[7]);
  EXPECT_FLOAT_EQ(ele_c[8], C.elements[8]);
  EXPECT_FLOAT_EQ(ele_c[9], C.elements[9]);
  EXPECT_FLOAT_EQ(ele_c[10], C.elements[10]);
  EXPECT_FLOAT_EQ(ele_c[11], C.elements[11]);
};

TEST(MatrixMult, __Tiling__) {
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

  test_parser.matmul_tiling(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
  EXPECT_FLOAT_EQ(ele_c[3], C.elements[3]);
  EXPECT_FLOAT_EQ(ele_c[4], C.elements[4]);
  EXPECT_FLOAT_EQ(ele_c[5], C.elements[5]);
  EXPECT_FLOAT_EQ(ele_c[6], C.elements[6]);
  EXPECT_FLOAT_EQ(ele_c[7], C.elements[7]);
  EXPECT_FLOAT_EQ(ele_c[8], C.elements[8]);
  EXPECT_FLOAT_EQ(ele_c[9], C.elements[9]);
  EXPECT_FLOAT_EQ(ele_c[10], C.elements[10]);
  EXPECT_FLOAT_EQ(ele_c[11], C.elements[11]);
};

TEST(MatrixMult, __Coalescing__) {
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

  test_parser.matmul_coalescing(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
  EXPECT_FLOAT_EQ(ele_c[3], C.elements[3]);
  EXPECT_FLOAT_EQ(ele_c[4], C.elements[4]);
  EXPECT_FLOAT_EQ(ele_c[5], C.elements[5]);
  EXPECT_FLOAT_EQ(ele_c[6], C.elements[6]);
  EXPECT_FLOAT_EQ(ele_c[7], C.elements[7]);
  EXPECT_FLOAT_EQ(ele_c[8], C.elements[8]);
  EXPECT_FLOAT_EQ(ele_c[9], C.elements[9]);
  EXPECT_FLOAT_EQ(ele_c[10], C.elements[10]);
  EXPECT_FLOAT_EQ(ele_c[11], C.elements[11]);
};

TEST(MatrixMult, __Computation_Omp__) {
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

  test_parser.matmul_comopt(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
  EXPECT_FLOAT_EQ(ele_c[3], C.elements[3]);
  EXPECT_FLOAT_EQ(ele_c[4], C.elements[4]);
  EXPECT_FLOAT_EQ(ele_c[5], C.elements[5]);
  EXPECT_FLOAT_EQ(ele_c[6], C.elements[6]);
  EXPECT_FLOAT_EQ(ele_c[7], C.elements[7]);
  EXPECT_FLOAT_EQ(ele_c[8], C.elements[8]);
  EXPECT_FLOAT_EQ(ele_c[9], C.elements[9]);
  EXPECT_FLOAT_EQ(ele_c[10], C.elements[10]);
  EXPECT_FLOAT_EQ(ele_c[11], C.elements[11]);

};

TEST(MatrixMult, __Unroll__) {
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
  test_parser.matmul_unroll(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
  EXPECT_FLOAT_EQ(ele_c[3], C.elements[3]);
  EXPECT_FLOAT_EQ(ele_c[4], C.elements[4]);
  EXPECT_FLOAT_EQ(ele_c[5], C.elements[5]);
  EXPECT_FLOAT_EQ(ele_c[6], C.elements[6]);
  EXPECT_FLOAT_EQ(ele_c[7], C.elements[7]);
  EXPECT_FLOAT_EQ(ele_c[8], C.elements[8]);
  EXPECT_FLOAT_EQ(ele_c[9], C.elements[9]);
  EXPECT_FLOAT_EQ(ele_c[10], C.elements[10]);
  EXPECT_FLOAT_EQ(ele_c[11], C.elements[11]);
};