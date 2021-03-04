#include <iostream>
#include "gtest/gtest.h"
#include "../src/parser.h"

using namespace std;
int TXTtoArrary( float* &points_array , string file_name , int num_feature = 3)
{
  ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  vector<float> temp_points;
  string c;

  while (!InFile.eof())
  {
      InFile >> c;

      temp_points.push_back(atof(c.c_str()));
  }
  points_array = new float[temp_points.size()];
  for (int i = 0 ; i < temp_points.size() ; ++i) {
    points_array[i] = temp_points[i];
  }

  InFile.close();  
  return temp_points.size() / num_feature;
  // printf("Done");
};

TEST(REDUCTION, __Inter_Div__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1,
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {12, \
                    20, \
                    36};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.reduce_interdiv(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};

TEST(REDUCTION, __Inter_Bank__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5, 
                      1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {12, \
                    20, \
                    36};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.reduce_interbank(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};


TEST(REDUCTION, __Seque_Naive__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {12, \
                    20, \
                    36};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.reduce_seqnaive(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};

TEST(REDUCTION, __Seque_Halve__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {12, \
                    20, \
                    36};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.reduce_seqhalve(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};

TEST(REDUCTION, __Seque_Unroll__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {12, \
                    20, \
                    36};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.reduce_sequnroll(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};

TEST(REDUCTION, __Complete_Unroll__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {12, \
                    20, \
                    36};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.complete_unroll(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};

TEST(REDUCTION, __Multiple_thread__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {12, \
                    20, \
                    36};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.multiple_thread(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};

TEST(REDUCTION, __TOTAL_SUM__) {
  // Two random matrices:
  float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                      2.0, 5.0, 4.0, 9.0, \
                      7.0, 6.0, 10., 13. };
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[1] = {68};   
  matrix A(3,4,ele_a);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(1,1);
  test_parser.total_sum(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
};


TEST(REDUCTION, __Points_Mean__) {
  //random 10000,3 array :
  std::string file_name ="/home/hova/cuda_template/test/data/points.txt";
  float* points_array;
  int in_num_points;
  in_num_points = TXTtoArrary(points_array,file_name); //(3 , 10000)
  // not used for reduction
  float ele_b[16] = {5.0, 8.0, 0.0, 6.6, 
                     4.0, 6.0, 3.5, 0.1, 
                     3.0, 7.0, 2.4, 9.5,
                     1.0, 0.5, 1.0, 7.4};

  float ele_c[3] = {5.00769048, 4.9880652, 4.96252771};   
  matrix A(3,in_num_points,points_array);
  matrix B(4,4,ele_b);
  parser test_parser(A, B);
  matrix C(3,1);
  test_parser.points_mean(C);

  EXPECT_FLOAT_EQ(ele_c[0], C.elements[0]);
  EXPECT_FLOAT_EQ(ele_c[1], C.elements[1]);
  EXPECT_FLOAT_EQ(ele_c[2], C.elements[2]);
};