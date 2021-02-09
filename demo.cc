#include <iostream>

#include "matmul/parser.h" // important to include .cu file, not header file

int main() {

    float ele_a[12] = {8.0, 3.0, 0.0, 1.0, \
                       2.0, 5.0, 4.0, 9.0, \
                       7.0, 6.0, 10., 13. };
    
    float ele_b[16] = {5.0, 8.0, 0.0, 6.6, \
                       4.0, 6.0, 3.5, 0.1, \
                       3.0, 7.0, 2.4, 9.5, \
                       1.0, 0.5, 1.0, 7.4};
    
    matrix A(3,4,ele_a);
    matrix B(4,4,ele_b);
    parser test_parser(A, B);
    matrix C(3,4);
    test_parser.naive(C);
    
	return 1;
}