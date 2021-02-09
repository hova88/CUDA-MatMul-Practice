#pragma once 

#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 1


#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

struct matrix
{
    int row , col;
    float* elements;
    matrix(int row , int col) :
    row(row) , col(col) {
        elements = new float[row * col]();
    }

    matrix(int row , int col , float* elements) :
    row(row) , col(col) , elements(elements){}
    ~matrix(){
        // delete[] elements_;
    }
};

class parser 
{
    private:
        matrix A,B;
    public:
        parser(matrix& a , matrix& b ) :
        A(a), B(b) {  }
        ~parser(){}
        // float*
        matrix naive(matrix& C);

};
