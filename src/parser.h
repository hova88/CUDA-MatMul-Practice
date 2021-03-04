#pragma once 

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>

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
        A(a), B(b) { /*printf("build parser...\n");*/ }
        ~parser(){}

        void matmul_naive(matrix& C);
        void matmul_tiling(matrix& C);
        void matmul_coalescing(matrix& C);
        void matmul_comopt(matrix& C);
        void matmul_unroll(matrix& C);

        void reduce_interdiv(matrix& C);
        void reduce_interbank(matrix& C);
        void reduce_seqnaive(matrix& C);
        void reduce_seqhalve(matrix& C);
        void reduce_sequnroll(matrix& C);
        void complete_unroll(matrix& C);
        void multiple_thread(matrix& C);
        void total_sum(matrix& C);
        void points_mean(matrix& C);

};
