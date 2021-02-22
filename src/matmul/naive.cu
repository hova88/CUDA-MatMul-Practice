#include "parser.h"
#include <stdio.h>


__global__ void
naive_kernel( float* C, float* A, float* B, int interDim)
{
  // Matrix A index
  
  int A_head = threadIdx.x * interDim;// {0,1,2}  *   4
  int B_head = threadIdx.y ;// {0,1,2,3}

  // Accumulate row i of A and column j of B
  float element = 0.0;
  for(int k=0; k<interDim; ++k){ 
    element +=  A[A_head + k] * B[ B_head + interDim * k];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[threadIdx.x * blockDim.y + threadIdx.y ] = element;
}

matrix parser::naive( matrix& C) {
	float* dev_a;
	cudaMalloc(&dev_a, A.row * A.col * sizeof(float));
	cudaMemcpy(dev_a, A.elements,  A.row * A.col * sizeof(float), cudaMemcpyHostToDevice);
    
  float* dev_b;
  cudaMalloc(&dev_b, B.row * B.col * sizeof(float));
	cudaMemcpy(dev_b, B.elements,  B.row * B.col * sizeof(float), cudaMemcpyHostToDevice);
    
  float* dev_c;
	cudaMalloc(&dev_c, C.row * C.col * sizeof(float));
	
    
    dim3 block_size(3,4);
    // dim3 grid_size(1);
    naive_kernel<<< 1 , block_size >>>(dev_c, dev_a, dev_b , 4);
    
    cudaDeviceSynchronize();
    

    cudaMemcpy(C.elements, dev_c, C.row * C.col * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return C;
}