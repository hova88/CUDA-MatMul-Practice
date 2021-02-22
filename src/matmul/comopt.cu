// perform outer product instead of inner product.//

/***
matrix A is stored in shared memory, but matrix B and C are stored in registers.
The outer product does not require sharing of matrix B and matrix C, 
therefore, each thread only stores one element of B and one column of the tile of C in the register.
The "computation-to-memory ratio" of the outer product is the same as the inner product.
***/


/*
 * 5KK73
 * Eindhoven University of Technology
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

 #include <stdio.h>
 #include "parser.h"

 ////////////////////////////////////////////////////////////////////////////////
 //! Matrix multiplication on the device: C = A * B
 //! wA is A's width and wB is B's width
 ////////////////////////////////////////////////////////////////////////////////
 __global__ void
 comopt_kernel( float* C, float* A, float* B,  int interDim)
 {

     // Declaration of the shared memory array As used to
     // store the sub-matrix of A
    __shared__ float As_trans[12];
    // As[threadIdx.x * blockDim.y + threadIdx.y] = A[threadIdx.x * blockDim.y + threadIdx.y];
    As_trans[threadIdx.y * blockDim.x + threadIdx.x] = A[threadIdx.x * blockDim.y + threadIdx.y]; //使用转置，让索引之间里的更近，加速访问[coalescing]
    
    __syncthreads();
    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    float cv[12] =  {0,0,0,0, \
                     0,0,0,0, \
                     0,0,0,0};

    // 使用外循环的方式来替代内循环
    // 1.提高shared memory的利用率
    // 2.简化流处理器的计算指令
    for (int i = 0 ; i < interDim; ++i) {
        cv[threadIdx.x * interDim + threadIdx.y] += B[i * interDim +threadIdx.y] \
                                                 * As_trans[i * blockDim.x + threadIdx.x];
    }
    __syncthreads();
 
     // Write the block sub-matrix to device memory;
     // each thread writes one element
    C[threadIdx.x * blockDim.y + threadIdx.y] = cv[threadIdx.x * blockDim.y + threadIdx.y];
 }
 
 matrix parser::comopt( matrix& C) {
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
    comopt_kernel<<< 1 , block_size , 2 * sizeof(float)>>>(dev_c, dev_a, dev_b , 4);

    cudaDeviceSynchronize();

    cudaMemcpy(C.elements, dev_c, C.row * C.col * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return C;
}