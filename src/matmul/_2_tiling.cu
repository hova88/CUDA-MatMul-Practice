/*
 * 5KK73
 * Eindhoven University of Technology
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

 #include <stdio.h>
 #include "../parser.h"

 ////////////////////////////////////////////////////////////////////////////////
 //! Matrix multiplication on the device: C = A * B
 //! wA is A's width and wB is B's width
 ////////////////////////////////////////////////////////////////////////////////
 __global__ void
 tiling_kernel( float* C, float* A, float* B,  int interDim)
 {

     // Declaration of the shared memory array As used to
     // store the sub-matrix of A
    __shared__ float As[2 * 6];
 
     // Declaration of the shared memory array Bs used to
     // store the sub-matrix of B
    __shared__ float Bs[2 * 8];
 
    // Index of the first sub-matrix of A&B processed by the block
    int A_head = threadIdx.x * interDim; // {0,1,2}  *   4
    int B_head = threadIdx.y ; // {0,1,2,3}
    // Index of the last sub-matrix of A processed by the block
    int A_tail = A_head + 4 -1 ;  // width of As = 2
    // Step size used to iterate through the sub-matrices of A&B
    int A_step = 2;
    int B_step = 2 * interDim ; 

     // Csub is used to store the element of the block sub-matrix
     // that is computed by the thread
     float Csub = 0;
 
     // Loop over all the sub-matrices of A and B
     // required to compute the block sub-matrix
     for (int a = A_head, b = B_head;
              a <= A_tail;
              a += A_step, b += B_step) {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[0 + a] = A[a];
        As[1 + a]  = A[a + 1];    

        Bs[0 + b] = B[b];
        Bs[1 * interDim + b] = B[b + 1 * interDim];
        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 1) {
        //     printf("-b: %d ; -Bs : [ %f , %f ] ; threadIdx.x : %d ; threadIdx.y : %d ; \n" ,\
        //              b , Bs[0] , Bs[1] , threadIdx.x , threadIdx.y );
        // }
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        // for (int k = 0; k < 12; ++k)
            Csub += As[0 + a]*Bs[0 + b] + As[1 + a] * Bs[1 * interDim + b] ;
         // Synchronize to make sure that the preceding
         // computation is done before loading two new
         // sub-matrices of A and B in the next iteration
         __syncthreads();
     }
 
     // Write the block sub-matrix to device memory;
     // each thread writes one element
     C[threadIdx.x * blockDim.y + threadIdx.y ] = Csub;
 }
 
void parser::matmul_tiling( matrix& C) {
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
    tiling_kernel<<< 1 , block_size , 2 * sizeof(float)>>>(dev_c, dev_a, dev_b , 4);

    cudaDeviceSynchronize();


    cudaMemcpy(C.elements, dev_c, C.row * C.col * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return;
}