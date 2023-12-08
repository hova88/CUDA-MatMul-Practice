#pragma once

#include <cuda_runtime.h>

/* 
 * Implements a naive matrix multiplication kernel.
 * C = alpha * A * B + beta * C
 * A is MxK, B is KxN, C is MxN
 */
template <const uint BLOCKSIZE>
__global__ void matmul_shared_mem_block(const float* A, const float* B, float* C, 
                             int M, int N, int K, float alpha, float beta) {
  int row = blockIdx.x;
  int col = blockIdx.y;

  // allocate buffer for current block in shared memory
  // shared memory is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the thread's row and column indices of the current block
  int thread_row = threadIdx.x % BLOCKSIZE; // threads_row contiguous within a warp, but our memory useage for it is not contiguous. 
  int thread_col = threadIdx.x / BLOCKSIZE;  
  if (row * BLOCKSIZE + thread_row >= M || col * BLOCKSIZE + thread_col >= N) {return;}

  // advance pointers to the starting position
  A += row * BLOCKSIZE * K; // (M,K) = (row,0)
  B += BLOCKSIZE * col;     // (K,N) = (0,col)
  C += row * BLOCKSIZE * N + BLOCKSIZE * col; // (M,N) = (row,col)

  float sum = 0.0f;
  for (int i=0; i < K; i += BLOCKSIZE) {
    // load block of A and B from global memory into shared memory
    As[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
    Bs[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];
    __syncthreads();

    // compute partial sum
    for (int j=0; j < BLOCKSIZE; ++j) {
      sum += As[thread_row * BLOCKSIZE + j] * Bs[j * BLOCKSIZE + thread_col];
    }
    __syncthreads();

    // advance pointers to the next block
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
  }

  // C = α*(A@B)+β*C
  C[thread_row * N + thread_col] = alpha * sum + beta * C[thread_row * N + thread_col];
  
}


void matmul_shared_mem_block_launcher(const float* A, const float* B, float* C, 
                                    int M, int N, int K, float alpha, float beta) {

  dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  dim3 blockDim(32 * 32);

  matmul_shared_mem_block<32><<< gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

