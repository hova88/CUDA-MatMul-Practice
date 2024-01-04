#pragma once

#include <cuda_runtime.h>

/* 
 * Implements a naive matrix multiplication kernel.
 * C = alpha * A * B + beta * C
 * A is MxK, B is KxN, C is MxN
 */
template <const uint BLOCKSIZE>
__global__ void matmul_global_mem_coalesce(const float* A, const float* B, float* C, 
                                           int M, int N, int K, float alpha, float beta) {
  int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE); // conitguous in memory
  if (row >= M || col >= N) {return;}

  float sum = 0.0f;
  for (int i = 0; i < K; ++i) {
    sum += A[row * K + i] * B[i * N + col];
  }
  // C = α*(A@B)+β*C
  C[row * N + col] = alpha * sum + beta * C[row * N + col];
}


void matmul_global_mem_coalesce_launcher(const float* A, const float* B, float* C, 
                           int M, int N, int K, float alpha, float beta) {

  dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  dim3 blockDim(32 * 32);

  matmul_global_mem_coalesce<32><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}
