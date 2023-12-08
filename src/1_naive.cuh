#pragma once

#include <cuda_runtime.h>

/* 
 * Implements a naive matrix multiplication kernel.
 * C = alpha * A * B + beta * C
 * A is MxK, B is KxN, C is MxN
 */
template <const uint BLOCKSIZE>
__global__ void matmul_naive(const float* A, const float* B, float* C, 
                             int M, int N, int K, float alpha, float beta) {
  int row = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE); // conitguous in memory
  int col = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  if (row >= M || col >= N) {return;}

  // if (row < M && col < N) {
  float sum = 0.0f;
  const float* A_row = A + row * K;
  const float* B_col = B + col;
  for (int i = 0; i < K; ++i) {
    sum += A_row[i] * B_col[i * N];
  }
  // C = α*(A@B)+β*C
  C[row * N + col] = alpha * sum + beta * C[row * N + col];
  // }
}


void matmul_naive_launcher(const float* A, const float* B, float* C, 
                           int M, int N, int K, float alpha, float beta) {

  dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  dim3 blockDim(32 * 32);

  matmul_naive<32><<< gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

