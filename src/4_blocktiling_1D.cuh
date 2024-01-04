#pragma once

#include <cuda_runtime.h>
#include <cassert>

/* 
 * C = alpha * A * B + beta * C
 * A is MxK, B is KxN, C is MxN
 */
template <const int BM, const int BN, const int BK, const int TM>
__global__ void matmul_smem_blocktiling1d(const float* A, const float* B, float* C, 
                int M, int N, int K, float alpha, float beta) {
  const uint cCol = blockIdx.x;
  const uint cRow = blockIdx.y;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadRow = threadIdx.x / BN;
  const int threadCol = threadIdx.x % BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerRowA = threadIdx.x / BK; // warp-level GMEM coalescing
  const uint innerColA = threadIdx.x % BK;

  const uint innerRowB = threadIdx.x / BN; // warp-level GMEM coalescing
  const uint innerColB = threadIdx.x % BN; 

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    int index = (threadRow * TM + resIdx) * N + threadCol;

    C[index] = alpha * threadResults[resIdx] + beta * C[index];

  }
}

void matmul_smem_blocktiling1d_launcher(const float* A, const float* B, float* C, 
                                    int M, int N, int K, float alpha, float beta) {

  // SMEM cache size of BM*BK + BN*BK = 64*8 + 64*8 = 1024 floats, 
  // for a total of 4KB per block.
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8; // Tile size of M
  dim3 gridDim((N + BN-1) / BN, (M + BM-1) / BM);
  dim3 blockDim((BM * BN) / TM);

  matmul_smem_blocktiling1d<BM,BN,BK,TM><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

