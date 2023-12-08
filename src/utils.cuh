#pragma once

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

// This macro definition simplifies error handling for CUDA API functions by printing an error message
// with the location of the error and aborting the program.
#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    fprintf(stderr, "Cuda failure: %s at line %d in file %s error status: %d\n", \
            cudaGetErrorString(status), __LINE__, __FILE__, status); \
    abort();                                                      \
  }                                                               \
}


void CudaDeviceInfo() {

  int count = 0;
  cudaDeviceProp prop;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);

  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Compute Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Memory Bus Width: %d\n", prop.memoryBusWidth);
    printf("  ----------------\n");
    printf("  Total Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Shared Memory Per Block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  Shared Memory Per MultiProcessor: %luKB\n", prop.sharedMemPerMultiprocessor >> 10);
    printf("  Total Constant Memory: %luKB\n", prop.totalConstMem >> 10);
    printf("  ----------------\n");
    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Regsters Per Block: %d\n", prop.regsPerBlock);
    printf("  Total MultiProcessors: %d\n", prop.multiProcessorCount);
    printf("  Max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max Regsters Per MultiProcessor: %d\n", prop.regsPerMultiprocessor);
    printf("  ----------------\n");
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("---------------------------\n");
  }
  printf("\n");
};

void random_init_matrix(float* mat, int N) {
  for (int i = 0; i < N; i++) {
    int randomInt = rand();
    float tmp = (float)randomInt /RAND_MAX;
    mat[i] = tmp;
  }
}

void copy_matrix(const float* src, float* dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++) {dest[i] = src[i];}

  if (i != N) {
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
  }
}

void print_matrix(const float* mat, int row, int col) {
  for (int i = 0; i < row; i++) {
    printf("[");
    for (int j = 0; j < col; j++) {
      printf("%5.2f ", mat[i * col + j]);
    }
    printf("]\n");
  }
}

bool verify_matrix(const float* A, const float* B, int N) {
  for (int i = 0; i < N; i++) {
    if (fabs(A[i] - B[i]) > 1e-4) {
      printf("C[%d] = %f, C_ref[%d] = %f, C - C_ref = %f \n", i, A[i], i, B[i], fabs(A[i] - B[i]));
      return false;
    }
  }
  return true;
}


