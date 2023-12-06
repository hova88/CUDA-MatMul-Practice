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
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
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
    if (fabs(A[i] - B[i]) > 1e-5) {
      printf("A[%d] = %f, B[%d] = %f, A-B = %f \n", i, A[i], i, B[i], fabs(A[i] - B[i]));
      return false;
    }
  }
  return true;
}


int main(int argc, char** argv) {

  CudaDeviceInfo();

  int MatSize = 1024;
  int MatBytes = MatSize * MatSize * sizeof(float);
  float alpha = 1.0f, beta = 1.0f; // GEMM input parameters, C=α*AB+β*C

  float *A, *B, *C, *C_ref; // host matrices
  float *d_A, *d_B, *d_C, *d_C_ref; // device matrices

  // allocate host memory
  A = (float*)malloc(MatBytes);
  B = (float*)malloc(MatBytes);
  C = (float*)malloc(MatBytes);
  C_ref = (float*)malloc(MatBytes);

  // allocate device memory
  checkCudaErrors(cudaMalloc((void**)&d_A, MatBytes));
  checkCudaErrors(cudaMalloc((void**)&d_B, MatBytes));
  checkCudaErrors(cudaMalloc((void**)&d_C, MatBytes));
  checkCudaErrors(cudaMalloc((void**)&d_C_ref, MatBytes));

  // initialize host matrices
  random_init_matrix(A, MatSize * MatSize);
  random_init_matrix(B, MatSize * MatSize);
  random_init_matrix(C, MatSize * MatSize);

  // copy host matrices to device
  checkCudaErrors(cudaMemcpy(d_A, A, MatBytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, B, MatBytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C, C, MatBytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C_ref, C, MatBytes, cudaMemcpyHostToDevice));

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_ref);
	return 0;
}
