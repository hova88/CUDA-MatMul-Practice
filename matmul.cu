#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include <cuda_runtime.h>

#include "./src/utils.cuh"
#include "./src/1_naive.cuh"
#include "./src/2_global_memory_coalescing.cuh"
#include "./src/3_shared_memeory.cuh"
#include "./src/4_blocktiling_1D.cuh"


/**
 * O(N^3) naive GEMM implementation (inner product)
 * 
 * for large square matrices where M=N=K, 
 * the number of math operations in a product of matrices is O(N3) 
 * while the amount of data needed is O(N2), 
 * yielding a compute intensity on the order of N.
 */
void matmul_cpu(const float* A, const float* B, float* C, float* C_result,
                      int M, int N, int K, float alpha, float beta) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++){
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C_result[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
}

void launch_and_verify_kernel(const char* kernel_info,
                              void (*kernel_launcher)(const float*, const float*, float*, int, int, int, float, float),
                              float* d_A, float* d_B, float* d_C_ref,
                              float* C, float* C_ref, float* C_after_gemm_cpu,
                              int M, int N, int K,
                              float alpha, float beta) {
  checkCudaErrors(cudaMemcpy(d_C_ref, C, M * N * sizeof(float), cudaMemcpyHostToDevice));
  kernel_launcher(d_A, d_B, d_C_ref, M, N, K, alpha, beta);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  if (verify_matrix(C_after_gemm_cpu, C_ref, M * N)) {
    printf("[%s]: Correct!\n", kernel_info);
  } else {
    printf("[%s]: Wrong!\n", kernel_info);
  }
}

void time_and_print_kernel(const char* kernel_info,
                           void (*kernel_launcher)(const float*, const float*, float*, int, int, int, float, float),
                           float* d_A, float* d_B, float* d_C, int M, int N, int K,
                           float alpha, float beta, int repeat_times) {
  cudaEvent_t beg, end;
  float elapsed_time;

  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  cudaEventRecord(beg);
  for (int i = 0; i < repeat_times; i++) {
    kernel_launcher(d_A, d_B, d_C, M, N, K, alpha, beta);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);

  printf("%s: %f ms\n", kernel_info, elapsed_time / repeat_times);

  cudaEventDestroy(beg);
  cudaEventDestroy(end);
}

int main(int argc, char** argv) {

  CudaDeviceInfo();

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  int repeat_times = 1000;

  int M = 1024, N = 1024, K = 1024;
  float alpha = 1.0f, beta = 1.0f; // GEMM input parameters, C=α*AB+β*C

  float *A, *B, *C, *C_ref, *C_after_gemm_cpu; // host matrices
  float *d_A, *d_B, *d_C, *d_C_ref; // device matrices

  // allocate host memory
  A = (float*)malloc(M * K * sizeof(float));
  B = (float*)malloc(K * N * sizeof(float));
  C = (float*)malloc(M * N * sizeof(float));
  C_ref = (float*)malloc(M * N * sizeof(float));
  C_after_gemm_cpu = (float*)malloc(M * N * sizeof(float));

  // allocate device memory
  checkCudaErrors(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_C_ref, M * N * sizeof(float)));

  // initialize host matrices
  random_init_matrix(A, M * K);
  random_init_matrix(B, K * N);
  random_init_matrix(C, M * N);


  // copy host matrices to device
  checkCudaErrors(cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice));

  // verify correctness of naive GEMM
  matmul_cpu(A, B, C, C_after_gemm_cpu, M, N, K, alpha, beta);

  launch_and_verify_kernel("kernel_1(Naive GEMM)", 
                           matmul_naive_launcher, 
                           d_A, d_B, d_C_ref, 
                           C, C_ref, C_after_gemm_cpu, M, N, K, alpha, beta);

  launch_and_verify_kernel("kernel_2(Global Memory coalesce)", 
                           matmul_naive_launcher, 
                           d_A, d_B, d_C_ref, 
                           C, C_ref, C_after_gemm_cpu, M, N, K, alpha, beta);

  launch_and_verify_kernel("kernel_3(Shared Memory + Global Memory Coalesce)", 
                           matmul_naive_launcher, 
                           d_A, d_B, d_C_ref, 
                           C, C_ref, C_after_gemm_cpu, M, N, K, alpha, beta);

  launch_and_verify_kernel("kernel_4(Shared Memory + Global Memory Coalesce + Block Tiling 1D)", 
                           matmul_naive_launcher, 
                           d_A, d_B, d_C_ref, 
                           C, C_ref, C_after_gemm_cpu, M, N, K, alpha, beta);

  // warm up
  printf("Warming up...\n");
  for (int i = 0; i < repeat_times; i++) {
    matmul_naive_launcher(d_A, d_B, d_C, M, N, K, alpha, beta);
    matmul_global_mem_coalesce_launcher(d_A, d_B, d_C, M, N, K, alpha, beta);
    matmul_shared_mem_block_launcher(d_A, d_B, d_C, M, N, K, alpha, beta);
  }

  // let's time it
  printf(" --- GEMM Performance ---\n");
  time_and_print_kernel("1. Naive GEMM", 
                        matmul_naive_launcher, 
                        d_A, d_B, d_C, M, N, K, alpha, beta, repeat_times);

  time_and_print_kernel("2. Global Memory coalesce", 
                        matmul_global_mem_coalesce_launcher, 
                        d_A, d_B, d_C, M, N, K, alpha, beta, repeat_times);

  time_and_print_kernel("3. Shared Memory + Global Memory Coalesce", 
                        matmul_shared_mem_block_launcher, 
                        d_A, d_B, d_C, M, N, K, alpha, beta, repeat_times);

  time_and_print_kernel("4. Shared Memory + Global Memory Coalesce + Block Tiling 1D", 
                        matmul_smem_blocktiling1d_launcher, 
                        d_A, d_B, d_C, M, N, K, alpha, beta, repeat_times);

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  free(C_after_gemm_cpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_ref);
	return 0;
}
