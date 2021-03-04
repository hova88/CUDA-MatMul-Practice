/*
 * Mark Harris
 * NVIDIA Developer Technology
 */
 #include <stdio.h>
 #include "../parser.h"
template<unsigned int blocksize>
__device__ void warpReduce(volatile float* sdata , int tid) {
    if (blocksize >= 64) sdata[tid] += sdata[tid + 32];
    if (blocksize >= 32) sdata[tid] += sdata[tid + 16];
    if (blocksize >= 16) sdata[tid] += sdata[tid +  8];
    if (blocksize >=  8) sdata[tid] += sdata[tid +  4];
    if (blocksize >=  4) sdata[tid] += sdata[tid +  2];
    if (blocksize >=  2) sdata[tid] += sdata[tid +  1];
}

template<unsigned int blocksize>
__global__ void
multiple_thread_kernel(float* g_odata, float* g_idata)
{
    extern __shared__ float sdata[];
    
    // each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * ( blocksize * 2 ) + threadIdx.x;
    unsigned int gridsize = blocksize * 2 * gridDim.x;
    unsigned int n = 12; // number of elements
    sdata[tid] = 0;
    
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+blocksize];
        i += gridsize;
    }
    __syncthreads();

    // do reduction in shared memory
    if (blocksize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256] ;} __syncthreads();}
    if (blocksize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128] ;} __syncthreads();}
    if (blocksize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid +   64] ;} __syncthreads();}
    
    
    if (tid < 32) warpReduce<blocksize>(sdata , tid);
    // write result for this block to global memory
    if (tid == 0 ) g_odata[blockIdx.x] = sdata[0];

}

  
void parser::multiple_thread( matrix& C) {
    
	float* dev_a;
	cudaMalloc(&dev_a, A.row * A.col * sizeof(float));
	cudaMemcpy(dev_a, A.elements,  A.row * A.col * sizeof(float), cudaMemcpyHostToDevice);
    
    float* dev_c;
    cudaMalloc(&dev_c, C.row  * sizeof(float));
    // switch (threads)
    multiple_thread_kernel<2><<< 3 , 2 , 32 * sizeof(float)>>>(dev_c, dev_a);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C.elements, dev_c, C.row * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_c);
    return;
}