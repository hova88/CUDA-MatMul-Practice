/*
 * Yan haixu
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
points_mean_kernel(float* g_odata, float* g_idata , unsigned int in_num_points)
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x; // 0,1,2,...,255
    unsigned int i = blockIdx.x  * in_num_points  + threadIdx.x; 
    //                {0, 1, 2}  *    10240     + {0,1,2,...,255}       
    sdata[tid] = 0;
    
    while (i < in_num_points * (blockIdx.x + 1)) {
        sdata[tid] += g_idata[i] + g_idata[i+blocksize];
        i += blocksize * 2;
    }
    __syncthreads();
    
    // do reduction in shared memory
    if (blocksize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256] ;} __syncthreads();}
    if (blocksize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128] ;} __syncthreads();}
    if (blocksize >= 128) {
        if (tid < 64) {  sdata[tid] += sdata[tid +  64] ;} __syncthreads();}

    if (tid < 32) warpReduce<blocksize>(sdata , tid);
    __syncthreads();

    // write result for this block to global memory
    if (tid == 0 ) g_odata[ blockIdx.x ] = sdata[0] / in_num_points;

}

  
void parser::points_mean( matrix& C) {
    
	float* dev_a;
	cudaMalloc(&dev_a, A.row * A.col * sizeof(float));
	cudaMemcpy(dev_a, A.elements,  A.row * A.col * sizeof(float), cudaMemcpyHostToDevice);
    
    float* dev_c;
    cudaMalloc(&dev_c, C.row  * sizeof(float));
    // switch (threads)
    // dim3 blocksize( 256 , 3);

    points_mean_kernel<512><<< 3  , 512 , 512 * sizeof(float)>>>(dev_c, dev_a , 10240);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C.elements, dev_c, C.row * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_c);
    return;
}