/*
 * Mark Harris
 * NVIDIA Developer Technology
 */

 #include <stdio.h>
 #include "../parser.h"

__global__ void
interadd_divbranch_kernel(float* g_odata, float* g_idata)
{
    extern __shared__ float sdata[];
    
    // each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = 1 ; s < blockDim.x ; s *= 2) {
        if (tid % (2 * s) == 0 ) {
            sdata[tid] += sdata[tid +s];
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0 ) g_odata[blockIdx.x] = sdata[0];

}

  
void parser::reduce_interdiv( matrix& C) {
    
	float* dev_a;
	cudaMalloc(&dev_a, A.row * A.col * sizeof(float));
	cudaMemcpy(dev_a, A.elements,  A.row * A.col * sizeof(float), cudaMemcpyHostToDevice);
    
    float* dev_c;
    cudaMalloc(&dev_c, C.row  * sizeof(float));
    
    interadd_divbranch_kernel<<< 3 , 4 , 32 * sizeof(float)>>>(dev_c, dev_a);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C.elements, dev_c, C.row * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_c);
    return;
}