# CUDA Matrix Multiplication Step-by-Step
![](./doc/jpgs/cover.jpg)
My rewrite version about how to optimize Matmul or GEMM from scratch.

$GEMM := C = \alpha AB + \beta C$

## Require
```
CUDA 10+
CMake 3.10+
```

## Usage

```bash
mkdir build && cd build
cmake .. && make -j6
./matmul 
```

## Result

# Reference
[1] cuda-cmake-gtest-gbench-starter. 
at: https://github.com/PhDP/cuda-cmake-gtest-gbench-starter.

[2] Matrix Multiplication CUDA. 
at: https://ecatue.gitlab.io/gpu2018/pages/Cookbook/matrix_multiplication_cuda.html#8.

[3] Optimizing Parallel Reduction in CUDA - Nvidia
at: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

[4] How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog
at: https://siboehm.com/articles/22/CUDA-MMM
