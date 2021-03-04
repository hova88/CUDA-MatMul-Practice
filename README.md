# cuda-template
![](./doc/jpgs/cover.jpgs)
The repo mainly provides a basic template related to the application of CUDA in C++ project with `CMakeLists.txt`. 
and takes the realization of matrix multiplication in various ways as an example.

## Install

```
CUDA 10+
CMake 3.4+
```

## Usage

```bash
mkdir build && cd build
cmake .. && make -j6
./test/test_matmul 
```

## Result

```bash
[==========] Running 14 tests from 2 test cases.
[----------] Global test environment set-up.
[----------] 5 tests from MatrixMult
[ RUN      ] MatrixMult.__Naive__
[       OK ] MatrixMult.__Naive__ (75 ms)
[ RUN      ] MatrixMult.__Tiling__
[       OK ] MatrixMult.__Tiling__ (0 ms)
[ RUN      ] MatrixMult.__Coalescing__
[       OK ] MatrixMult.__Coalescing__ (0 ms)
[ RUN      ] MatrixMult.__Computation_Omp__
[       OK ] MatrixMult.__Computation_Omp__ (0 ms)
[ RUN      ] MatrixMult.__Unroll__
[       OK ] MatrixMult.__Unroll__ (0 ms)
[----------] 5 tests from MatrixMult (75 ms total)

[----------] 9 tests from REDUCTION
[ RUN      ] REDUCTION.__Inter_Div__
[       OK ] REDUCTION.__Inter_Div__ (1 ms)
[ RUN      ] REDUCTION.__Inter_Bank__
[       OK ] REDUCTION.__Inter_Bank__ (0 ms)
[ RUN      ] REDUCTION.__Seque_Naive__
[       OK ] REDUCTION.__Seque_Naive__ (1 ms)
[ RUN      ] REDUCTION.__Seque_Halve__
[       OK ] REDUCTION.__Seque_Halve__ (0 ms)
[ RUN      ] REDUCTION.__Seque_Unroll__
[       OK ] REDUCTION.__Seque_Unroll__ (0 ms)
[ RUN      ] REDUCTION.__Complete_Unroll__
[       OK ] REDUCTION.__Complete_Unroll__ (0 ms)
[ RUN      ] REDUCTION.__Multiple_thread__
[       OK ] REDUCTION.__Multiple_thread__ (0 ms)
[ RUN      ] REDUCTION.__TOTAL_SUM__
[       OK ] REDUCTION.__TOTAL_SUM__ (0 ms)
[ RUN      ] REDUCTION.__Points_Mean__
[       OK ] REDUCTION.__Points_Mean__ (4 ms)
[----------] 9 tests from REDUCTION (6 ms total)

[----------] Global test environment tear-down
[==========] 14 tests from 2 test cases ran. (81 ms total)
[  PASSED  ] 14 tests.
```

# Reference
[1] cuda-cmake-gtest-gbench-starter. 
at: https://github.com/PhDP/cuda-cmake-gtest-gbench-starter.

[2] Matrix Multiplication CUDA. 
at: https://ecatue.gitlab.io/gpu2018/pages/Cookbook/matrix_multiplication_cuda.html#8.

[3] Optimizing Parallel Reduction in CUDA - Nvidia
at: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
## License

MIT © Richard McRichface
