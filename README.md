# cuda-template

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
[==========] Running 4 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 4 tests from MatrixMult
[ RUN      ] MatrixMult.__Naive__
[       OK ] MatrixMult.__Naive__ (144 ms)
[ RUN      ] MatrixMult.__Tiling__
[       OK ] MatrixMult.__Tiling__ (0 ms)
[ RUN      ] MatrixMult.__Coalescing__
[       OK ] MatrixMult.__Coalescing__ (0 ms)
[ RUN      ] MatrixMult.__Computation_Omp__
[       OK ] MatrixMult.__Computation_Omp__ (0 ms)
[----------] 4 tests from MatrixMult (144 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test case ran. (144 ms total)
[  PASSED  ] 4 tests.
```

# Reference
[1] cuda-cmake-gtest-gbench-starter. at: https://github.com/PhDP/cuda-cmake-gtest-gbench-starter.

[2] Matrix Multiplication CUDA. at: https://ecatue.gitlab.io/gpu2018/pages/Cookbook/matrix_multiplication_cuda.html#8.
## License

MIT © Richard McRichface
