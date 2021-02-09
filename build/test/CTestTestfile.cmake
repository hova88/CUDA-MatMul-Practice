# CMake generated Testfile for 
# Source directory: /home/hova/cuda_template/test
# Build directory: /home/hova/cuda_template/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests "/home/hova/cuda_template/build/test/test_matmul")
set_tests_properties(tests PROPERTIES  _BACKTRACE_TRIPLES "/home/hova/cuda_template/test/CMakeLists.txt;21;add_test;/home/hova/cuda_template/test/CMakeLists.txt;0;")
subdirs("gtest")
