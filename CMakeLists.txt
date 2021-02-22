cmake_minimum_required(VERSION 3.4)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
project(demo) #VERSION 0.1.0 


option(BUILD_SHARED_LIBS "Build as a shared library"  ON)
option(USE_DOUBLE        "Use double as kernel_type" OFF)
option(TEST              "BUILD TEST"                 ON)
option(BENCH             "BUILD BENCH"               OFF)

if(USE_DOUBLE)
    message("Use double as kernel_type")
endif()
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

############################################################
# add 3th_part library
############################################################
# find_package (PCL REQUIRED)
# include_directories (${PCL_INCLUDE_DIRS})

# find_package(catkin REQUIRED COMPONENTS rosbag roscpp sensor_msgs)
# include_directories(${catkin_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

find_package(Threads)
find_package(OpenMP REQUIRED)
add_definitions("-DELPP_NO_LOG_TO_FILE")

find_package(CUDA REQUIRED QUIET)
if (CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()
list(APPEND LINK_LIBRARY ${CUDA_cusparse_LIBRARY})

# Correct error for GCC 5 and CUDA 7.5
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")

if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

set(COMMON_INCLUDES ${CMAKE_CURRENT_BINARY_DIR})

# set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
# configure_file(include/matmul/config.h.in matmul/config.h)

############################################################
# Create a library in subdirectory
############################################################
set(PROJECT_LIB_NAME ${PROJECT_NAME})
include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/src/matmul)


############################################################
# Create an executable
############################################################
add_executable(${PROJECT_NAME}-demo demo.cc ${COMMON_INCLUDES})

target_link_libraries(${PROJECT_NAME}-demo ${LINK_LIBRARY} ${PROJECT_LIB_NAME})


if (TEST)
    message("Building tests")
    enable_testing()
    add_subdirectory(test)
endif ()

# if (BENCH)
#     message("Building benchmark")
#     add_subdirectory(bench)
# endif ()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build shared libs    : ${BUILD_SHARED_LIBS}")
message(STATUS "  Use double for kernel: ${USE_DOUBLE}")
message(STATUS "  Build tests          : ${TEST}")
message(STATUS "  Build benchmarks     : ${BENCH}")
message(STATUS "  FoundCUDA            : ${CUDA_FOUND}")
message(STATUS "  Boost include dirs   : ${Boost_INCLUDE_DIRS}")
message(STATUS "")


