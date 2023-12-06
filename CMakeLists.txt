cmake_minimum_required(VERSION 3.19)
project(CUDA_MATMUL LANGUAGES CXX CUDA)  
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# set cuda compute capability
set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
# aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(matmul matmul.cu ${SRC})
target_link_libraries(matmul ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

