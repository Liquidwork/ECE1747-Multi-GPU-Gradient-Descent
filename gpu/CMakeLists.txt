project(gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER nvcc)

# set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(gpu
        gpu.cu
        read.cpp
)

# CUDA include directories and libraries
target_include_directories(gpu PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu PRIVATE ${CUDA_LIBRARIES})
