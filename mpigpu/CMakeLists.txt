project(mpigpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER nvcc)

# set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(mpigpu
        read.cpp
        mpigpu.cpp
        gpuhelper.cu
)

# CUDA include directories and libraries
target_include_directories(mpigpu PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(mpigpu PRIVATE ${CUDA_LIBRARIES})

target_link_libraries(mpigpu PRIVATE MPI::MPI_CXX)
