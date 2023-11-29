project(project2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER nvcc)

# set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(project2
        #        local.cpp
        read.cpp
        mpigpu.cpp
        gpuhelper.cu
        #        mpi.cpp
)

# CUDA include directories and libraries
target_include_directories(project2 PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(project2 PRIVATE ${CUDA_LIBRARIES})

target_link_libraries(project2 PRIVATE MPI::MPI_CXX)
