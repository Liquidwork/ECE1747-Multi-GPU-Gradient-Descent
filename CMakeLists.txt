cmake_minimum_required(VERSION 3.26)
project(project2)

set(CMAKE_CXX_STANDARD 17)

add_executable(project2 local.cpp
        read.cpp)
find_package(MPI REQUIRED)

set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(project2
#        local.cpp
        read.cpp
        mpi.cpp
)

target_link_libraries(project2 PRIVATE MPI::MPI_CXX)
