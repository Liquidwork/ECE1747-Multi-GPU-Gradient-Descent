project(mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

# set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(mpi
        read.cpp
        mpi.cpp
)

target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
