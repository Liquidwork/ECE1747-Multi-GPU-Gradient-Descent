project(local LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(local
        local.cpp
        read.cpp
)
