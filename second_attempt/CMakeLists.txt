cmake_minimum_required(VERSION 3.10)
project(MCTS_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_TORCH "Enable Torch support" OFF)

# Conditionally set CMAKE_PREFIX_PATH and find_package
if(USE_TORCH)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
    find_package(Torch REQUIRED)

    # Define a macro TORCH for use in your C++ code
    add_definitions(-DTORCH)
endif()

# Add compiler optimizations for release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Source files
set(SOURCES
    src/Gamestate.cpp
    src/main.cpp
    src/Node.cpp
    src/MCTS.cpp
    src/model.c
)


# Add the executable
add_executable(quoridor ${SOURCES})

# target_compile_options(quoridor PRIVATE -Wall -Wextra -Wuninitialized)

if(USE_TORCH)
    target_link_libraries(quoridor "${TORCH_LIBRARIES}")
endif()

# Include directories
target_include_directories(quoridor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include) 
