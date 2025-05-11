cmake_minimum_required(VERSION 3.10)
project(MCTS_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler optimizations for release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Source files
set(SOURCES
    game_state_tuple.cpp
    MCTS.cpp
    main.cpp
)

# Add the executable
add_executable(quoridor ${SOURCES})

# Include directories
target_include_directories(quoridor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}) 