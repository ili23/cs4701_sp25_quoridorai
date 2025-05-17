cmake_minimum_required(VERSION 3.10)
project(MCTS_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_ASAN "Enable AddressSanitizer" ON)

if (FALSE)
    message(STATUS "Building with AddressSanitizer")
    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${ASAN_FLAGS}")
endif()

option(USE_TORCH "Enable Torch support" OFF)

# Conditionally set CMAKE_PREFIX_PATH and find_package
if(FALSE)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
    find_package(Torch REQUIRED)
    message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
    message(STATUS "Torch CXX Flags: ${TORCH_CXX_FLAGS}")
    message(STATUS "Torch C Flags: ${CMAKE_C_FLAGS}")
    message(STATUS "Torch CXX Flags Release: ${TORCH_CXX_FLAGS_RELEASE}")

    # Add Torch flags to the build
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    # Add Torch include directories to the compile flags
    include_directories(${TORCH_INCLUDE_DIRS})

    # Add the library path explicitly
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib")
    
    # Set rpath to find dynamic libraries at runtime
    set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    
    # On macOS, use the install name path
    if(APPLE)
        set(CMAKE_INSTALL_NAME_DIR "@rpath")
        set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR TRUE)
    endif()

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

# Use the new TRUE option consistently
if(FALSE)
    # Simple and direct approach to linking Torch
    target_link_libraries(quoridor ${TORCH_LIBRARIES})
    
    # Make sure we're using the correct include directories
    target_include_directories(quoridor PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

# Include directories
target_include_directories(quoridor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include) 
