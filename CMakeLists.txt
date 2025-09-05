cmake_minimum_required(VERSION 3.16)
project(stand_go2)

SET(CMAKE_BUILD_TYPE Release)

# Aggressive runtime optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math")

# Enable Link Time Optimization for cross-module optimizations
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# Set default build type to Debug if not specified
# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE Debug)
# endif()

# Add debug flags for Debug builds
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#     set(CMAKE_C_FLAGS_DEBUG "-g -O0")
# endif()

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/home/atari/workspace/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Print some information about found packages
message(STATUS "Found Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")


add_executable(run_controller run_controller.cpp)
target_link_libraries(run_controller 
    unitree_sdk2
    ${TORCH_LIBRARIES}
)

