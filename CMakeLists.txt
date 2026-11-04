cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 23)

project(capstone)

# keep executables at the top of the build dir rather than under src/<app>/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_subdirectory(external/ev2)

add_subdirectory(src/shared)
add_subdirectory(src/training)
add_subdirectory(src/transform)
