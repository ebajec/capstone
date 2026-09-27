cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 23)

set(TARGET covq)
project(${TARGET})

file(GLOB SOURCES "*.cpp" "*.c")

add_executable(${TARGET}
	src/main.cpp
	src/covq.cpp
	src/stb_image.cpp	
)

target_compile_definitions(${TARGET} PRIVATE -DPROJECT_MOUNT="cap")
target_compile_definitions(${TARGET} PRIVATE -DRESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resource")

# Eigen's hand-unrolled AVX-512 GEMM kernels (enabled by -march=native on AVX-512
# CPUs) take ~60s per TU to optimize in clang when debug info is on. The generic
# gebp kernel still uses AVX-512 and is plenty for our small matrix products.
target_compile_definitions(${TARGET} PRIVATE EIGEN_USE_AVX512_GEMM_KERNELS=0)

target_include_directories(${TARGET}
	PRIVATE
	src
)

target_compile_options(${TARGET}
	PRIVATE
	-O3
	-march=native
)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

target_link_libraries(${TARGET} PRIVATE
	Eigen3::Eigen
	OpenMP::OpenMP_CXX
	ev2-editor
	ev2-engine
)

add_subdirectory(external/ev2)

ev2_add_mount(${TARGET} NAME cap DIR resource)
