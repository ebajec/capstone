add_library(shared STATIC
	covq.cpp
	stb_image.cpp
)

target_include_directories(shared
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)

# public so every TU that includes Eigen agrees on the kernel config
target_compile_definitions(shared PUBLIC EIGEN_USE_AVX512_GEMM_KERNELS=0)

target_compile_options(shared PRIVATE
	-O3
)

# must be PUBLIC: every TU that includes Eigen has to see the same SIMD flags,
# otherwise EIGEN_MAX_ALIGN_BYTES differs between TUs (64 vs 16) and the
# linker mixes 16-byte-aligned allocations with 64-byte-aligned AVX-512 stores
#
# guarded because older Apple Clang on arm64 rejects -march=native
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
	target_compile_options(shared PUBLIC
		-march=native
	)
endif()


target_link_libraries(shared PUBLIC
	Eigen3::Eigen
	OpenMP::OpenMP_CXX
)
