cmake_minimum_required(VERSION 3.27)

# --- Project ---

set(TARGET covq)
project(${TARGET} LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

# --- Platform ---

include(GNUInstallDirs)

if(UNIX)
    set(BUILD_SHARED_LIBS OFF)
    
    set(CMAKE_INSTALL_BINDIR bin)
    set(CMAKE_INSTALL_LIBDIR lib)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Target
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Shared libs
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Static libs

    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

    if(APPLE)
        # Fix for Xcode
        foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
            string(TOUPPER ${CONFIG} CONFIG_UPPER)
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
            set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
        endforeach()

        # Ensure homebrew packages are discoverable
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
        set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")

        # Fix for homebrew's libomp
        if(NOT DEFINED OpenMP_ROOT)
            find_program(HOMEBREW_EXECUTABLE brew)
            if(HOMEBREW_EXECUTABLE)
                execute_process(
                    COMMAND "${HOMEBREW_EXECUTABLE}" --prefix libomp
                    OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                    RESULT_VARIABLE HOMEBREW_LIBOMP_RESULT
                )
                if(HOMEBREW_LIBOMP_RESULT EQUAL 0)
                    set(OpenMP_ROOT "${HOMEBREW_LIBOMP_PREFIX}")
                endif()
            endif()
        endif()
    endif()
endif()

# --- Sources and Dependencies ---

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
	-g
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


# --- Subdirs ---

add_subdirectory(external/ev2)

# --- Resources ---

ev2_add_mount(${TARGET} NAME cap DIR resource)

