cmake_minimum_required(VERSION 3.27)

# --- Project ---

project(capstone LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "" FORCE)
endif()

# --- Platform ---

include(GNUInstallDirs)

# keep executables in one place rather than under src/<app>/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Shared libs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Static libs

if(UNIX)
    set(BUILD_SHARED_LIBS OFF)

    set(CMAKE_INSTALL_BINDIR bin)
    set(CMAKE_INSTALL_LIBDIR lib)

    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

    if(APPLE)
        # Fix for Xcode (multi-config generators append the config name otherwise)
        foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
            string(TOUPPER ${CONFIG} CONFIG_UPPER)
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
            set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
        endforeach()

        # Ensure homebrew packages are discoverable
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
        set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")

        # Fix for homebrew's libomp (keg-only, so not on the default prefix path)
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

# --- Dependencies ---

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# --- Subdirs ---

add_subdirectory(external/ev2)

add_subdirectory(src/shared)
add_subdirectory(src/training)
add_subdirectory(src/transform)
