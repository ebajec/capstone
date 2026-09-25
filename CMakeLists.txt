cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 23)

set(TARGET covq)
project(${TARGET})

file(GLOB SOURCES "*.cpp" "*.c")

add_executable(${TARGET}
	src/main.cpp	
	src/stb_image.cpp	
)

target_include_directories(${TARGET}
	PUBLIC
	src
)

target_compile_options(${TARGET}
	PUBLIC
	-O1
	-march=native
)

find_package(Eigen3 REQUIRED NO_MODULE)

target_link_libraries(${TARGET} PRIVATE
	Eigen3::Eigen
)
