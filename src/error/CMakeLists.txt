# --- Target ---

set(TARGET error)

add_executable(${TARGET})

# --- Dependencies ---

target_compile_definitions(${TARGET} PRIVATE -DPROJECT_MOUNT="cap")
target_compile_definitions(${TARGET} PRIVATE -DRESOURCE_DIR="${PROJECT_SOURCE_DIR}/resource")

target_link_libraries(${TARGET} PRIVATE
	shared
	ev2-editor
	ev2-engine
)

ev2_add_mount(${TARGET} NAME cap DIR ${PROJECT_SOURCE_DIR}/resource)

# --- Sources ----

file(GLOB CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
file(GLOB CXX_HEADERS ${CMAKE_CURRENT_SOURCE_DIR} "*.h[pp]")

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(${TARGET} PRIVATE ${CXX_SOURCES} ${CXX_HEADERS})

