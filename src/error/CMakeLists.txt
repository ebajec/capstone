# --- Target ---

set(TARGET error)

add_executable(${TARGET})

# --- Dependencies ---

target_link_libraries(${TARGET} PRIVATE
	shared
)

# --- Sources ----

file(GLOB CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
file(GLOB CXX_HEADERS ${CMAKE_CURRENT_SOURCE_DIR} "*.h[pp]")

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(${TARGET} PRIVATE ${CXX_SOURCES} ${CXX_HEADERS})

