add_executable(transform
	transform.cpp
)

target_compile_options(transform PUBLIC
	-g
	-O0
)

target_link_libraries(transform PRIVATE
	shared
	ev2-editor
	ev2-engine
)

ev2_add_mount(transform NAME cap DIR ${PROJECT_SOURCE_DIR}/resource)
