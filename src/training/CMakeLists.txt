add_executable(training
	training.cpp
)

target_compile_definitions(training PRIVATE -DPROJECT_MOUNT="cap")
target_compile_definitions(training PRIVATE -DRESOURCE_DIR="${PROJECT_SOURCE_DIR}/resource")

target_link_libraries(training PRIVATE
	shared
	ev2-editor
	ev2-engine
)

ev2_add_mount(training NAME cap DIR ${PROJECT_SOURCE_DIR}/resource)
