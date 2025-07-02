# Get the name of the current directory
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})
message(STATUS "Processing: ${PROJECT_NAME}")

# Add the executable
file(GLOB EXE_SOURCES "*.cpp" "*.hpp")

source_group("Source Files" FILES ${EXE_SOURCES})
add_executable(${PROJECT_NAME} ${EXE_SOURCES})

# Link libraries and include directories
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvutils
  nvpro2::nvvk
)

add_project_definitions(${PROJECT_NAME})


#------------------------------------------------------------------------------------------------------------------------------
# Installation, copy files
copy_to_runtime_and_install( ${PROJECT_NAME}
    AUTO
)
