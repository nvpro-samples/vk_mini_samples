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
)

add_project_definitions(${PROJECT_NAME})

# Include directory for generated files
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})

#------------------------------------------------------------------------------------------------------------------------------
# Compile shaders
file(GLOB SHADER_SLANG_FILES "shaders/*.slang")
file(GLOB SHADER_GLSL_FILES "shaders/*.glsl")
file(GLOB SHADER_H_FILES "shaders/*.h")

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/_autogen")


compile_slang(
    "${SHADER_SLANG_FILES}"
     "${SHADER_OUTPUT_DIR}"
    GENERATED_SHADER_SLANG_HEADERS
    EXTRA_FLAGS "-I${NVSHADERS_DIR};-I${COMMON_DIR}"
)
compile_glsl(
   "${SHADER_GLSL_FILES}"
   "${SHADER_OUTPUT_DIR}"
   GENERATED_SHADER_GLSL_HEADERS
   EXTRA_FLAGS "-I${NVSHADERS_DIR};-I${COMMON_DIR}"
 )

# Add shader files to the project
source_group("Shaders" FILES ${SHADER_SLANG_FILES} ${SHADER_GLSL_FILES} ${SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS})

# Add the output shader headers (target) directly to the executable
# This allows to compile the shaders when the executable is built
target_sources(${PROJECT_NAME} PRIVATE ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS} ${SHADER_H_FILES})

#------------------------------------------------------------------------------------------------------------------------------
# Installation, copy files

# Installation setup
set(INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
file(MAKE_DIRECTORY ${INSTALL_DIR})

copy_to_runtime_and_install(${PROJECT_NAME}
    LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
    AUTO
)

