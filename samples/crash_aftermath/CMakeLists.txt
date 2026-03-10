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
  nvpro2::nvaftermath
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvslang
  nvpro2::nvshaders_host
  nvpro2::nvutils
  nvpro2::nvvk
  tinygltf
  vk_mini_samples_common
)

add_project_definitions(${PROJECT_NAME})

# Include directory for generated files
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR})

#------------------------------------------------------------------------------------------------------------------------------
# Compile shaders
file(GLOB SHADER_SLANG_FILES "shaders/*.slang")
file(GLOB SHADER_GLSL_FILES "shaders/*.glsl")
file(GLOB SHADER_H_FILES "shaders/*.h")

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/_autogen")



compile_slang(
    "${SHADER_SLANG_FILES}"
    "${SHADER_OUTPUT_DIR}"
    HEADERS_VAR GENERATED_SHADER_SLANG_HEADERS
    SPVS_VAR GENERATED_SHADER_SLANG_SPVS
    EXTRA_FLAGS "-I${NVSHADERS_DIR};-I${COMMON_DIR}"
    DEBUG_LEVEL 2 
)

compile_glsl(
   "${SHADER_GLSL_FILES}"
   "${SHADER_OUTPUT_DIR}"
   GENERATED_SHADER_GLSL_HEADERS
   EXTRA_FLAGS "-I${NVSHADERS_DIR};-I${COMMON_DIR}"
)

# GLSL build produces .spv in _autogen; list them for copy to runtime (Inspector loads by content)
set(GENERATED_SHADER_GLSL_SPVS
  ${SHADER_OUTPUT_DIR}/crash_aftermath.vert.glsl.spv
  ${SHADER_OUTPUT_DIR}/crash_aftermath.frag.glsl.spv
  ${SHADER_OUTPUT_DIR}/crash_aftermath.comp.glsl.spv
)

# Add shader files to the project
source_group("Shaders" FILES ${SHADER_SLANG_FILES} ${SHADER_GLSL_FILES} ${SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS})

# Add the output shader headers (target) directly to the executable
# This allows to compile the shaders when the executable is built
target_sources(${PROJECT_NAME} PRIVATE ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS} ${SHADER_H_FILES})

#------------------------------------------------------------------------------------------------------------------------------
# Installation, copy files

copy_to_runtime_and_install( ${PROJECT_NAME}
    LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
    FILES ${NsightAftermath_DLLS}
    AUTO
)

# Copy generated .spv to runtime so the Crash Dump Inspector can load them (it hashes by content).
# These are generated at build time so we use POST_BUILD; copy_to_runtime checks EXISTS at configure time.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${GENERATED_SHADER_SLANG_SPVS} ${GENERATED_SHADER_GLSL_SPVS}
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
  COMMENT "Copying .spv shaders to runtime for Aftermath Monitor / Crash Dump Inspector"
)
