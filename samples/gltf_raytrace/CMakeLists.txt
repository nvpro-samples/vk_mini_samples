# Get the name of the current directory
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})
message(STATUS "Processing: ${PROJECT_NAME}")

# Adding all sources
file(GLOB EXE_SOURCES "*.cpp" "*.hpp")
source_group("Source Files" FILES ${EXE_SOURCES})

# Add the executable
add_executable(${PROJECT_NAME} ${EXE_SOURCES})

# Link libraries and include directories
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvslang
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvshaders_host
  nvpro2::nvaftermath
  nvpro2::nvvkgltf
  vk_mini_samples_common
)

add_project_definitions(${PROJECT_NAME})

# Include directory for generated files
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR})

#------------------------------------------------------------------------------------------------------------------------------
# Compile shaders
file(GLOB SHADER_SLANG_FILES "shaders/*.slang")
file(GLOB SHADER_H_FILES "shaders/*.h")

# Adding local pre-compiled version of the tonemapper
list(APPEND SHADER_SLANG_FILES ${NVSHADERS_DIR}/nvshaders/tonemapper.slang)

compile_slang(
    "${SHADER_SLANG_FILES}"
    "${CMAKE_CURRENT_LIST_DIR}/_autogen"
    GENERATED_SHADER_HEADERS
    EXTRA_FLAGS "-I${NVSHADERS_DIR};-I${COMMON_DIR}"
)

# Add shader files to the project
source_group("Shaders" FILES ${SHADER_SLANG_FILES} ${SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_HEADERS})

# Add the output shader headers (target) directly to the executable
# This allow to compile the shaders when the executable is built
target_sources(${PROJECT_NAME} PRIVATE ${GENERATED_SHADER_HEADERS} ${SHADER_H_FILES})

#------------------------------------------------------------------------------------------------------------------------------
# Installation, copy files

# Copy .jpg images next to the executable
copy_to_runtime_and_install( ${PROJECT_NAME}
    FILES ${NsightAftermath_DLLS}
    LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
    AUTO
)
