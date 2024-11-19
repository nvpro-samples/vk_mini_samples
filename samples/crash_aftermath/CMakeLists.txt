# Get the name of the current directory
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Find all source files
file(GLOB_RECURSE SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

# Find all shader files
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB_RECURSE SHADER_SOURCES ${SHD_DIR}/*.glsl)
file(GLOB_RECURSE SHADER_HLSL ${SHD_DIR}/*.hlsl)
file(GLOB_RECURSE SHADER_SLANG ${SHD_DIR}/*.slang)
file(GLOB_RECURSE SHADER_HEADERS ${SHD_DIR}/*.h)

# Create the sample
create_sample(${PROJECT_NAME} 
  SOURCES           ${SOURCES}
  SHADER_SOURCES    ${SHADER_SOURCES}
  SHADER_HEADERS    ${SHADER_HEADERS}
  SHADER_HLSL       ${SHADER_HLSL}
  SHADER_SLANG      ${SHADER_SLANG}
)

if(NOT NSIGHT_AFTERMATH_SDK)
  message(       "  - This project should use NSight Aftermath.")
  message(STATUS "    Install NSight Aftermath SDK and set NSIGHT_AFTERMATH_SDK path (CMake).")
endif()