set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(crash_aftermath
  SOURCES 
    src/crash_aftermath.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/raster.vert.glsl
    ${SHD_DIR}/raster.frag.glsl
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
  SHADER_HLSL
    ${SHD_DIR}/raster.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/raster.slang
)

if(NOT NSIGHT_AFTERMATH_SDK)
  message(       "  - This project should use NSight Aftermath.")
  message(STATUS "    Install NSight Aftermath SDK and set NSIGHT_AFTERMATH_SDK path (CMake).")
endif()