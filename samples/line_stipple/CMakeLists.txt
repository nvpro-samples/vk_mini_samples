
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(line_stipple
  SOURCES 
    line_stipple.cpp 
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

