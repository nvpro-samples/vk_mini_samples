
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(texture_3d
  SOURCES 
    texture_3d.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/raster.vert.glsl
    ${SHD_DIR}/raster.frag.glsl
    ${SHD_DIR}/perlin.comp.glsl
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
  SHADER_HLSL
    ${SHD_DIR}/raster.hlsl 
    ${SHD_DIR}/perlin.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/raster.slang
    ${SHD_DIR}/perlin.slang 
)

