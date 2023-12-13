
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(tiny_shader_toy
  SOURCES 
    tiny_shader_toy.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/raster.vert
    ${SHD_DIR}/raster.frag
  SHADER_HEADERS
    ${SHD_DIR}/buffer_a.glsl
    ${SHD_DIR}/common.glsl
    ${SHD_DIR}/image.glsl
)
