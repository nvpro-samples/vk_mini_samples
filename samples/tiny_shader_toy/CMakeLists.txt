
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

# Install - copying the media directory
install(DIRECTORY "shaders" 
        CONFIGURATIONS Release 
        DESTINATION "bin_${ARCH}/tiny_shader_toy")
install(DIRECTORY "shaders" 
        CONFIGURATIONS Debug 
        DESTINATION "bin_${ARCH}_debug/tiny_shader_toy")
