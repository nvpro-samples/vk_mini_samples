
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

create_sample(${DIRNAME}
  SOURCES 
    gltf_raytrace.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/gltf_pathtrace.comp.glsl
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
    ${SHD_DIR}/dh_bindings.h
  SHADER_HLSL
    ${SHD_DIR}/gltf_pathtrace.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/gltf_pathtrace.slang
)

