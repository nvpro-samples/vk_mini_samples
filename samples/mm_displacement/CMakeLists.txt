
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(mm_displacement
  SOURCES 
    dmm_process.cpp 
    dmm_process.hpp 
    mm_displacement.cpp
    vk_nv_micromesh.h
  SHADER_SOURCES 
    ${SHD_DIR}/raytrace.rgen.glsl
    ${SHD_DIR}/raytrace.rchit.glsl
    ${SHD_DIR}/raytrace.rmiss.glsl
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
    ${SHD_DIR}/payload.h
    ${SHD_DIR}/dh_bindings.h
  SHADER_HLSL
    ${SHD_DIR}/raytrace.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/raytrace.slang
)



