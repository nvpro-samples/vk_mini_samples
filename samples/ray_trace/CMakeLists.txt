
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(ray_trace
  SOURCES 
    ray_trace.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/raytrace.rchit.glsl
    ${SHD_DIR}/raytrace.rgen.glsl
    ${SHD_DIR}/raytrace.rmiss.glsl
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
    ${SHD_DIR}/dh_bindings.h
    ${SHD_DIR}/payload.h
  SHADER_HLSL
    ${SHD_DIR}/raytrace.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/raytrace.slang
)

