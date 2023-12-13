
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(ray_tracing_position_fetch
  SOURCES 
    ray_tracing_position_fetch.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/raytrace.rchit
    ${SHD_DIR}/raytrace.rgen
    ${SHD_DIR}/raytrace.rmiss
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
    ${SHD_DIR}/dh_bindings.h
    ${SHD_DIR}/payload.h
  SHADER_HLSL
    ${SHD_DIR}/raytrace.hlsl 
)

