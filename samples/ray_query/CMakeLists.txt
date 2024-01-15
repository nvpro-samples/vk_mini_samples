
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(ray_query
  SOURCES 
    ray_query.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/ray_query.comp
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
  SHADER_HLSL
    ${SHD_DIR}/ray_query.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/ray_query.slang
)

