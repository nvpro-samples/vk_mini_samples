
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(ray_query_position_fetch
  SOURCES 
    ray_query_position_fetch.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/ray_query.comp
  SHADER_HEADERS
    ${SHD_DIR}/dh_bindings.h
)

