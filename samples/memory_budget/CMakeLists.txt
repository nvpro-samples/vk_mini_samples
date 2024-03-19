
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(${PROJECT_NAME}
  SOURCES 
    memory_budget.cpp 
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

unset(PROJECT_NAME)