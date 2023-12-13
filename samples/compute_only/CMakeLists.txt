
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(compute_only
  SOURCES 
    compute_only.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/shader.comp
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
  SHADER_HLSL
    ${SHD_DIR}/shader.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/shader.slang
)
