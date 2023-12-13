
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(ser_pathtrace
  SOURCES 
    ser_pathtrace.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/pathtrace.rgen
    ${SHD_DIR}/pathtrace.rchit
    ${SHD_DIR}/pathtrace.rmiss
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
    ${SHD_DIR}/dh_bindings.h
    ${SHD_DIR}/gethit.h
    ${SHD_DIR}/payload.h
    ${SHD_DIR}/temperature.glsl
  SHADER_HLSL
    ${SHD_DIR}/pathtrace.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/pathtrace.slang
)

