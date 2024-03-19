
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(mm_opacity
  SOURCES 
    mm_process.cpp 
    mm_process.hpp 
    mm_opacity.cpp
    ${SAMPLES_COMMON_DIR}/bird_curve_helper.cpp 
	  ${SAMPLES_COMMON_DIR}/bird_curve_helper.hpp
	  ${SAMPLES_COMMON_DIR}/bit_packer.hpp
  SHADER_SOURCES 
    ${SHD_DIR}/raytrace.rgen.glsl
    ${SHD_DIR}/raytrace.rchit.glsl
    ${SHD_DIR}/raytrace.rahit.glsl
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




