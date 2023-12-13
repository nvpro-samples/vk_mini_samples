
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(mm_displacement
  SOURCES 
    dmm_process.cpp 
    dmm_process.hpp 
    mm_displacement.cpp
    vk_nv_micromesh.h
    ${SAMPLES_COMMON_DIR}/bird_curve_helper.cpp 
    ${SAMPLES_COMMON_DIR}/bird_curve_helper.hpp
    ${SAMPLES_COMMON_DIR}/bit_packer.hpp
  SHADER_SOURCES 
    ${SHD_DIR}/raytrace.rgen
    ${SHD_DIR}/raytrace.rchit
    ${SHD_DIR}/raytrace.rmiss
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
    ${SHD_DIR}/payload.h
    ${SHD_DIR}/dh_bindings.h
  SHADER_HLSL
    ${SHD_DIR}/raytrace.hlsl 
  SHADER_SLANG
    ${SHD_DIR}/raytrace.slang
)



