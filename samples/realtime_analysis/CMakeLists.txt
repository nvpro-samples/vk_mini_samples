
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(realtime_analysis
  SOURCES 
    realtime_analysis.cpp 
    realtime_analysis.h 
  SHADER_SOURCES 
    ${SHD_DIR}/external_forces.comp.glsl
    ${SHD_DIR}/update_spatial_hash.comp.glsl
    ${SHD_DIR}/update_positions.comp.glsl
    ${SHD_DIR}/calculate_densities.comp.glsl
    ${SHD_DIR}/calculate_pressure_force.comp.glsl
    ${SHD_DIR}/calculate_viscosity.comp.glsl
    ${SHD_DIR}/bitonic_sort.comp.glsl
    ${SHD_DIR}/bitonic_sort_offsets.comp.glsl
    ${SHD_DIR}/raster.vert.glsl
    ${SHD_DIR}/raster.frag.glsl
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
    ${SHD_DIR}/fluid_sim_2D.glsl
    ${SHD_DIR}/fluid_maths_2D.glsl
    ${SHD_DIR}/spatial_hash.glsl
    ${SHD_DIR}/layouts.glsl
  SHADER_SLANG
    ${SHD_DIR}/fluid_sim_2D.slang
    ${SHD_DIR}/raster.slang
)
