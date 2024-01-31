
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(realtime_analysis
  SOURCES 
    realtime_analysis.cpp 
    realtime_analysis.h 
  SHADER_SOURCES 
    ${SHD_DIR}/external_forces.comp
    ${SHD_DIR}/update_spatial_hash.comp
    ${SHD_DIR}/update_positions.comp
    ${SHD_DIR}/calculate_densities.comp
    ${SHD_DIR}/calculate_pressure_force.comp
    ${SHD_DIR}/calculate_viscosity.comp
    ${SHD_DIR}/bitonic_sort.comp
    ${SHD_DIR}/bitonic_sort_offsets.comp
    ${SHD_DIR}/raster.vert
    ${SHD_DIR}/raster.frag
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
