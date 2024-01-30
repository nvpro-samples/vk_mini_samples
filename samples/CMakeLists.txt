# This is the list of all samples supported in GLSL, Slang and HLSL

add_subdirectory(compute_only)
add_subdirectory(gpu_monitor)
add_subdirectory(image_ktx)
add_subdirectory(image_viewer)
add_subdirectory(mm_displacement)
add_subdirectory(mm_opacity)
add_subdirectory(msaa)
add_subdirectory(offscreen)
add_subdirectory(ray_query)
add_subdirectory(shader_printf)
add_subdirectory(ray_trace)
add_subdirectory(rectangle)
add_subdirectory(shader_object)
add_subdirectory(simple_polygons)
add_subdirectory(solid_color)
add_subdirectory(texture_3d)

# Following are restricted samples

if(NOT USE_SLANG) 
  add_subdirectory(aftermath) # Couldn't make it to crash :-) 
endif()

if(NOT USE_HLSL) 
  add_subdirectory(barycentric_wireframe) # Bug with EXT_fragment_shader_barycentric
  add_subdirectory(ray_query_position_fetch)   # No support for fetch
  add_subdirectory(ray_tracing_position_fetch) # No support for fetch
  add_subdirectory(ser_pathtrace) # no support for SER
endif()

if(NOT USE_HLSL) # Not yet implemented
  add_subdirectory(tiny_shader_toy)
endif()
