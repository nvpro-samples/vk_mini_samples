# This is the list of all samples supported in GLSL, Slang and HLSL

add_subdirectory(barycentric_wireframe)
add_subdirectory(compute_only)
add_subdirectory(compute_multi_threaded)
add_subdirectory(crash_aftermath) 
add_subdirectory(gltf_raytrace)
add_subdirectory(gpu_monitor)
add_subdirectory(image_ktx)
add_subdirectory(image_viewer)
add_subdirectory(line_stipple)
add_subdirectory(memory_budget) 
add_subdirectory(mm_opacity)
add_subdirectory(msaa)
add_subdirectory(offscreen)
add_subdirectory(ray_query)
add_subdirectory(ray_trace)
add_subdirectory(rectangle)
add_subdirectory(ser_pathtrace)
add_subdirectory(shader_object)
add_subdirectory(shader_printf)
add_subdirectory(simple_polygons)
add_subdirectory(solid_color)
add_subdirectory(texture_3d)

# Following are restricted samples
if(NOT USE_HLSL) 
  add_subdirectory(ray_query_position_fetch)   # No support for fetch
  add_subdirectory(ray_tracing_position_fetch) # No support for fetch
  add_subdirectory(realtime_analysis) # Missing some functions to make realtime analysis
  add_subdirectory(ray_trace_motion_blur) # no motion
  add_subdirectory(tiny_shader_toy)
endif()

if(NOT USE_HLSL AND NOT USE_SLANG) # Not yet implemented
 # add_subdirectory(tiny_shader_toy)
endif()
