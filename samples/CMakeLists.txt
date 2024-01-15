if(NOT USE_SLANG) # Couldn't make it to crash :-) 
add_subdirectory(aftermath)
endif()

if(NOT USE_HLSL) # Bug with EXT_fragment_shader_barycentric
add_subdirectory(barycentric_wireframe)
endif()

add_subdirectory(compute_only)
add_subdirectory(gpu_monitor)
add_subdirectory(image_ktx)
add_subdirectory(image_viewer)
add_subdirectory(mm_displacement)
add_subdirectory(mm_opacity)
add_subdirectory(msaa)
add_subdirectory(offscreen)

#if(NOT USE_SLANG) # No buffer reference
add_subdirectory(ray_query)
#endif()

if(NOT USE_SLANG AND NOT USE_HLSL) # Only supported with GLSL
add_subdirectory(ray_query_position_fetch)
endif()

if(NOT USE_HLSL) # Not supported
add_subdirectory(ray_tracing_position_fetch)
endif()

add_subdirectory(shader_printf)
add_subdirectory(ray_trace)
add_subdirectory(rectangle)

if(NOT USE_HLSL) # New issue with InterlockedMax
add_subdirectory(ser_pathtrace)
endif()

add_subdirectory(shader_object)
add_subdirectory(simple_polygons)
add_subdirectory(solid_color)
add_subdirectory(texture_3d)

#if(NOT USE_SLANG AND NOT USE_HLSL) # Not yet implemented
if(NOT USE_HLSL) # Not yet implemented
add_subdirectory(tiny_shader_toy)
endif()