cmake_minimum_required(VERSION 3.22...3.31)
project(vk_mini_samples VERSION 2.0 LANGUAGES C CXX)

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# Add common library first as other samples might depend on it
add_subdirectory(common)

add_subdirectory(samples/barycentric_wireframe)
add_subdirectory(samples/compute_multi_threaded)
add_subdirectory(samples/compute_only)
add_subdirectory(samples/crash_aftermath)
add_subdirectory(samples/gltf_raytrace)
add_subdirectory(samples/gpu_monitor)
add_subdirectory(samples/image_ktx)
add_subdirectory(samples/image_viewer)
add_subdirectory(samples/line_stipple)
add_subdirectory(samples/memory_budget)
add_subdirectory(samples/mm_opacity)
add_subdirectory(samples/msaa)
add_subdirectory(samples/offscreen)
add_subdirectory(samples/ray_query)
add_subdirectory(samples/ray_query_position_fetch)
add_subdirectory(samples/ray_trace)
add_subdirectory(samples/ray_trace_motion_blur)
add_subdirectory(samples/ray_tracing_position_fetch)
add_subdirectory(samples/realtime_analysis)
add_subdirectory(samples/rectangle)
add_subdirectory(samples/ser_pathtrace)
add_subdirectory(samples/shader_object)
add_subdirectory(samples/shader_printf)
add_subdirectory(samples/simple_polygons)
add_subdirectory(samples/solid_color)
# add_subdirectory(samples/swept_sphere)
add_subdirectory(samples/texture_3d)
add_subdirectory(samples/tiny_shader_toy)


# Make Visual Studio use this project as the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ray_trace)

# Install the resources and nvshaders folders
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/resources" DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/common/common_shaders" DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY "${NVSHADERS_DIR}/nvshaders" DESTINATION ${CMAKE_INSTALL_PREFIX})


