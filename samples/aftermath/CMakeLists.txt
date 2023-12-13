set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

create_sample(aftermath
  SOURCES 
    src/aftermath.cpp 
    src/NsightAftermathGpuCrashTracker.cpp
    src/NsightAftermathGpuCrashTracker.h
    src/NsightAftermathHelpers.h
  SHADER_SOURCES 
    ${SHD_DIR}/raster.vert 
    ${SHD_DIR}/raster.frag
  SHADER_HEADERS
    ${SHD_DIR}/device_host.h
  SHADER_HLSL
    ${SHD_DIR}/raster.hlsl 
  # SHADER_SLANG
  #   ${SHD_DIR}/raster.slang
)
