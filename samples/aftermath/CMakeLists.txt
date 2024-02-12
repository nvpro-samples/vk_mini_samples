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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# Aftermath
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Copy the Aftermath dll where the project executable is
macro(copy_dll dlls)
    add_custom_command(TARGET aftermath POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${dlls} $<TARGET_FILE_DIR:aftermath>)
endmacro()

# Searching for the Aftermath SDK
set(AFTERMATH_SDK ${CMAKE_CURRENT_SOURCE_DIR}/aftermath/aftermath_sdk)
if ((NOT DEFINED ENV{NSIGHT_AFTERMATH_SDK}) AND (EXISTS ${AFTERMATH_SDK}/include))
    set(NSIGHT_AFTERMATH_SDK ${AFTERMATH_SDK} CACHE PATH "AftermathSDK" FORCE)
endif()
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(NsightAftermath)

if(NsightAftermath_FOUND)
    message(STATUS "Found Aftermath under:" ${NsightAftermath_LIBRARY})
    target_compile_definitions(aftermath PRIVATE USE_NSIGHT_AFTERMATH)
    target_link_libraries (aftermath ${NsightAftermath_LIBRARIES})
    target_include_directories(aftermath PRIVATE ${NsightAftermath_INCLUDE_DIRS})
    string(REPLACE ".lib" ".dll" NsightAftermath_DLL ${NsightAftermath_LIBRARY})
    copy_dll(${NsightAftermath_DLL})
    install(FILES ${NsightAftermath_DLL} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
    install(FILES ${NsightAftermath_DLL} CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
else()
    message("\n\n============= ERROR =============")
    message("Path to the Aftermath SDK is missing.")
    message("Please follow steps in aftermath/README.md")
    message("============= ERROR =============\n\n")
endif()

