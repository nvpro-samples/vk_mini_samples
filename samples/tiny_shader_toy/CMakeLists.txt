
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

set(PROJECT_NAME tiny_shader_toy)

create_sample(${PROJECT_NAME}
  SOURCES 
    tiny_shader_toy.cpp 
  SHADER_SOURCES 
    ${SHD_DIR}/raster.vert.glsl
    ${SHD_DIR}/raster.frag.glsl
  SHADER_HEADERS
    ${SHD_DIR}/buffer_a.glsl
    ${SHD_DIR}/common.glsl
    ${SHD_DIR}/image.glsl
 SHADER_SLANG
   ${SHD_DIR}/raster.slang
)

# Install - copying the media directory
install(DIRECTORY "shaders" 
        CONFIGURATIONS Release 
        DESTINATION "bin_${ARCH}/tiny_shader_toy")
install(DIRECTORY "shaders" 
        CONFIGURATIONS Debug 
        DESTINATION "bin_${ARCH}_debug/tiny_shader_toy")


        # On Windows, linking release shaderc with a debug app produces linker errors.
# So we use release shaderc_shared on Windows and release shaderc_combined on Linux.
if(WIN32 AND USE_SLANG)
    set(EXECUTABLE_NAME "vk_mini_${PROJECT_NAME}")
  add_custom_command(TARGET ${EXECUTABLE_NAME} POST_BUILD 
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Slang_DLL} ${Slang_glslang_DLL} $<TARGET_FILE_DIR:${EXECUTABLE_NAME}> VERBATIM
      )
  install(FILES ${Slang_DLL} ${Slang_glslang_DLL} DESTINATION bin_${ARCH}$<IF:$<CONFIG:Debug>,_debug,>)
endif()
