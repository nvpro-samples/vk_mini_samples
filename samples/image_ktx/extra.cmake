# HLSL
if(USE_HLSL) 
  compile_hlsl_file(
    SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raster.hlsl 
    FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
    )
  target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
endif()

if(USE_SLANG)
  compile_slang_file(
    SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raster.slang
    FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
    )
  target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
endif()

if(USE_HLSL OR USE_SLANG) 
    # Adding the HLSL header to the Visual Studio project
    file(GLOB SHARE_HLSL ${SAMPLES_COMMON_DIR}/shaders/functions.hlsli)
    target_sources(${PROJECT_NAME} PRIVATE ${SHARE_HLSL})
    source_group("shaders/shared" FILES ${SHARE_HLSL})
endif()

# KTX needs extra libraries
_add_package_KTX()
target_link_libraries (${PROJECT_NAME} basisu)