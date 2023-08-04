if(USE_HLSL)
# HLSL
compile_hlsl_file(
    SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raster.hlsl 
    DST ${SAMPLE_FOLDER}/_autogen/raster_vertexMain.spirv.h
    )
target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
endif()

if(USE_SLANG)
  # SLANG
  compile_slang_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raster.slang
      )
  target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
endif()
