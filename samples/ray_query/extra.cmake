# HLSL
if(USE_HLSL)
  compile_hlsl_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/ray_query.hlsl 
      FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
      )
  target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
endif()

if(USE_SLANG)
  compile_slang_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/ray_query.slang
      FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
      )
  target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
endif()

if(USE_HLSL) 
    # Adding the HLSL header to the Visual Studio project
    file(GLOB SHARE_HLSL ${SAMPLES_COMMON_DIR}/shaders/*.hlsl ${SAMPLES_COMMON_DIR}/shaders/*.hlsli ${SAMPLES_COMMON_DIR}/shaders/*.h)
    target_sources(${PROJECT_NAME} PRIVATE ${SHARE_HLSL})
    source_group("shaders/shared" FILES ${SHARE_HLSL})
endif()
