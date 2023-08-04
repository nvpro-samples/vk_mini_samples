if(USE_HLSL)
  # HLSL
  compile_hlsl_file(
    SOURCE_FILE ${SAMPLE_FOLDER}/shaders/pathtrace.hlsl 
    FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
    )
  target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
endif()

if(USE_SLANG)
  # SLANG
  compile_slang_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/pathtrace.slang
      FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
      )
  target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
endif()

if(USE_HLSL OR USE_SLANG) 
    # Adding the HLSL header to the Visual Studio project
    file(GLOB SHARE_HLSL ${SAMPLES_COMMON_DIR}/shaders/*.hlsl ${SAMPLES_COMMON_DIR}/shaders/*.hlsli ${SAMPLES_COMMON_DIR}/shaders/*.h)
    target_sources(${PROJECT_NAME} PRIVATE ${SHARE_HLSL})
    source_group("shaders/shared" FILES ${SHARE_HLSL})
endif()
