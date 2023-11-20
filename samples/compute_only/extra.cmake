# HLSL
if(USE_HLSL)
  compile_hlsl_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/shader.hlsl 
      FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
      )
  target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
endif()

if(USE_SLANG)
  compile_slang_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/shader.slang
      FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
      )
  target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
endif()

