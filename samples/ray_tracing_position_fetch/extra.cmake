if(USE_HLSL)
# HLSL
compile_hlsl_file(
    SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raytrace.hlsl 
    DST ${SAMPLE_FOLDER}/_autogen/raytrace_rgenMain.spv
    FLAGS "--extra_arguments=-I ${SAMPLES_COMMON_DIR}/shaders"
    )
target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
endif()

if(USE_SLANG)
  # SLANG
  compile_slang_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raytrace.slang
      FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
      )
  target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
endif()
