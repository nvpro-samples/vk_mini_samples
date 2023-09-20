# HLSL
compile_hlsl_file(
    SOURCE_FILE ${SAMPLE_FOLDER}/shaders/ray_query.hlsl 
    DST ${SAMPLE_FOLDER}/_autogen/ray_query_computeMain.spv
    FLAGS "--extra_arguments=-I ${SAMPLES_COMMON_DIR}/shaders"
    )

if(USE_HLSL) 
    # Adding the HLSL header to the Visual Studio project
    file(GLOB SHARE_HLSL ${SAMPLES_COMMON_DIR}/shaders/*.hlsl ${SAMPLES_COMMON_DIR}/shaders/*.hlsli ${SAMPLES_COMMON_DIR}/shaders/*.h)
    target_sources(${PROJECT_NAME} PRIVATE ${SHARE_HLSL})
    source_group("shaders/shared" FILES ${SHARE_HLSL})
endif()
