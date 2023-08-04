# Adding the micromesh prototype
set(COMMON_SRC
	${SAMPLES_COMMON_DIR}/bird_curve_helper.cpp 
	${SAMPLES_COMMON_DIR}/bird_curve_helper.hpp
	${SAMPLES_COMMON_DIR}/bit_packer.hpp
	)
target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SRC})
source_group(common FILES ${COMMON_SRC})

# HLSL
if(USE_HLSL)
  compile_hlsl_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raytrace.hlsl 
      FLAGS -I ${SAMPLES_COMMON_DIR}/shaders
      )
  target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
endif()

# SLANG
if(USE_SLANG)
  compile_slang_file(
      SOURCE_FILE ${SAMPLE_FOLDER}/shaders/raytrace.slang
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
