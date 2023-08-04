
function(makeRelative FROM TO OUT)
    file(RELATIVE_PATH _TMP_STR "${FROM}" "${TO}")
    set (${OUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

function(buildSample PROJECT_NAME)

    # Executable
    #get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message(STATUS "-------------------------------")
    message(STATUS "Processing Project ${PROJECT_NAME}")
    add_executable(${PROJECT_NAME})
    
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    SET(SAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME})

    # Sources for the project
    file(GLOB SOURCE_FILES 
        ${SAMPLE_FOLDER}/src/*.cpp 
        ${SAMPLE_FOLDER}/src/*.hpp 
        ${SAMPLE_FOLDER}/src/*.h)
    target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
    target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SOURCE_FILES})  # Extra source from nvpro-core based on options
    target_sources(${PROJECT_NAME} PRIVATE ${PACKAGE_SOURCE_FILES}) # Extra source from nvpro-core based on options
    source_group(TREE ${SAMPLE_FOLDER} FILES ${SOURCE_FILES})
    source_group("Other" FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

    # Readme 
    target_sources(${PROJECT_NAME} PRIVATE ${SAMPLE_FOLDER}/README.md)

    # Include paths
    target_include_directories(${PROJECT_NAME} PRIVATE ${SAMPLES_COMMON_DIR} ${SAMPLE_FOLDER})

    # Linking with other libraries
    target_link_libraries(${PROJECT_NAME} 
        optimized ${LIBRARIES_OPTIMIZED}
        debug ${LIBRARIES_DEBUG}
        ${PLATFORM_LIBRARIES}
        nvpro_core
         ${UNIXLINKLIBS}
    )

    # Compile definitions
    # the "config" directory doesn't really exist but serves as place holder
    makeRelative("${OUTPUT_PATH}/config" "${SAMPLE_FOLDER}" TO_CURRENT_SOURCE_DIR)
    makeRelative("${OUTPUT_PATH}/config" "${DOWNLOAD_TARGET_DIR}" TO_DOWNLOAD_TARGET_DIR)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_RELDIRECTORY="${TO_CURRENT_SOURCE_DIR}/")
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_DOWNLOAD_RELDIRECTORY="${TO_DOWNLOAD_TARGET_DIR}/")

    # other properties
    set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

	# Using GLSL (by default)
	if(USE_GLSL) 
		# Vulkan shaders in project
		set(SHD_DIR ${SAMPLE_FOLDER}/shaders)
		file(GLOB SHD_HDR  ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h)
		file(GLOB SHD_SRC 
			${SHD_DIR}/*.vert 
			${SHD_DIR}/*.frag 
			${SHD_DIR}/*.tesc
			${SHD_DIR}/*.tese
			${SHD_DIR}/*.geom
			${SHD_DIR}/*.comp
			${SHD_DIR}/*.rgen 
			${SHD_DIR}/*.rchit 
			${SHD_DIR}/*.rmiss 
			${SHD_DIR}/*.rahit
			${SHD_DIR}/*.rint
			${SHD_DIR}/*.rcall
		)

		# Compiling shaders to Spir-V header
		compile_glsl(
			SOURCE_FILES ${SHD_SRC}
			HEADER_FILES ${SHD_HDR}
			DST "${SAMPLE_FOLDER}/_autogen" 
			VULKAN_TARGET "vulkan1.3"
			HEADER ON
			DEPENDENCY ${VULKANSDK_BUILD_DEPENDENCIES}
			FLAGS -I${SHD_DIR} -I${NVPRO_CORE_DIR} -g -D__glsl
			)


		target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
		source_group(TREE ${SAMPLE_FOLDER} FILES ${GLSL_SOURCES} ${GLSL_HEADERS})
		target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=0)
	endif()

    if(USE_HLSL) 
        # Adding the HLSL header to the Visual Studio project
        file(GLOB HLSL ${SAMPLE_FOLDER}/shaders/*.hlsl ${SAMPLE_FOLDER}/shaders/*.hlsli ${SAMPLE_FOLDER}/shaders/*.h)
        target_sources(${PROJECT_NAME} PRIVATE ${HLSL})
        source_group("shaders" FILES ${HLSL})
		target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=1)
    endif()

    if(USE_SLANG) 
        # Adding the HLSL header to the Visual Studio project
        file(GLOB SLANG ${SAMPLE_FOLDER}/shaders/*.slang ${SAMPLE_FOLDER}/shaders/*.hlsli ${SAMPLE_FOLDER}/shaders/*.h)
        target_sources(${PROJECT_NAME} PRIVATE ${SLANG})
        source_group("shaders" FILES ${SLANG})
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SLANG=1)
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=0)
    endif()


    # Extra Cmake 
    set(EXTRA_CMAKE ${SAMPLE_FOLDER}/extra.cmake)
    if(EXISTS ${EXTRA_CMAKE})
        include(${EXTRA_CMAKE})
        target_sources(${PROJECT_NAME} PRIVATE ${EXTRA_CMAKE})
    endif()

    # Copy binary
    _finalize_target( ${PROJECT_NAME} )

endfunction()

# List of all samples
set(SAMPLES 
    barycentric_wireframe
    image_ktx
    image_viewer
    mm_displacement
    mm_opacity
    msaa
    offscreen
    shader_printf
    ray_trace
    rectangle
    ser_pathtrace
    simple_polygons
    solid_color
    texture_3d
    )

# Sample only available in HLSL AND GLSL
if(USE_GLSL OR USE_HLSL)
    list(APPEND SAMPLES
        aftermath
        ray_query
    )
endif()


# Sample only available in GLSL
if(USE_GLSL)
    list(APPEND SAMPLES
        ray_tracing_position_fetch
        tiny_shader_toy
    )
endif()

foreach(SAMPLE ${SAMPLES})
    buildSample(${SAMPLE})
endforeach()
