cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(vk_mini_samples LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

# FetchContent is used for downloading content
include(FetchContent)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(SLANG_VERSION  "2024.0.5")
find_package(Slang)

# Adding HLSL compiler function
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hlsl.cmake)

# Adding SLANG compiler function
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/slang.cmake)


#--------------------------------------------------------------------------------------------------
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()

# Set the install diectory in the project directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/_install" CACHE PATH "folder in which INSTALL will put everything needed to run the binaries" FORCE)

# Various functions and macros REQUIRED
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()



# Various Paths
set(SAMPLES_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)

if(MSVC)
    add_definitions(/wd26812)  # 'enum class' over 'enum'
    add_definitions(/wd26451)  # Arithmetic overflow, casting 4 byte value to 8 byte value
endif()

# Packages shared by all projects
# _add_package_VulkanSDK()
_add_package_ZLIB()
_add_package_ImGUI()
_add_package_NVML()
_add_package_ShaderC() # implicitly including Vulkan

set(VULKANSDK_BUILD_DEPENDENCIES FALSE CACHE BOOL "Build dependecies for GLSL shaders")

option(USE_HLSL "Use the HLSL shaders for this sample")
option(USE_SLANG "Use the Slang shaders for this sample")

if(${USE_HLSL} AND ${USE_SLANG})
  message(FATAL_ERROR "Cannot use both HLSL and Slang simultaneously.")
endif()

set(USE_GLSL 1)
if(${USE_HLSL} OR ${USE_SLANG})
    set(USE_GLSL 0)
endif()


# Need to be added last, as it uses defines from packages
_add_nvpro_core_lib()

# Copying media files to the EXE destination
message(STATUS "COPY ${CMAKE_CURRENT_SOURCE_DIR}/media  to  ${OUTPUT_PATH}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION ${OUTPUT_PATH})



function(makeRelative FROM TO OUT)
    file(RELATIVE_PATH _TMP_STR "${FROM}" "${TO}")
    set (${OUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

function(create_sample  PROJECT_NAME)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES SHADER_SOURCES SHADER_HEADERS SHADER_HLSL SHADER_SLANG FLAGS)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Executable
    #get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message(STATUS "-------------------------------")
    message(STATUS "Processing Project ${PROJECT_NAME}")
    add_executable(${PROJECT_NAME})
    
    # Force good level of warnings
    target_compile_options(${PROJECT_NAME} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/W3>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
    )

    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    SET(SAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})

    # Sources for the project
    target_sources(${PROJECT_NAME} PRIVATE ${ARGS_SOURCES})
    target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SOURCE_FILES})  # Extra source from nvpro-core based on options
    target_sources(${PROJECT_NAME} PRIVATE ${PACKAGE_SOURCE_FILES}) # Extra source from nvpro-core based on options
    target_sources(${PROJECT_NAME} PRIVATE ${ARGS_SHADER_HEADERS})

    # Folders for Visual Studio
    source_group("Other" FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
    source_group("Shaders" FILES 
      ${ARGS_SHADER_SOURCES} 
      ${ARGS_SHADER_HEADERS}
      ${ARGS_SHADER_HLSL}
      ${ARGS_SHADER_SLANG}
    )

    # Readme 
    target_sources(${PROJECT_NAME} PRIVATE ${SAMPLE_FOLDER}/README.md)

    # Include paths
    target_include_directories(${PROJECT_NAME} PRIVATE ${SAMPLES_COMMON_DIR} ${SAMPLE_FOLDER})

    # Linking with other libraries
    target_link_libraries(${PROJECT_NAME} 
        nvpro_core
        optimized ${LIBRARIES_OPTIMIZED}
        debug ${LIBRARIES_DEBUG}
        ${PLATFORM_LIBRARIES}
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
        # GLSL compilation flags
        set(_GLSL_FLAGS 
          -I${SAMPLE_FOLDER}/shaders 
          -I${NVPRO_CORE_DIR} 
          -g -D_glsl)
        list(APPEND _GLSL_FLAGS ${ARGS_FLAGS})

        # Compiling shaders to Spir-V header
        compile_glsl(
            SOURCE_FILES ${ARGS_SHADER_SOURCES}
            HEADER_FILES ${ARGS_SHADER_HEADERS}
            DST "${SAMPLE_FOLDER}/_autogen" 
            VULKAN_TARGET "vulkan1.3"
            HEADER ON
            DEPENDENCY ON
            FLAGS ${_GLSL_FLAGS}
            )

        target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
    endif()

    if(USE_HLSL AND ARGS_SHADER_HLSL) 
        foreach(HLSL_FILE ${ARGS_SHADER_HLSL})
          compile_hlsl_file(
            SOURCE_FILE ${HLSL_FILE}
            FLAGS -I${SAMPLES_ROOT_DIR} ${ARGS_FLAGS}
            )
          target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
          source_group("Shaders" FILES ${SLANG_OUTPUT_FILES})
        endforeach()
    endif()

    if(USE_SLANG)
      target_include_directories(${PROJECT_NAME} PRIVATE ${SLANG_SDK})
      target_link_libraries(${PROJECT_NAME} ${SLANG_LIB})
      if(ARGS_SHADER_SLANG) 
          foreach(SLANG_FILE ${ARGS_SHADER_SLANG})
              compile_slang_file(
                SOURCE_FILE ${SLANG_FILE}
                FLAGS -I${SAMPLES_ROOT_DIR} ${ARGS_FLAGS}
                )
              target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
              source_group("Shaders" FILES ${SLANG_OUTPUT_FILES})
          endforeach()
      endif()
    endif()

    # target definitions for GLSL, HLSL and SLANG
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GLSL=${USE_GLSL})
    if(${USE_GLSL})
      target_compile_definitions(${PROJECT_NAME} PRIVATE SHADER_LANGUAGE_STR="GLSL")
    endif()
    if(${USE_SLANG})
      target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SLANG=1)
      target_compile_definitions(${PROJECT_NAME} PRIVATE SHADER_LANGUAGE_STR="Slang")
    else()
      target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SLANG=0)
    endif()
    if(${USE_HLSL})
      target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=1)
      target_compile_definitions(${PROJECT_NAME} PRIVATE SHADER_LANGUAGE_STR="HLSL")
    else()
      target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=0)
    endif()


    # Copy binary
    _finalize_target( ${PROJECT_NAME} )

endfunction()


# Sub examples
add_subdirectory(samples)

# Install - copying the media directory
install(DIRECTORY "media" 
        CONFIGURATIONS Release 
        DESTINATION "bin_${ARCH}")
install(DIRECTORY "media" 
        CONFIGURATIONS Debug 
        DESTINATION "bin_${ARCH}_debug")


# Fake project to add files
set(EXTRAFILES 
  CMakeLists.txt 
  README.md
  test.py
  cmake/hlsl.cmake
  cmake/slang.cmake
  cmake/find_entrypoints.cmake
  cmake/FindSlang.cmake
  samples/CMakeLists.txt
)
add_custom_target(Settings SOURCES ${EXTRAFILES})
set_property(TARGET Settings PROPERTY FOLDER "_cmake")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${EXTRAFILES})

