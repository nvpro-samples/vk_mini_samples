cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(vk_mini_samples LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

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
  if(NOT BASE_DIRECTORY)
    message(FATAL_ERROR "Could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
  endif()
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

# Adding Slang
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(Slang_VERSION  "2025.6.1")
find_package(Slang)

# Adding HLSL compiler function
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hlsl.cmake)

# Adding SLANG compiler function
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/slang.cmake)

# Various Paths
set(SAMPLES_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)

if(MSVC)
    add_definitions(/wd26812)  # 'enum class' over 'enum'
    add_definitions(/wd26451)  # Arithmetic overflow, casting 4 byte value to 8 byte value
endif()


if(NOT VULKAN_BUILD_DEPENDENCIES)
  set(VULKAN_BUILD_DEPENDENCIES OFF CACHE BOOL "Create dependencies on GLSL files")
endif()

# Packages shared by all projects
_add_package_VulkanSDK()        # All
_add_package_ImGUI()            # All
_add_package_NsightAftermath()  # crash_aftermath
_add_package_ZLIB()             # image_ktx
_add_package_KTX()              # image_ktx
_add_package_NVML()             # gpu_monitor
_add_package_NVAPI()            # compute_mutil_threaded
_add_package_ShaderC()          # tiny_shader_toy

# Initialize variables
set(USE_GLSL OFF )
set(USE_HLSL OFF)
set(USE_SLANG OFF)

# Define the available choices for shader languages
set(CHOICES "GLSL" "HLSL" "SLANG")
set(USE_SHADER_LANGUAGE "GLSL" CACHE STRING "Choose the shading language to use.")
set_property(CACHE USE_SHADER_LANGUAGE PROPERTY STRINGS ${CHOICES}) # Provide the choices to the user in the CMake GUI
if(USE_SHADER_LANGUAGE STREQUAL "GLSL")
   set(USE_GLSL ON)
elseif(USE_SHADER_LANGUAGE STREQUAL "HLSL")
    set(USE_HLSL ON)
elseif(USE_SHADER_LANGUAGE STREQUAL "SLANG")
    set(USE_SLANG ON)
else()
    message(FATAL_ERROR "Invalid choice for USE_SHADER_LANGUAGE: ${USE_SHADER_LANGUAGE}")
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
    set(EXECUTABLE_NAME "vk_mini_${PROJECT_NAME}")
    add_executable(${EXECUTABLE_NAME})
    
    # Force good level of warnings
    target_compile_options(${EXECUTABLE_NAME} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/W3>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
    )

    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    SET(SAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})

    # Sources for the project
    target_sources(${EXECUTABLE_NAME} PRIVATE ${ARGS_SOURCES})
    target_sources(${EXECUTABLE_NAME} PRIVATE ${COMMON_SOURCE_FILES})  # Extra source from nvpro-core based on options
    target_sources(${EXECUTABLE_NAME} PRIVATE ${PACKAGE_SOURCE_FILES}) # Extra source from nvpro-core based on options
    target_sources(${EXECUTABLE_NAME} PRIVATE ${ARGS_SHADER_HEADERS})

    # Folders for Visual Studio
    source_group("Other" FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
    source_group("Shaders" FILES 
      ${ARGS_SHADER_SOURCES} 
      ${ARGS_SHADER_HEADERS}
      ${ARGS_SHADER_HLSL}
      ${ARGS_SHADER_SLANG}
    )

    # Readme 
    target_sources(${EXECUTABLE_NAME} PRIVATE ${SAMPLE_FOLDER}/README.md)

    # Include paths
    target_include_directories(${EXECUTABLE_NAME} PRIVATE ${SAMPLES_ROOT_DIR} ${SAMPLE_FOLDER})

    # Linking with other libraries
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
        nvpro_core
        vk_mini_samples_common
        optimized ${LIBRARIES_OPTIMIZED}
        debug ${LIBRARIES_DEBUG}
        ${PLATFORM_LIBRARIES}
        ${UNIXLINKLIBS}
    )

    # Compile definitions
    # the "config" directory doesn't really exist but serves as place holder
    makeRelative("${OUTPUT_PATH}/config" "${SAMPLE_FOLDER}" TO_CURRENT_SOURCE_DIR)
    makeRelative("${OUTPUT_PATH}/config" "${DOWNLOAD_TARGET_DIR}" TO_DOWNLOAD_TARGET_DIR)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PROJECT_NAME="${EXECUTABLE_NAME}")
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PROJECT_RELDIRECTORY="${TO_CURRENT_SOURCE_DIR}/")
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PROJECT_DOWNLOAD_RELDIRECTORY="${TO_DOWNLOAD_TARGET_DIR}/")

    # TinyGLTF
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE TINYGLTF_NO_EXTERNAL_IMAGE)

    # other properties
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY FOLDER "vk_mini_samples")

    # Using GLSL (by default)
    if(USE_GLSL) 
        # GLSL compilation flags
        set(_GLSL_FLAGS 
          -I${SAMPLE_FOLDER}/shaders 
          -I${NVPRO_CORE_DIR} 
          -g # Tell the compiler to embed the source code in the binary
          -D_glsl)
        list(APPEND _GLSL_FLAGS ${ARGS_FLAGS})

        # Compiling shaders to Spir-V header
        compile_glsl(
            SOURCE_FILES ${ARGS_SHADER_SOURCES}
            HEADER_FILES ${ARGS_SHADER_HEADERS}
            DST "${SAMPLE_FOLDER}/_autogen" 
            VULKAN_TARGET "vulkan1.3"
            HEADER ON
            DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
            FLAGS ${_GLSL_FLAGS}
            )

        target_sources(${EXECUTABLE_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
    endif()

    if(USE_HLSL AND ARGS_SHADER_HLSL) 
        foreach(HLSL_FILE ${ARGS_SHADER_HLSL})
          compile_hlsl_file(
            SOURCE_FILE ${HLSL_FILE}
            FLAGS -I${SAMPLES_ROOT_DIR} ${ARGS_FLAGS}
            )
          target_sources(${EXECUTABLE_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
          source_group("Shaders" FILES ${SLANG_OUTPUT_FILES})
        endforeach()
    endif()

    if(USE_SLANG)
      target_include_directories(${EXECUTABLE_NAME} PRIVATE ${Slang_SDK}/include)
      target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${Slang_LIBRARY})
      if(ARGS_SHADER_SLANG) 
          foreach(SLANG_FILE ${ARGS_SHADER_SLANG})
              compile_slang_file(
                SOURCE_FILE ${SLANG_FILE}
                FLAGS -I${SAMPLES_ROOT_DIR} -I${NVPRO_CORE_DIR} ${ARGS_FLAGS}
                )
              target_sources(${EXECUTABLE_NAME} PRIVATE ${SLANG_FILE})
              source_group("Shaders" FILES ${SLANG_FILE})
          endforeach()
      endif()
    endif()

    # target definitions for GLSL, HLSL and SLANG
    if(USE_GLSL)
      target_compile_definitions(${EXECUTABLE_NAME} PRIVATE USE_GLSL=1 USE_HLSL=0 USE_SLANG=0)
      target_compile_definitions(${EXECUTABLE_NAME} PRIVATE SHADER_LANGUAGE_STR="GLSL")
    elseif(USE_HLSL)
      target_compile_definitions(${EXECUTABLE_NAME} PRIVATE USE_GLSL=0 USE_HLSL=1 USE_SLANG=0)
      target_compile_definitions(${EXECUTABLE_NAME} PRIVATE SHADER_LANGUAGE_STR="HLSL")
    elseif(USE_SLANG)
      target_compile_definitions(${EXECUTABLE_NAME} PRIVATE USE_GLSL=0 USE_HLSL=0 USE_SLANG=1)
      target_compile_definitions(${EXECUTABLE_NAME} PRIVATE SHADER_LANGUAGE_STR="Slang")
    endif()

    # Copy binary
    _finalize_target( ${EXECUTABLE_NAME} )

endfunction()



# Sub examples
add_subdirectory(common)
add_subdirectory(samples)

#--------------------------------------------------------------------------------------------------
# Install - copying the media directory
message(STATUS "COPY ${CMAKE_CURRENT_SOURCE_DIR}/media  to  ${OUTPUT_PATH}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION ${OUTPUT_PATH})
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION "${OUTPUT_PATH}/Debug")
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION "${OUTPUT_PATH}/Release")
install(DIRECTORY "media" CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
install(DIRECTORY "media" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")


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
add_custom_target(vk_mini_samples_settings SOURCES ${EXTRAFILES})
set_property(TARGET vk_mini_samples_settings PROPERTY FOLDER "_cmake")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${EXTRAFILES})

