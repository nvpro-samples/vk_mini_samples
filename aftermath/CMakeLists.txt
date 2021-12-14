#*****************************************************************************
# Copyright 2020 NVIDIA Corporation. All rights reserved.
#*****************************************************************************

cmake_minimum_required(VERSION 3.9.6 FATAL_ERROR)

#--------------------------------------------------------------------------------------------------
# Project setting
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(PROJNAME vk_${PROJNAME})
project(${PROJNAME} LANGUAGES C CXX)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJNAME}:")
set(COMMON_DIR ${SAMPLES_ROOT_DIR}/common)

#--------------------------------------------------------------------------------------------------
# C++ target and defines
set(CMAKE_CXX_STANDARD 17)
add_executable(${PROJNAME})
_add_project_definitions(${PROJNAME})

#--------------------------------------------------------------------------------------------------
# Shaders in project
#
set(SHD_COMMON_DIR ${COMMON_DIR}/shaders)
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Common shaders
file(GLOB SHD_COMMON_HDR  ${SHD_COMMON_DIR}/*.glsl)
set(SHD_COMMON_SRC 
    ${SHD_COMMON_DIR}/pathtrace.rahit
    ${SHD_COMMON_DIR}/pathtrace.rgen
    ${SHD_COMMON_DIR}/pathtrace.rchit
    ${SHD_COMMON_DIR}/pathtrace.rmiss
    ${SHD_COMMON_DIR}/passthrough.vert
    ${SHD_COMMON_DIR}/post.frag
    ${SHD_COMMON_DIR}/raster.vert
)

# Local shader
file(GLOB SHD_HDR  ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h)
set(SHD_SRC 
    ${SHD_DIR}/raster.frag
    )

# Compiling shaders to Spir-V header
compile_glsl(
    SOURCE_FILES ${SHD_COMMON_SRC} ${SHD_SRC}
    HEADER_FILES ${SHD_COMMON_HDR} ${SHD_HDR}
    DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
    VULKAN_TARGET "vulkan1.2"
    HEADER ON
    FLAGS "-I${SHD_DIR} -I${SAMPLES_ROOT_DIR}"
    )


macro(copy_dll dlls)
    add_custom_command(TARGET ${PROJNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${dlls} $<TARGET_FILE_DIR:${PROJNAME}>)
endmacro()

#####################################################################################
# nSight Aftermath
macro(_add_package_Aftermath)
    if (NOT DEFINED $ENV{NSIGHT_AFTERMATH_SDK})
      if(UNIX)
        set(AFTERMATH_URL "https://developer.nvidia.com/rdp/assets/nsight-aftermath-sdk-2021_1-linux-package")
        set(AFTERMATH_FILE "${DOWNLOAD_TARGET_DIR}/aftermath.tgz")
      else()
        set(AFTERMATH_URL "https://developer.nvidia.com/rdp/assets/nsight-aftermath-sdk-2021_1-windows-package")
        set(AFTERMATH_FILE "${DOWNLOAD_TARGET_DIR}/aftermath.zip")
      endif()

      set(AFTERMATH_DIR "${DOWNLOAD_TARGET_DIR}/aftermath")

      if (NOT EXISTS ${AFTERMATH_DIR})
        if (NOT EXISTS ${AFTERMATH_FILE})
          message(STATUS "Downloading Aftermath SDK...")
          file(DOWNLOAD ${AFTERMATH_URL}
              ${AFTERMATH_FILE}
              SHOW_PROGRESS
              STATUS STAT
              LOG  log)
    #     message(STATUS "Status: ${STAT}
    #                    Log: ${log}")
        endif()
        file(MAKE_DIRECTORY ${AFTERMATH_DIR})
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${AFTERMATH_FILE}
                        WORKING_DIRECTORY ${AFTERMATH_DIR})
      endif()
      set(NSIGHT_AFTERMATH_SDK ${AFTERMATH_DIR})
    else()
      set(NSIGHT_AFTERMATH_SDK  $ENV{NSIGHT_AFTERMATH_SDK} CACHE STRING "Path to the Aftermath SDK")
    endif()

    find_package(NsightAftermath)

    if(NsightAftermath_FOUND)
        add_definitions(-DNVVK_SUPPORTS_AFTERMATH)
        include_directories(${NsightAftermath_INCLUDE_DIRS})
        LIST(APPEND LIBRARIES_OPTIMIZED ${NsightAftermath_LIBRARIES})
        LIST(APPEND LIBRARIES_DEBUG ${NsightAftermath_LIBRARIES})
    endif(NsightAftermath_FOUND)

endmacro(_add_package_Aftermath)


#--------------------------------------------------------------------------------------------------
# Source files for this project
#
set(EXTRA_COMMON ${COMMON_DIR}/src/vulkan_sample.cpp ${COMMON_DIR}/src/vulkan_sample.hpp)
list(APPEND COMMON_SOURCE_FILES ${EXTRA_COMMON})
file(GLOB SOURCE_FILES src/*.*)

include_directories(${SAMPLES_ROOT_DIR})

target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${COMMON_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${PACKAGE_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${GLSL_SOURCES} ${GLSL_HEADERS})

#--------------------------------------------------------------------------------------------------
# Sub-folders in Visual Studio
#
source_group("Common"           FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group("Shader Common"    FILES ${SHD_COMMON_SRC} ${SHD_COMMON_HDR})
source_group("Shader Local"     FILES ${SHD_SRC} ${SHD_HDR})
source_group("Source Files"     FILES ${SOURCE_FILES})

#--------------------------------------------------------------------------------------------------
# Aftermath
#
if (NOT DEFINED $ENV{NSIGHT_AFTERMATH_SDK})
    set(DOWNLOAD_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")

    if (NOT DEFINED $ENV{NSIGHT_AFTERMATH_SDK})
        # Version of the SDK to download
        if(UNIX)
            set(AFTERMATH_URL "https://developer.nvidia.com/rdp/assets/nsight-aftermath-sdk-2021_1-linux-package")
            set(AFTERMATH_FILE "${DOWNLOAD_TARGET_DIR}/aftermath.tgz")
        else()
            set(AFTERMATH_URL "https://developer.nvidia.com/rdp/assets/nsight-aftermath-sdk-2021_1-windows-package")
            set(AFTERMATH_FILE "${DOWNLOAD_TARGET_DIR}/aftermath.zip")
        endif()

        set(AFTERMATH_DIR "${DOWNLOAD_TARGET_DIR}/aftermath")

        # Download Aftermath SDK
        if (NOT EXISTS ${AFTERMATH_DIR})
            if (NOT EXISTS ${AFTERMATH_FILE})
              message(STATUS "Downloading Aftermath SDK...")
              file(DOWNLOAD ${AFTERMATH_URL}
                  ${AFTERMATH_FILE}
                  SHOW_PROGRESS
                  STATUS STAT
                  LOG  log)
            endif()
            file(MAKE_DIRECTORY ${AFTERMATH_DIR})
            # Unzip package
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${AFTERMATH_FILE} WORKING_DIRECTORY ${AFTERMATH_DIR})
        endif()
        set(NSIGHT_AFTERMATH_SDK ${AFTERMATH_DIR} CACHE PATH "Path to the Aftermath SDK")
    else()
      set(NSIGHT_AFTERMATH_SDK  $ENV{NSIGHT_AFTERMATH_SDK} CACHE PATH "Path to the Aftermath SDK")
    endif()

    find_package(NsightAftermath)

    if(NsightAftermath_FOUND)
        add_definitions(-DNVVK_SUPPORTS_AFTERMATH)
        # list(APPEND PLATFORM_LIBRARIES NsightAftermath::NsightAftermath)
        list(APPEND LIBRARIES_OPTIMIZED ${NsightAftermath_LIBRARIES})
        list(APPEND LIBRARIES_DEBUG ${NsightAftermath_LIBRARIES})
        include_directories(${NsightAftermath_INCLUDE_DIRS})
        string(REPLACE ".lib" ".dll" NsightAftermath_DLL ${NsightAftermath_LIBRARY})
        copy_dll(${NsightAftermath_DLL})
    else()
        message("\n\n============= ERROR =============")
        message("Path to the Aftermath SDK is missing.")
        message("Please follow steps in aftermath/README.md")
        message("============= ERROR =============\n\n")
    endif()
endif()

#--------------------------------------------------------------------------------------------------
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} nvpro_core)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#--------------------------------------------------------------------------------------------------
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target( ${PROJNAME} )

if(NsightAftermath_FOUND)
    install(FILES ${NsightAftermath_DLL}  CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
    install(FILES ${NsightAftermath_DLL}  CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
endif()

