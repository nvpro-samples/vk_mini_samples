#*****************************************************************************
# Copyright 2020 NVIDIA Corporation. All rights reserved.
#*****************************************************************************

cmake_minimum_required(VERSION 3.9.6 FATAL_ERROR)

#--------------------------------------------------------------------------------------------------
# Project setting
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
    ${SHD_COMMON_DIR}/pathtrace.rchit
    ${SHD_COMMON_DIR}/pathtrace.rmiss
    ${SHD_COMMON_DIR}/passthrough.vert
    ${SHD_COMMON_DIR}/post.frag
    ${SHD_COMMON_DIR}/raster.frag
    ${SHD_COMMON_DIR}/raster.vert
)

# Local shader
file(GLOB SHD_HDR  ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h)
set(SHD_SRC 
    ${SHD_DIR}/pathtrace.rgen
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
