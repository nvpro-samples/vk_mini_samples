#*****************************************************************************
# Copyright 2020 NVIDIA Corporation. All rights reserved.
#*****************************************************************************

#--------------------------------------------------------------------------------------------------
# Project setting
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(PROJNAME vk_${PROJNAME})
project(${PROJNAME} LANGUAGES C CXX)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJNAME}:")


#--------------------------------------------------------------------------------------------------
# C++ target and defines
add_executable(${PROJNAME})
_add_project_definitions(${PROJNAME})

#--------------------------------------------------------------------------------------------------
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.*)

file(GLOB EXTRA_COMMON ${SAMPLES_ROOT_DIR}/common/*.*)
list(APPEND COMMON_SOURCE_FILES ${EXTRA_COMMON})
include_directories(${SAMPLES_ROOT_DIR}/common)

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
# GLSL to SPIR-V custom build
set(SAMPLES_AFTERMATH_DEPENDENCIES OFF CACHE BOOL "Create dependencies on GLSL files")
compile_glsl_directory(
	SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
	DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
	VULKAN_TARGET "vulkan1.2"
	HEADER ON
	DEPENDENCY ${SAMPLES_AFTERMATH_DEPENDENCIES}
	)


#--------------------------------------------------------------------------------------------------
# Sources
target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${COMMON_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${PACKAGE_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${GLSL_SOURCES} ${GLSL_HEADERS})

#--------------------------------------------------------------------------------------------------
# Sub-folders in Visual Studio
#
source_group("Common"           FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group("Shader Sources"   FILES ${GLSL_SOURCES})
source_group("Shader Headers"   FILES ${GLSL_HEADERS})
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

