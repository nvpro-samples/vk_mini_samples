cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(vk_mini_samples LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

#configure the external projects
set(EXTERNALS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${EXTERNALS_DIR}/_build")


# FetchContent is used for downloading content
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALS_DIR}  CACHE PATH "Directory under which to collect all populated content" FORCE)
set(FETCHCONTENT_QUIET FALSE CACHE BOOL "Enables QUIET option for all content population" FORCE)

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

# Adding HLSL compiler function
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hlsl.cmake)

# Adding SLANG compiler function
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/slang.cmake)

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
_add_package_ShaderC() # implicitly including Vulkan
find_package(Python REQUIRED Interpreter) # Find Python executable
message(STATUS "--> using Python from: ${Python_EXECUTABLE}")

set(VULKANSDK_BUILD_DEPENDENCIES FALSE CACHE BOOL "Build dependecies for GLSL shaders")

option(USE_HLSL "Use the HLSL shaders for this sample")
option(USE_SLANG "Use the Slang shaders for this sample")

if(${USE_HLSL} AND ${USE_SLANG})
  message(FATAL_ERROR "Cannot use both HLSL and Slang simultaneously.")
endif()

if(${USE_HLSL} OR ${USE_SLANG})
	set(USE_GLSL FALSE)
else()
	set(USE_GLSL TRUE)
endif()


# Need to be added last, as it uses defines from packages
_add_nvpro_core_lib()

# Copying media files to the EXE destination
message(STATUS "COPY ${CMAKE_CURRENT_SOURCE_DIR}/media  to  ${OUTPUT_PATH}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION ${OUTPUT_PATH})

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
set(EXTRAFILES CMakeLists.txt 
	README.md
    test.py
    cmake/hlsl.cmake
    cmake/slang.cmake
)
add_custom_target(Settings SOURCES ${EXTRAFILES})
set_property(TARGET Settings PROPERTY FOLDER "Extra")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${EXTRAFILES})

