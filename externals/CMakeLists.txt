cmake_minimum_required(VERSION 3.10)

include(ExternalProject)

project(Externals)

message("Building Externals in ${PROJECT_SOURCE_DIR}")

set(EXTERNAL_INSTALL_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/install)

if(WIN32)
set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v2023.1.1/slang-2023.1.1-win64.zip")
else()
set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v2023.1.1/slang-2023.1.1-linux-x86_64.zip")
endif()

message(STATUS "SLANG_URL: ${SLANG_URL}")

#Slang is prebuilt, just i.e. headers and prebuilt binaries
ExternalProject_Add(
    SLANG
    URL ${SLANG_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    PREFIX ${CMAKE_CURRENT_SOURCE_DIR}
	CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
	INSTALL_COMMAND ""
)

add_custom_target(external_all DEPENDS SLANG)
