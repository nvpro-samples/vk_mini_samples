# FindSlang.cmake

# Usage: 
# set(SLANG_VERSION "2023.4.9")
# find_package(Slang)

# Set the default Slang version
set(SLANG_DEFAULT_VERSION "2023.5.5")

# Parse optional arguments
set(SLANG_VERSION ${SLANG_DEFAULT_VERSION} CACHE INTERNAL "")

# Download Slang SDK
if(WIN32)
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-win64.zip")
else()
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip")
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    set(_EXTRA_OPTIONS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
else()
    set(_EXTRA_OPTIONS)
endif()

FetchContent_Declare(slangsdk
    URL ${SLANG_URL}
    ${_EXTRA_OPTIONS}
)

message(STATUS "Looking for Slang ${SLANG_VERSION}")
if(NOT DEFINED SLANG_COMPILER) 
    message(STATUS "Downloading Slang ${SLANG_VERSION}")
    FetchContent_Populate(slangsdk)
    message(STATUS "Done")

    set(SLANG_SDK ${slangsdk_SOURCE_DIR} CACHE PATH "Path to Slang SDK root directory")
    
    if(slangsdk_SOURCE_DIR)
        # Use the one downloaded
        find_program(SLANG_COMPILER 
            NAMES slangc 
            PATHS ${slangsdk_SOURCE_DIR}/bin/windows-x64/release ${slangsdk_SOURCE_DIR}/bin/linux-x64/release NO_DEFAULT_PATH
        )

        find_library(SLANG_LIB 
            NAMES slang
            PATHS ${slangsdk_SOURCE_DIR}/bin/windows-x64/release ${slangsdk_SOURCE_DIR}/bin/linux-x64/release NO_DEFAULT_PATH
        )

    endif()
    
    if(NOT SLANG_COMPILER)
        message(ERROR "Slang not found")
    endif()
     
    # Provide the Slang compiler and SDK paths to the user
    set(SLANG_COMPILER ${SLANG_COMPILER} CACHE FILEPATH "Path to Slang compiler")
    set(SLANG_SDK ${slangsdk_SOURCE_DIR} CACHE PATH "Path to Slang SDK root directory")
endif()

# Test to see if compiler is working
message(STATUS "--> using SLANGC under: ${SLANG_COMPILER}")
execute_process(COMMAND ${SLANG_COMPILER} "-v" ERROR_VARIABLE SLANG_VERSION RESULTS_VARIABLE EXTRACT_RESULT)
message(STATUS "--> SLANGC version: ${SLANG_VERSION}")
if(NOT EXTRACT_RESULT EQUAL 0)
    message("Envoking Slang compiler failed with error code: ${EXTRACT_RESULT}")
endif()