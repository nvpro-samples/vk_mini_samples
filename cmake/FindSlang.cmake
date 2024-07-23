# FindSlang.cmake

# Usage: 
# set(Slang_VERSION "2024.1.30")
# find_package(Slang)

# Set the default Slang version
set(Slang_DEFAULT_VERSION "2024.1.30")

# Parse optional arguments
set(Slang_VERSION ${Slang_DEFAULT_VERSION} CACHE INTERNAL "")

# Download Slang SDK
if(WIN32)
    set(Slang_URL "https://github.com/shader-slang/slang/releases/download/v${Slang_VERSION}/slang-${Slang_VERSION}-windows-x86_64.zip")
else()
    set(Slang_URL "https://github.com/shader-slang/slang/releases/download/v${Slang_VERSION}/slang-${Slang_VERSION}-linux-x86_64.zip")
endif()

download_package(
  NAME Slang
  URL ${Slang_URL}
  VERSION ${Slang_VERSION}
  LOCATION Slang_SOURCE_DIR
)

set(Slang_SDK ${Slang_SOURCE_DIR} CACHE PATH "Path to Slang SDK root directory")
mark_as_advanced(Slang_SDK)

# Use the one downloaded
find_program(Slang_slangc_EXE
    NAMES slangc 
    HINTS ${Slang_SOURCE_DIR}/bin NO_DEFAULT_PATH
)
# Provide the Slang compiler and SDK paths to the user
set(Slang_slangc_EXE ${Slang_slangc_EXE} CACHE FILEPATH "Path to Slang compiler")

find_library(Slang_LIBRARY
    NAMES slang
    HINTS ${Slang_SOURCE_DIR}/lib NO_DEFAULT_PATH
)
mark_as_advanced(Slang_LIBRARY)

if(WIN32)
    find_file(Slang_DLL
        NAMES slang.dll
        HINTS ${Slang_SOURCE_DIR}/bin
        NO_DEFAULT_PATH
    )
  find_file(Slang_glslang_DLL
        NAMES slang-glslang.dll
        HINTS ${Slang_SOURCE_DIR}/bin 
        NO_DEFAULT_PATH
  )
elseif(UNIX)
    find_file(Slang_DLL
        NAMES libslang.so
        HINTS ${Slang_SOURCE_DIR}/lib
        NO_DEFAULT_PATH
    )
    find_file(Slang_glslang_DLL
        NAMES libslang-glslang.so
        HINTS ${Slang_SOURCE_DIR}/lib
        NO_DEFAULT_PATH
    )
endif()
mark_as_advanced(Slang_DLL)
mark_as_advanced(Slang_glslang_DLL)

# Test to see if compiler is working
message(STATUS "--> using SLANGC under: ${Slang_slangc_EXE}")
execute_process(COMMAND ${Slang_slangc_EXE} "-v" ERROR_VARIABLE Slang_VERSION RESULTS_VARIABLE EXTRACT_RESULT)
message(STATUS "--> SLANGC version: ${Slang_VERSION}")
if(NOT EXTRACT_RESULT EQUAL 0)
  message("Envoking Slang compiler failed with error code: ${EXTRACT_RESULT}")
endif()
