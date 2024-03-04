# FindSlang.cmake

# Usage: 
# set(Slang_VERSION "2023.4.9")
# find_package(Slang)

# Set the default Slang version
set(Slang_DEFAULT_VERSION "2023.5.5")

# Parse optional arguments
set(Slang_VERSION ${Slang_DEFAULT_VERSION} CACHE INTERNAL "")

# Download Slang SDK
if(WIN32)
    set(Slang_URL "https://github.com/shader-slang/slang/releases/download/v${Slang_VERSION}/slang-${Slang_VERSION}-win64.zip")
else()
    set(Slang_URL "https://github.com/shader-slang/slang/releases/download/v${Slang_VERSION}/slang-${Slang_VERSION}-linux-x86_64.zip")
endif()

CPMAddPackage(
  NAME Slang
  URL ${Slang_URL}
  VERSION ${Slang_VERSION}
)

if(Slang_ADDED)
  set(Slang_SDK ${Slang_SOURCE_DIR} CACHE PATH "Path to Slang SDK root directory")
  mark_as_advanced(Slang_SDK)

  # Use the one downloaded
  find_program(Slang_slangc_EXE
      NAMES slangc 
      HINTS ${Slang_SOURCE_DIR}/bin/windows-x64/release ${Slang_SOURCE_DIR}/bin/linux-x64/release NO_DEFAULT_PATH
  )
  # Provide the Slang compiler and SDK paths to the user
  set(Slang_slangc_EXE ${Slang_slangc_EXE} CACHE FILEPATH "Path to Slang compiler")

  find_library(Slang_LIBRARY
      NAMES slang
      HINTS ${Slang_SOURCE_DIR}/bin/windows-x64/release ${Slang_SOURCE_DIR}/bin/linux-x64/release NO_DEFAULT_PATH
  )
  mark_as_advanced(Slang_LIBRARY)

  find_file(Slang_DLL
        NAMES slang.dll
        HINTS ${Slang_SOURCE_DIR}/bin/windows-x64/release ${Slang_SOURCE_DIR}/bin/linux-x64/release NO_DEFAULT_PATH
  )
  mark_as_advanced(Slang_DLL)

  find_file(Slang_glslang_DLL
        NAMES slang-glslang.dll
        HINTS ${Slang_SOURCE_DIR}/bin/windows-x64/release ${Slang_SOURCE_DIR}/bin/linux-x64/release NO_DEFAULT_PATH
  )
  mark_as_advanced(Slang_glslang_DLL)

  # Test to see if compiler is working
  message(STATUS "--> using SLANGC under: ${Slang_slangc_EXE}")
  execute_process(COMMAND ${Slang_slangc_EXE} "-v" ERROR_VARIABLE Slang_VERSION RESULTS_VARIABLE EXTRACT_RESULT)
  message(STATUS "--> SLANGC version: ${Slang_VERSION}")
  if(NOT EXTRACT_RESULT EQUAL 0)
    message("Envoking Slang compiler failed with error code: ${EXTRACT_RESULT}")
  endif()

else()
    message(ERROR "Slang not found")
endif()
  
