
#
# CMAKE to deal with Slang shader files
#

include(${CMAKE_CURRENT_LIST_DIR}/find_entrypoints.cmake)

set(CMAKE_PATH ${CMAKE_CURRENT_LIST_DIR})

# -----------------------------------------------------------------------------
# Function to compile a Slang file
#
# Example:
# compile_slang_file SOURCE_FILE foo.slang
#   FLAGS "-I ${SAMPLES_COMMON_DIR}/shaders"
# )
# target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
#
# Inputs:
# SOURCE_FILE (string): The file to compile.
# FLAGS (list of strings): Optional flags to add to the slangc command line.
#
# Outputs (all in the parent scope):
# SLANG_OUTPUT_FILES: Files that will be generated. Set new each time, not appended to.
#
# Flags:
# DEBUG: If specified, will generate .glsl instead.
function(compile_slang_file)
    set(options DEBUG)
    set(oneValueArgs SOURCE_FILE)
    set(multiValueArgs FLAGS)
    cmake_parse_arguments(COMPILE  "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    
    set(SLANG_OUTPUT_FILES "")
    
    # Do nothing if Slang isn't used
    if(NOT USE_SLANG)       
        return()
    endif()
    
    if(NOT COMPILE_SOURCE_FILE)
      message(FATAL_ERROR "compile_slang_file called without a source file!")
    endif()
    
    # Get only the filename without path or extension
    get_filename_component(_FILE_STEM ${COMPILE_SOURCE_FILE} NAME_WE)
    
    # Create the output directory if it does not exist
    set(_OUT_DIR ${SAMPLE_FOLDER}/_autogen)
    file(MAKE_DIRECTORY ${_OUT_DIR})
    
    # This will contain a list of COMMAND arguments.
    set(_SLANG_COMMANDS )
    
    set(_TARGET "spirv")

    ## # Call the function to find entrypoints
    ## find_entrypoints(${COMPILE_SOURCE_FILE} ENTRYPOINTS_LIST STAGES_LIST)
    ## 
    ## # Get the length of the list
    ## list(LENGTH ENTRYPOINTS_LIST ENTRYPOINTS_LIST_LENGTH)
    ## math(EXPR ENTRYPOINTS_LIST_LENGTH "${ENTRYPOINTS_LIST_LENGTH} - 1")
    ## 
    ## # Iterate over the indices
    ## foreach(INDEX RANGE ${ENTRYPOINTS_LIST_LENGTH})
    ##   # Access the element at the current index
    ##   list(GET ENTRYPOINTS_LIST ${INDEX} _ENTRY_NAME)
    ##   list(GET STAGES_LIST ${INDEX} _STAGE)
    ##   message(STATUS "  - Found entrypoint: ${_ENTRY_NAME} (${_STAGE})")
    ## 
    ##   set(_VAR_NAME "${_FILE_STEM}_${_ENTRY_NAME}") # Variable name in header files
    ##   set(_OUT_STEM "${_OUT_DIR}/${_VAR_NAME}") # Path to the output without extensions
    ##   
    ## 
    ## 
    ##   # Common Slang compiler flags
    ##   set(_SLANG_FLAGS 
    ##     -entry ${_ENTRY_NAME} 
    ##     -target ${_TARGET}
    ##     # -emit-spirv-directly
    ##     -g3
    ##     -line-directive-mode glsl 
    ##     -profile glsl_460 
    ##     -D__slang 
    ##     -force-glsl-scalar-layout
    ##   )
    ## 
    ##   # Adding external flags, like -I ... 
    ##   list(APPEND _SLANG_FLAGS ${COMPILE_FLAGS})
    ## 
    ##   # _OUT_ARG is the -o argument passed to Slang
    ##   set(_OUT_ARG "${_OUT_STEM}.${_TARGET}") 
    ## 
    ##   ## DEBUG output 
    ##   if(COMPILE_DEBUG)
    ##     list(APPEND _SLANG_COMMANDS
    ##       COMMAND ${SLANG_COMPILER} ${_SLANG_FLAGS} -o ${_OUT_ARG} ${COMPILE_SOURCE_FILE}
    ##     )
    ##   endif()
    ## 
    ##   set(_OUT_FILE "${_OUT_ARG}.h")
    ## 
    ##   # Embed the Spir-V in a header
    ##   list(APPEND _SLANG_FLAGS -source-embed-style text -source-embed-name ${_VAR_NAME})
    ## 
    ##   # Compile shader
    ##   list(APPEND _SLANG_COMMANDS
    ##     COMMAND ${CMAKE_COMMAND} -E echo ${SLANG_COMPILER} ${_SLANG_FLAGS} -o ${_OUT_ARG} ${COMPILE_SOURCE_FILE}
    ##     COMMAND ${SLANG_COMPILER} ${_SLANG_FLAGS} -o ${_OUT_ARG} ${COMPILE_SOURCE_FILE}
    ##   )
    ## 
    ##   list(APPEND SLANG_OUTPUT_FILES ${_OUT_FILE})
    ## endforeach()

    # !! Compiling all entry points in a single compilation
    set(_OUT_ARG "${_OUT_DIR}/${_FILE_STEM}_slang.h")
    set(_SLANG_FLAGS
        -profile glsl_460
        -target ${_TARGET} 
        -emit-spirv-directly
        -force-glsl-scalar-layout
        -fvk-use-entrypoint-name
        -g3
        -D__slang
        -source-embed-style text 
        -source-embed-name ${_FILE_STEM}Slang
        -o ${_OUT_ARG}
    )
    list(APPEND _SLANG_FLAGS ${COMPILE_FLAGS})

    list(APPEND _SLANG_COMMANDS
         COMMAND ${CMAKE_COMMAND} -E echo ${SLANG_COMPILER} ${_SLANG_FLAGS} ${COMPILE_SOURCE_FILE}  
         COMMAND ${SLANG_COMPILER} ${_SLANG_FLAGS} ${COMPILE_SOURCE_FILE}
    )
    # list(APPEND SLANG_OUTPUT_FILES ${_OUT_ARG})
    set(SLANG_OUTPUT_FILES ${_OUT_ARG})

    # We emit a single large add_custom_command so that it's possible to
    # right-click on a .slang file in Visual Studio and build all its outputs.
    # nbickford imagines doing this using add_custom_command(... APPEND) one day,
    # but CMake doesn't support that yet according to its 3.27.1 docs.
    add_custom_command(
      OUTPUT ${SLANG_OUTPUT_FILES}
      ${_SLANG_COMMANDS}
      MAIN_DEPENDENCY ${COMPILE_SOURCE_FILE}
      VERBATIM COMMAND_EXPAND_LISTS
    )
    set(SLANG_OUTPUT_FILES ${SLANG_OUTPUT_FILES} PARENT_SCOPE)
endfunction()

