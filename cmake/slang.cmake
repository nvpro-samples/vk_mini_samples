cmake_minimum_required(VERSION 3.12)
#project(shader_compiler)


# Find Python executable
find_package(Python REQUIRED)

# Finding Slang
if (NOT SLANG_SDK)
  if(DEFINED ENV{SLANG_SDK})
    set(SLANG_SDK "$ENV{SLANG_SDK}" CACHE PATH "Path to SDK")
  else()
      set(SLANG_SDK "SLANG_SDK-NOTFOUND" CACHE PATH "Path to SDK")
  endif()
endif()
   
if(SLANG_SDK)
    find_program(SLANG_EXE NAMES slangc PATHS ${SLANG_SDK}/bin/windows-x64/release
                                              ${SLANG_SDK}/bin/linux-x64/release)
endif()

if(NOT SLANG_EXE)
    message(ERROR " Slang not found: set SLANG_SDK environment variable.")
else()
    message(STATUS "SLANGC found under: ${SLANG_EXE}")
endif()


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
    
    # Read the input file and search through it to find entrypoints.
    file(READ ${COMPILE_SOURCE_FILE} _CONTENT)
    
    # How we do this is not the focus of the mini_samples, but may be useful for
    # debugging.
    # This regex has 3 groups: 1 is the shader stage, and 3 contains the name of
    # the entrypoint. For instance, in
    # ```
    # [shader("compute")]
    # [numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
    # void computeMain(uint3 threadIdx : SV_DispatchThreadID)
    # ```
    # group 0 is `compute` and group 1 is `computeMain`.
    #
    # To do this, it searches for strings that look like this:
    # [shader("group_1_text"...
    # [text_we_ignore] (0 or more times)
    # line starting declarations group_3_text(
    #
    set(_REGEX [=[\[shader\(\"(\w+)\"[^\n]*[\n](\[[^\n]*\][\n])*[\w+[ \t]+]*(\w+)\(]=] )
    # To break this down into further detail (notes on quirks below)
    # [=[               This "bracket argument" opens a raw string. See https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#id19.
    # \[shader\(\"      Literal [shader("
    # (                 Open capture group 1
    #   \w+               Any sequence of "word characters"
    # )                 Close capture group 1
    # \"                Literal "
    # [^\n]*[\n]        The rest of the line
    # (                 Open capture group 2
    #   \[[^\n]*\][\n]    Line starts with [, ends with ], has any characters in between
    # )*                Close capture group 2; repeated any number of times.
    # [\w+[ \t]+]*      Skip to the start of the function name
    # (                 Open capture group 3
    #   \w+               Any sequence of word characters
    # )                 Close capture group 3
    # \(                Literal (
    # ]=]               Bracket argument closing raw string
    #
    # CMake has its own regex engine, which has some quirks that affect the
    # design of this regex. Notably, it doesn't support the \w escape sequence,
    # and uses tab, Carriage Return, and Line Feed characters instead of \t, \r,
    # and \n. (nbickford considers this to be rather cursed.) So we replace those:
    string(REPLACE [=[\n]=] "\r\n" _REGEX "${_REGEX}")
    string(REPLACE [=[\t]=] "\t" _REGEX "${_REGEX}")
    string(REPLACE [=[\w]=] "[a-zA-Z0-9_]" _REGEX "${_REGEX}")
    # Additionally, CMake's regex engine does not support lazy `*?` or
    # non capturing-groups `(?:`. This is why clauses in the regex above follow
    # a "match everything except the end character; end character" pattern, and
    # why we must skip over group 2.
    # Note that there's an open issue to improve this in CMake:
    # https://gitlab.kitware.com/cmake/cmake/-/issues/17686
    # Also, note that this regex takes quadratic time on adversarial strings.
    
    # We iterate over each match by finding the first match, removing it from
    # the string, and repeating. This is because if we used CMake's MATCHALL,
    # the information we're looking for would be in groups 1, 3, 4, 6, 7, 9, ...
    # but CMake has only 10 CMAKE_MATCH slots, so we wouldn't be able to find
    # more than 3 entrypoints per file.
    # Unfortunately, this approach takes time quadratic in the number of entrypoints.
    while(TRUE)
      string(REGEX MATCH "${_REGEX}" _REGEX_MATCH ${_CONTENT})
      if(NOT _REGEX_MATCH)
        break()
      endif()
      
      if(NOT CMAKE_MATCH_1)
        message(FATAL_ERROR "Could not find shader stage!")
      endif()
      
      if(NOT CMAKE_MATCH_3)
        message(FATAL_ERROR "Could not find shader entrypoint entry name!")
      endif()
      
      set(_STAGE ${CMAKE_MATCH_1})
      set(_ENTRY_NAME ${CMAKE_MATCH_3})
      
      # Remove the part of the string up to and including the point where the
      # entrypoint declaration was found
      string(FIND "${_CONTENT}" "${_REGEX_MATCH}" _ENTRYPOINT_DECL_START)
      string(LENGTH "${_REGEX_MATCH}" _ENTRYPOINT_DECL_LENGTH)
      math(EXPR _SUBSTR_START ${_ENTRYPOINT_DECL_START}+${_ENTRYPOINT_DECL_LENGTH})
      string(SUBSTRING "${_CONTENT}" "${_SUBSTR_START}" -1 _CONTENT)
      
      set(_VAR_NAME "${_FILE_STEM}_${_ENTRY_NAME}") # Variable name in header files
      set(_OUT_STEM "${_OUT_DIR}/${_VAR_NAME}") # Path to the output without extensions
      
      if(COMPILE_DEBUG)
        set(_TARGET "glsl")
      else()
        set(_TARGET "spirv")
      endif() 

      set(_SLANG_FLAGS 
        -entry ${_ENTRY_NAME} 
        -target ${_TARGET}
        -g3 
        -line-directive-mode standard 
        -profile glsl_460 
        -D__slang 
        -force-glsl-scalar-layout
      )

      set(_OUT_ARG "${_OUT_STEM}.${_TARGET}") # _OUT_ARG is the -o argument passed to Slang
      if(COMPILE_DEBUG)
        set(_OUT_FILE "${_OUT_ARG}") # _OUT_FILE is the file it will write to
      else()
        set(_OUT_FILE "${_OUT_ARG}.h")
        list(APPEND _SLANG_FLAGS -source-embed-style text -source-embed-name ${_VAR_NAME})
      endif()
      
      list(APPEND _SLANG_FLAGS ${COMPILE_FLAGS})

      list(APPEND _SLANG_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ${SLANG_EXE} ${_SLANG_FLAGS} -o ${_OUT_ARG} ${COMPILE_SOURCE_FILE}
        COMMAND ${SLANG_EXE} ${_SLANG_FLAGS} -o ${_OUT_ARG} ${COMPILE_SOURCE_FILE}
      )
      list(APPEND SLANG_OUTPUT_FILES ${_OUT_FILE})
    endwhile()
    
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
