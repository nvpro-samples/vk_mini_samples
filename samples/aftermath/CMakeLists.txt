DefaultBasicMakefile()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# Aftermath
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy the Aftermath dll where the project excutable is
macro(copy_dll dlls)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${dlls} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endmacro()

# Searching for the Aftermath SDK
set(AFTERMATH_SDK ${CMAKE_CURRENT_SOURCE_DIR}/aftermath_sdk)
if (NOT DEFINED ENV{NSIGHT_AFTERMATH_SDK})
  set(ENV{NSIGHT_AFTERMATH_SDK} ${AFTERMATH_SDK})
endif()
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(NsightAftermath)

if(NsightAftermath_FOUND)
    message(STATUS "Found Aftermath under:" ${NsightAftermath_LIBRARY})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NSIGHT_AFTERMATH)
    target_link_libraries (${PROJECT_NAME} ${NsightAftermath_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${NsightAftermath_INCLUDE_DIRS})
    string(REPLACE ".lib" ".dll" NsightAftermath_DLL ${NsightAftermath_LIBRARY})
    copy_dll(${NsightAftermath_DLL})
    install(FILES ${NsightAftermath_DLL} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
    install(FILES ${NsightAftermath_DLL} CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
else()
    message("\n\n============= ERROR =============")
    message("Path to the Aftermath SDK is missing.")
    message("Please follow steps in aftermath/README.md")
    message("============= ERROR =============\n\n")
endif()

