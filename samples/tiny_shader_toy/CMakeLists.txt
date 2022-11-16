DefaultBasicMakefile()

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
_add_project_definitions(${PROJECT_NAME})
target_link_libraries (${PROJECT_NAME} ${VULKANSDK_SHADERC_LIB}) # Adding ShaderC


#--------------------------------------------------------------------------------------------------
# Install - copying the media directory
install(DIRECTORY "shaders" 
        CONFIGURATIONS Release 
        DESTINATION "bin_${ARCH}/${PROJECT_NAME}")
install(DIRECTORY "shaders" 
        CONFIGURATIONS Debug 
        DESTINATION "bin_${ARCH}_debug/${PROJECT_NAME}")
