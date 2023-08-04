
target_link_libraries (${PROJECT_NAME} ${VULKANSDK_SHADERC_LIB}) # Adding ShaderC

#--------------------------------------------------------------------------------------------------
# Install - copying the media directory
install(DIRECTORY  ${SAMPLE_FOLDER}/shaders 
        CONFIGURATIONS Release 
        DESTINATION "bin_${ARCH}/${PROJECT_NAME}")
install(DIRECTORY ${SAMPLE_FOLDER}/shaders
        CONFIGURATIONS Debug 
        DESTINATION "bin_${ARCH}_debug/${PROJECT_NAME}")
