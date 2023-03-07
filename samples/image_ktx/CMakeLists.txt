DefaultBasicMakefile()

# KTX needs extra libraries
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
_add_package_KTX()
target_link_libraries (${PROJECT_NAME} basisu)
