set(LIB_NAME vk_mini_samples_common)

file(GLOB LIB_HEADERS *.hpp)
file(GLOB SHADERS_SOURCES common_shaders/*.*)

# Since this is a header-only library, we use INTERFACE.
add_library(${LIB_NAME} INTERFACE)
# However, adding sources here makes it appear in Visual Studio:
target_sources(${LIB_NAME} PRIVATE ${LIB_HEADERS} ${SHADERS_SOURCES})

# Add the vk_mini_samples directory to the include path, so that these files
# can be included using a "common/" prefix.
target_include_directories(${LIB_NAME} INTERFACE ${PROJECT_SOURCE_DIR})
# And include GLM:
target_include_directories(${LIB_NAME} INTERFACE ${NVPRO_CORE_DIR}/third_party/glm)

# Make headers show up in IDE
source_group("Headers" FILES ${LIB_HEADERS})
source_group("Shaders" FILES ${SHADERS_SOURCES})

# Since this is actually header-only, we don't need to build anything
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
