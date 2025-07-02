set(LIB_NAME vk_mini_samples_common)

file(GLOB LIB_HEADERS *.hpp)
file(GLOB SHADERS_SOURCES common_shaders/*.*)

# Create a STATIC library to make it visible in Visual Studio
add_library(${LIB_NAME} STATIC ${LIB_HEADERS} ${SHADERS_SOURCES})

# Add the current directory to include path for the library
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${LIB_NAME} PUBLIC ${NVPRO_CORE_DIR}/third_party/glm)

# Make headers show up in IDE
source_group("Headers" FILES ${LIB_HEADERS})
source_group("Shaders" FILES ${SHADERS_SOURCES})

# Since this is actually header-only, we don't need to build anything
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

