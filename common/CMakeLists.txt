
set(LIB_NAME vk_mini_samples_common)

file(GLOB LIB_SOURCES *.cpp *.hpp)
file(GLOB SHADERS_SOURCES shaders/*.*)

# Collect all source files in the current directory
add_library(${LIB_NAME} ${LIB_SOURCES} ${SHADERS_SOURCES})

# Add the current directory to include path for the library
# target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${LIB_NAME} PRIVATE ${NVPRO_CORE_DIR}/third_party/glm)

source_group("Shaders" FILES ${SHADERS_SOURCES}) # Group shaders in IDE

set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "vk_mini_samples")
