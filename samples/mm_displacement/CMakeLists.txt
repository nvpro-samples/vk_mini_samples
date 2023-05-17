DefaultBasicMakefile()

# Get name
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Adding the micromesh prototype
set(COMMON_SRC
	${SAMPLES_COMMON_DIR}/bird_curve_helper.cpp 
	${SAMPLES_COMMON_DIR}/bird_curve_helper.hpp
	${SAMPLES_COMMON_DIR}/bit_packer.hpp
	)
target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SRC})
source_group(common FILES ${COMMON_SRC})

