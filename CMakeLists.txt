cmake_minimum_required(VERSION 3.5)
set(PROJNAME vk_timeline_semaphore)
project(${PROJNAME})

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../..
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

#####################################################################################
# Add sample packages
#
_add_package_VulkanSDK()
_add_package_IMGUI()
_add_package_shaderc()
_add_nvpro_core_lib()

#####################################################################################
_add_project_definitions(${PROJNAME})

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.c)
file(GLOB HEADER_FILES *.h *.hpp shaders/*.h)
file(GLOB SHADER_FILES shaders/*.vert shaders/*.frag shaders/*.comp shaders/*.glsl)

#####################################################################################
# Executable
#
add_executable(${PROJNAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${SHADER_FILES} ${HEADER_FILES})

#####################################################################################
# Source code groups
#
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})
source_group("Shader Files" FILES ${SHADER_FILES})


#####################################################################################
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} nvpro_core)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies files that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target( ${PROJNAME} )
install(DIRECTORY ./shaders CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/")
install(DIRECTORY ./shaders CONFIGURATIONS Debug   DESTINATION "bin_${ARCH}_debug/${PROJNAME}/")
