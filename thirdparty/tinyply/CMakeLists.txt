cmake_minimum_required(VERSION 2.13)

# Compiler requirements
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

project(tinyply LANGUAGES CXX)
set(PROJECT_VERSION 2.3)

set(CMAKE_DEBUG_POSTFIX "d")

# Library
set(SHARED_LIB false CACHE BOOL "Build as shared library")
if(${SHARED_LIB})
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_library(tinyply SHARED source/tinyply.cpp source/tinyply.h)
else()
    add_library(tinyply STATIC source/tinyply.cpp source/tinyply.h)
endif()

set(BUILD_TESTS false CACHE BOOL "Build tests")

# Example Application
add_executable(example source/example.cpp)
target_link_libraries(example PRIVATE tinyply)

# Test Application
if(${BUILD_TESTS})
  add_executable(tests source/tests.cpp)
  target_link_libraries(tests PRIVATE tinyply)
endif()

# pkg-config support
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(UNIX)
  configure_file(cmake/tinyply.pc.in ${generated_dir}/tinyply.pc @ONLY)
  install(FILES "${generated_dir}/tinyply.pc" DESTINATION lib/pkgconfig)
endif()

# CMake package config support
include(CMakePackageConfigHelpers)

set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "")

write_basic_package_version_file(
  "${version_config}"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "cmake/Config.cmake.in"
  "${project_config}"
  INSTALL_DESTINATION "${config_install_dir}"
)

target_include_directories(tinyply PUBLIC "source")
target_include_directories(tinyply PUBLIC "third-party")
        
