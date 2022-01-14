cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(TestOptiX LANGUAGES CXX CUDA)
set(EXE ${PROJECT_NAME})

set(MY_PTX_PATH ${CMAKE_BINARY_DIR}/cudaLib.dir/Debug/ray.ptx)
add_compile_definitions(MY_PTX_PATH=\"${MY_PTX_PATH}\")

message(STATUS ".ptx path: ${MY_PTX_PATH}")

include_directories("C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/include")
#list(APPEND CMAKE_MODULE_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/SDK/CMake")
#find_package(OptiX REQUIRED)

add_executable(${EXE} main.cpp)

find_package(CUDA REQUIRED)

add_library(cudaLib OBJECT ray.cu ray.cuh)

# Pour générer des PTX au lieu de OBJ (PTX = fichier bytecode intermédiaire OptiX)
set_property(TARGET cudaLib PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET cudaLib PROPERTY CUDA_STANDARD 14)
target_link_libraries(${EXE} ${CUDA_LIBRARIES})

# Le PTX est un fichier texte qu'il faudra lire
# On définit la macro MY_PTX_PATH qui contient le chemin
get_target_property(PTX_DIR cudaLib BINARY_DIR)