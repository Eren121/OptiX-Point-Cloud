cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(TestOptiX LANGUAGES CXX CUDA)
set(EXE ${PROJECT_NAME})

add_definitions(-DNOMINMAX) # évite une erreur de compilation sur windows

set(CMAKE_CUDA_ARCHITECTURES 61-virtual) # -virtual pour éviter une erreur de compilation
set(CMAKE_CXX_STANDARD 17)

set(BUILD_TESTS OFF)
set(MY_PTX_PATH ${CMAKE_BINARY_DIR}/cudaLib.dir/Debug/ray.ptx)
add_compile_definitions(MY_PTX_PATH=\"${MY_PTX_PATH}\")

find_package(CUDA REQUIRED)

# .ply parser
add_subdirectory(thirdparty/tinyply)

message(STATUS ".ptx path: ${MY_PTX_PATH}")

include_directories("C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/include")
#list(APPEND CMAKE_MODULE_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/SDK/CMake")
#find_package(OptiX REQUIRED)

# Fichiers sources du projet
list(APPEND SRC main.cpp stb.cpp Application.cpp Application.h ray.cuh common.hpp common.cpp
    SimpleGLRect.cpp SimpleGLRect.hpp OpenGL.hpp PointsCloud.cpp PointsCloud.hpp Camera.hpp OrbitalControls.cpp OrbitalControls.hpp
    Scene.hpp Scene.cpp
    Distribution.hpp Distribution.cpp
    Gui.hpp Gui.cpp)

include_directories(.)

add_executable(${EXE} ${SRC})

find_package(CUDA REQUIRED)

add_library(cudaLib OBJECT ray.cu ray.cuh)
target_link_libraries(cudaLib ${CUDA_LIBRARIES})

target_compile_options(cudaLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)
                       
# Pour générer des PTX au lieu de OBJ (PTX = fichier bytecode intermédiaire OptiX)
set_property(TARGET cudaLib PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET cudaLib PROPERTY CUDA_STANDARD 14)
target_link_libraries(${EXE} ${CUDA_LIBRARIES})

set_property(TARGET ${EXE} PROPERTY CUDA_STANDARD 14)

# Le PTX est un fichier texte qu'il faudra lire
# On définit la macro MY_PTX_PATH qui contient le chemin
get_target_property(PTX_DIR cudaLib BINARY_DIR)

message(STATUS "PTX dir: ${PTX_DIR}")
message(STATUS )

target_compile_definitions(${EXE} PRIVATE _USE_MATH_DEFINES)

target_compile_definitions(cudaLib PRIVATE _USE_MATH_DEFINES)

# ---------------------------- Télécharge, compile et link GLFW automatiquement

include(FetchContent)

# -- Désactive certaines options pour compiler GLFW plus rapidement
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# OpenGL wrapper
add_subdirectory(thirdparty/glad)

 # GLFW v3.3.6: Pour les fenêtres
add_subdirectory(thirdparty/glfw) # Avant ImGUI car ImGUI dépend de GLFW

# ImGUI v1.86: Pour l'interface
# Les sources de ImGUI ont été modifiées pour intégrer GLM
add_subdirectory(thirdparty/imgui)

target_link_libraries(${EXE} imgui glfw glad tinyply glm)
target_link_libraries(cudaLib glm)