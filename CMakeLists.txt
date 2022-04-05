cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(TestOptiX LANGUAGES CXX CUDA)

# Options globales

    set(EXE ${PROJECT_NAME})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 61-virtual) # -virtual pour éviter une erreur de compilation
    set(CMAKE_CXX_STANDARD 17)
    set(BUILD_TESTS OFF)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG") # définir la macro DEBUG en mode Debug
    set(SRC_DIR "${PROJECT_SOURCE_DIR}/src/")

    find_package(CUDA REQUIRED)
    include_directories("C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/include")

# Bibliothèques

    add_subdirectory(thirdparty/tinyply) # .ply parser

    # OpenGL wrapper
    add_subdirectory(thirdparty/glad)

    # -- Désactive certaines options pour compiler GLFW plus rapidement
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

    # GLFW v3.3.6: Pour les fenêtres
    # Avant ImGUI car ImGUI dépend de GLFW
    add_subdirectory(thirdparty/glfw)

    # ImGUI v1.86: Pour l'interface
    # Les sources de ImGUI ont été modifiées pour intégrer GLM
    add_subdirectory(thirdparty/imgui)

# Fichiers sources du projet
# communs à au programme exécutable et aux shaders

    list(APPEND COMMON_SRC stb.cpp Application.cpp Application.h ray.cuh common.hpp common.cpp
        SimpleGLRect.cpp SimpleGLRect.hpp OpenGL.hpp PointsCloud.cpp PointsCloud.hpp Camera.hpp OrbitalControls.cpp OrbitalControls.hpp
        Scene.hpp Scene.cpp
        Distribution.hpp Distribution.cpp
        Gui.hpp Gui.cpp
        core/cuda/check.cpp
        core/cuda/managed_device_ptr.cpp
        core/cuda/StreamScatter.cpp
        core/optix/stubs.cpp
        core/optix/check.cpp
        core/utility/time.cpp
        ssaa/SuperSampling.cu)

    list(TRANSFORM COMMON_SRC PREPEND ${SRC_DIR})
    add_library(commonLib ${COMMON_SRC})
    target_include_directories(commonLib PUBLIC ${SRC_DIR})
    target_compile_definitions(commonLib PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(commonLib PUBLIC NOMINMAX) # évite une erreur de compilation sur windows
    target_compile_definitions(commonLib PUBLIC MY_PTX_PATH=\"ray.ptx\")
    target_link_libraries(commonLib PUBLIC imgui glfw glad tinyply glm ${CUDA_LIBRARIES})

# Exécutable
    
    add_executable(${EXE} ${SRC} ${SRC_DIR}/main.cpp)
    target_link_libraries(${EXE} commonLib)

# Shaders (fichiers .ptx)
# A compiler séparémment
# shadersLib contient tous les .ptx précompilés
# que l'on copiera dans le répertoire de sortie de l'exe
# Les .ptx sont un simple language assembleur qui n'est pas linké

    list(APPEND SHADERS_SRC
        ray.cu)
    list(TRANSFORM SHADERS_SRC PREPEND ${SRC_DIR})

    add_library(shadersLib OBJECT ${SHADERS_SRC})
    target_link_libraries(shadersLib commonLib)

    # Permet de compiler les PTX quand on recompile juste l'exe
    # et que le code des PTX change
    # Pour une raison target_link_libraries(${EXE} shadersLib) ne fonctionne pas
    add_dependencies(${EXE} shadersLib)
    
    # options de debugging + optimisation
    target_compile_options(shadersLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info
                        --use_fast_math
                        >)
                       
    # Pour générer des PTX au lieu de OBJ (PTX = fichier bytecode intermédiaire OptiX)
    set_target_properties(shadersLib PROPERTIES CUDA_PTX_COMPILATION ON)

# Compilation des exemples
# Chaque fichier dans example/*.cpp possède un main, qui contient un test / exemple

    enable_testing()

    file(GLOB examples_file_cpp "${SRC_DIR}/examples/*.cpp")
    foreach(cpp_path ${examples_file_cpp})

        get_filename_component(target ${cpp_path} NAME_WE)

        add_executable(${target} ${SRC} ${cpp_path})
        target_link_libraries(${target} commonLib)

        message(STATUS ">>> Create test target '${target}' (${cpp_path})")
        add_test(NAME ${target} COMMAND ${target})

    endforeach()

# Copier les PTX dans le répertoire de l'exe automatiquement

    # Message
    add_custom_command(
        TARGET ${EXE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            echo ">>> Copy shaders files:" $<TARGET_OBJECTS:shadersLib> "to executable directory ($<TARGET_FILE_DIR:${EXE}>)")

    # Copie
    add_custom_command(
        TARGET ${EXE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            copy_if_different $<TARGET_OBJECTS:shadersLib> $<TARGET_FILE_DIR:${EXE}>)