# imgui ne fournit pas de CMake, on doit donc créer le CMakeLists.txt nous-même c'est-à-dire:
#   - Ajouter les sources à la main
#   - Ajouter les sources spécifiques au Back-end, ici OpenGL/GLFW
#
# Ce répertoire est le même que le repo github v1.86, avec les modifications suivantes
#   - répertoires doc, examples supprimés
#   - ajout du CMakeLists.txt (wrapper CMake)
#   - ajout de misc/single_file/imgui_single_file.cpp (implémentation single-file)

set(SRC
    # Sources imgui communes à tous les back-ends
    misc/single_file/imgui_single_file.cpp
    misc/single_file/imgui_single_file.h

    # Sources imgui backend
    backends/imgui_impl_opengl3.cpp # OpenGL
    backends/imgui_impl_opengl3.h
    backends/imgui_impl_opengl3_loader.h

    backends/imgui_impl_glfw.cpp # GLFW
    backends/imgui_impl_glfw.h    
)

add_library(imgui ${SRC})

target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/backends)

# ImGUI dépend de GLFW et GLAD, qui doit être disponible quand ce projet est parsé par CMake
# Concrètement, imgui est le dernier add_subdirectory() après GLFW et GLAD
target_link_libraries(imgui glfw glad)