#include "Application.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

void Application::errorCallbackGLFW(int error, const char *description)
{
    fprintf(stderr, "[GLFW] Error %d: %s\n", error, description);
}

void GLAPIENTRY Application::messageCallbackGL(GLenum source,
    GLenum type, GLuint id, GLenum severity, GLsizei length,
    const GLchar* message, const void* userParam)
{
    // Source:
    // https://www.khronos.org/opengl/wiki/OpenGL_Error#:~:text=If%20the%20parameters%20of%20a,presented%20as%20an%20error%20code.

    if(type == GL_DEBUG_TYPE_ERROR)
    {
        fprintf(stderr,
            "[OpenGL] Error: type = 0x%x, severity = 0x%x, message = %s\n",
            type, severity, message);
    }
}

Application::Application(int width, int height, const char *title)
{
    // Demande à GLFW un contexte OpenGL avec cette version
    // Si possible, sion il peut utiliser une autre version proche
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, MY_GL_VERSION_MAJOR);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, MY_GL_VERSION_MINOR);

    // Initialise GLFW au début de l'application
    if(!glfwInit())
    {
        // Une erreur s'est produite lors de l'initialisation de GLFW,
        // on affiche un message et on quitte le programme.
        std::cerr << "Erreur d'initialisation de GLFW." << std::endl;
        std::exit(1);
    }

    // Redirige toutes les erreurs de GLFW à une fonction pour les afficher
    glfwSetErrorCallback(errorCallbackGLFW);

    // Créé la fenêtre
    m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if(!m_window)
    {
        // Erreur inattendue lors de la création de la fenêtre,
        // on affiche un message et on quitte le programme.
        std::cerr << "Erreur lors de la création d'une fênetre GLFW." << std::endl;
        std::exit(1);
    }

    // Nécessaire pour pouvoir utiliser les fonctions OpenGL
    // Tant que le contexte n'est pas modifié, on peut dessiner maintenant dans la fenêtre
    glfwMakeContextCurrent(m_window);

    // Evite le "screen tearing"
    glfwSwapInterval(1);
    
    // Permet d'utiliser les fonctions modernes d'OpenGL
    // A appeler après avoir bind le contexte OpenGL
    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));

    // Permet d'afficher les erreurs OpenGL quand il y en a
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(messageCallbackGL, nullptr);
}

Application::~Application()
{
    // Détruit la fenêtre GLFW si elle existe et le contexte OpenGL
    if(m_window)
    {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    // Termine GLFW à la fin de l'application
    glfwTerminate();
}

void Application::display()
{
    glfwMakeContextCurrent(m_window);

    // Initialise ImGUI (après avoir bind le contexte OpenGL)
    initializeImGUI();

    // Tant que l'utilisateur n'a pas demandé à fermer la fenêtre, on boucle l'affichage
    // On affiche une frame par itération
    while (!glfwWindowShouldClose(m_window))
    {
        // Re-bind le contexte à chaque frame au cas où un code l'aurait changé
        // Typiquement s'il y a plusieurs fenêtres
        glfwMakeContextCurrent(m_window);

        // Lit les entrées utilisateurs (clavier, souris, etc...)
        glfwPollEvents();

        // Affichage
        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);
        glViewport(0, 0, width, height);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        onFrameBeginImGUI();
        
        onDraw();

        onFrameRenderImGUI();
        
        // Switch les tampons du double-buffering
        glfwSwapBuffers(m_window);
    }

    destroyImGUI();
}

void Application::initializeImGUI()
{
    // Tuto: https://blog.conan.io/2019/06/26/An-introduction-to-the-Dear-ImGui-library.html
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init(MY_GL_VERSION_GLSL);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
}

void Application::onFrameBeginImGUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Application::onFrameRenderImGUI()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::destroyImGUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}