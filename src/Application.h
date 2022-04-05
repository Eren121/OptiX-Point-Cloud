#ifndef WINDOW_H
#define WINDOW_H

// Implémentation basique de GLFW.
// Doc de GLFW: https://www.glfw.org/docs/3.3/quick.html

#define GLFW_INCLUDE_NONE // utiliser notre propre wrapper OpenGL (GLAD) et pas celui de GLFW
#include "OpenGL.hpp"
#include <GLFW/glfw3.h>

#include <functional>

/**
 * @brief Gère une fenêtre, et gère l'interoperabilité GFLW <=> OptiX, permet aussi d'utiliser ImGUI.
 *
 * @remarks
 * Pour gérer l'interopérabilité avec OptiX, le contexte OpenGL ne fait que afficher une texture sur tout l'écran à chaque Frame.
 * Les pixels de cette texture seont actualisés par CUDA/OptiX.
 *
 * @remarks
 * On a pas besoin de gérer l'interopérabilité avec OptiX, car OptiX est une surcouche CUDA
 * et OptiX ne gère jamais la mémoire lui-même.
 */
class Application
{
public:
    /**
     * @brief Construit une fenêtre, mais ne l'affiche pas encore.
     * Pour cela, appeler display().
     * 
     * @param width La largeur de la fenêtre.
     * @param height La hauteur de la fenêtre.
     * @param title Le titre de la fenêtre.
     */
    Application(int width, int height, const char *title = "");
    ~Application();

    // ---- Rend la classe non-copiable
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    // ----

    /**
     * @brief Affiche la fenêtre et bloque jusqu'à ce que la fenêtre se ferme.
     */
    void display();
    
    /**
     * Ferme la fenêtre après la prochaine frame.
     * Utile avec un outil de profiling comme nsight pour profiler 1 seule frame.
     */
    void stopAfterNextFrame() const;

private:
    /**
     * @brief Callback pour les erreurs OpenGL. Facilite le debugging et évite de gérer chaque appel de fonction d'OpenGL.
     */
    static void GLAPIENTRY messageCallbackGL(GLenum source,
        GLenum type, GLuint id, GLenum severity, GLsizei length,
        const GLchar* message, const void* userParam);


    /**
     * @brief Callback pour les erreurs GLFW.
     */
    static void errorCallbackGLFW(int error, const char *description);

    
public:
    static void noop() {}

    GLFWwindow* getWindow() const { return m_window; }
    
    /**
     * @brief Fonction de callback appelée à chaque affichage. L'utilisateur peut la rédéfinir et afficher ici la scène.
     */
    std::function<void()> onDraw = noop;

private:
    void initializeImGUI();
    void onFrameBeginImGUI();
    void onFrameRenderImGUI();
    void destroyImGUI();

    /**
     * @brief Contient à la fois la fenêtre et le contexte OpenGL.
     */
    GLFWwindow *m_window = nullptr;
};

#endif /* WINDOW_H */
