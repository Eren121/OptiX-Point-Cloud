#ifndef SIMPLEGLRECT_HPP
#define SIMPLEGLRECT_HPP

#include "OpenGL.hpp"
#include <vector>

/**
 * @brief Simple carré OpenGL qui contient un vbo, un vao, et un shader pour afficher une texture dans toute la fenêtre
 */
class SimpleGLRect
{
public:
    /**
     * @param width, height Taille de la texture.
     */
    SimpleGLRect(int width, int height);
    ~SimpleGLRect();

    SimpleGLRect(const SimpleGLRect&) = delete;
    SimpleGLRect& operator=(const SimpleGLRect&) = delete;

    /**
     * Affiche la texture (ou une partie si UV max < 1)
     */
    void draw();

    GLuint getTexture() const
    {
        return m_texture;
    }
    
    /**
     * N'utiliser qu'une sous-partie de la texture dans l'affichage
     * La texture interne (taille passée au constructeur) n'est pas redimensionnable
     */
    void setRenderSize(int width, int height);

private:
    // Remplit les données du VBO + setup du VAO
    void fillBuffers(float uv_x_max, float uv_y_max);
    void allocTexture();

    // Créer les objets OpenGL
    void initGLObjects();
    void createTexture();
    void createProgram();

    static std::vector<float> getSquareCoords(float min, float max);
    static std::vector<float> getRectCoords(float min_x, float max_x, float min_y, float max_y);
    

    int m_width = 0;
    int m_height = 0;

    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_texture = 0;
    GLuint m_program = 0;
};

#endif /* SIMPLEGLRECT_HPP */
