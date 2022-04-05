#ifndef SIMPLEGLRECT_HPP
#define SIMPLEGLRECT_HPP

#include "OpenGL.hpp"

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

    void draw();

    GLuint getTexture() const
    {
        return m_texture;
    }

private:
    void createVAO();
    void createVBO();
    void createTexture();
    void createProgram();

    int m_width = 0;
    int m_height = 0;

    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_texture = 0;
    GLuint m_program = 0;
};

#endif /* SIMPLEGLRECT_HPP */
