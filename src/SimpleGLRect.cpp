#include "SimpleGLRect.hpp"
#include <iostream>
#include <vector>

SimpleGLRect::SimpleGLRect(int width, int height)
    : m_width(width), m_height(height)
{
    createVAO();
    createVBO();
    createTexture();
    createProgram();
}

SimpleGLRect::~SimpleGLRect()
{
    glDeleteVertexArrays(1, &m_vao);
    m_vao = 0;

    glDeleteBuffers(1, &m_vbo);
    m_vbo = 0;

    glDeleteTextures(1, &m_texture);
    m_texture = 0;

    glDeleteProgram(m_program);
    m_program = 0;
}

void SimpleGLRect::createVBO()
{
     // Coordonnées d'un carré pour afficher la texture à chaque frame sur tout l'écran
     // Par défaut, OpenGL affiche dans le cube [-1; 1]^3
    const float min = -1.0f, max = 1.0f;
    static const float vertices[] = {
        min, min, // Triangle 1
        max, min,
        min, max,

        max, min, // Triangle 2
        max, max,
        min, max
    };
    

    glBindVertexArray(m_vao);

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Active l'attribut 0 qui contient les coordonnées xy (float[2])
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
}

void SimpleGLRect::createVAO()
{
    // Index 0: coordonnées xy, (float[2])
    // Pas besoin de coordonnées de texture UV, on considère que se ce les mêmes que xy car on affiche sur tout l'écran

    glGenVertexArrays(1, &m_vao);
}

void SimpleGLRect::createTexture()
{
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    // Pour tester si la texture fonctionne bien, on l'initialise avec une couleur blanche
    std::vector<unsigned char> textureData(m_width * m_height * 3, 255);

    // Une texture vide suffirait (nullptr), pas besoin de donner de données
    // car elles seront écrites à chaque frame
    // Note: Pour fonctionner avec CUDA, on doit choisir le format interne RGB8
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData.data());
    
    // Filtre
    // IMPORTANTS pour ne pas être faussé par l'antialiasing effectué par OpenGL
    // alors qu'OptiX n'en fait pas
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Normalement on a jamais besoin des mipmaps car la texture a la même taille que l'écran,
    // mais sans la texture ne fonctionne pas
    glGenerateMipmap(GL_TEXTURE_2D);
}

void SimpleGLRect::draw()
{
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glUseProgram(m_program);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void SimpleGLRect::createProgram()
{
    // Inspiré de https://learnopengl.com/Getting-started/Hello-Triangle

    static const char *vertexShaderSource = R"(
        #version 430 core
        layout (location = 0) in vec2 aPos;
        out vec2 uv;

        void main()
        {
           // La position est dans [-1;1] mais les uv sont dans [0;1]
           uv = (1.0 + aPos) * 0.5;
           gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        }
    )";

    static const char *fragmentShaderSource = R"(
        #version 430 core
        out vec4 FragColor;
        in vec2 uv;

        uniform sampler2D uTexture;

        void main()
        {
            FragColor = texture(uTexture, uv);

            // La texture ne stocke pas de alpha normalement
            // (valeur 1.0 par défaut, donc cette ligne est optionnel)
            FragColor.a = 1.0;
        }
    )";

    int success;
    char infoLog[512];

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    m_program = glCreateProgram();
    glAttachShader(m_program, vertexShader);
    glAttachShader(m_program, fragmentShader);
    glLinkProgram(m_program);

    glGetProgramiv(m_program, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(m_program, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
    }
    else {
        std::cerr << "[OpenGL] Programme compile avec succes" << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Link l'indice de texture "0" au shader
    glUseProgram(m_program);
    glUniform1i(glGetUniformLocation(m_program, "uTexture"), 0);
}