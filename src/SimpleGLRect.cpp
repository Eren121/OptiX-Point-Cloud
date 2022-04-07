#include "SimpleGLRect.hpp"
#include <iostream>
#include <vector>

SimpleGLRect::SimpleGLRect(int width, int height)
    : m_width(width), m_height(height)
{
    initGLObjects();
    createProgram();
    fillBuffers(1.0f, 1.0f);
    allocTexture();
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

void SimpleGLRect::setRenderSize(int width, int height)
{
    // On modifie les UVs en calculant selon la taille interne de la texture et la taille voulue

    // Ex. : Le buffer interne est 1920x1080 (m_width=1920, m_height=1080),
    // mais on ne veut utiliser qu'une partie 500x500 (width=height=500)
    // uv_x_max = 500/1920 = 0.26,
    // uv_y_max = 500/1080 = 0.46

    float uv_x_max = static_cast<float>(width) / m_width;
    float uv_y_max = static_cast<float>(height) / m_height;

    fillBuffers(uv_x_max, uv_y_max);
}

void SimpleGLRect::initGLObjects()
{
    glGenBuffers(1, &m_vbo);
    glGenVertexArrays(1, &m_vao);
    glGenTextures(1, &m_texture);
    m_program = glCreateProgram();
}

std::vector<float> SimpleGLRect::getSquareCoords(float min, float max)
{
    // Renvoit un carré comme 2 triangles
    return {
        min, min, // Triangle 1
        max, min,
        min, max,

        max, min, // Triangle 2
        max, max,
        min, max
    };
}

std::vector<float> SimpleGLRect::getRectCoords(float min_x, float max_x, float min_y, float max_y)
{
    // Renvoit un rectangle comme 2 triangles
    return {
        min_x, min_y, // Triangle 1
        max_x, min_y,
        min_x, max_y,

        max_x, min_y, // Triangle 2
        max_x, max_y,
        min_x, max_y
    };
}

void SimpleGLRect::fillBuffers(float uv_x_max, float uv_y_max)
{
    // Pour éviter d'utiliser 2 VBO on stocke tout dans 1 seul VBO
    // on concatène dedans les coordonnées xy et les coordonnées uv
    // layout = xyxyxy...uvuvuv...
    //          <------->
    //         uv_offset
    std::vector<float> all;
    
    // Coordonnées d'un carré pour afficher la texture à chaque frame sur tout l'écran
    // Par défaut, OpenGL affiche dans le cube [-1; 1]^3
    std::vector<float> vertices = getSquareCoords(-1.0f, 1.0f);
    std::vector<float> uvs = getRectCoords(0.0f, uv_x_max, 0.0f, uv_y_max);

    all.insert(all.end(), vertices.begin(), vertices.end());
    all.insert(all.end(), uvs.begin(), uvs.end());
    
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, all.size() * sizeof(float), all.data(), GL_STATIC_DRAW);

    // Active l'attribut 0 qui contient les coordonnées xy (float[2])
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    // Active l'attribut 1 qui contient les coordonnées de texture uv (float[2])
    // Avec un offset (les coord. uvs sont situés à la fin des coord. xy)
    const void* uv_offset = reinterpret_cast<void*>(vertices.size() * sizeof(float));
    glVertexAttribPointer(1, 2, GL_FLOAT, false, 2 * sizeof(float), uv_offset);
    glEnableVertexAttribArray(1);
}

void SimpleGLRect::allocTexture()
{
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
        layout (location = 1) in vec2 aUV;
        out vec2 uv;

        void main()
        {
           uv = aUV;
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