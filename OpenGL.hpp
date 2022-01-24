#ifndef OPENGL_HPP
#define OPENGL_HPP

#include "common.hpp"

// Ne pas modifier, correspond à la version que l'on a choisir pour GLAD
// Utile pour ImGUI
#define MY_GL_VERSION_MAJOR 4
#define MY_GL_VERSION_MINOR 3

// Par exemple pour v4.3, MY_GL_VERSION == 430
#define MY_GL_VERSION CAT(CAT(MY_GL_VERSION_MAJOR, MY_GL_VERSION_MINOR), 0)
#define MY_GL_VERSION_GLSL "#version " STR(MY_GL_VERSION)

#include <glad.h>

#endif /* OPENGL_HPP */
