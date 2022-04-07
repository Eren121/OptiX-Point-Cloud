#pragma once

#include <glad.h>
#include <cstdio>

#define GL_CHECK(e) do { e; glCheckError(); } while(false)

inline void glCheckError()
{
    GLenum err;
    if((err = glGetError()) != GL_NO_ERROR)
    {
        fprintf(stderr, "*** OpenGL error %d ***", err);
    }
}