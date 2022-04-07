#include "Pbo.h"
#include <glad.h>
#include <cuda_gl_interop.h> // For a reason, this should be included after GL includes, otherwise error
#include "check.h"

Pbo::Pbo(int width, int height)
{
    glGenBuffers(1, &m_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Indique à OpenGL que la texture sera utilisée à la fois par CUDA et OpenGL
    // On veut écrire l'image avec CUDA et l'afficher avec OpenGL d'où CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD.
    // Aide: https://forums.developer.nvidia.com/t/reading-opengl-texture-data-from-cuda/110370/3
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda, m_id, cudaGraphicsRegisterFlagsWriteDiscard));
}

Pbo::~Pbo()
{
    if(m_cuda)
    {
        CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda));
        m_cuda = nullptr;
    }
    if(m_id)
    {
        glDeleteBuffers(1, &m_id);
        m_id = 0;
    }
}

void Pbo::resize(int width, int height)
{
    // On doit ré-allouer le PBO quand on le redimensionne
    // pour que les données restent contiguës (sinon tout est décalé dans les kernels)

    CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Re-register à la fin
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda, m_id, cudaGraphicsRegisterFlagsWriteDiscard));
}

void* Pbo::map(cudaStream_t stream)
{
    void* d_image = nullptr;
    size_t sizeInBytes = 0;
  
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda, stream));

    // d_image contient le PBO
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&d_image, &sizeInBytes, m_cuda));

    return d_image;
}

void Pbo::unmap(cudaStream_t stream)
{
    // Unbind la ressource d'interoperabilité CUDA, à chaque frame
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda, stream));
}