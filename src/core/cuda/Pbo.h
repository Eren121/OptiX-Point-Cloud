#pragma once

#include <cuda_runtime.h>
#include "core/utility/no_copy.h"

/**
 * Pixel Buffer Object (RGB, uchar8).
 *
 * Et interfaçage avec CUDA (écriture avec CUDA et lecture avec OpenGL).
 * Permet de le redimensionner.
 */
class Pbo
{
public:
    Pbo(int width, int height);
    ~Pbo();

    NO_COPY(Pbo)

    void resize(int width, int height);

    unsigned int id() const { return m_id; }

    ////////
    // CUDA interop (each frame)
    void* map(cudaStream_t stream = 0); // Return PBO as device pointer to pixels
    void unmap(cudaStream_t stream = 0);
    ////////

private:
    int m_width;
    int m_height;
    unsigned int m_id = 0;

    // Contient les pixels de la fenêtre (= framebuffer) sous forme d'un pointeur vers le GPU.
    cudaGraphicsResource* m_cuda = nullptr;
};