#pragma once

#include "core/cuda/managed_device_ptr.h"
#include <cuda_runtime.h>

/**
 * Implémente et simplifie l'utilisation du super-sampling,
 * une méthode d'anti-aliasing.
 *
 * @remarks
 * Cette classe est là pour tester la parallélisation,
 * on peut faire aussi une simple boucle pour tuos les sous-pixels dans le shader
 * mais ce n'est potentiellement pas parallélisé.
 */
class SuperSampling
{
public:
    using Pixel = uchar3;

    /**
     * @param imageWidth, imageHeight La taille de l'image cible de sortie
     * @param subPixelsCount Le nombre de sous-pixels par pixel.
     */
    SuperSampling(int imageWidth, int imageHeight, int subPixelsCount);

    /**
     * Effectue l'interpolation linéaire du buffer temporaire et stocke l'interpolation dans une image.
     * @param d_output Image de sortie Pixel* sur le device imageWidth x imageHeight
     */
    void interpolate(Pixel* d_output);

    managed_device_ptr& getDeviceBuffer() { return m_d_buffer; }
    
private:
    /**
     * Contient le buffer vers l'image, sauf que contient plusieurs
     * sous-pixel par pixel.
     * 
     * Le but va être :
     *    1) De remplir ce buffer avec un pattern (qui n'est pas géré par cette classe,
     *       donc on peut utiliser n'importe quel pattern)
     *    2) Interpoler linéairement le buffer pour produire l'image finale
     *
     * 
     */
    managed_device_ptr m_d_buffer;

    size_t m_width;
    size_t m_height;
    size_t m_subPixelsCount;
};
