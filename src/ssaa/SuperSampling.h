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
     * Modifie le nombre de rayons par pixel.
     *
     * Cela n'alloue aucune donnée GPU.
     * Si le stockage actuel n'est pas assez grand, cela lance une erreur.
     *
     * Si le stockage actuel est trop grand, alors cela ne fait qu'utiliser une partie
     * du stockage, et la fin du stockage ne sera pas utilisée.
     *
     * Dans les faits, seulement la taille des ArrayView seront modifiées.
     */
    void setNumRays(int numRays);

    /**
     * Modifie la taille des données.
     * Cette taille doit être inférieure à la capacité du buffer interne
     * (arguments du constructeur).
     */
    void setSize(int width, int height);

    /**
     * Effectue l'interpolation linéaire du buffer temporaire et stocke l'interpolation dans une image.
     * @param d_output Image de sortie Pixel* sur le device imageWidth x imageHeight
     */
    void interpolate(Pixel* d_output);

    /**
     * Effectue l'interpolation en utilisant plusieurs streams.
     * Peut améliorer les performances.
     * @param streams Le tableau de streams
     * @param count Le nombre de streams
     */
    void interpolate(Pixel* d_output, cudaStream_t* streams, int count);

    uchar3* getBufferDeviceData() { return m_d_buffer.as<uchar3>(); }
    managed_device_ptr& getBufferDevice() { return m_d_buffer; }

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
