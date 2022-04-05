#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * Agglomère toutes les informations passées à un pattern SSAA (dependency container).
 * Evite de changer chaque fonction de pattern à chaque fois qu'on veut ajouter un argument.
 */
struct SsaaContext
{
    int2 numRays;

    __device__ __host__ int flattenNumRays() const
    {
        return numRays.x * numRays.y;
    }

    /**
     * Varie entre [0;numRays).
     */
    int2 id;

    /**
     * @return id mais on se considère en 1D
     * Certains algorithmes n'ont pas besoin de la 2D (par ex. la version totalement aléatoire)
     */
    __device__ __host__ int flattenID() const
    {
        return id.y * numRays.x + id.x;
    }

    /**
     * Donnée de sortie:
     * contiendra la position dans [0;1]^2.
     * Cela indiquera la position à relativement à l'intérieur du pixel.
     * Il est possible de déborder sur les pixels d'à côté si > 1 ou < 0.
     */
    float2 out_pos;

    curandState_t* rand;
};