#pragma once

#include "Options.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * Agglomère toutes les informations passées à un pattern SSAA (dependency container).
 * Evite de changer chaque fonction de pattern à chaque fois qu'on veut ajouter un argument.
 */
struct SsaaContext
{
    SsaaOptions* options;
    int numRays;
    int numRaysSqrt;
    
    __device__ __host__ int2 id2D() const
    {
        return {id % numRaysSqrt, id / numRaysSqrt};
    }

    __device__ __host__ int2 numRays2D() const
    {
        return {numRaysSqrt, numRaysSqrt};
    }

    void setNumRays(int numRays)
    {
        numRays = numRays;
        numRaysSqrt = static_cast<int>(sqrt(static_cast<double>(numRays)));
    }

    /**
     * Varie entre [0;numRays).
     */
    int id;

    /**
     * Donnée de sortie:
     * contiendra la position dans [0;1]^2.
     * Cela indiquera la position à relativement à l'intérieur du pixel.
     * Il est possible de déborder sur les pixels d'à côté si > 1 ou < 0.
     */
    float2 out_pos;

    curandState_t* rand;
};