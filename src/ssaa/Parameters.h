#pragma once

#include "Type.h"
#include "Options.h"
#include <cmath>

#define SSAA_NUM_RAYS_MAX 64

/**
 * Classe à envoyer au kernel, contient tous les paramètres pour le SSAA.
 */
class SsaaParameters
{ 
private:
    int m_numRays = 1;
    int m_numRaysSqrt = 1; // Cela économise de calculer sqrt() sur le GPU
    
public:
    SsaaType type = SSAA_REGULAR;
    SsaaOptions options = {};

    void setNumRays(int numRays)
    {
        m_numRays = numRays;
        m_numRaysSqrt = static_cast<int>(sqrt(static_cast<double>(numRays)));
    }

    __device__ __host__ int numRays() const { return m_numRays; }
    __device__ __host__ int numRaysSqrt() const { return m_numRaysSqrt; }
};

// Returns true if params was changed by user
bool drawGui(SsaaParameters& params);