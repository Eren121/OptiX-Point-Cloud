#pragma once

#include "Context.h"
#include "core/cuda/math.h"

/**
 * Grille régulière
 */
inline __device__ __host__ void ssaa_regular(SsaaContext& ctxt)
{
    const float2 id = make_float2(ctxt.id2D());
    const float2 numRays = make_float2(ctxt.numRays2D());

    // On ne touche jamais les bords d'un pixel pour être régulier (d'où +0.5f)
    ctxt.out_pos = (0.5f + id) / numRays;
}

/**
 * Totalement aléatoire, en restant à l'intérieur du pixel
 */
inline __device__ __host__ void ssaa_random(SsaaContext& ctxt)
{
    const float dispersion = ctxt.options->random.dispersion;

    ctxt.out_pos.x = 0.5f + unNormalize(curand_uniform(ctxt.rand), -dispersion, dispersion);
    ctxt.out_pos.y = 0.5f + unNormalize(curand_uniform(ctxt.rand), -dispersion, dispersion);
}