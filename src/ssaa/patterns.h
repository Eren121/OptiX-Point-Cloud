#pragma once

#include "Context.h"

/**
 * Grille régulière
 */
inline __device__ __host__ void ssaa_regular(SsaaContext& ctxt)
{
    const float2 id = make_float2(ctxt.id);
    const float2 numRays = make_float2(ctxt.numRays);

    // On ne touche jamais les bords d'un pixel pour être régulier (d'où +0.5f)
    ctxt.out_pos = (0.5f + id) / numRays;
}

/**
 * Totalement aléatoire, en restant à l'intérieur du pixel
 */
inline __device__ __host__ void ssaa_random(SsaaContext& ctxt)
{
    const float2 id = make_float2(ctxt.id);
    const float2 numRays = make_float2(ctxt.numRays);

    ctxt.out_pos.x = curand_uniform(ctxt.rand);
    ctxt.out_pos.y = curand_uniform(ctxt.rand);
}