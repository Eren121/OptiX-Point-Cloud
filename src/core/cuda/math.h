#pragma once

#include "detail/helper_math.h"

// Ajoute quelques éléments qui ne sont pas dans l'utilitaire fournit par Nvidia

using uchar = unsigned char;

/**
 * Division arrondie vers le haut, on en a besoin pour calculer la taille d'une grille lors
 * d'un lancement de kernel
 *
 * @return ceil(num / den)
 *
 * @remarks uint Car c'est le type de dim3.x, etc de la classe de CUDA dim2 ou dim3.
 */
inline __host__ __device__ uint ceil_div(uint num, uint den)
{
    return (num + den - 1u) / den;
}

/**
 * @param x 0.0 <= x <= 1.0
 * @return x scalé dans l'intervalle [start;end]
 */
template<typename T>
inline __host__ __device__ T unNormalize(T x, T start, T end)
{
    return x * (end - start) + start;
}


inline __host__ __device__ uchar3 make_uchar3(uchar s)
{
    return make_uchar3(s, s, s);
}

inline __host__ __device__ uchar4 make_uchar4(uchar s)
{
    return make_uchar4(s, s, s, s);
}

////////////////////////////////////////////////////////////////////////////////
// different than
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ bool operator!=(const int2 &a, const int2 &b)
{
    return a.x != b.x && a.y != b.y;
}

////////////////////////////////////////////////////////////////////////////////
// vectors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float lengthSquared(const float2& a)
{
    return a.x * a.x + a.y * a.y;
}

inline __host__ __device__ float lengthSquared(const float3& a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}