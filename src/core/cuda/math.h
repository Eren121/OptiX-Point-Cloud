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

inline __host__ __device__ uchar3 make_uchar3(uchar s)
{
    return make_uchar3(s, s, s);
}

inline __host__ __device__ uchar4 make_uchar4(uchar s)
{
    return make_uchar4(s, s, s, s);
}