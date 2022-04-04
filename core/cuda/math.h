#pragma once

#include "helper_math.h"

/**
 * Division arrondie vers le haut, on en a besoin pour calculer la taille d'une grille lors
 * d'un lancement de kernel
 *
 * @return ceil(num / den)
 *
 * @remarks uint Car c'est le type de dim3.x, etc de la classe de CUDA dim3.
 */
inline uint ceil_div(uint num, uint den)
{
    return (num + den - 1u) / den;
}