#pragma once

#include "Type.h"

/**
 * Classe à envoyer au kernel, contient tous les paramètres pour le SSAA.
 */

struct SsaaParameters
{
    int2 numRays;
    SsaaType type;
};