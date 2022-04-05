#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "common.hpp"
#include "Camera.hpp"

#include "ssaa/Parameters.h"

#define OPTIMIZE_SUPERSAMPLE 1

// La structure est read-only sur le GPU!!
// Mais elle peut contenir un pointeur vers une zone où l'on peut écrire
struct Params
{
    // Données de l'image, en row-major (nécessaire pour stbi_write)
    uchar3* image = nullptr;

    /**
     * Décalage à prendre en compte pour projeter le rayon sur la caméra
     * Cela permet de lancer plusieurs shaders en parallèle dans différents streams
     */
    uint2 offsetIdx = {};

    /**
     * Taille de l'écran
     */
    uint width = 0;
    uint height = 0;

    // Contient toute les géométries de la scène
    OptixTraversableHandle traversableHandle = {};

    int missCount = {};

    // Convertit un indice 2D en offset dans l'image
    __device__ __host__ uchar3* at(int x, int y) {
        return &image[y * width + x];
    }

    // Temps en secondes écoulé
    float time = 0.0f;


    // Taille des points
    // Cela multiple le rayon des point par pointSize
    float pointSize = 1.0f;

    Camera camera;

    float3 lightDirection = normalize(make_float3(1.0f, -1.0f, 1.0f));

    bool shadowRayEnabled = true;

    /**
     * La taille des points est multiplié par ce nombre.
     * Attention, cela ne modifie pas la taille des AABB pour les collisions, mais uniquement
     * la taille une fois que l'on sait que l'on est dans l'AABB de la sphère.
     * Si pointRadiusModifier >= sqrt(3.0f), alors le point aura toujours l'apparence d'un cube,
     * car sqrt(3.0f) est la longueur de la diagonale d'un cube de côté 1
     * Sphère circonscrite au cube, le cube est entièrement contenu dans la sphère (AABB).
     */
    float pointRadiusModifier = 1.0f;
    
    unsigned long frame = 0;

    SsaaParameters ssaaParams = {};

    curandState* rand = nullptr;
};

extern "C" __constant__ Params params;
