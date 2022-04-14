#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "common.hpp"
#include "Camera.hpp"

#include "ssaa/Parameters.h"

// Différence entre la taille de la fenêtre sur le bureau
// Et la taille interne de la fenêtre:
// Pour rendre les calculs plus rapides, on stocke moins de pixels
// et ils sont affichés interpolés
// evidemment, le rendu sera moins bien et pixellisé
// On alloue initialemment un buffer de la taille max. possible pour ne pas réallouer à chaque fois
// Cette façon de faire sera effectuée plusieurs fois
// On considère du full HD 1920x1080, peut être changé à + pour du 4K
const int maxWinTexWidth = 1920;
const int maxWinTexHeight = 1080;
        
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
