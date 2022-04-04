#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "common.hpp"
#include "Camera.hpp"

#define OPTIMIZE_SUPERSAMPLE 1

// La structure est read-only sur le GPU!!
// Mais elle peut contenir un pointeur vers une zone où l'on peut écrire
struct Params
{
    // Données de l'image, en row-major (nécessaire pour stbi_write)
    uchar3* image = nullptr;

    unsigned int width, height;

    // Contient toute les géométries de la scène
    OptixTraversableHandle traversableHandle;

    int missCount = 0;

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
    
    /**
     * Combien de rayons tirer par pixels.
     * La distribution des rayons peut-être faite de façon régulière ou aléatoire (Monte-Carlo).
     * Au total, on a donc countRaysPerPixel.x * countRaysPerPixel.y rayons qui sont tirés par pixel.
     */
    glm::uvec2 countRaysPerPixel = {1, 1};


    unsigned long frame = 0;
};

extern "C" __constant__ Params params;
