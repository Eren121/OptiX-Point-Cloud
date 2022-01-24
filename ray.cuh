#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "Camera.hpp"

// Structure utilitaire, stocke le pointeur de données de l'AS et le pointeur OptiX
// La seule raison d'être de cette classe est de pouvoir désallouer l'AS en gardant un pointeur vers le stockage.
// Il existe une fonction pour convertir un pointeur en OptixTraversableHandle (optixConvertPointerToTraversableHandle())
// Mais il est plus simple de stocker les deux soi-même.
struct TraversableHandleStorage {

    // handle référence une case mémoire dans d_output
    OptixTraversableHandle handle = {};

    // Stockage. OptiX n'alloue aucune mémoire, on doit allouer nous même
    // Pour détruire l'AS, il suffira donc de désallouer d_output.
    CUdeviceptr d_output = {};  
};

// La structure est read-only sur le GPU!!
// Mais elle peut contenir un pointeur vers une zone où l'on peut écrire
struct Params
{
    // Données de l'image, en row-major (nécessaire pour stbi_write)
    uchar3 *image = nullptr;

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
};

extern "C" __constant__ Params params;
