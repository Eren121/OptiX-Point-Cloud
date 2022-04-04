#pragma once

 // On utilise plutôt les headers C que C++ car avec CUDA l'utilisation du C++ a été source de problème
 // (linkage, erreur de syntaxe...)
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <utility>

#define CAT2(x, y) x ## y
#define CAT(x, y) CAT2(x, y)
    
#define STR2(x) # x
#define STR(x) STR2(x)

#define CUDA_CHECK(expr) checkImplCuda((expr), __FILE__, __LINE__)
#define OPTIX_CHECK(expr) checkImplOptiX((expr), __FILE__, __LINE__)

namespace my
{
    #define my_inline __forceinline__ __device__ __host__

    static constexpr const float pi = 3.14159265358979323846;

    static my_inline float degrees(float radians)
    {
        return radians / pi * 180.0f;
    }

    static my_inline float radians(float degrees)
    {
        return degrees / 180.0f * pi;
    }

    /**
     * @brief Interpolation linéaire.
     * @param x Start.
     * @param y End.
     * @param a Quelle valeur à prendre entre x et y. Range: [0; 1].
     */
    template<typename T>
    my_inline T mix(T x, T y, float a)
    {
        return x * (1.0f - a) + y * a;
    }

    template<typename T>
    my_inline auto ddot(T a, T b)
    {
        return max(static_cast<decltype(a.x)>(0), dot(a, b));
    }
}

void checkImplCuda(cudaError_t result, const char *file, int line);
void checkImplOptiX(OptixResult result, const char *file, int line);

#include "core/cuda/managed_device_ptr.h"

// Structure utilitaire, stocke le pointeur de données de l'AS et le pointeur OptiX
// La seule raison d'être de cette classe est de pouvoir désallouer l'AS en gardant un pointeur vers le stockage.
// Il existe une fonction pour convertir un pointeur en OptixTraversableHandle (optixConvertPointerToTraversableHandle())
// Mais il est plus simple de stocker les deux soi-même.
struct TraversableHandleStorage {

    // handle référence une case mémoire dans d_output
    OptixTraversableHandle handle = {};

    // Stockage. OptiX n'alloue aucune mémoire, on doit allouer nous même
    // Pour détruire l'AS, il suffira donc de désallouer d_output (fait automatiquement à la destruction du TraversableHandleStorage)
    managed_device_ptr d_storage;
};