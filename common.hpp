#pragma once

 // On utilise plutôt les headers C que C++ car avec CUDA l'utilisation du C++ a été source de problème
 // (linkage, erreur de syntaxe...)
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#define CAT2(x, y) x ## y
#define CAT(x, y) CAT2(x, y)
    
#define STR2(x) # x
#define STR(x) STR2(x)

#define CUDA_CHECK(expr) checkImplCuda((expr), __FILE__, __LINE__)
#define OPTIX_CHECK(expr) checkImplOptiX((expr), __FILE__, __LINE__)

namespace my
{
    static constexpr const float pi = 3.14159265358979323846;

    inline float degrees(float radians)
    {
        return radians / pi * 180.0f;
    }

    inline float radians(float degrees)
    {
        return degrees / 180.0f * pi;
    }
}

void checkImplCuda(cudaError_t result, const char *file, int line);
void checkImplOptiX(OptixResult result, const char *file, int line);

/**
 * @brief Stocke un bloc de données de taille fixe sur le GPU et gère automatiquement la désallocation.
 * @details Sans cette classe, on devrait allourdir le code de nombreux cudaMalloc(), cudaMemcpy() et cudaFree().
 *
 * @remarks Explication différence CUdeviceptr / void*:
 *      L'API CUDA est séparée en deux API distinctes: l'API bas-niveau et l'API haut-niveau.
 *
 *      ---------------------------------------------------------------
 *      |                   |    API bas-niveau     | API haut-niveau |
 *      | Pointeur vers GPU |   CUdeviceptr (uint)  |      void*      |
 *      | Allocation mém.   |      cuMemAlloc()     |    cuMalloc()   |  ==> Les fonctions font la même chose
 *      |--------------------------------------------------------------
 *
 *      Dans les versions récentes de CUDA, ces deux API ont fusionnées, mais les deux versions existent
 *      toujours et en C++ ces types sont incompatibles entre eux,
 *      car CUdeviceptr est un type entier, même s'il s'agit exactement de la même valeur.
 *      On doit donc effectuer de nombreux reinterpret_cast() car certaines fonctions attendent un CUdeviceptr
 *      et d'autres attendent un void*. Cette classe permet aussi de gérer simplement les conversions.
 *
 *      Exemples:
 *          CUdeviceptr ptr1 = nullptr;             // Erreur, même si conceptuellement le type est un pointeur
 *          CUdeviceptr ptr2 = 0;                   // Ok
 *          void *ptr3 = nullptr;                   // Ok
 *          ptr3 = ptr1;                            // Erreur
 *          ptr3 = reinterpret_cast<void*>(ptr1);   // Ok
 *
 */
struct managed_device_ptr
{
    /**
     * Ce pointeur pointe vers une zone mémoire qui se trouve sur le GPU.
     * @remarks Initialisé dans le constructeur.
     * @remarks L'utilisateur de la classe ne doit pas le désallouer, ce sera fait automatiquement dans le destructeur.
     */
    CUdeviceptr device_ptr = 0;
   
    /**
     * @brief Construit un nouveau bloc mémoire sur le GPU copiée depuis une zone mémoire du CPU.
     * 
     * @param[in] data La donnée à copier sur le GPU
     * @param[in] size La taille de donnée à copier sur le GPU
     */
    managed_device_ptr(const void *data, size_t size)
    {
        CUDA_CHECK(cudaMalloc(&toVoidPtr(), size));
        copyToDevice(data, size);
    }

    /**
     * @brief Construit un nouveau bloc mémoire sur le GPU copié depuis un type générique.
     * 
     * @param[in] data La donnée à copier sur le GPU, on infère la taille grâce à sizeof(data).
     *
     * @remarks Le constructeur est annoté "explicit" pour éviter les conversions automatiques, ce qui est recommandé.
     */
    template<typename T>
    explicit managed_device_ptr(const T& data)
        : managed_device_ptr(&data, sizeof(T))
    {
    }

    /**
     * @brief Destructeur, détruit la zone mémoire allouée sur le GPU.
     */
    ~managed_device_ptr()
    {
        CUDA_CHECK(cudaFree(toVoidPtr()));
        device_ptr = 0;
    }

    /**
     * @brief Utilitaire pour convertir device_ptr en void*.
     * reinterpret_cast peut être dangereux, mais dans ce cas précis
     * les deux types sont toujours compatibles entre eux.
     *
     * @return device_ptr sous forme de void*.
     *
     * @remarks On retourne une référence car certaines fonctions attendent un tableau
     * de pointeur, et pour un tableau de taille 1 on peut donner &myManagedDevicePtr->toVoidPtr().
     * Comme void* et CUdeviceptr sont normalement de la même taille, cela ne devrait pas poser de problèmes.
     */
    void* & toVoidPtr()
    {
        return reinterpret_cast<void* &>(device_ptr);
    }

    /**
     * @brief Copie une données CPU sur le GPU en inférant la taille grâce à sizeof(data).
     * @param[in] data Le pointeur CPU à copier sur le GPU.
     */
    template<typename T>
    void copyToDevice(const T& data)
    {
        copyToDevice(&data, sizeof(data));
    }

    /**
     * @brief Copie une donnée CPU sur la mémoire allouée sur le GPU.
     * 
     * @param[in] data Le pointeur CPU à copier sur le GPU.
     * @param[in] size Le nombre de bytes de data à copier sur le GPU.
     *
     * @remarks N'alloue pas de données, en fait que copier. Il faut donc que la taille
     * de data ne dépasse pas celle allouée dans le constructeur.
     */
    void copyToDevice(const void *data, size_t size)
    {
        CUDA_CHECK(cudaMemcpy(toVoidPtr(), data, size, cudaMemcpyHostToDevice));
    }
};