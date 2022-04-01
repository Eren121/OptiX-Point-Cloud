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
class managed_device_ptr
{
public:
    managed_device_ptr(const managed_device_ptr&) = delete;
    managed_device_ptr& operator=(const managed_device_ptr&) = delete;

    managed_device_ptr(managed_device_ptr&& rhs)
    {
        using std::swap;

        swap(m_device_ptr, rhs.m_device_ptr);
        swap(m_size, rhs.m_size);
    }

    managed_device_ptr& operator=(managed_device_ptr&& rhs)
    {
        using std::swap;

        swap(m_device_ptr, rhs.m_device_ptr);
        swap(m_size, rhs.m_size);

        return* this;
    }

    template<typename T>
    static managed_device_ptr copyFrom(const T& cpuData)
    {
        return managed_device_ptr(&cpuData, sizeof(T));
    }

    /**
     * @brief Construit un nullptr sur le GPU.
     */
    managed_device_ptr() = default;
   
    /**
     * @brief Construit un nouveau bloc mémoire sur le GPU copiée depuis une zone mémoire du CPU.
     * 
     * @²[in] data La donnée à copier sur le GPU
     * @param[in] size La taille de donnée à copier sur le GPU
     */
    managed_device_ptr(const void *data, size_t size)
        : managed_device_ptr(size)
    {
        copyToDevice(data, size);
    }

    /**
     * @brief Construit un nouveau bloc mémoire en allouant seulement la taille sans copie.
     */
    explicit managed_device_ptr(size_t size)
        : m_size(size)
    {
        CUDA_CHECK(cudaMalloc(&toVoidPtr(), size));
    }

    /**
     * @brief Destructeur, détruit la zone mémoire allouée sur le GPU.
     */
    ~managed_device_ptr()
    {
        CUDA_CHECK(cudaFree(toVoidPtr()));
        m_device_ptr = 0;
        m_size = 0;
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
        return reinterpret_cast<void* &>(m_device_ptr);
    }

    void* toVoidPtr() const
    {
        return reinterpret_cast<void*>(m_device_ptr);
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

    /**
     * @brief Conversion implicite vers un pointeur sur le GPU.
     */
    operator void*() const
    {
        return toVoidPtr();
    }

    /**
     * @brief Conversion implicite vers CUdeviceptr.
     * @return Une référence vers le pointeur stocké, pour permettre par exemple si une fonction attend un CUdeviceptr* comme aabbBuffers.
     */
    operator const CUdeviceptr&() const
    {
        return m_device_ptr;
    }

    size_t size() const
    {
        return m_size;
    }

private:
    /**
     * Buffer alloué sur le GPU.
     */
    CUdeviceptr m_device_ptr = 0;

    /**
     * Le nombre de bytes stockés dans m_device_ptr.
     */
    size_t m_size = 0;
};

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