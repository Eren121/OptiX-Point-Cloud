#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "core/utility/no_copy.h"

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
    NO_COPY(managed_device_ptr);

    /**
     * Construit un nullptr sur le GPU.
     */
    managed_device_ptr() = default;
   
    /**
     * Construit un nouveau bloc mémoire sur le GPU copiée depuis une zone mémoire du CPU.
     * 
     * @param[in] data La donnée à allouer et à copier sur le GPU.
     * @param[in] size La taille de donnée à copier sur le GPU.
     * @param stream Si != 0, la copie est effectuée de façon asynchrone sur ce stream.
     */
    managed_device_ptr(const void *data, size_t size, cudaStream_t stream = 0);

    /**
     * Construit un nouveau bloc mémoire en allouant seulement la taille sans copie
     * ni initialisation du buffer.
     *
     * @param[in] size La taille du buffer à allouer.
     */
    explicit managed_device_ptr(size_t size);
    
    managed_device_ptr(managed_device_ptr&& rhs);
    managed_device_ptr& operator=(managed_device_ptr&& rhs);

    /**
     * Créer un buffer sur le device, le remplit et le retourne.
     *
     * @param cpuData La donnée à copier sur le GPU, la taille est déduite par sizeof(T).
     * @param stream Si != 0, la copie est effectuée de façon asynchrone sur ce stream.
     *
     * @remarks Factory et pas constructeur pour éviter les ambiguités, les constructeurs
     * prennent toujours une taille en byte et une donnée en void*, mais ici la taille est déduite
     * par le type.
     */
    template<typename T>
    static managed_device_ptr create_from(const T& cpuData, cudaStream_t stream = 0)
    {
        return managed_device_ptr(&cpuData, sizeof(T), stream);
    }

    /**
     * @brief Destructeur, détruit la zone mémoire allouée sur le GPU.
     */
    ~managed_device_ptr();

    /**
     * Interpète les données sur le GPU selon un type
     * et évite un code trop verbeux avec pleins de reinterpret_cast.
     */
    /// @{
    template<typename T>
    T* as()
    {
        return reinterpret_cast<T*>(m_device_ptr);
    }

    template<typename T>
    const T* as() const
    {
        return reinterpret_cast<const T*>(m_device_ptr);
    }
    /// @}
    
    /**
     * @brief Utilitaire pour convertir device_ptr en void*.
     * reinterpret_cast peut être dangereux, mais dans ce cas précis
     * les deux types sont toujours compatibles entre eux.
     *
     * @return device_ptr sous forme de void*.
     *
     * @remarks On retourne une référence car certaines fonctions attendent un tableau
     * de pointeur, et pour un tableau de taille 1 on peut donner &myManagedDevicePtr->to_void_ptr().
     * Comme void* et CUdeviceptr sont normalement de la même taille, cela ne devrait pas poser de problèmes.
     */
    void*& to_void_ptr();
    void* to_void_ptr() const;
    
    /**
     * Copie une donnée CPU sur la mémoire allouée sur le GPU.
     * @param[in] data Le pointeur CPU à copier sur le GPU.
     * @param[in] size Le nombre de bytes de data à copier sur le GPU.
     * @param stream Si != 0, la copie est effectuée de façon asynchrone sur ce stream.
     * @remarks N'alloue pas de données, en fait que copier. Il faut donc que la taille
     * de data ne dépasse pas celle allouée dans le constructeur.
     */
    void fill(const void* data, size_t size, cudaStream_t stream = 0);

    /**
     * Copier size bytes de data vers le GPU,
     * avec un décalage d'où commencer à écrire.
     *
     * @param offset Décalage dans les données GPU pour commencer à copier
     * @param size Taille de data
     */
    void subfill(const void* data, size_t size, size_t offset);


    /**
     * Récupère les données sur le CPU.
     * @param[out] data Les données écrites. Doit être assez grand pour stocker le buffer.
     * @param stream Si != 0, la copie est effectuée de façon asynchrone sur ce stream.
     */
    void download(void *data, cudaStream_t stream = 0) const;

    /**
     * @brief Conversion implicite vers un pointeur sur le GPU.
     */
    operator void*() const;

    /**
     * @brief Conversion implicite vers CUdeviceptr.
     * @return Une référence vers le pointeur stocké, pour permettre par exemple si une fonction attend un CUdeviceptr* comme aabbBuffers.
     */
    operator const CUdeviceptr&() const;

    /**
     * @return La taille en bytes du buffer.
     */
    size_t size() const;

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