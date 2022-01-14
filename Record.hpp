#pragma once

#include "common.hpp"

// https://raytracing-docs.nvidia.com/optix7/guide/index.html#shader_binding_table#shader-binding-table
template<typename T>
struct Record
{
    // Nvidia (doc officielle) utilise __align__
    // Mais C++11 à alignas() qui doit être portable
    alignas(OPTIX_SBT_RECORD_ALIGNMENT)
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]; // Header

    T data; // Data-block

    // On stocke aussi un pointeur sur le device qui contient this
    // même durée de vie que this
    // pour simplifier le code de l'utilsateur de la classe
    // Ce pointeur est dans la classe, mais il ne fait pas vraiment partie
    // de la classe : on le "concatène" ici, mais quand on le copie sur
    // la mémoire GPU, on ne l'inclus pas.
    CUdeviceptr d_this = 0;

    // Remplit automatiquement le header d'après programGroup
    // Alloue automatiquement le Record sur gpu (désalloué au destructeur)
    Record(OptixProgramGroup programGroup)
    {
        // C++ garanti qu'il n'y a pas de padding avant le premier membre
        // this == this->header
        optixSbtRecordPackHeader(programGroup, this);
        
        
        CUDA_CHECK(cudaMalloc(
            // CUdevicedptr est un entier, on doit le convertir en pointeur
            &reinterpret_cast<void*>(d_this),
            
            sizeof(*this)
        ));
    }

    void copyToDevice() const
    {
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_this),
            this, sizeof(*this),
            cudaMemcpyHostToDevice));
    }

    CUdeviceptr getDevicePtr() const
    {
        return d_this;
    }
    
    ~Record()
    {
        // Désallouer le pointeur mémoire
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_this)));
    }
};