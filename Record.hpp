#pragma once

#include "common.hpp"

// https://raytracing-docs.nvidia.com/optix7/guide/index.html#shader_binding_table#shader-binding-table
template<typename T>
struct Record
{
    /**
     * @brief Le premier élément de la classe doit être un header d'une taille fixe.
     * C'est pour OptiX, nous ne gérerons pas cette donnée mais elle doit être présente.
     * On devra la remplir le header nous-même avec la fonction optixSbtRecordPackHeader().
     *
     * @remarks Doit être aligné.
     *  Nvidia (doc officielle) utilise __align__.
     *  Mais il est préférable d'utiliser alignas() du C++11 qui est portable.
     *  Cela effectue la même chose.
     * 
     */
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    

    // Block de données utilisateur
    // Peut-être n'importe quoi
    T data;

    // Remplit automatiquement le header d'après programGroup
    void fill(OptixProgramGroup programGroup)
    {
        // C++ garanti qu'il n'y a pas de padding avant le premier membre
        // this == &this->header
        optixSbtRecordPackHeader(programGroup, this);
    }
};

using RaygenRecord = Record<uchar3>; // Triangle RGB