#include "common.hpp"

// Cet include, nécessaire, définit une variable globale OptiX "g_optixFunctionTable"
#include <optix_function_table_definition.h>

void checkImplCuda(cudaError_t result, const char *file, int line)
{
    if(result != cudaSuccess)
    {
        fprintf(stderr, "%s:%d\n", file, line);
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(result));
        exit(1);
    }
}

void checkImplOptiX(OptixResult result, const char *file, int line)
{
    if(result != OPTIX_SUCCESS)
    {
        fprintf(stderr, "%s:%d\n", file, line);
        fprintf(stderr, "OptiX error: %s", optixGetErrorString(result));
        exit(1);
    }
}