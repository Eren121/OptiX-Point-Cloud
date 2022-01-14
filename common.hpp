#pragma once

#include <iostream>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define CUDA_CHECK(expr) cudaCheckImpl((expr), __FILE__, __LINE__)
#define OPTIX_CHECK(expr) optixCheckImpl((expr), __FILE__, __LINE__)

void cudaCheckImpl(cudaError_t result, const char *file, int line)
{
    if(result != cudaSuccess)
    {
        std::cerr << file << ":" << line << std::endl;
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

void optixCheckImpl(OptixResult result, const char *file, int line)
{
    if(result != OPTIX_SUCCESS)
    {
        std::cerr << file << ":" << line << std::endl;
        std::cerr << "OptiX error: " << optixGetErrorString(result) << std::endl;
        exit(1);
    }
}

