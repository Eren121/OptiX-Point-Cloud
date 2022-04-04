#include "check.h"
#include <cstdio>
#include <cstdlib>

void checkImplCuda(cudaError_t result, const char* file, int line)
{
    if(result != cudaSuccess)
    {
        std::fprintf(stderr, "%s:%d\n", file, line);
        std::fprintf(stderr, "CUDA error: %s", cudaGetErrorString(result));
        std::exit(1);
    }
}