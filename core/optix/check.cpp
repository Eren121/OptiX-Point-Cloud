#include "check.h"
#include <cstdio>
#include <cstdlib>

void checkImplOptiX(OptixResult result, const char* file, int line)
{
    if(result != OPTIX_SUCCESS)
    {
        std::fprintf(stderr, "%s:%d\n", file, line);
        std::fprintf(stderr, "OptiX error: %s", optixGetErrorString(result));
        std::exit(1);
    }
}