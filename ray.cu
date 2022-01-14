#include "ray.cuh"
#include <optix.h>

extern "C"
{
    __global__ void __raygen__my_program()
    {
        const uint3 idx = optixGetLaunchIndex();
        const uint3 dim = optixGetLaunchDimensions();
        params.image[0] = make_uchar4(1, 2, 3, 4);
    }
}