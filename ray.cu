#include "ray.cuh"
#include <optix.h>

extern "C"
{
    __global__ void __raygen__my_program()
    {
        const uint3 idx = optixGetLaunchIndex();
        const uint3 dim = optixGetLaunchDimensions();

        //printf("%d, %d\n", params.width, params.height);
        params.image[0] = make_uchar4(1, 2, 3, 4);
    }

    __global__ void __intersection_my_program()
    {

    }

    __global__ void __anyhit__my_program()
    {

    }

    __global__ void __closesthit_my_program()
    {

    }

    __global__ void __miss__my_program()
    {
        
    }
}