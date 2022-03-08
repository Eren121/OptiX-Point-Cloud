#ifndef HELPER_CONVERSION_H
#define HELPER_CONVERSION_H

#include <cuda_runtime.h>

inline __host__ __device__ int uchar4_as_int(uchar4 u)
{
    return u.x | u.y << 8 | u.z << 16 | u.w << 24;
}

inline __host__ __device__ uchar4 int_as_uchar4(int i)
{
    uchar4 u;
    u.x = i & 0xff;
    u.y = (i >> 8) & 0xff;
    u.z = (i >> 16) & 0xff;
    u.w = (i >> 24) & 0xff;
    return u;
}

inline __host__ __device__ int uchar3_as_int(uchar3 u)
{
    return u.x | u.y << 8 | u.z << 16;
}

inline __host__ __device__ uchar3 int_as_uchar3(int i)
{
    uchar3 u;
    u.x = i & 0xff;
    u.y = (i >> 8) & 0xff;
    u.z = (i >> 16) & 0xff;
    return u;
}

#define declare_vector_conversion_function(type) \
    template<typename T> \
    __device__ __host__ type as_##type(const T& vec) \
    { return make_##type(vec.x, vec.y, vec.z); }

declare_vector_conversion_function(int3)
declare_vector_conversion_function(uchar3)
declare_vector_conversion_function(float3)

#endif /* HELPER_CONVERSION_H */
