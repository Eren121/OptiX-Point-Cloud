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

#endif /* HELPER_CONVERSION_H */
