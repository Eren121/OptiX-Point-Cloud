#pragma once

#include "cuda_runtime.h"
#include <glm/glm.hpp>
#include "core/cuda/math.h"

inline __device__ __host__ uchar3 convertFloatToCharColor(const float3& color)
{
    return make_uchar3(
        glm::clamp(color.x, 0.0f, 1.0f) * 255.0f,
        glm::clamp(color.y, 0.0f, 1.0f) * 255.0f,
        glm::clamp(color.z, 0.0f, 1.0f) * 255.0f
    );
}

inline __device__ __host__ float3 convertCharToFloatColor(const uchar3& color)
{
    return make_float3(
        color.x / 255.0f,
        color.y / 255.0f,
        color.z / 255.0f
    );
}