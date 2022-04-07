#pragma once

#include "patterns.h"
#include "Context.h"

enum SsaaType {
    SSAA_REGULAR,
    SSAA_RANDOM,

    SSAA_Count
};

extern const char* const ssaaNames[SSAA_Count];


inline __device__ __host__ void ssaaApply(SsaaType type, SsaaContext& context)
{
    switch(type)
    {
        case SSAA_REGULAR: ssaa_regular(context); break;
        case SSAA_RANDOM: ssaa_random(context); break;
    }
}

inline __device__ __host__ bool ssaaIs2D(SsaaType type)
{
    switch(type)
    {
        case SSAA_REGULAR: return true;
        case SSAA_RANDOM: return false;
        default: return false;
    }
}